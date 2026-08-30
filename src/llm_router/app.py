import hashlib
import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from llm_router.admission import (
    AdmissionController,
    AdmissionRejectedError,
    QuotaExceededError,
    SlidingWindowQuota,
)
from llm_router.backends import (
    BackendResult,
    BackendUnavailableError,
    InferenceBackend,
    MockInferenceBackend,
    VLLMBackend,
)
from llm_router.caching import (
    CachedCompletion,
    CacheStore,
    InMemoryCacheStore,
    PrefixTracker,
    RouterDecisionCache,
    SemanticCache,
    build_cache_key,
    catalog_fingerprint,
    exact_cache_eligible,
    semantic_cache_eligible,
)
from llm_router.config import Settings, get_settings
from llm_router.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    RouteDecision,
    Usage,
)
from llm_router.observability import Metrics
from llm_router.redis_state import RedisCacheStore, RedisFixedWindowQuota, RedisLike
from llm_router.registry import Registry, RegistryError, catalog_revisions, load_registry
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles


def _redis_client(settings: Settings) -> RedisLike | None:
    """Build a shared-state client when a Redis URL is configured."""

    if not settings.redis_url:
        return None
    from redis.asyncio import Redis  # imported lazily so the extra stays optional

    client: RedisLike = Redis.from_url(settings.redis_url)
    return client


def _load_catalog(path: str) -> Registry | None:
    """Load the governed catalog, falling back to built-in profiles when absent."""

    if not Path(path).exists():
        return None
    return load_registry(path)


def create_app(
    settings: Settings | None = None,
    *,
    backend: InferenceBackend | None = None,
    metrics: Metrics | None = None,
    cache_store: CacheStore | None = None,
    registry: Registry | None = None,
    redis_client: RedisLike | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    catalog = registry if registry is not None else _load_catalog(runtime_settings.registry_path)
    profiles = catalog.profiles() if catalog is not None else default_model_profiles()
    policy_version = (
        catalog.policy.version if catalog is not None else runtime_settings.routing_policy_version
    )
    router = Router(
        profiles=profiles,
        external_fallback_enabled=runtime_settings.external_fallback_enabled,
        registry=catalog,
    )
    admission = AdmissionController(
        runtime_settings.max_concurrency,
        runtime_settings.admission_timeout_seconds,
    )
    shared_state = redis_client if redis_client is not None else _redis_client(runtime_settings)
    quota = SlidingWindowQuota(runtime_settings.quota_requests_per_minute)
    shared_quota = (
        RedisFixedWindowQuota(shared_state, runtime_settings.quota_requests_per_minute)
        if shared_state is not None
        else None
    )
    engine_client = (
        httpx.AsyncClient() if backend is None and runtime_settings.backend == "vllm" else None
    )
    inference_backend: InferenceBackend = backend or (
        VLLMBackend(
            base_url=runtime_settings.vllm_base_url,
            client=engine_client,
            request_timeout_seconds=runtime_settings.backend_timeout_seconds,
        )
        if engine_client is not None
        else MockInferenceBackend()
    )
    telemetry = metrics if metrics is not None else Metrics()
    exact_cache: CacheStore = (
        cache_store
        or (
            RedisCacheStore(shared_state, ttl_seconds=runtime_settings.cache_ttl_seconds)
            if shared_state is not None
            else None
        )
        or InMemoryCacheStore(
            max_entries=runtime_settings.cache_max_entries,
            ttl_seconds=runtime_settings.cache_ttl_seconds,
        )
    )
    semantic_cache = SemanticCache(threshold=runtime_settings.semantic_similarity_threshold)
    decision_cache = RouterDecisionCache(policy_version=policy_version)
    prefix_tracker = PrefixTracker()
    revisions = (
        catalog_revisions(catalog)
        if catalog is not None
        else (profile.revision for profile in router.profiles)
    )
    catalog_version = catalog_fingerprint(revisions, policy_version)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = True
        yield
        app.state.ready = False
        if engine_client is not None:
            await engine_client.aclose()

    app = FastAPI(
        title="Local LLM Inference Router",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def authenticate(authorization: str | None = Header(default=None)) -> str:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.removeprefix(prefix)
        if not any(
            secrets.compare_digest(token, candidate)
            for candidate in runtime_settings.accepted_api_keys
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return hashlib.sha256(token.encode()).hexdigest()

    @app.exception_handler(NoEligibleModelError)
    async def no_model_handler(_: Request, error: NoEligibleModelError) -> JSONResponse:
        telemetry.record_rejection("no_eligible_model")
        return JSONResponse(status_code=422, content={"error": {"message": str(error)}})

    @app.exception_handler(AdmissionRejectedError)
    async def admission_handler(_: Request, error: AdmissionRejectedError) -> JSONResponse:
        telemetry.record_rejection("overloaded")
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "1"},
            content={"error": {"message": str(error), "type": "overloaded"}},
        )

    @app.exception_handler(BackendUnavailableError)
    async def backend_handler(_: Request, error: BackendUnavailableError) -> JSONResponse:
        telemetry.record_rejection("backend_unavailable")
        return JSONResponse(
            status_code=502,
            headers={"Retry-After": "5"},
            content={"error": {"message": str(error), "type": "backend_unavailable"}},
        )

    @app.exception_handler(QuotaExceededError)
    async def quota_handler(_: Request, error: QuotaExceededError) -> JSONResponse:
        telemetry.record_rejection("quota_exceeded")
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"error": {"message": str(error), "type": "quota_exceeded"}},
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/readyz")
    async def readiness(request: Request) -> dict[str, str]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="not ready")
        if not await inference_backend.healthy():
            raise HTTPException(status_code=503, detail="inference backend is unhealthy")
        return {"status": "ready"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        payload, content_type = telemetry.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models() -> dict[str, object]:
        cards = {card.id: card for card in catalog.servable_models()} if catalog else {}
        visible: list[dict[str, object]] = []
        for profile in router.profiles:
            if not profile.local and not runtime_settings.external_fallback_enabled:
                continue
            entry: dict[str, object] = {
                "id": profile.id,
                "object": "model",
                "owned_by": "local" if profile.local else "external-policy",
                "revision": profile.revision,
                "healthy": profile.healthy,
                "context_limit": profile.context_limit,
            }
            card = cards.get(profile.id)
            if card is not None:
                entry.update(
                    {
                        "tier": card.tier.value,
                        "license": card.license,
                        "tokenizer": card.tokenizer,
                        "quantization": card.quantization.value,
                        "stage": card.stage.value,
                    }
                )
            visible.append(entry)
        return {"object": "list", "data": visible}

    @app.get("/v1/registry/models/{model_id}", dependencies=[Depends(authenticate)])
    async def model_card(model_id: str) -> dict[str, object]:
        if catalog is None:
            raise HTTPException(status_code=404, detail="no catalog is configured")
        try:
            card = catalog.model_card(model_id)
        except RegistryError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        payload = card.model_dump(mode="json")
        payload["benchmarks"] = [
            run.model_dump(mode="json") for run in catalog.benchmarks_for(card.revision)
        ]
        return payload

    @app.get("/v1/registry/adapters", dependencies=[Depends(authenticate)])
    async def adapters() -> dict[str, object]:
        if catalog is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [adapter.model_dump(mode="json") for adapter in catalog.servable_adapters()],
        }

    @app.get("/v1/registry/deployments", dependencies=[Depends(authenticate)])
    async def deployments() -> dict[str, object]:
        if catalog is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [
                {
                    **revision.model_dump(mode="json"),
                    "rollback_target": (
                        target.id
                        if (target := catalog.rollback_target(revision.id)) is not None
                        else None
                    ),
                }
                for revision in catalog.deployments
            ],
        }

    def _completion_response(
        *,
        model_id: str,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        routing: dict[str, object],
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="length" if finish_reason == "length" else "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            routing=routing,
        )

    def _cached_stream(entry: CachedCompletion, hit_name: str) -> StreamingResponse:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        async def iterator() -> AsyncIterator[str]:
            yield _chunk(completion_id, created, entry.model_id, delta={"role": "assistant"})
            yield _chunk(completion_id, created, entry.model_id, delta={"content": entry.text})
            yield _chunk(completion_id, created, entry.model_id, delta={}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            iterator(),
            media_type="text/event-stream",
            headers={
                "X-Cache": hit_name,
                "X-Route-Model": entry.model_id,
                "X-Route-Revision": entry.model_revision,
            },
        )

    async def _lookup_cache(
        payload: ChatCompletionRequest, prompt: str, cache_key: str, tenant: str
    ) -> tuple[str, CachedCompletion] | None:
        if not runtime_settings.cache_enabled:
            return None
        if exact_cache_eligible(payload):
            entry = await exact_cache.get(cache_key)
            telemetry.record_cache_event("exact", "hit" if entry is not None else "miss")
            if entry is not None:
                return "exact", entry
        if (
            runtime_settings.semantic_cache_enabled
            and payload.routing.task is not None
            and semantic_cache_eligible(payload, payload.routing.task)
        ):
            scope = semantic_cache.scope(payload, tenant=tenant, model_revision=catalog_version)
            match = semantic_cache.lookup(scope, prompt)
            telemetry.record_cache_event("semantic", "hit" if match is not None else "miss")
            if match is not None:
                return "semantic", match
        return None

    async def _store_cache(
        payload: ChatCompletionRequest,
        prompt: str,
        cache_key: str,
        tenant: str,
        decision: RouteDecision,
        result: BackendResult,
    ) -> None:
        if not runtime_settings.cache_enabled:
            return
        entry = CachedCompletion(
            text=result.text,
            model_id=decision.profile.id,
            model_revision=decision.profile.revision,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        if exact_cache_eligible(payload):
            await exact_cache.set(cache_key, entry)
        if runtime_settings.semantic_cache_enabled and semantic_cache_eligible(
            payload, decision.task
        ):
            scope = semantic_cache.scope(payload, tenant=tenant, model_revision=catalog_version)
            semantic_cache.store(scope, prompt, entry)

    async def _consume_quota(subject: str) -> None:
        if shared_quota is None:
            await quota.consume(subject)
            return
        window = int(time.time() // 60)
        if not await shared_quota.consume(subject, window=window):
            raise QuotaExceededError("request quota exceeded")

    def _route_headers(decision: RouteDecision, cache_state: str) -> dict[str, str]:
        headers = {
            "X-Cache": cache_state,
            "X-Route-Model": decision.profile.id,
            "X-Route-Revision": decision.profile.revision,
            "X-Route-Reason": decision.reason,
        }
        if decision.adapter_id is not None:
            headers["X-Route-Adapter"] = decision.adapter_id
        return headers

    def _chunk(completion_id: str, created: int, model_id: str, **choice: object) -> str:
        document = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, **choice}],
        }
        return f"data: {json.dumps(document)}\n\n"

    async def _stream_completion(
        payload: ChatCompletionRequest,
        prompt: str,
        cache_key: str,
        subject: str,
        decision: RouteDecision,
        started: float,
        queue_seconds: float,
    ) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_id = decision.profile.id
        collected: list[str] = []

        yield _chunk(completion_id, created, model_id, delta={"role": "assistant"})
        async for delta in inference_backend.stream(payload, decision):
            collected.append(delta)
            yield _chunk(completion_id, created, model_id, delta={"content": delta})
        yield _chunk(completion_id, created, model_id, delta={}, finish_reason="stop")
        yield "data: [DONE]\n\n"

        text = "".join(collected)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        telemetry.record_completion(
            decision,
            latency_seconds=time.perf_counter() - started,
            queue_seconds=queue_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await _store_cache(
            payload,
            prompt,
            cache_key,
            subject,
            decision,
            BackendResult(
                text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        payload: ChatCompletionRequest,
        response: Response,
        subject: str = Depends(authenticate),
    ) -> ChatCompletionResponse | StreamingResponse:
        started = time.perf_counter()
        await _consume_quota(subject)
        prompt = payload.prompt
        cache_key = build_cache_key(
            payload, tenant=subject, model_revision=catalog_version, prompt=prompt
        )

        cached = await _lookup_cache(payload, prompt, cache_key, subject)
        if cached is not None:
            hit_name, entry = cached
            if payload.stream:
                return _cached_stream(entry, hit_name)
            response.headers["X-Cache"] = hit_name
            response.headers["X-Route-Model"] = entry.model_id
            response.headers["X-Route-Revision"] = entry.model_revision
            return _completion_response(
                model_id=entry.model_id,
                text=entry.text,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                routing={
                    "model_revision": entry.model_revision,
                    "cache": hit_name,
                    "reason": f"served from the {hit_name} cache",
                },
            )

        cached_task = decision_cache.get(prompt) if runtime_settings.cache_enabled else None
        telemetry.record_cache_event("router", "hit" if cached_task is not None else "miss")
        decision = router.select(payload, task=cached_task)
        if runtime_settings.cache_enabled:
            decision_cache.set(prompt, decision.task)
        telemetry.record_route(decision, privacy=payload.routing.privacy.value)
        telemetry.record_cache_event(
            "prefix",
            "hit"
            if prefix_tracker.observe(prompt, model_revision=decision.profile.revision)
            else "miss",
        )

        telemetry.queued_requests.inc()
        try:
            async with admission.slot():
                telemetry.queued_requests.dec()
                queue_seconds = time.perf_counter() - started
                telemetry.inflight_requests.inc()
                try:
                    if payload.stream:
                        return StreamingResponse(
                            _stream_completion(
                                payload,
                                prompt,
                                cache_key,
                                subject,
                                decision,
                                started,
                                queue_seconds,
                            ),
                            media_type="text/event-stream",
                            headers=_route_headers(decision, "miss"),
                        )
                    result = await inference_backend.generate(payload, decision)
                finally:
                    telemetry.inflight_requests.dec()
        except AdmissionRejectedError:
            telemetry.queued_requests.dec()
            raise

        telemetry.record_completion(
            decision,
            latency_seconds=time.perf_counter() - started,
            queue_seconds=queue_seconds,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

        await _store_cache(payload, prompt, cache_key, subject, decision, result)

        response.headers["X-Cache"] = "miss"
        if decision.adapter_id is not None:
            response.headers["X-Route-Adapter"] = decision.adapter_id
        response.headers["X-Route-Model"] = decision.profile.id
        response.headers["X-Route-Revision"] = decision.profile.revision
        response.headers["X-Route-Reason"] = decision.reason
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=decision.profile.id,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason="length" if result.finish_reason == "length" else "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            routing={
                "model_revision": decision.profile.revision,
                "adapter_id": decision.adapter_id,
                "adapter_revision": decision.adapter_revision,
                "task": decision.task.value,
                "reason": decision.reason,
                "score": decision.score,
                "candidate_count": decision.candidate_count,
            },
        )

    return app


app = create_app()
