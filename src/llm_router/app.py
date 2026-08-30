import hashlib
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from llm_router.admission import (
    AdmissionController,
    AdmissionRejectedError,
    QuotaExceededError,
    SlidingWindowQuota,
)
from llm_router.backends import BackendResult, InferenceBackend, MockInferenceBackend
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
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles


def create_app(
    settings: Settings | None = None,
    *,
    backend: InferenceBackend | None = None,
    metrics: Metrics | None = None,
    cache_store: CacheStore | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    router = Router(
        profiles=default_model_profiles(),
        external_fallback_enabled=runtime_settings.external_fallback_enabled,
    )
    admission = AdmissionController(
        runtime_settings.max_concurrency,
        runtime_settings.admission_timeout_seconds,
    )
    quota = SlidingWindowQuota(runtime_settings.quota_requests_per_minute)
    inference_backend = backend or MockInferenceBackend()
    telemetry = metrics if metrics is not None else Metrics()
    exact_cache: CacheStore = cache_store or InMemoryCacheStore(
        max_entries=runtime_settings.cache_max_entries,
        ttl_seconds=runtime_settings.cache_ttl_seconds,
    )
    semantic_cache = SemanticCache(threshold=runtime_settings.semantic_similarity_threshold)
    decision_cache = RouterDecisionCache(policy_version=runtime_settings.routing_policy_version)
    prefix_tracker = PrefixTracker()
    catalog_version = catalog_fingerprint(
        (profile.revision for profile in router.profiles),
        runtime_settings.routing_policy_version,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ready = True
        yield
        app.state.ready = False

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
        return {"status": "ready"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        payload, content_type = telemetry.render()
        return Response(content=payload, media_type=content_type)

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models() -> dict[str, object]:
        visible = [
            {
                "id": profile.id,
                "object": "model",
                "owned_by": "local" if profile.local else "external-policy",
                "revision": profile.revision,
                "healthy": profile.healthy,
            }
            for profile in router.profiles
            if profile.local or runtime_settings.external_fallback_enabled
        ]
        return {"object": "list", "data": visible}

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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        payload: ChatCompletionRequest,
        response: Response,
        subject: str = Depends(authenticate),
    ) -> ChatCompletionResponse:
        started = time.perf_counter()
        await quota.consume(subject)
        prompt = payload.prompt
        cache_key = build_cache_key(
            payload, tenant=subject, model_revision=catalog_version, prompt=prompt
        )

        cached = await _lookup_cache(payload, prompt, cache_key, subject)
        if cached is not None:
            hit_name, entry = cached
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
                "task": decision.task.value,
                "reason": decision.reason,
                "score": decision.score,
                "candidate_count": decision.candidate_count,
            },
        )

    return app


app = create_app()
