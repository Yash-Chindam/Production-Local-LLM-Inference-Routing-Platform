import hashlib
import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace import TracerProvider

from llm_router.admission import (
    AdmissionController,
    AdmissionRejectedError,
    QuotaExceededError,
    SlidingWindowQuota,
)
from llm_router.backends import (
    BackendOutOfMemoryError,
    BackendResult,
    BackendUnavailableError,
    DispatchingBackend,
    ExternalDispatchRefusedError,
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
from llm_router.canary import CanaryMonitor, canary_plans
from llm_router.classifier import TaskClassifier, load_classifier
from llm_router.config import Settings, get_settings
from llm_router.credentials import (
    CredentialError,
    TokenVerifier,
    build_verifier,
    looks_like_a_token,
)
from llm_router.engine_stats import ColdStartTracker, EngineStatsCollector
from llm_router.evaluation import structured_output_valid
from llm_router.load import LoadTracker
from llm_router.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    PrivacyClass,
    RouteDecision,
    Usage,
)
from llm_router.observability import Metrics
from llm_router.redis_state import RedisCacheStore, RedisFixedWindowQuota, RedisLike
from llm_router.registry import (
    Registry,
    RegistryError,
    TenantRecord,
    catalog_revisions,
    load_registry,
    strictest_privacy,
)
from llm_router.resilience import CircuitBreaker, EngineCircuitOpenError, ResilientBackend
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles
from llm_router.tracing import RequestSpan, Tracing, build_tracer_provider


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: which tenant, and which credential proved it.

    Quota and cache scope use the tenant rather than the credential, so
    rotating a key neither resets a tenant's quota nor orphans its cache.
    """

    tenant_id: str
    credential_fingerprint: str


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


def _load_task_classifier(path: str) -> TaskClassifier | None:
    """Train the routing classifier, or fall back to keyword rules when absent."""

    if not Path(path).exists():
        return None
    return load_classifier(path)


def create_app(
    settings: Settings | None = None,
    *,
    backend: InferenceBackend | None = None,
    metrics: Metrics | None = None,
    cache_store: CacheStore | None = None,
    registry: Registry | None = None,
    redis_client: RedisLike | None = None,
    engine_stats: EngineStatsCollector | None = None,
    tracer_provider: TracerProvider | None = None,
    token_verifier: TokenVerifier | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    verifier = token_verifier or build_verifier(
        issuer=runtime_settings.jwt_issuer,
        audience=runtime_settings.jwt_audience,
        jwks=runtime_settings.jwt_jwks,
        jwks_url=runtime_settings.jwt_jwks_url,
        max_lifetime_seconds=runtime_settings.jwt_max_lifetime_seconds,
        tenant_claim=runtime_settings.jwt_tenant_claim,
    )
    catalog = registry if registry is not None else _load_catalog(runtime_settings.registry_path)
    profiles = catalog.profiles() if catalog is not None else default_model_profiles()
    policy_version = (
        catalog.policy.version if catalog is not None else runtime_settings.routing_policy_version
    )
    load = LoadTracker()
    plans = canary_plans(catalog) if catalog is not None else ()
    canary = CanaryMonitor(plans)
    router = Router(
        profiles=profiles,
        external_fallback_enabled=runtime_settings.external_fallback_enabled,
        registry=catalog,
        classifier=_load_task_classifier(runtime_settings.task_classifier_path),
        load=load,
        canary=canary,
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
    raw_backend: InferenceBackend = backend or (
        VLLMBackend(
            base_url=runtime_settings.vllm_base_url,
            client=engine_client,
            request_timeout_seconds=runtime_settings.backend_timeout_seconds,
        )
        if engine_client is not None
        else MockInferenceBackend()
    )
    circuit = CircuitBreaker(
        failure_threshold=runtime_settings.engine_failure_threshold,
        cooldown_seconds=runtime_settings.engine_cooldown_seconds,
    )
    external_client = (
        httpx.AsyncClient() if backend is None and runtime_settings.external_base_url else None
    )
    # An injected or mock backend answers external routes too, which keeps
    # tests and local development free of a provider.
    raw_external: InferenceBackend = (
        VLLMBackend(
            base_url=runtime_settings.external_base_url,
            client=external_client,
            request_timeout_seconds=runtime_settings.backend_timeout_seconds,
            api_key=runtime_settings.external_api_key,
            health_path="/health/liveliness",
        )
        if external_client is not None
        else raw_backend
    )
    # Each target has its own circuit: a lost GPU node must not close the
    # route to the provider, nor a provider outage the route to the engine.
    external_circuit = CircuitBreaker(
        failure_threshold=runtime_settings.engine_failure_threshold,
        cooldown_seconds=runtime_settings.engine_cooldown_seconds,
    )
    inference_backend: InferenceBackend = DispatchingBackend(
        local=ResilientBackend(raw_backend, circuit),
        external=ResilientBackend(raw_external, external_circuit),
    )
    telemetry = metrics if metrics is not None else Metrics()
    engine_telemetry = engine_stats or (
        EngineStatsCollector(base_url=runtime_settings.vllm_base_url, client=engine_client)
        if engine_client is not None
        else None
    )
    cold_start = ColdStartTracker()
    tracing = Tracing(
        tracer_provider or build_tracer_provider(runtime_settings),
        record_prompt_content=runtime_settings.trace_prompt_content,
    )
    # One gateway deployment faces one engine target, so engine-level telemetry
    # and cold starts are attributed to that target rather than to a model.
    engine_label = runtime_settings.backend
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
        # Readiness fails first so no new traffic arrives, then admitted
        # requests are given the grace period to finish before the engine
        # client they depend on is closed underneath them.
        app.state.ready = False
        await admission.drain(runtime_settings.shutdown_grace_seconds)
        if engine_client is not None:
            await engine_client.aclose()
        if external_client is not None:
            await external_client.aclose()

    app = FastAPI(
        title="Local LLM Inference Router",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def authenticate(authorization: str | None = Header(default=None)) -> Principal:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.removeprefix(prefix)
        if verifier is not None and looks_like_a_token(token):
            try:
                verified = await verifier.verify(token)
            except CredentialError as error:
                # A token that fails is refused outright; it is never retried
                # as a static key.
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(error),
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                ) from error
            # The subject, not the token, identifies the caller: a renewed
            # token is the same caller.
            identity = f"{verifier.issuer}|{verified.subject}"
            return Principal(
                tenant_id=verified.tenant_id,
                credential_fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
            )
        if runtime_settings.require_short_lived_credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="a short-lived token is required; static keys are not accepted",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Every candidate is compared so the work does not depend on which
        # credential matched, and the match itself stays constant time.
        matched: str | None = None
        for candidate, tenant_id in runtime_settings.tenant_by_key.items():
            if secrets.compare_digest(token, candidate):
                matched = tenant_id
        if matched is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal(
            tenant_id=matched,
            credential_fingerprint=hashlib.sha256(token.encode()).hexdigest(),
        )

    def _entitlement(principal: Principal) -> TenantRecord | None:
        return catalog.tenant(principal.tenant_id) if catalog is not None else None

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

    @app.exception_handler(ExternalDispatchRefusedError)
    async def refused_handler(_: Request, error: ExternalDispatchRefusedError) -> JSONResponse:
        # Reaching here means routing chose an external model for data that
        # must stay local. The request is refused, and counted so it is seen.
        telemetry.record_rejection("external_dispatch_refused")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(error), "type": "policy_violation"}},
        )

    @app.exception_handler(BackendOutOfMemoryError)
    async def out_of_memory_handler(_: Request, error: BackendOutOfMemoryError) -> JSONResponse:
        # Not a retry-as-is condition: the same request at the same size will
        # fail again, so the message says what to change.
        telemetry.record_rejection("engine_out_of_memory")
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "10"},
            content={"error": {"message": str(error), "type": "engine_out_of_memory"}},
        )

    @app.exception_handler(EngineCircuitOpenError)
    async def circuit_handler(_: Request, error: EngineCircuitOpenError) -> JSONResponse:
        telemetry.record_rejection("engine_unavailable")
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(max(1, round(error.retry_after_seconds)))},
            content={"error": {"message": str(error), "type": "engine_unavailable"}},
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
        healthy = await inference_backend.healthy()
        # The probe is the one place that sees the engine go from loading to
        # serving, so cold start is measured here instead of being configured.
        loaded_seconds = cold_start.observe(healthy=healthy, now=time.perf_counter())
        if loaded_seconds is not None:
            telemetry.record_model_load(engine_label, loaded_seconds)
        if not healthy:
            raise HTTPException(status_code=503, detail="inference backend is unhealthy")
        return {"status": "ready"}

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        # Engine and accelerator state is pulled through on scrape so the
        # gateway stays the single scrape target for the whole serving path and
        # no background poller runs when nobody is collecting.
        telemetry.record_circuit_state(circuit.state, engine=engine_label)
        telemetry.record_circuit_state(external_circuit.state, engine="external")
        if engine_telemetry is not None:
            stats = await engine_telemetry.sample()
            if stats is not None:
                telemetry.record_engine_stats(stats, engine=engine_label)
                load.observe_engine(stats)
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

    @app.get("/v1/registry/variants", dependencies=[Depends(authenticate)])
    async def variants() -> dict[str, object]:
        """Every optimization variant with its measured delta, or none yet."""

        if catalog is None:
            return {"object": "list", "data": []}
        data: list[dict[str, object]] = []
        for variant in catalog.variants:
            verdict = catalog.variant_verdict(variant)
            data.append(
                {
                    **variant.model_dump(mode="json"),
                    "verdict": (
                        None
                        if verdict is None
                        else {**verdict.model_dump(mode="json"), "accepted": verdict.accepted}
                    ),
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/registry/canaries", dependencies=[Depends(authenticate)])
    async def canaries() -> dict[str, object]:
        """Every canary plan by track, with live state for the adapter track."""

        return {
            "object": "list",
            "data": [
                {
                    **plan.model_dump(mode="json"),
                    "live": (
                        canary.status(plan.subject) if plan.subject in canary.subjects else None
                    ),
                }
                for plan in plans
            ],
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
        # A canary's responses are never cached: if it is rolled back, nothing
        # it produced may keep being served from the cache afterwards.
        if not runtime_settings.cache_enabled or decision.canary_arm == "canary":
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

    def _record_canary(
        decision: RouteDecision, *, ok: bool, started: float, quality: float | None = None
    ) -> None:
        """Feed a canaried route's outcome to the monitor, suspending on failure."""

        if decision.canary_subject is None or decision.canary_arm is None:
            return
        on_canary = decision.canary_arm == "canary"
        telemetry.record_canary(decision.canary_subject, arm=decision.canary_arm, ok=ok)
        verdict = canary.record(
            decision.canary_subject,
            canary=on_canary,
            ok=ok,
            latency_ms=(time.perf_counter() - started) * 1000,
            quality=quality,
        )
        if verdict is not None and verdict.action == "rollback":
            telemetry.record_canary_rollback(decision.canary_subject)

    def _fallback_route(
        payload: ChatCompletionRequest,
        failed: RouteDecision,
        tenant: TenantRecord | None,
        privacy_raised_from: PrivacyClass | None,
    ) -> RouteDecision | None:
        """Find an approved external route for a request the local engine failed.

        Only a local failure falls back, and only to a model the request was
        already entitled to: the same privacy, tenant, operator, and opt-in
        rules apply as on the first attempt.
        """

        if not failed.profile.local:
            return None
        try:
            return router.select(
                payload,
                task=failed.task,
                permitted_models=(
                    catalog.permitted_models_for(tenant) if catalog is not None else None
                ),
                quality_floor=tenant.quality_floor if tenant is not None else 0.0,
                tenant_allows_external=(
                    tenant.allow_external_fallback is not False if tenant is not None else True
                ),
                privacy_raised_from=privacy_raised_from,
                fallback_from=failed.profile.id,
            )
        except NoEligibleModelError:
            return None

    async def _consume_quota(subject: str, tenant: TenantRecord | None) -> None:
        # A tenant may carry its own limit; absent one the platform default
        # applies, which is why None means "defer" rather than "unlimited".
        limit = tenant.quota_requests_per_minute if tenant is not None else None
        if shared_quota is None:
            await quota.consume(subject, limit=limit)
            return
        window = int(time.time() // 60)
        if not await shared_quota.consume(subject, window=window, limit=limit):
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

    class _StreamLease:
        """What a live stream holds until it ends: a slot, a gauge, a span."""

        def __init__(self, span: RequestSpan) -> None:
            self._span = span
            self._closed = False

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            telemetry.inflight_requests.dec()
            admission.release()
            self._span.end()

    class _LeasedStreamingResponse(StreamingResponse):
        """Releases the lease even if the body iterator is never started.

        A generator that is never iterated never runs its own cleanup, which
        happens when the client disconnects before the first chunk. The
        response is always invoked, so it closes the lease as well.
        """

        def __init__(self, *args: Any, lease: _StreamLease, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._lease = lease

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                self._lease.close()

    async def _stream_completion(
        payload: ChatCompletionRequest,
        prompt: str,
        cache_key: str,
        subject: str,
        decision: RouteDecision,
        started: float,
        queue_seconds: float,
        span: RequestSpan,
        lease: "_StreamLease",
    ) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_id = decision.profile.id
        collected: list[str] = []

        # The stream outlives the handler, so it owns the span from here and
        # ends it whether generation finishes, fails, or the client leaves.
        try:
            yield _chunk(completion_id, created, model_id, delta={"role": "assistant"})
            async for delta in inference_backend.stream(payload, decision):
                collected.append(delta)
                yield _chunk(completion_id, created, model_id, delta={"content": delta})
            yield _chunk(completion_id, created, model_id, delta={}, finish_reason="stop")
            yield "data: [DONE]\n\n"

            text = "".join(collected)
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)
            span.set_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            validity: float | None = None
            if payload.routing.structured:
                valid = structured_output_valid(text)
                validity = 1.0 if valid else 0.0
                telemetry.record_structured_output(decision, valid=valid)
            _record_canary(decision, ok=True, started=started, quality=validity)
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
        except Exception as error:
            span.fail(error)
            if isinstance(error, BackendUnavailableError):
                _record_canary(decision, ok=False, started=started)
            raise
        finally:
            lease.close()

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        payload: ChatCompletionRequest,
        response: Response,
        principal: Annotated[Principal, Depends(authenticate)],
    ) -> ChatCompletionResponse | StreamingResponse:
        span = tracing.start_request()
        try:
            result = await _complete(payload, response, principal, span)
        except Exception as error:
            span.fail(error)
            raise
        finally:
            if not span.handed_off:
                span.end()
        trace_id = span.trace_id
        if trace_id is not None:
            # A response returned directly does not inherit injected headers.
            target = result if isinstance(result, Response) else response
            target.headers["X-Trace-Id"] = trace_id
        return result

    async def _complete(
        payload: ChatCompletionRequest,
        response: Response,
        principal: Principal,
        span: RequestSpan,
    ) -> ChatCompletionResponse | StreamingResponse:
        started = time.perf_counter()
        tenant = _entitlement(principal)
        # Quota and cache are scoped to the tenant, not the credential, so
        # rotating a key neither resets a quota nor orphans a cache.
        subject = principal.tenant_id

        declared_privacy = payload.routing.privacy
        if tenant is not None:
            effective_privacy = strictest_privacy(declared_privacy, tenant.minimum_privacy)
            if effective_privacy is not declared_privacy:
                # Raised before anything reads the class, so cache eligibility,
                # the cache key, and routing all agree on one value and a
                # public-declared request can never reach restricted entries.
                payload = payload.model_copy(
                    update={
                        "routing": payload.routing.model_copy(update={"privacy": effective_privacy})
                    }
                )
        privacy_raised_from = (
            declared_privacy if payload.routing.privacy is not declared_privacy else None
        )
        # Described before the quota is charged so a rejected request is still
        # attributed to its tenant, and after the floor so the trace is redacted
        # under the effective class rather than the declared one.
        span.set_request(payload, tenant_id=subject, declared_privacy=declared_privacy)
        await _consume_quota(subject, tenant)

        prompt = payload.prompt
        cache_key = build_cache_key(
            payload, tenant=subject, model_revision=catalog_version, prompt=prompt
        )

        cached = await _lookup_cache(payload, prompt, cache_key, subject)
        span.set_cache("miss" if cached is None else cached[0])
        if cached is not None:
            hit_name, entry = cached
            span.set_served_from_cache(model_id=entry.model_id, model_revision=entry.model_revision)
            span.set_usage(
                prompt_tokens=entry.prompt_tokens, completion_tokens=entry.completion_tokens
            )
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
        decision = router.select(
            payload,
            task=cached_task,
            permitted_models=(
                catalog.permitted_models_for(tenant) if catalog is not None else None
            ),
            quality_floor=tenant.quality_floor if tenant is not None else 0.0,
            tenant_allows_external=(
                tenant.allow_external_fallback is not False if tenant is not None else True
            ),
            privacy_raised_from=privacy_raised_from,
            canary_key=f"{subject}|{prompt}",
        )
        span.set_route(decision)
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
            await admission.acquire()
        finally:
            telemetry.queued_requests.dec()
        queue_seconds = time.perf_counter() - started
        # What this request actually waited becomes the next request's
        # estimate for the same model.
        load.observe_queue(decision.profile.id, queue_seconds * 1000)
        telemetry.inflight_requests.inc()

        if payload.stream:
            # The stream holds its admission slot until it ends. Returning it
            # from inside a slot block would free the slot before generation
            # began, and streamed work would escape the concurrency bound.
            lease = _StreamLease(span)
            span.handed_off = True
            return _LeasedStreamingResponse(
                _stream_completion(
                    payload,
                    prompt,
                    cache_key,
                    subject,
                    decision,
                    started,
                    queue_seconds,
                    span,
                    lease,
                ),
                lease=lease,
                media_type="text/event-stream",
                headers=_route_headers(decision, "miss"),
            )

        try:
            try:
                result = await inference_backend.generate(payload, decision)
            except BackendUnavailableError as error:
                _record_canary(decision, ok=False, started=started)
                fallback = _fallback_route(payload, decision, tenant, privacy_raised_from)
                if fallback is None:
                    raise
                # Declared, attributed, and counted: never a silent switch.
                telemetry.record_fallback(
                    from_model=decision.profile.id,
                    to_model=fallback.profile.id,
                    cause=type(error).__name__,
                )
                telemetry.record_route(fallback, privacy=payload.routing.privacy.value)
                decision = fallback
                span.set_route(decision)
                result = await inference_backend.generate(payload, decision)
        finally:
            telemetry.inflight_requests.dec()
            admission.release()

        span.set_usage(
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens
        )
        validity: float | None = None
        if payload.routing.structured:
            valid = structured_output_valid(result.text)
            validity = 1.0 if valid else 0.0
            telemetry.record_structured_output(decision, valid=valid)
        _record_canary(decision, ok=True, started=started, quality=validity)
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
                "task_source": decision.task_source,
                "task_confidence": decision.task_confidence,
                "complexity": decision.complexity,
                "canary_arm": decision.canary_arm,
                "reason": decision.reason,
                "score": decision.score,
                "candidate_count": decision.candidate_count,
            },
        )

    return app


app = create_app()
