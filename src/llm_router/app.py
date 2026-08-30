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
from llm_router.backends import InferenceBackend, MockInferenceBackend
from llm_router.config import Settings, get_settings
from llm_router.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from llm_router.observability import Metrics
from llm_router.routing import NoEligibleModelError, Router, default_model_profiles


def create_app(
    settings: Settings | None = None,
    *,
    backend: InferenceBackend | None = None,
    metrics: Metrics | None = None,
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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        payload: ChatCompletionRequest,
        response: Response,
        subject: str = Depends(authenticate),
    ) -> ChatCompletionResponse:
        started = time.perf_counter()
        await quota.consume(subject)
        decision = router.select(payload)
        telemetry.record_route(decision, privacy=payload.routing.privacy.value)

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
