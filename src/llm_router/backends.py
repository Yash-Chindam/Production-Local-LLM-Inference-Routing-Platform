"""Inference backends behind the routing decision (sections 7.4 and 10).

`MockInferenceBackend` keeps continuous integration deterministic and GPU-free.
`VLLMBackend` talks to a real vLLM OpenAI-compatible server, serving a LoRA
adapter by name when the router selected one.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from llm_router.models import ChatCompletionRequest, PrivacyClass, RouteDecision


class BackendUnavailableError(RuntimeError):
    """Raised when an inference engine is unreachable or returns an error."""


class BackendOutOfMemoryError(BackendUnavailableError):
    """Raised when the engine reports it ran out of GPU memory for a request."""


OUT_OF_MEMORY_MARKERS = ("out of memory", "outofmemoryerror", "cuda oom")


def engine_failure(status_code: int, body: str, context: str) -> BackendUnavailableError:
    """Classify an engine error response without repeating what it said.

    The body is inspected for an out-of-memory report and then discarded: an
    engine error can echo the prompt it rejected, so it is never forwarded.
    """

    if any(marker in body.lower() for marker in OUT_OF_MEMORY_MARKERS):
        return BackendOutOfMemoryError(
            f"inference engine ran out of GPU memory {context}; "
            "retry with a shorter prompt or a smaller max_tokens"
        )
    return BackendUnavailableError(f"inference engine returned {status_code} {context}")


@dataclass(frozen=True)
class BackendResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class InferenceBackend(Protocol):
    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult: ...

    def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]: ...

    async def healthy(self) -> bool: ...


def served_model_name(decision: RouteDecision) -> str:
    """vLLM serves an adapter under its own name over the shared base model."""

    return decision.adapter_id or decision.profile.id


class MockInferenceBackend:
    """Deterministic backend used until vLLM deployments are configured."""

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        response = f"[{served_model_name(decision)}] accepted {decision.task.value} request"
        prompt_tokens = max(1, len(request.prompt) // 4)
        completion_tokens = max(1, len(response) // 4)
        return BackendResult(
            text=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        result = await self.generate(request, decision)
        for token in result.text.split(" "):
            yield f"{token} "

    async def healthy(self) -> bool:
        return True


@dataclass
class VLLMBackend:
    """Client for a vLLM OpenAI-compatible server, optionally behind Ray Serve."""

    base_url: str
    client: httpx.AsyncClient
    request_timeout_seconds: float = 60.0
    # The same OpenAI-compatible client reaches the LiteLLM proxy that fronts
    # external providers, which authenticates and has its own health path.
    api_key: str = ""
    health_path: str = "/health"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _payload(
        self, request: ChatCompletionRequest, decision: RouteDecision, *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": served_model_name(decision),
            "messages": [message.model_dump() for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=self._payload(request, decision, stream=False),
                headers=self._headers,
                timeout=self.request_timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise BackendUnavailableError(f"inference engine unreachable: {error}") from error

        if response.status_code >= 400:
            raise engine_failure(
                response.status_code, response.text, f"for {served_model_name(decision)}"
            )

        body = response.json()
        try:
            choice = body["choices"][0]
            usage = body.get("usage", {})
            return BackendResult(
                text=choice["message"]["content"],
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                finish_reason=str(choice.get("finish_reason", "stop")),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise BackendUnavailableError("inference engine returned an unusable body") from error

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        payload = self._payload(request, decision, stream=True)
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=self.request_timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise engine_failure(response.status_code, body, "while streaming")
                async for line in response.aiter_lines():
                    delta = _parse_stream_line(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as error:
            raise BackendUnavailableError(f"inference engine unreachable: {error}") from error

    async def healthy(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}{self.health_path}", headers=self._headers, timeout=2.0
            )
        except httpx.HTTPError:
            return False
        return response.status_code < 400


class ExternalDispatchRefusedError(RuntimeError):
    """Raised when a non-public request reaches the external dispatch boundary."""


@dataclass
class DispatchingBackend:
    """Sends each decision to the local engine or the external provider proxy.

    Routing already refuses to pick an external model for private or
    restricted data. The same rule is enforced again here, at the last point
    before a prompt could leave the private environment, so a routing defect
    cannot become a disclosure.
    """

    local: InferenceBackend
    external: InferenceBackend | None = None

    def _target(self, request: ChatCompletionRequest, decision: RouteDecision) -> InferenceBackend:
        if decision.profile.local:
            return self.local
        if request.routing.privacy is not PrivacyClass.PUBLIC:
            raise ExternalDispatchRefusedError(
                f"refused to send a {request.routing.privacy.value} request to an external model"
            )
        if self.external is None:
            raise BackendUnavailableError("no external provider is configured")
        return self.external

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        return await self._target(request, decision).generate(request, decision)

    async def stream(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> AsyncIterator[str]:
        async for delta in self._target(request, decision).stream(request, decision):
            yield delta

    async def healthy(self) -> bool:
        """Readiness follows the local engine; the external provider is optional."""

        return await self.local.healthy()


def _parse_stream_line(line: str) -> str:
    """Extract the content delta from one server-sent-event line."""

    if not line.startswith("data:"):
        return ""
    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return ""
    try:
        document = json.loads(payload)
        content = document["choices"][0]["delta"].get("content")
    except (ValueError, KeyError, IndexError, TypeError):
        return ""
    return str(content) if content else ""
