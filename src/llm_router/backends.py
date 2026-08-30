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

from llm_router.models import ChatCompletionRequest, RouteDecision


class BackendUnavailableError(RuntimeError):
    """Raised when an inference engine is unreachable or returns an error."""


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
                timeout=self.request_timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise BackendUnavailableError(f"inference engine unreachable: {error}") from error

        if response.status_code >= 400:
            raise BackendUnavailableError(
                f"inference engine returned {response.status_code} for "
                f"{served_model_name(decision)}"
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
                timeout=self.request_timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    raise BackendUnavailableError(
                        f"inference engine returned {response.status_code} while streaming"
                    )
                async for line in response.aiter_lines():
                    delta = _parse_stream_line(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as error:
            raise BackendUnavailableError(f"inference engine unreachable: {error}") from error

    async def healthy(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400


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
