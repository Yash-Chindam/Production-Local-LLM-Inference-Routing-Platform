from dataclasses import dataclass
from typing import Protocol

from llm_router.models import ChatCompletionRequest, RouteDecision


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


class MockInferenceBackend:
    """Deterministic backend used until vLLM deployments are configured."""

    async def generate(
        self, request: ChatCompletionRequest, decision: RouteDecision
    ) -> BackendResult:
        response = f"[{decision.profile.id}] accepted {decision.task.value} request"
        prompt_tokens = max(1, len(request.prompt) // 4)
        completion_tokens = max(1, len(response) // 4)
        return BackendResult(
            text=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
