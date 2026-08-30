from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TaskClass(StrEnum):
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    RAG = "rag"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CRITIQUE = "critique"
    GENERAL = "general"


class PrivacyClass(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class RoutingOptions(BaseModel):
    task: TaskClass | None = None
    domain: str | None = Field(default=None, max_length=64)
    privacy: PrivacyClass = PrivacyClass.PRIVATE
    latency_tier: Literal["interactive", "standard", "batch"] = "standard"
    quality_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    allow_external_fallback: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    routing: RoutingOptions = Field(default_factory=RoutingOptions)

    @model_validator(mode="after")
    def reject_streaming_for_initial_slice(self) -> "ChatCompletionRequest":
        if self.stream:
            raise ValueError("streaming is not available in the initial control-plane slice")
        return self

    @property
    def prompt(self) -> str:
        return "\n".join(message.content for message in self.messages)


class ModelProfile(BaseModel):
    id: str
    revision: str
    local: bool
    healthy: bool = True
    context_limit: int
    supported_tasks: frozenset[TaskClass]
    quality: float = Field(ge=0.0, le=1.0)
    estimated_queue_ms: int = Field(default=0, ge=0)
    cost_weight: float = Field(default=0.0, ge=0.0)


class RouteDecision(BaseModel):
    profile: ModelProfile
    task: TaskClass
    reason: str
    score: float
    candidate_count: int
    adapter_id: str | None = None
    adapter_revision: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    routing: dict[str, Any]
