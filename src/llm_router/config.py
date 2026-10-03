from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables prefixed with ROUTER_."""

    model_config = SettingsConfigDict(env_prefix="ROUTER_", extra="ignore")

    environment: str = "development"
    api_keys: str = "dev-key"
    # Binds credentials to tenant ids as "tenant:key,tenant:key". Entitlements
    # are governance and live in the catalog; the secret that proves which
    # tenant is calling stays in the environment and is never committed.
    tenant_keys: str = ""
    default_tenant: str = "default"
    max_concurrency: int = Field(default=32, ge=1)
    admission_timeout_seconds: float = Field(default=0.25, gt=0)
    quota_requests_per_minute: int = Field(default=120, ge=1)
    external_fallback_enabled: bool = False
    backend: Literal["mock", "vllm"] = "mock"
    vllm_base_url: str = "http://127.0.0.1:8001"
    backend_timeout_seconds: float = Field(default=60.0, gt=0)
    # Consecutive engine failures before requests fail fast, and how long
    # the gateway waits before letting a trial request through again.
    engine_failure_threshold: int = Field(default=5, ge=1)
    engine_cooldown_seconds: float = Field(default=30.0, gt=0)
    # How long shutdown waits for admitted requests before closing the engine
    # client; keep it under the pod's termination grace period.
    shutdown_grace_seconds: float = Field(default=20.0, ge=0)
    registry_path: str = "config/registry.yaml"
    routing_policy_version: str = "v1"
    # Labelled prompts the task and complexity classifier is trained on at
    # start-up; keyword rules decide the task when the file is absent.
    task_classifier_path: str = "config/routing/task-classifier-v1.jsonl"
    redis_url: str = ""
    # Traces are exported only when a collector endpoint is set. Prompt content
    # is never recorded unless an operator opts in, and then only for public
    # requests; restricted and private prompts stay out of traces regardless.
    otlp_endpoint: str = ""
    trace_prompt_content: bool = False
    cache_enabled: bool = True
    cache_ttl_seconds: float = Field(default=300.0, gt=0)
    cache_max_entries: int = Field(default=1024, ge=1)
    semantic_cache_enabled: bool = False
    semantic_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def reject_development_key_in_shared_environments(self) -> "Settings":
        if self.environment not in {"development", "test"} and "dev-key" in self.accepted_api_keys:
            raise ValueError("ROUTER_API_KEYS must be set outside development and test")
        return self

    @property
    def accepted_api_keys(self) -> frozenset[str]:
        return frozenset(self.tenant_by_key)

    @property
    def tenant_by_key(self) -> dict[str, str]:
        """Map every accepted credential to the tenant it authenticates.

        Keys listed without a tenant belong to the default tenant, so an
        existing ROUTER_API_KEYS deployment keeps working unchanged.
        """

        bindings = {
            key.strip(): self.default_tenant for key in self.api_keys.split(",") if key.strip()
        }
        for entry in self.tenant_keys.split(","):
            tenant, separator, key = entry.partition(":")
            if separator and tenant.strip() and key.strip():
                bindings[key.strip()] = tenant.strip()
        return bindings


@lru_cache
def get_settings() -> Settings:
    return Settings()
