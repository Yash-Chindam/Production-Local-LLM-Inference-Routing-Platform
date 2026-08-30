from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables prefixed with ROUTER_."""

    model_config = SettingsConfigDict(env_prefix="ROUTER_", extra="ignore")

    environment: str = "development"
    api_keys: str = "dev-key"
    max_concurrency: int = Field(default=32, ge=1)
    admission_timeout_seconds: float = Field(default=0.25, gt=0)
    quota_requests_per_minute: int = Field(default=120, ge=1)
    external_fallback_enabled: bool = False
    routing_policy_version: str = "v1"
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
        return frozenset(key.strip() for key in self.api_keys.split(",") if key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
