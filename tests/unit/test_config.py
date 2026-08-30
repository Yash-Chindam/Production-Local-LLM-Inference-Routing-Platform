import pytest
from pydantic import ValidationError

from llm_router.config import Settings


def test_api_keys_are_normalized() -> None:
    settings = Settings(environment="test", api_keys=" first, second ,, ")
    assert settings.accepted_api_keys == frozenset({"first", "second"})


def test_production_rejects_default_development_key() -> None:
    with pytest.raises(ValidationError, match="ROUTER_API_KEYS"):
        Settings(environment="production")


def test_production_accepts_explicit_key() -> None:
    settings = Settings(environment="production", api_keys="production-key")
    assert settings.accepted_api_keys == frozenset({"production-key"})
