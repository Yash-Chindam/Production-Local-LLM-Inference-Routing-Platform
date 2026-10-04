"""Short-lived credentials: tokens signed with real keys, verified by the gateway."""

import json
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from pydantic import ValidationError

from llm_router.app import create_app
from llm_router.config import Settings
from llm_router.credentials import (
    CredentialError,
    RemoteKeys,
    StaticKeys,
    TokenVerifier,
    build_verifier,
    http_key_fetcher,
    load_key_set,
    looks_like_a_token,
)

pytestmark = pytest.mark.integration

ISSUER = "https://identity.example"
AUDIENCE = "llm-gateway"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
ROTATED_KEY = ec.generate_private_key(ec.SECP256R1())
STRANGER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def public_jwk(key: Any, key_id: str) -> dict[str, Any]:
    algorithm = RSAAlgorithm if isinstance(key, rsa.RSAPrivateKey) else ECAlgorithm
    return {**json.loads(algorithm.to_jwk(key.public_key())), "kid": key_id, "use": "sig"}


JWKS = {"keys": [public_jwk(SIGNING_KEY, "current")]}


def token(
    *,
    key: Any = SIGNING_KEY,
    key_id: str | None = "current",
    algorithm: str = "RS256",
    lifetime: int = 600,
    issued: float | None = None,
    **claims: Any,
) -> str:
    now = int(time.time() if issued is None else issued)
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "service:claims-pipeline",
        "tenant": "support-tooling",
        "iat": now,
        "exp": now + lifetime,
        **claims,
    }
    payload = {name: value for name, value in payload.items() if value is not None}
    headers = {"kid": key_id} if key_id else None
    return jwt.encode(payload, key, algorithm=algorithm, headers=headers)


def verifier(keys: Any = None, max_lifetime: int = 3600) -> TokenVerifier:
    return TokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        keys=keys or StaticKeys(JWKS),
        max_lifetime_seconds=max_lifetime,
    )


@pytest.mark.asyncio
async def test_a_current_token_from_the_issuer_names_its_subject_and_tenant() -> None:
    verified = await verifier().verify(token())

    assert verified.subject == "service:claims-pipeline"
    assert verified.tenant_id == "support-tooling"
    assert verified.expires_at - verified.issued_at == 600


@pytest.mark.parametrize(
    ("credential", "reason"),
    [
        (lambda: token(lifetime=-120), "token has expired"),
        (lambda: token(lifetime=86_400), "lifetime exceeds the 3600 seconds allowed"),
        (lambda: token(key=STRANGER_KEY), "failed verification"),
        (lambda: token(key_id="retired"), "signed with an unknown key"),
        (lambda: token(aud="some-other-service"), "failed verification"),
        (lambda: token(iss="https://impostor.example"), "failed verification"),
        (lambda: token(tenant=None), "does not name a tenant"),
        (lambda: token(tenant=["support-tooling"]), "does not name a tenant"),
        (lambda: token(sub=None), "failed verification"),
        (
            lambda: token(key="shared-secret-of-enough-length-to-sign", algorithm="HS256"),
            "accepted algorithm",
        ),
        (lambda: jwt.encode({"sub": "x"}, None, algorithm="none"), "accepted algorithm"),
        (lambda: "not.a.token", "malformed token"),
    ],
)
@pytest.mark.asyncio
async def test_a_token_that_should_not_be_trusted_is_refused(credential: Any, reason: str) -> None:
    with pytest.raises(CredentialError, match=reason):
        await verifier().verify(credential())


@pytest.mark.asyncio
async def test_a_token_missing_its_issue_time_cannot_prove_it_is_short_lived() -> None:
    now = int(time.time())
    unbounded = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "tenant": "default", "exp": now + 60},
        SIGNING_KEY,
        algorithm="RS256",
        headers={"kid": "current"},
    )

    with pytest.raises(CredentialError, match="failed verification"):
        await verifier().verify(unbounded)


@pytest.mark.asyncio
async def test_a_single_key_set_serves_tokens_that_name_no_key() -> None:
    assert (await verifier().verify(token(key_id=None))).tenant_id == "support-tooling"

    two = StaticKeys({"keys": [*JWKS["keys"], public_jwk(ROTATED_KEY, "next")]})
    with pytest.raises(CredentialError, match="unknown key"):
        await verifier(two).verify(token(key_id=None))


def test_a_key_set_must_hold_a_usable_key() -> None:
    with pytest.raises(ValueError, match="no usable verification key"):
        StaticKeys({"keys": [{"kty": "nonsense"}]})
    with pytest.raises(ValueError, match="not valid JSON"):
        load_key_set("{")
    with pytest.raises(ValueError, match="JSON object"):
        load_key_set("[]")
    assert looks_like_a_token("a.b.c") and not looks_like_a_token("static-key")


class Issuer:
    """A key endpoint whose published keys can change, and which can go down."""

    def __init__(self) -> None:
        self.document: dict[str, Any] = JWKS
        self.fetches = 0
        self.down = False

    async def fetch(self) -> dict[str, Any]:
        self.fetches += 1
        if self.down:
            raise httpx.ConnectError("issuer unreachable")
        return self.document


@pytest.mark.asyncio
async def test_a_rotated_signing_key_is_picked_up_without_a_restart() -> None:
    issuer, now = Issuer(), [0.0]
    remote = verifier(RemoteKeys(issuer.fetch, clock=lambda: now[0]))
    rotated = token(key=ROTATED_KEY, key_id="next", algorithm="ES256")

    await remote.verify(token())
    await remote.verify(token())
    assert issuer.fetches == 1

    issuer.document = {"keys": [*JWKS["keys"], public_jwk(ROTATED_KEY, "next")]}
    # Too soon after the last fetch: unknown keys may not hammer the issuer.
    now[0] = 5.0
    with pytest.raises(CredentialError, match="unknown key"):
        await remote.verify(rotated)
    assert issuer.fetches == 1

    now[0] = 11.0
    assert (await remote.verify(rotated)).tenant_id == "support-tooling"
    assert issuer.fetches == 2


@pytest.mark.asyncio
async def test_an_unreachable_issuer_does_not_empty_the_key_cache() -> None:
    issuer, now = Issuer(), [0.0]
    remote = verifier(RemoteKeys(issuer.fetch, clock=lambda: now[0]))
    await remote.verify(token())

    issuer.down = True
    now[0] = 600.0
    assert (await remote.verify(token())).subject == "service:claims-pipeline"
    assert issuer.fetches == 2

    issuer.down = False
    issuer.document = {"keys": []}
    now[0] = 1200.0
    assert (await remote.verify(token())).subject == "service:claims-pipeline"


@pytest.mark.asyncio
async def test_keys_are_fetched_from_the_issuers_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://identity.example/keys"
        return httpx.Response(200, json=JWKS)

    fetch = http_key_fetcher("https://identity.example/keys", httpx.MockTransport(handler))

    assert await fetch() == JWKS
    assert (await verifier(RemoteKeys(fetch)).verify(token())).tenant_id == "support-tooling"


def settings(**overrides: Any) -> Settings:
    return Settings(
        **{
            "api_keys": "static-key",
            "jwt_issuer": ISSUER,
            "jwt_audience": AUDIENCE,
            "jwt_jwks": json.dumps(JWKS),
            **overrides,
        }
    )


def chat(client: TestClient, credential: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {credential}"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Classify this ticket"}],
            "routing": {"privacy": "public"},
        },
    )


def test_the_gateway_serves_a_caller_holding_a_current_token() -> None:
    with TestClient(create_app(settings())) as client:
        response = chat(client, token())

    assert response.status_code == 200
    # The tenant in the token is the entitlement applied: this tenant's
    # floor raises a request declared public to private.
    assert "raised from declared public by the tenant floor" in (response.headers["X-Route-Reason"])


@pytest.mark.parametrize(
    ("credential", "detail"),
    [
        (lambda: token(lifetime=-120), "token has expired"),
        (lambda: token(lifetime=86_400), "token lifetime exceeds the 3600 seconds allowed"),
        (lambda: token(key=STRANGER_KEY), "token failed verification"),
    ],
)
def test_the_gateway_refuses_a_bad_token_and_says_why(credential: Any, detail: str) -> None:
    with TestClient(create_app(settings())) as client:
        response = chat(client, credential())

    assert response.status_code == 401
    assert response.json()["detail"] == detail
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


def test_static_keys_keep_working_until_short_lived_credentials_are_required() -> None:
    with TestClient(create_app(settings())) as client:
        assert chat(client, "static-key").status_code == 200

    strict = settings(require_short_lived_credentials=True, environment="production")
    with TestClient(create_app(strict)) as client:
        refused = chat(client, "static-key")
        accepted = chat(client, token())

    assert refused.status_code == 401
    assert "static keys are not accepted" in refused.json()["detail"]
    assert accepted.status_code == 200


def test_a_renewed_token_is_the_same_caller_for_quota() -> None:
    limited = settings(quota_requests_per_minute=2)
    with TestClient(create_app(limited)) as client:
        first = chat(client, token(tenant="default"))
        second = chat(client, token(tenant="default", issued=time.time() - 5))
        third = chat(client, token(tenant="default", issued=time.time() - 9))

    assert (first.status_code, second.status_code) == (200, 200)
    assert third.status_code == 429


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"jwt_issuer": "", "require_short_lived_credentials": True},
            "ROUTER_JWT_ISSUER must be set",
        ),
        ({"jwt_audience": ""}, "ROUTER_JWT_AUDIENCE must be set"),
        ({"jwt_jwks": ""}, "exactly one of ROUTER_JWT_JWKS and ROUTER_JWT_JWKS_URL"),
        ({"jwt_jwks_url": "https://identity.example/keys"}, "exactly one of"),
    ],
)
def test_an_incomplete_token_configuration_is_refused_at_start_up(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        settings(**overrides)


def test_requiring_tokens_removes_the_need_for_any_static_key() -> None:
    strict = Settings(
        environment="production",
        require_short_lived_credentials=True,
        jwt_issuer=ISSUER,
        jwt_audience=AUDIENCE,
        jwt_jwks_url="https://identity.example/keys",
    )

    assert build_verifier(
        issuer=strict.jwt_issuer,
        audience=strict.jwt_audience,
        jwks=strict.jwt_jwks,
        jwks_url=strict.jwt_jwks_url,
        max_lifetime_seconds=strict.jwt_max_lifetime_seconds,
        tenant_claim=strict.jwt_tenant_claim,
    )
    assert (
        build_verifier(
            issuer="", audience="", jwks="", jwks_url="", max_lifetime_seconds=60, tenant_claim="t"
        )
        is None
    )
