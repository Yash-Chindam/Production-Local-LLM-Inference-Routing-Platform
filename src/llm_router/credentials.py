"""Short-lived credentials for users and services (section 14).

A static API key is valid until someone remembers to rotate it. A signed token
from the organization's identity provider says who is calling, which tenant
they belong to, and when the claim stops being true. The gateway verifies the
signature against the issuer's published keys and refuses any token that
lives longer than the configured limit, so a leaked credential expires on its
own.
"""

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import jwt
from jwt import PyJWK

# Asymmetric algorithms only. The gateway holds public keys, so it can verify
# a token but never mint one; a shared-secret or unsigned token is refused.
ALLOWED_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "PS256"})
CLOCK_SKEW_SECONDS = 30
KEY_CACHE_SECONDS = 300.0
# A token signed by an unknown key triggers a refetch, but not more often than
# this, so a stream of bad tokens cannot be turned into load on the issuer.
KEY_REFRESH_INTERVAL_SECONDS = 10.0


class CredentialError(Exception):
    """A token that cannot be accepted. The message is safe to return to the caller."""


@dataclass(frozen=True)
class VerifiedToken:
    subject: str
    tenant_id: str
    issued_at: int
    expires_at: int


class KeySource(Protocol):
    async def key(self, key_id: str | None) -> PyJWK | None: ...


def _parse_key_set(document: dict[str, Any]) -> dict[str | None, PyJWK]:
    keys: dict[str | None, PyJWK] = {}
    for entry in document.get("keys", []):
        try:
            parsed = PyJWK.from_dict(entry)
        except jwt.PyJWTError:
            # One unusable entry must not take the usable ones down with it.
            continue
        keys[entry.get("kid")] = parsed
    return keys


class StaticKeys:
    """Verification keys supplied as a JWKS document, for example from a secret."""

    def __init__(self, document: dict[str, Any]) -> None:
        self._keys = _parse_key_set(document)
        if not self._keys:
            raise ValueError("the key set holds no usable verification key")

    async def key(self, key_id: str | None) -> PyJWK | None:
        if key_id is None and len(self._keys) == 1:
            return next(iter(self._keys.values()))
        return self._keys.get(key_id)


class RemoteKeys:
    """Verification keys fetched from the issuer's JWKS endpoint and cached.

    The cache is refreshed when it ages out, and early when a token names a key
    it does not hold, which is how a rotated signing key is picked up without
    a restart.
    """

    def __init__(
        self,
        fetch: Callable[[], Awaitable[dict[str, Any]]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._clock = clock
        self._keys: dict[str | None, PyJWK] = {}
        self._fetched_at: float | None = None

    async def _refresh(self) -> None:
        self._fetched_at = self._clock()
        try:
            self._keys = _parse_key_set(await self._fetch()) or self._keys
        except Exception:
            # Keep verifying with the keys already held while the issuer is
            # unreachable; an empty cache would turn its outage into ours.
            return

    async def key(self, key_id: str | None) -> PyJWK | None:
        age = None if self._fetched_at is None else self._clock() - self._fetched_at
        stale = age is None or age >= KEY_CACHE_SECONDS
        unknown = key_id not in self._keys
        if stale or (unknown and age is not None and age >= KEY_REFRESH_INTERVAL_SECONDS):
            await self._refresh()
        return self._keys.get(key_id)


class TokenVerifier:
    """Accepts a bearer token only if it is signed, current, short-lived and ours."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        keys: KeySource,
        max_lifetime_seconds: int,
        tenant_claim: str = "tenant",
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.keys = keys
        self.max_lifetime_seconds = max_lifetime_seconds
        self.tenant_claim = tenant_claim

    async def verify(self, token: str) -> VerifiedToken:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as error:
            raise CredentialError("malformed token") from error
        algorithm = header.get("alg")
        if algorithm not in ALLOWED_ALGORITHMS:
            raise CredentialError("token is not signed with an accepted algorithm")
        key = await self.keys.key(header.get("kid"))
        if key is None:
            raise CredentialError("token is signed with an unknown key")
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=self.audience,
                issuer=self.issuer,
                leeway=CLOCK_SKEW_SECONDS,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.ExpiredSignatureError as error:
            raise CredentialError("token has expired") from error
        except jwt.PyJWTError as error:
            raise CredentialError("token failed verification") from error

        issued_at, expires_at = int(claims["iat"]), int(claims["exp"])
        if expires_at - issued_at > self.max_lifetime_seconds:
            raise CredentialError(
                f"token lifetime exceeds the {self.max_lifetime_seconds} seconds allowed"
            )
        tenant = claims.get(self.tenant_claim)
        if not isinstance(tenant, str) or not tenant:
            # No default: a token that does not say which tenant it is for
            # must not inherit one.
            raise CredentialError("token does not name a tenant")
        return VerifiedToken(
            subject=str(claims["sub"]),
            tenant_id=tenant,
            issued_at=issued_at,
            expires_at=expires_at,
        )


def looks_like_a_token(credential: str) -> bool:
    """Whether a bearer credential has the three-part shape of a signed token."""

    return credential.count(".") == 2


def load_key_set(document: str) -> StaticKeys:
    """Parse a JWKS document supplied as configuration."""

    try:
        parsed = json.loads(document)
    except ValueError as error:
        raise ValueError("ROUTER_JWT_JWKS is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("ROUTER_JWT_JWKS must be a JSON object with a keys list")
    return StaticKeys(parsed)


def http_key_fetcher(
    url: str, transport: httpx.AsyncBaseTransport | None = None
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """Fetch the issuer's key set over HTTP, one short-lived connection per refresh."""

    async def fetch() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0, transport=transport) as client:
            response = await client.get(url)
            response.raise_for_status()
            document: dict[str, Any] = response.json()
            return document

    return fetch


def build_verifier(
    *,
    issuer: str,
    audience: str,
    jwks: str,
    jwks_url: str,
    max_lifetime_seconds: int,
    tenant_claim: str,
) -> TokenVerifier | None:
    """The verifier the settings describe, or None when no issuer is configured."""

    if not issuer:
        return None
    keys: KeySource = load_key_set(jwks) if jwks else RemoteKeys(http_key_fetcher(jwks_url))
    return TokenVerifier(
        issuer=issuer,
        audience=audience,
        keys=keys,
        max_lifetime_seconds=max_lifetime_seconds,
        tenant_claim=tenant_claim,
    )
