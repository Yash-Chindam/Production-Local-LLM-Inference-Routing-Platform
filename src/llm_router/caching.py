"""Cache tiers defined in section 12 of the design specification.

Every cache key includes the tenant, the immutable model revision, and the
generation parameters, so a cache entry can never cross a tenant boundary or
survive a model promotion. Semantic caching is additionally restricted to
non-sensitive, deterministic task classes.
"""

import hashlib
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from llm_router.models import ChatCompletionRequest, PrivacyClass, TaskClass

SEMANTIC_CACHE_ELIGIBLE_TASKS = frozenset(
    {TaskClass.CLASSIFICATION, TaskClass.EXTRACTION, TaskClass.SUMMARIZATION}
)


@dataclass(frozen=True)
class CachedCompletion:
    text: str
    model_id: str
    model_revision: str
    prompt_tokens: int
    completion_tokens: int


class CacheStore(Protocol):
    """Minimal async key-value contract implemented by memory and Redis stores."""

    async def get(self, key: str) -> CachedCompletion | None: ...

    async def set(self, key: str, value: CachedCompletion) -> None: ...


class InMemoryCacheStore:
    """Bounded TTL store used for single-replica deployments and tests."""

    def __init__(self, *, max_entries: int = 1024, ttl_seconds: float = 300.0) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, CachedCompletion]] = OrderedDict()

    async def get(self, key: str) -> CachedCompletion | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    async def set(self, key: str, value: CachedCompletion) -> None:
        self._entries[key] = (time.monotonic() + self._ttl_seconds, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(word for word in text.lower().split() if word)


def similarity(left: str, right: str) -> float:
    """Jaccard similarity over normalized tokens, used for semantic lookups."""

    left_tokens, right_tokens = _tokenize(left), _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def build_cache_key(
    request: ChatCompletionRequest,
    *,
    tenant: str,
    model_revision: str,
    prompt: str | None = None,
) -> str:
    """Derive an exact-response key from tenant, revision, parameters, and prompt."""

    material = "|".join(
        (
            tenant,
            model_revision,
            f"max_tokens={request.max_tokens}",
            f"temperature={request.temperature}",
            f"privacy={request.routing.privacy.value}",
            prompt if prompt is not None else request.prompt,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()


def prefix_key(prompt: str, *, model_revision: str, prefix_chars: int = 512) -> str:
    """Key for repeated instruction and context prefixes bound to one revision."""

    return hashlib.sha256(f"{model_revision}|{prompt[:prefix_chars]}".encode()).hexdigest()


def exact_cache_eligible(request: ChatCompletionRequest) -> bool:
    """Exact reuse requires deterministic generation and a non-restricted class."""

    return request.temperature == 0.0 and request.routing.privacy != PrivacyClass.RESTRICTED


def semantic_cache_eligible(request: ChatCompletionRequest, task: TaskClass) -> bool:
    """Semantic reuse is limited to public, deterministic, approved task classes."""

    return (
        request.temperature == 0.0
        and request.routing.privacy == PrivacyClass.PUBLIC
        and task in SEMANTIC_CACHE_ELIGIBLE_TASKS
    )


@dataclass
class SemanticCache:
    """Similarity cache scoped by tenant, revision, and generation parameters."""

    threshold: float = 0.92
    max_entries: int = 256
    _entries: OrderedDict[str, list[tuple[str, CachedCompletion]]] = field(
        default_factory=OrderedDict
    )

    def scope(self, request: ChatCompletionRequest, *, tenant: str, model_revision: str) -> str:
        return build_cache_key(request, tenant=tenant, model_revision=model_revision, prompt="")

    def lookup(self, scope: str, prompt: str) -> CachedCompletion | None:
        candidates: Iterable[tuple[str, CachedCompletion]] = self._entries.get(scope, [])
        best: tuple[float, CachedCompletion] | None = None
        for stored_prompt, value in candidates:
            score = similarity(stored_prompt, prompt)
            if score >= self.threshold and (best is None or score > best[0]):
                best = (score, value)
        return None if best is None else best[1]

    def store(self, scope: str, prompt: str, value: CachedCompletion) -> None:
        bucket = self._entries.setdefault(scope, [])
        bucket.append((prompt, value))
        self._entries.move_to_end(scope)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


@dataclass
class RouterDecisionCache:
    """Caches stable task classifications and invalidates on policy changes."""

    policy_version: str
    max_entries: int = 512
    _entries: OrderedDict[str, TaskClass] = field(default_factory=OrderedDict)

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.policy_version}|{prompt}".encode()).hexdigest()

    def get(self, prompt: str) -> TaskClass | None:
        key = self._key(prompt)
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def set(self, prompt: str, task: TaskClass) -> None:
        key = self._key(prompt)
        self._entries[key] = task
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def invalidate(self, policy_version: str) -> None:
        if policy_version != self.policy_version:
            self.policy_version = policy_version
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def catalog_fingerprint(revisions: Iterable[str], policy_version: str) -> str:
    """Stable identity for the active model catalog and routing policy.

    Cache lookups happen before routing, so keys are bound to this fingerprint:
    promoting any model or policy version invalidates every dependent entry.
    """

    material = "|".join((policy_version, *sorted(revisions)))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


class PrefixTracker:
    """Tracks reuse of repeated instruction prefixes for cache-hit reporting."""

    def __init__(self, *, max_entries: int = 1024, prefix_chars: int = 512) -> None:
        self._max_entries = max_entries
        self._prefix_chars = prefix_chars
        self._seen: OrderedDict[str, None] = OrderedDict()

    def observe(self, prompt: str, *, model_revision: str) -> bool:
        key = prefix_key(prompt, model_revision=model_revision, prefix_chars=self._prefix_chars)
        hit = key in self._seen
        self._seen[key] = None
        self._seen.move_to_end(key)
        while len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)
        return hit
