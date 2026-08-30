# Production Local-LLM Inference & Routing Platform

An OpenAI-compatible control plane for routing requests across local model tiers. The
current inference backend is deterministic for CI and is replaceable by Ray Serve and
vLLM deployments.

The complete architecture and design targets are documented in
[`02-production-local-llm-inference-routing-platform.md`](02-production-local-llm-inference-routing-platform.md).

## Development

Requires Python 3.11 or newer and Node.js 20 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
npm ci
python -m uvicorn llm_router.app:app --app-dir src --reload
```

The development bearer token is `dev-key`. Override it in every shared environment.
Production startup rejects that development key.

```bash
ROUTER_API_KEYS="replace-me" python -m uvicorn llm_router.app:app --app-dir src
```

Example request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Extract invoice fields"}],"routing":{"privacy":"restricted"}}'
```

Every response records the selected model, immutable revision, inferred task, candidate
count, policy score, and route reason.

## Verification

```bash
ruff format --check .
ruff check .
mypy
pytest tests/unit tests/integration --cov=llm_router --cov-report=term-missing
npx playwright install chromium
npm run test:e2e
docker build -t local-llm-router:dev .
```

CI reports unit/static analysis, integration, and Playwright end-to-end tests separately.
The release-image build starts only after all three test layers pass.

After a successful CI run on `main`, CD creates a versioned OCI image artifact. Registry
or cluster publication remains disabled until an explicit deployment destination is
configured. Dependabot maintains Python, npm, and GitHub Actions dependencies, while the
PR labeler classifies API, test, CI/CD, documentation, and dependency changes.

Successful PR CI runs are merged automatically only for trusted same-repository authors
and Dependabot. Forks, drafts, and untrusted author associations are deliberately skipped;
repository branch-protection and review requirements continue to apply.

## Observability

`GET /metrics` returns Prometheus exposition text and is intentionally unauthenticated so
in-cluster scrapers can read it; restrict it with network policy rather than a bearer token.

| Metric | Purpose |
|---|---|
| `router_request_latency_seconds` | End-to-end latency histogram per model. |
| `router_time_to_first_token_seconds` | Admission-to-first-token delay. |
| `router_time_per_output_token_seconds` | Mean generation time per output token. |
| `router_tokens_total` | Prompt and completion tokens per model. |
| `router_inflight_requests` / `router_queued_requests` | Live capacity and queue depth. |
| `router_routes_total` | Requests per route with task and privacy class. |
| `router_external_fallback_total` | Fallback frequency. |
| `router_queue_delay_prediction_error_ms` | Predicted versus observed queue delay. |
| `router_rejections_total` | Quota, overload, and policy rejections. |
| `router_cache_events_total` | Cache lookups by cache and result. |
| `router_model_load_seconds` | Model load and cold-start duration. |

## Caching

Cache keys bind the tenant, the active model-catalog fingerprint, and the generation
parameters, so promoting any model revision or routing policy version invalidates every
dependent entry. Responses carry `X-Cache: miss | exact | semantic`.

| Cache | Eligibility |
|---|---|
| Exact response | Deterministic requests (`temperature = 0`) that are not `restricted`. |
| Prefix | Reported per model revision for repeated instruction prefixes. |
| Semantic | Disabled by default; requires `public` privacy, deterministic generation, and an extraction, classification, or summarization task. |
| Router decision | Reuses stable task classification; cleared when the policy version changes. |

## Runtime settings

All settings use the `ROUTER_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `ROUTER_API_KEYS` | `dev-key` | Comma-separated bearer tokens. |
| `ROUTER_MAX_CONCURRENCY` | `32` | Maximum in-flight requests. |
| `ROUTER_ADMISSION_TIMEOUT_SECONDS` | `0.25` | Time allowed to wait for capacity. |
| `ROUTER_QUOTA_REQUESTS_PER_MINUTE` | `120` | Per-token sliding-window quota. |
| `ROUTER_EXTERNAL_FALLBACK_ENABLED` | `false` | Operator gate for external fallback. |
| `ROUTER_ROUTING_POLICY_VERSION` | `v1` | Invalidates router and response caches when changed. |
| `ROUTER_CACHE_ENABLED` | `true` | Master switch for all cache tiers. |
| `ROUTER_CACHE_TTL_SECONDS` | `300` | Exact-response entry lifetime. |
| `ROUTER_CACHE_MAX_ENTRIES` | `1024` | Bound on cached responses. |
| `ROUTER_SEMANTIC_CACHE_ENABLED` | `false` | Enables similarity reuse for approved tasks. |
| `ROUTER_SEMANTIC_SIMILARITY_THRESHOLD` | `0.92` | Minimum similarity for a semantic hit. |

External routing also requires public data and request-level opt-in. Private and restricted
requests are never eligible for an external route.
