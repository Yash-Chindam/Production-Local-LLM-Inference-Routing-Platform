# Production Local-LLM Inference & Routing Platform

An OpenAI-compatible control plane for routing requests across local model tiers. This
repository currently contains the first runnable vertical slice: authentication, quotas,
bounded admission, deterministic privacy/capability routing, route attribution, and a
mock inference backend suitable for CI. The mock boundary is intentionally replaceable by
Ray Serve and vLLM deployments in the next implementation phase.

The complete architecture and design targets are in
[`02-production-local-llm-inference-routing-platform.md`](02-production-local-llm-inference-routing-platform.md).

## Quick start

Requires Python 3.11+ and Node.js 20+.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
npm ci
python -m uvicorn llm_router.app:app --app-dir src --reload
```

The development bearer token is `dev-key`. Override it in every shared environment:

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

Every response identifies the chosen model, immutable revision, inferred task, policy
score, candidate count, and route reason. The same attribution is returned in `X-Route-*`
headers for operational tracing.

## Verification

```bash
ruff format --check .
ruff check .
mypy
pytest tests/unit tests/integration --cov=llm_router --cov-report=term-missing
pytest tests/integration
npx playwright install chromium
npm run test:e2e
docker build -t local-llm-router:dev .
```

CI keeps unit, integration, and Playwright end-to-end tests as separate required signals.
On a successful main-branch CI run, CD creates a versioned OCI image artifact. Registry or
cluster publication stays disabled until an explicit deployment destination is configured.
Successful PR CI runs are merged automatically only for trusted same-repository authors and
Dependabot; repository branch protection and review requirements still apply.

## Runtime settings

All settings use the `ROUTER_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `ROUTER_API_KEYS` | `dev-key` | Comma-separated bearer tokens. |
| `ROUTER_MAX_CONCURRENCY` | `32` | Maximum in-flight inference requests. |
| `ROUTER_ADMISSION_TIMEOUT_SECONDS` | `0.25` | Time allowed to wait for capacity. |
| `ROUTER_QUOTA_REQUESTS_PER_MINUTE` | `120` | Per-token sliding-window request quota. |
| `ROUTER_EXTERNAL_FALLBACK_ENABLED` | `false` | Operator gate for approved external fallback. |

External routing additionally requires public data and request-level opt-in. Private and
restricted requests are never eligible for an external route.
