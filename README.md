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

## Runtime settings

All settings use the `ROUTER_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `ROUTER_API_KEYS` | `dev-key` | Comma-separated bearer tokens. |
| `ROUTER_MAX_CONCURRENCY` | `32` | Maximum in-flight requests. |
| `ROUTER_ADMISSION_TIMEOUT_SECONDS` | `0.25` | Time allowed to wait for capacity. |
| `ROUTER_QUOTA_REQUESTS_PER_MINUTE` | `120` | Per-token sliding-window quota. |
| `ROUTER_EXTERNAL_FALLBACK_ENABLED` | `false` | Operator gate for external fallback. |

External routing also requires public data and request-level opt-in. Private and restricted
requests are never eligible for an external route.
