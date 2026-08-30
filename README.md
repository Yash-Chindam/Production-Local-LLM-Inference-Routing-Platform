# Production Local-LLM Inference & Routing Platform

Policy-aware routing components for a production local-model inference platform.

The complete architecture and design targets are documented in
[`02-production-local-llm-inference-routing-platform.md`](02-production-local-llm-inference-routing-platform.md).

## Development

Requires Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
mypy
pytest tests/unit --cov=llm_router --cov-report=term-missing
```

This first delivery slice contains deterministic routing, privacy restrictions, quotas,
and bounded admission. API ingress and model-serving adapters are delivered separately.
