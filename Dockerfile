FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROUTER_ENVIRONMENT=production

RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels

WORKDIR /app
COPY config ./config
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["uvicorn", "llm_router.app:app", "--host", "0.0.0.0", "--port", "8000"]

