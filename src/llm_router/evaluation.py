"""Quality, latency, and cost measurement from section 16.

Quality and latency are always reported together, and every comparison states
its regressions explicitly so an optimization cannot be accepted on latency
alone. Nothing here samples production traffic; runs are driven from committed
dataset and workload definitions.
"""

import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from llm_router.engine_stats import GPU_MEMORY_USED, parse_exposition
from llm_router.models import TaskClass

DRAFT_ACCEPTANCE_RATE = "vllm:spec_decode_draft_acceptance_rate"


class EvaluationError(RuntimeError):
    """Raised when a dataset or comparison is unusable."""


@dataclass(frozen=True)
class EvaluationCase:
    """One graded example from a committed benchmark dataset."""

    id: str
    task: TaskClass
    prompt: str
    expected: str
    structured: bool = False


@dataclass(frozen=True)
class CaseOutcome:
    case: EvaluationCase
    output: str
    latency_ms: float
    gpu_seconds: float = 0.0
    succeeded: bool = True


@dataclass(frozen=True)
class EvaluationReport:
    """Quality and latency reported together, never separately."""

    model_id: str
    model_revision: str
    adapter_id: str | None
    quantization: str
    cases: int
    successes: int
    quality_score: float
    structured_validity: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    gpu_seconds_per_successful_request: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        adapter = self.adapter_id or "none"
        return (
            f"{self.model_id}@{self.model_revision} adapter={adapter} "
            f"quantization={self.quantization} quality={self.quality_score:.3f} "
            f"structured={self.structured_validity:.3f} p95={self.latency_p95_ms:.1f}ms "
            f"gpu_s/req={self.gpu_seconds_per_successful_request:.4f}"
        )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; deterministic and stable for small samples."""

    if not values:
        raise EvaluationError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def structured_output_valid(output: str) -> bool:
    """Structured tasks must return a JSON object, not prose that looks like one."""

    try:
        return isinstance(json.loads(output), dict)
    except ValueError:
        return False


def score_case(outcome: CaseOutcome) -> float:
    """Exact match for constrained tasks, token overlap for generative ones."""

    if not outcome.succeeded:
        return 0.0
    case = outcome.case
    if case.structured and not structured_output_valid(outcome.output):
        return 0.0
    if case.task in {TaskClass.CLASSIFICATION, TaskClass.EXTRACTION}:
        return 1.0 if outcome.output.strip() == case.expected.strip() else 0.0
    expected = set(case.expected.lower().split())
    produced = set(outcome.output.lower().split())
    if not expected:
        return 0.0
    return len(expected & produced) / len(expected | produced)


def load_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Read a JSON Lines dataset of graded cases."""

    cases: list[EvaluationCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
            cases.append(
                EvaluationCase(
                    id=str(document["id"]),
                    task=TaskClass(document["task"]),
                    prompt=str(document["prompt"]),
                    expected=str(document["expected"]),
                    structured=bool(document.get("structured", False)),
                )
            )
        except (ValueError, KeyError) as error:
            raise EvaluationError(f"{path}:{number} is not a usable case: {error}") from error
    if not cases:
        raise EvaluationError(f"{path} contains no cases")
    return tuple(cases)


def build_report(
    outcomes: Iterable[CaseOutcome],
    *,
    model_id: str,
    model_revision: str,
    adapter_id: str | None = None,
    quantization: str = "none",
    wall_clock_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    collected = list(outcomes)
    if not collected:
        raise EvaluationError("cannot report on an empty run")

    latencies = [outcome.latency_ms for outcome in collected]
    successes = [outcome for outcome in collected if outcome.succeeded]
    structured = [outcome for outcome in collected if outcome.case.structured]
    elapsed = wall_clock_seconds if wall_clock_seconds else sum(latencies) / 1000 or 1e-9

    return EvaluationReport(
        model_id=model_id,
        model_revision=model_revision,
        adapter_id=adapter_id,
        quantization=quantization,
        cases=len(collected),
        successes=len(successes),
        quality_score=sum(score_case(outcome) for outcome in collected) / len(collected),
        structured_validity=(
            sum(structured_output_valid(outcome.output) for outcome in structured) / len(structured)
            if structured
            else 1.0
        ),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        latency_p99_ms=percentile(latencies, 0.99),
        throughput_rps=len(collected) / elapsed,
        gpu_seconds_per_successful_request=(
            sum(outcome.gpu_seconds for outcome in collected) / len(successes) if successes else 0.0
        ),
        metadata=metadata or {},
    )


def execute(
    cases: Iterable[EvaluationCase],
    invoke: Callable[[EvaluationCase], tuple[str, bool, float]],
) -> tuple[CaseOutcome, ...]:
    """Run every case through a caller-supplied transport and time each one.

    The transport returns the produced text, whether the request succeeded, and
    the GPU seconds it consumed; latency is measured here so every harness run
    reports it the same way.
    """

    outcomes: list[CaseOutcome] = []
    for case in cases:
        started = time.perf_counter()
        output, succeeded, gpu_seconds = invoke(case)
        outcomes.append(
            CaseOutcome(
                case=case,
                output=output,
                latency_ms=(time.perf_counter() - started) * 1000,
                gpu_seconds=gpu_seconds,
                succeeded=succeeded,
            )
        )
    return tuple(outcomes)


@dataclass(frozen=True)
class Comparison:
    """Variant versus baseline, with the quality cost of any speedup stated."""

    baseline: EvaluationReport
    variant: EvaluationReport
    quality_delta: float
    latency_p95_delta_ms: float
    cost_delta_gpu_seconds: float
    regressions: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.regressions


def compare(
    baseline: EvaluationReport,
    variant: EvaluationReport,
    *,
    quality_tolerance: float = 0.01,
    structured_tolerance: float = 0.0,
) -> Comparison:
    """Reject a variant that buys latency with quality, however small the loss."""

    quality_delta = variant.quality_score - baseline.quality_score
    regressions: list[str] = []
    if quality_delta < -quality_tolerance:
        regressions.append(
            f"quality fell by {abs(quality_delta):.3f}, beyond the "
            f"{quality_tolerance:.3f} tolerance"
        )
    structured_delta = variant.structured_validity - baseline.structured_validity
    if structured_delta < -structured_tolerance:
        regressions.append(f"structured-output validity fell by {abs(structured_delta):.3f}")
    if variant.successes < baseline.successes:
        regressions.append(
            f"successful requests fell from {baseline.successes} to {variant.successes}"
        )

    return Comparison(
        baseline=baseline,
        variant=variant,
        quality_delta=quality_delta,
        latency_p95_delta_ms=variant.latency_p95_ms - baseline.latency_p95_ms,
        cost_delta_gpu_seconds=(
            variant.gpu_seconds_per_successful_request - baseline.gpu_seconds_per_successful_request
        ),
        regressions=tuple(regressions),
    )


def render_comparison(comparison: Comparison) -> str:
    verdict = "accepted" if comparison.accepted else "rejected"
    lines = [
        f"baseline: {comparison.baseline.summary()}",
        f"variant:  {comparison.variant.summary()}",
        f"quality delta: {comparison.quality_delta:+.3f}",
        f"p95 latency delta: {comparison.latency_p95_delta_ms:+.1f} ms",
        f"gpu seconds per successful request delta: {comparison.cost_delta_gpu_seconds:+.4f}",
        f"verdict: {verdict}",
    ]
    lines.extend(f"regression: {reason}" for reason in comparison.regressions)
    return "\n".join(lines)


def endpoint_transport(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    gpu_count: int = 1,
    max_tokens: int = 256,
) -> Callable[[EvaluationCase], tuple[str, bool, float]]:
    """Build a transport that runs a case against an OpenAI-compatible endpoint.

    GPU seconds are the request's wall time multiplied by the accelerators the
    model occupies. That is exact for the sequential runs this harness makes
    and an upper bound once requests share a batch.
    """

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def invoke(case: EvaluationCase) -> tuple[str, bool, float]:
        started = time.perf_counter()
        try:
            response = client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": case.prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            text = str(response.json()["choices"][0]["message"]["content"])
            succeeded = response.status_code < 400
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            text, succeeded = "", False
        return text, succeeded, (time.perf_counter() - started) * gpu_count

    return invoke


def engine_measurements(client: httpx.Client, base_url: str) -> dict[str, float]:
    """Read memory and draft acceptance from the engine after a run, if published."""

    try:
        response = client.get(f"{base_url}/metrics")
    except httpx.HTTPError:
        return {}
    if response.status_code >= 400:
        return {}
    samples = parse_exposition(response.text)
    measured: dict[str, float] = {}
    memory = [value for _, value in samples.get(GPU_MEMORY_USED, [])]
    if memory:
        # DCGM reports mebibytes per accelerator; the catalog records gigabytes.
        measured["gpu_memory_gb"] = round(sum(memory) / 1024, 2)
    acceptance = [value for _, value in samples.get(DRAFT_ACCEPTANCE_RATE, [])]
    if acceptance:
        measured["draft_acceptance_rate"] = round(sum(acceptance) / len(acceptance), 4)
    return measured


def benchmark_document(
    report: EvaluationReport,
    cases: Sequence[EvaluationCase],
    *,
    benchmark_id: str,
    dataset_version: str,
    hardware: str,
    task: TaskClass | None = None,
    measurements: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Shape a report as a catalog benchmark record, ready to commit as evidence."""

    lengths = sorted(max(1, len(case.prompt) // 4) for case in cases)
    document: dict[str, Any] = {
        "id": benchmark_id,
        "dataset_version": dataset_version,
        "workload_version": "sequential-1",
        "hardware": hardware,
        "driver": str(report.metadata.get("driver", "unrecorded")),
        "container_digest": str(report.metadata.get("container_digest", "unrecorded")),
        "engine_revision": str(report.metadata.get("engine_revision", "unrecorded")),
        "model_revision": report.model_revision,
        "concurrency": 1,
        "prompt_tokens_p50": int(percentile(lengths, 0.50)),
        "prompt_tokens_p95": int(percentile(lengths, 0.95)),
        "quality_score": round(report.quality_score, 4),
        "latency_p95_ms": round(report.latency_p95_ms, 2),
        "throughput_rps": round(report.throughput_rps, 3),
        "gpu_seconds_per_request": round(report.gpu_seconds_per_successful_request, 4),
    }
    if task is not None:
        document["task"] = task.value
    document.update(measurements or {})
    return document


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - command-line wrapper
    """Benchmark a live endpoint and print a catalog record for the result."""

    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model or experiment name")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--task", choices=[item.value for item in TaskClass])
    parser.add_argument("--api-key", default="")
    parser.add_argument("--gpu-count", type=int, default=1)
    arguments = parser.parse_args(argv)

    cases = load_dataset(arguments.dataset)
    with httpx.Client(timeout=120.0) as client:
        started = time.perf_counter()
        outcomes = execute(
            cases,
            endpoint_transport(
                client,
                base_url=arguments.base_url,
                model=arguments.model,
                api_key=arguments.api_key,
                gpu_count=arguments.gpu_count,
            ),
        )
        report = build_report(
            outcomes,
            model_id=arguments.model,
            model_revision=arguments.model_revision,
            wall_clock_seconds=time.perf_counter() - started,
        )
        measurements = engine_measurements(client, arguments.base_url)
    document = benchmark_document(
        report,
        cases,
        benchmark_id=arguments.benchmark_id,
        dataset_version=arguments.dataset_version,
        hardware=arguments.hardware,
        task=TaskClass(arguments.task) if arguments.task else None,
        measurements=measurements,
    )
    print(yaml.safe_dump([document], sort_keys=False), end="")


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    main()
