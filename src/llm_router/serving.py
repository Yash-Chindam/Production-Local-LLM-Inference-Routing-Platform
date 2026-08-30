"""Ray Serve deployment configuration derived from the registry (section 7.3).

The control plane does not invent deployment topology at runtime. This module
renders a declarative Ray Serve LLM configuration from the governed catalog so
what is deployed is traceable to a model card, an adapter, and a GPU pool.
"""

from typing import Any

import yaml

from llm_router.registry import LifecycleStage, ModelCard, Quantization, Registry

DEFAULT_MIN_REPLICAS = 1
DEFAULT_MAX_REPLICAS = 4
DEFAULT_MAX_ONGOING_REQUESTS = 16
DEFAULT_TARGET_ONGOING_REQUESTS = 8


class ServingConfigError(RuntimeError):
    """Raised when the catalog cannot produce a deployable configuration."""


def _autoscaling_config(card: ModelCard) -> dict[str, Any]:
    """Latency-sensitive tiers keep a warm replica; heavier tiers scale wider."""

    warm = card.tier.value in {"small-specialist", "general-local"}
    return {
        "min_replicas": DEFAULT_MIN_REPLICAS if warm else 0,
        "max_replicas": DEFAULT_MAX_REPLICAS if warm else 2,
        "target_ongoing_requests": DEFAULT_TARGET_ONGOING_REQUESTS,
        "upscale_delay_s": 10,
        "downscale_delay_s": 300 if warm else 60,
    }


def _engine_kwargs(card: ModelCard, adapter_count: int) -> dict[str, Any]:
    engine: dict[str, Any] = {
        "max_model_len": card.context_limit,
        "tensor_parallel_size": card.hardware.tensor_parallel_size,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    }
    if card.quantization is not Quantization.NONE:
        engine["quantization"] = card.quantization.value
    if adapter_count:
        engine["enable_lora"] = True
        engine["max_loras"] = adapter_count
        engine["max_lora_rank"] = 32
    return engine


def build_serving_config(registry: Registry) -> dict[str, Any]:
    """Render one Ray Serve LLM application entry per servable local model."""

    applications: list[dict[str, Any]] = []
    for card in registry.servable_models():
        if not card.local:
            continue
        adapters = [
            adapter
            for adapter in registry.servable_adapters()
            if adapter.base_model_id == card.id and adapter.base_revision == card.revision
        ]
        entry: dict[str, Any] = {
            "model_id": card.id,
            "model_revision": card.revision,
            "stage": card.stage.value,
            "model_loading_config": {
                "model_id": card.id,
                "model_source": f"registry://{card.id}@{card.revision}",
                "tokenizer": card.tokenizer,
            },
            "accelerator_type": card.hardware.accelerator,
            "deployment_config": {
                "autoscaling_config": _autoscaling_config(card),
                "max_ongoing_requests": DEFAULT_MAX_ONGOING_REQUESTS,
                "ray_actor_options": {
                    "num_gpus": card.hardware.count,
                    "resources": {f"gpu_pool_{card.hardware.accelerator}": 0.001},
                },
            },
            "engine_kwargs": _engine_kwargs(card, len(adapters)),
        }
        if adapters:
            entry["lora_config"] = {
                "dynamic_lora_loading_path": f"registry://adapters/{card.id}",
                "adapters": [
                    {
                        "id": adapter.id,
                        "revision": adapter.adapter_revision,
                        "domain": adapter.domain,
                        "quantized": adapter.quantized,
                        "stage": adapter.stage.value,
                    }
                    for adapter in sorted(adapters, key=lambda item: item.id)
                ],
            }
        applications.append(entry)

    if not applications:
        raise ServingConfigError("catalog contains no servable local models")

    return {
        "policy_version": registry.policy.version,
        "applications": applications,
    }


def canary_config(registry: Registry, deployment_id: str) -> dict[str, Any]:
    """Describe a canary and the revision it rolls back to (section 13)."""

    current = next((item for item in registry.deployments if item.id == deployment_id), None)
    if current is None:
        raise ServingConfigError(f"unknown deployment {deployment_id}")
    target = registry.rollback_target(deployment_id)
    staged_adapters = [
        adapter.id
        for adapter in registry.servable_adapters()
        if adapter.stage is LifecycleStage.STAGING
    ]
    return {
        "deployment_id": current.id,
        "container_digest": current.container_digest,
        "gpu_pool": current.gpu_pool,
        "canary_traffic_percent": 10,
        "promote_after_successful_requests": 500,
        "rollback_to": None if target is None else target.id,
        "rollback_triggers": [
            "readiness probe failure",
            "p95 latency above the tier objective",
            "quality below the benchmark floor",
            "error rate above one percent",
        ],
        "staged_adapters": sorted(staged_adapters),
    }


def render_serving_config(registry: Registry) -> str:
    return yaml.safe_dump(build_serving_config(registry), sort_keys=False)


def main() -> None:  # pragma: no cover - thin command-line wrapper
    """Render the deployment configuration for the committed catalog."""

    import sys

    from llm_router.registry import load_registry

    catalog = sys.argv[1] if len(sys.argv) > 1 else "config/registry.yaml"
    sys.stdout.write(render_serving_config(load_registry(catalog)))


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    main()
