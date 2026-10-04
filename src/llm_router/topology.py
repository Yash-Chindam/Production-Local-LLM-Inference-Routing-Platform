"""Ray head and worker topology derived from the registry (sections 7.3 and 17).

One KubeRay ``RayService`` holds the whole serving plane: a head that
schedules and runs no model, and one worker group per accelerator type, so
GPU pools stay separate and each is sized from the models placed on it. Like
the serving configuration, the manifest is generated from the catalog and
never edited by hand.
"""

import json
from collections.abc import Sequence
from typing import Any

import yaml
from pydantic import BaseModel

from llm_router.governance import DEFAULT_ARTIFACT_ROOT, governed_versions
from llm_router.registry import Registry
from llm_router.serving import build_serving_config

NAMESPACE = "llm-routing"
SERVICE_NAME = "llm-serve"
POD_LABEL = "ray-serve"
RAY_VERSION = "2.51.0"
RAY_IMAGE = "docker.io/rayproject/ray-llm@sha256:REPLACE_ME"
SERVE_PORT = 8000
METRICS_PORT = 8080
DASHBOARD_PORT = 8265
# Node labels published by GPU feature discovery. They differ between clouds
# and card variants, so anything not listed falls back to NVIDIA-<TYPE>.
GPU_PRODUCT_LABELS = {"A100": "NVIDIA-A100-SXM4-80GB"}


class _Block(str):
    """A string rendered as a YAML literal block, so embedded YAML stays readable."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _Block,
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|"),
)


class WorkerPool(BaseModel):
    """The capacity one accelerator type needs for the models placed on it."""

    accelerator: str
    ray_accelerator: str
    gpus_per_worker: int
    memory_gb: int
    min_workers: int
    max_workers: int
    models: tuple[str, ...]

    @property
    def group_name(self) -> str:
        return f"{self.ray_accelerator.lower()}-pool"

    @property
    def resource(self) -> str:
        return f"gpu_pool_{self.accelerator}"


def ray_accelerator(accelerator: str) -> str:
    """Translate a catalog accelerator into the name Ray knows it by."""

    return accelerator.removeprefix("nvidia-").upper()


def worker_pools(registry: Registry) -> tuple[WorkerPool, ...]:
    """One pool per accelerator type, sized from its models' autoscaling bounds.

    A worker holds as many GPUs as the largest replica on its pool needs, so a
    tensor-parallel model always fits on one node. Experiments add headroom
    to the ceiling but never to the floor: they are never kept warm.
    """

    config = build_serving_config(registry)
    pools: dict[str, dict[str, Any]] = {}
    for entry in (*config["applications"], *config.get("experiments", ())):
        card = registry.model_card(entry.get("base_model_id", entry["model_id"]))
        scaling = entry["deployment_config"]["autoscaling_config"]
        pool = pools.setdefault(
            card.hardware.accelerator,
            {"gpus": 0, "memory": 0, "min": 0, "max": 0, "models": []},
        )
        pool["gpus"] = max(pool["gpus"], card.hardware.count)
        pool["memory"] = max(pool["memory"], card.hardware.minimum_memory_gb)
        pool["min"] += scaling["min_replicas"]
        pool["max"] += scaling["max_replicas"]
        pool["models"].append(entry["model_id"])
    return tuple(
        WorkerPool(
            accelerator=accelerator,
            ray_accelerator=ray_accelerator(accelerator),
            gpus_per_worker=pool["gpus"],
            memory_gb=pool["memory"],
            min_workers=pool["min"],
            max_workers=pool["max"],
            models=tuple(pool["models"]),
        )
        for accelerator, pool in sorted(pools.items())
    )


def build_serve_application(
    registry: Registry, artifact_root: str = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    """The Ray Serve LLM application: one OpenAI-compatible endpoint for every model.

    Only fields Ray's ``LLMConfig`` accepts are kept. Model weights are read
    from the location governance records for each revision, so what Ray loads
    is what MLflow says was promoted.
    """

    root = artifact_root.rstrip("/")
    sources = {item.name: item.source for item in governed_versions(registry, root)}
    config = build_serving_config(registry)
    llm_configs: list[dict[str, Any]] = []
    for entry in (*config["applications"], *config.get("experiments", ())):
        base = entry.get("base_model_id", entry["model_id"])
        card = registry.model_card(base)
        deployment = {
            **entry["deployment_config"],
            "ray_actor_options": {
                "num_gpus": card.hardware.count,
                "resources": {f"gpu_pool_{card.hardware.accelerator}": 0.001},
            },
        }
        llm_config: dict[str, Any] = {
            "model_loading_config": {"model_id": entry["model_id"], "model_source": sources[base]},
            "accelerator_type": ray_accelerator(card.hardware.accelerator),
            "deployment_config": deployment,
            "engine_kwargs": _engine_kwargs(entry["engine_kwargs"], sources),
            "log_engine_metrics": True,
        }
        if "lora_config" in entry:
            llm_config["lora_config"] = {
                # Ray resolves <path>/<adapter id>; the deploy step publishes
                # each promoted adapter revision there.
                "dynamic_lora_loading_path": f"{root}/adapters",
                "max_num_adapters_per_replica": len(entry["lora_config"]["adapters"]),
            }
        llm_configs.append(llm_config)
    return {
        "applications": [
            {
                "name": "llm_app",
                "route_prefix": "/",
                "import_path": "ray.serve.llm:build_openai_app",
                "args": {"llm_configs": llm_configs},
            }
        ]
    }


def _engine_kwargs(engine: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    """Point a speculative draft model at its recorded artifact as well."""

    speculative = engine.get("speculative_config")
    if speculative is None:
        return engine
    draft = str(speculative["model"]).removeprefix("registry://").split("@", 1)[0]
    return {**engine, "speculative_config": {**speculative, "model": sources[draft]}}


def _pod(
    *,
    image: str,
    cpu: str,
    memory: str,
    gpus: int = 0,
    node_selector: dict[str, str] | None = None,
    head: bool = False,
) -> dict[str, Any]:
    resources: dict[str, str] = {"cpu": cpu, "memory": memory}
    if gpus:
        resources["nvidia.com/gpu"] = str(gpus)
    ports = [{"name": "metrics", "containerPort": METRICS_PORT}]
    if head:
        ports += [
            {"name": "gcs", "containerPort": 6379},
            {"name": "dashboard", "containerPort": DASHBOARD_PORT},
            {"name": "serve", "containerPort": SERVE_PORT},
        ]
    else:
        ports.append({"name": "serve", "containerPort": SERVE_PORT})
    spec: dict[str, Any] = {
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "ray-head" if head else "ray-worker",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "ports": ports,
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "resources": {"requests": dict(resources), "limits": dict(resources)},
                "volumeMounts": [
                    {"name": "tmp", "mountPath": "/tmp"},
                    {"name": "cache", "mountPath": "/home/ray/.cache"},
                    {"name": "shm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {"name": "tmp", "emptyDir": {}},
            {"name": "cache", "emptyDir": {}},
            {"name": "shm", "emptyDir": {"medium": "Memory"}},
        ],
        # Long enough for a replica to drain its in-flight generations.
        "terminationGracePeriodSeconds": 60 if head else 120,
    }
    if node_selector:
        spec["nodeSelector"] = node_selector
        spec["tolerations"] = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ]
    return {
        "metadata": {
            "labels": {
                "app.kubernetes.io/name": POD_LABEL,
                "app.kubernetes.io/part-of": "local-llm-router",
            }
        },
        "spec": spec,
    }


def _worker_group(pool: WorkerPool, image: str) -> dict[str, Any]:
    product = GPU_PRODUCT_LABELS.get(pool.ray_accelerator, f"NVIDIA-{pool.ray_accelerator}")
    return {
        "groupName": pool.group_name,
        "replicas": pool.min_workers,
        "minReplicas": pool.min_workers,
        "maxReplicas": pool.max_workers,
        "rayStartParams": {
            "metrics-export-port": str(METRICS_PORT),
            # The custom resource pins a model to its own pool even when two
            # pools could both satisfy a bare GPU request.
            "resources": json.dumps(json.dumps({pool.resource: 1})),
        },
        "template": _pod(
            image=image,
            cpu=str(4 * pool.gpus_per_worker),
            memory=f"{pool.memory_gb}Gi",
            gpus=pool.gpus_per_worker,
            node_selector={"nvidia.com/gpu.product": product},
        ),
    }


def build_ray_service(
    registry: Registry,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    image: str = RAY_IMAGE,
) -> dict[str, Any]:
    """Render the RayService that serves every local model in the catalog."""

    serve_config = yaml.safe_dump(build_serve_application(registry, artifact_root), sort_keys=False)
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayService",
        "metadata": {
            "name": SERVICE_NAME,
            "namespace": NAMESPACE,
            "labels": {"app.kubernetes.io/part-of": "local-llm-router"},
            "annotations": {"llm-routing/policy-version": registry.policy.version},
        },
        "spec": {
            "serveConfigV2": _Block(serve_config),
            "rayClusterConfig": {
                "rayVersion": RAY_VERSION,
                "enableInTreeAutoscaling": True,
                "headGroupSpec": {
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "metrics-export-port": str(METRICS_PORT),
                        # The head schedules; it never runs a replica.
                        "num-cpus": "0",
                    },
                    "template": _pod(image=image, cpu="2", memory="8Gi", head=True),
                },
                "workerGroupSpecs": [_worker_group(pool, image) for pool in worker_pools(registry)],
            },
        },
    }


HEADER = (
    "# Generated from config/registry.yaml by `python -m llm_router.topology`.\n"
    "# Do not edit: change the catalog and regenerate. A test and CD both fail on drift.\n"
)


def render_ray_service(registry: Registry, artifact_root: str = DEFAULT_ARTIFACT_ROOT) -> str:
    manifest = build_ray_service(registry, artifact_root)
    return HEADER + yaml.dump(manifest, Dumper=_Dumper, sort_keys=False, width=1000)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the RayService manifest for the committed catalog."""

    import argparse
    import sys

    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--catalog", default="config/registry.yaml")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    arguments = parser.parse_args(argv)
    sys.stdout.write(render_ray_service(load_registry(arguments.catalog), arguments.artifact_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
