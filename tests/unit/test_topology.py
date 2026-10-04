"""The Ray head and worker topology, GPU support, dashboards and alerts (section 17)."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_router.app import create_app
from llm_router.config import Settings
from llm_router.governance import governed_versions
from llm_router.registry import load_registry
from llm_router.topology import (
    build_ray_service,
    build_serve_application,
    main,
    ray_accelerator,
    render_ray_service,
    worker_pools,
)

CATALOG = load_registry("config/registry.yaml")
BASE = Path("deploy/kubernetes")
OVERLAY = Path("deploy/overlays/ray")
SERVICE = build_ray_service(CATALOG)
CLUSTER = SERVICE["spec"]["rayClusterConfig"]
# The fields Ray's LLMConfig accepts; anything else is rejected at deploy time.
LLM_CONFIG_FIELDS = {
    "model_loading_config",
    "engine_kwargs",
    "accelerator_type",
    "deployment_config",
    "lora_config",
    "log_engine_metrics",
}


def llm_configs() -> dict[str, dict[str, Any]]:
    application = build_serve_application(CATALOG)["applications"][0]
    return {
        item["model_loading_config"]["model_id"]: item
        for item in application["args"]["llm_configs"]
    }


def worker_group(name: str) -> dict[str, Any]:
    return next(group for group in CLUSTER["workerGroupSpecs"] if group["groupName"] == name)


def pod_templates() -> list[dict[str, Any]]:
    return [
        CLUSTER["headGroupSpec"]["template"],
        *(group["template"] for group in CLUSTER["workerGroupSpecs"]),
    ]


def test_accelerators_are_named_the_way_ray_knows_them() -> None:
    assert ray_accelerator("nvidia-l4") == "L4"
    assert ray_accelerator("nvidia-a10g") == "A10G"


def test_each_accelerator_type_gets_its_own_pool_sized_from_its_models() -> None:
    pools = {pool.group_name: pool for pool in worker_pools(CATALOG)}

    assert set(pools) == {"l4-pool", "a10g-pool", "a100-pool"}
    assert (pools["l4-pool"].min_workers, pools["l4-pool"].max_workers) == (1, 4)
    # The experiment adds headroom to the ceiling and nothing to the floor.
    assert (pools["a10g-pool"].min_workers, pools["a10g-pool"].max_workers) == (1, 5)
    assert pools["a10g-pool"].models == ("general-local", "general-local--general-gptq")
    # The high-capability tier scales to zero and needs both GPUs on one node.
    assert (pools["a100-pool"].min_workers, pools["a100-pool"].max_workers) == (0, 3)
    assert pools["a100-pool"].gpus_per_worker == 2


def test_the_head_schedules_and_never_runs_a_model() -> None:
    head = CLUSTER["headGroupSpec"]
    container = head["template"]["spec"]["containers"][0]

    assert head["rayStartParams"]["num-cpus"] == "0"
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert "nodeSelector" not in head["template"]["spec"]
    assert CLUSTER["enableInTreeAutoscaling"] is True


def test_workers_are_pinned_to_their_accelerator_and_advertise_their_pool() -> None:
    group = worker_group("a100-pool")
    spec = group["template"]["spec"]

    assert spec["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"}
    assert spec["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert (group["minReplicas"], group["maxReplicas"]) == (0, 3)
    # KubeRay wants the resource map as a quoted JSON string.
    assert json.loads(json.loads(group["rayStartParams"]["resources"])) == {
        "gpu_pool_nvidia-a100": 1
    }
    assert worker_group("l4-pool")["template"]["spec"]["nodeSelector"] == {
        "nvidia.com/gpu.product": "NVIDIA-L4"
    }


def test_every_model_asks_for_the_pool_resource_a_worker_group_provides() -> None:
    provided = {
        resource
        for group in CLUSTER["workerGroupSpecs"]
        for resource in json.loads(json.loads(group["rayStartParams"]["resources"]))
    }

    for model_id, config in llm_configs().items():
        wanted = set(config["deployment_config"]["ray_actor_options"]["resources"])
        assert wanted <= provided, model_id


def test_the_serve_application_uses_only_fields_ray_accepts() -> None:
    application = build_serve_application(CATALOG)["applications"][0]

    assert application["import_path"] == "ray.serve.llm:build_openai_app"
    for model_id, config in llm_configs().items():
        assert set(config) <= LLM_CONFIG_FIELDS, model_id
        assert set(config["model_loading_config"]) == {"model_id", "model_source"}
    small = llm_configs()["small-specialist"]
    assert small["accelerator_type"] == "L4"
    assert small["lora_config"] == {
        "dynamic_lora_loading_path": "s3://llm-routing-artifacts/adapters",
        "max_num_adapters_per_replica": 3,
    }
    assert "lora_config" not in llm_configs()["general-local"]


def test_ray_loads_the_artifact_governance_recorded() -> None:
    recorded = {item.name: item.source for item in governed_versions(CATALOG, "s3://bucket")}
    application = build_serve_application(CATALOG, "s3://bucket/")["applications"][0]
    configs = {
        item["model_loading_config"]["model_id"]: item
        for item in application["args"]["llm_configs"]
    }

    for model_id in ("small-specialist", "general-local", "high-capability"):
        assert configs[model_id]["model_loading_config"]["model_source"] == recorded[model_id]
    # A variant serves its base model's weights, and a draft model its own.
    speculative = configs["high-capability--high-capability-speculative"]
    assert speculative["model_loading_config"]["model_source"] == recorded["high-capability"]
    assert (
        speculative["engine_kwargs"]["speculative_config"]["model"]
        == (recorded["small-specialist"])
    )
    assert "approved-external-fallback" not in configs


@pytest.mark.parametrize(
    "template", pod_templates(), ids=lambda item: item["spec"]["containers"][0]["name"]
)
def test_ray_pods_meet_the_same_contract_as_every_other_workload(template: dict[str, Any]) -> None:
    spec = template["spec"]
    container = spec["containers"][0]

    assert template["metadata"]["labels"]["app.kubernetes.io/name"] == "ray-serve"
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert "@sha256:" in container["image"]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert any(port["name"] == "metrics" for port in container["ports"])


def test_the_committed_manifest_is_what_the_catalog_generates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    committed = (OVERLAY / "ray-service.yaml").read_text(encoding="utf-8")

    assert committed == render_ray_service(CATALOG)
    assert main([]) == 0
    assert capsys.readouterr().out == committed
    embedded = yaml.safe_load(yaml.safe_load(committed)["spec"]["serveConfigV2"])
    assert embedded == build_serve_application(CATALOG)


def test_the_overlay_lists_files_that_exist_and_replaces_the_single_engine() -> None:
    kustomization = yaml.safe_load((OVERLAY / "kustomization.yaml").read_text(encoding="utf-8"))

    assert kustomization["resources"][0] == "../../kubernetes"
    for resource in kustomization["resources"][1:]:
        assert (OVERLAY / resource).is_file(), resource
    removed = {
        (patch["kind"], patch["metadata"]["name"])
        for patch in (
            yaml.safe_load(item["patch"]) for item in kustomization["patches"] if "patch" in item
        )
        if patch.get("$patch") == "delete"
    }
    assert removed == {
        ("Deployment", "vllm-serve"),
        ("Service", "vllm-serve"),
        ("NetworkPolicy", "vllm-serve"),
    }


def test_inference_reaches_ray_only_from_the_gateway() -> None:
    policy = yaml.safe_load((OVERLAY / "network-policy.yaml").read_text(encoding="utf-8"))["spec"]
    gateway = yaml.safe_load(
        (OVERLAY / "gateway-network-policy.patch.yaml").read_text(encoding="utf-8")
    )["spec"]

    serve_sources = [
        entry["from"]
        for entry in policy["ingress"]
        if any(port["port"] == 8000 for port in entry.get("ports", []))
    ]
    assert serve_sources == [
        [{"podSelector": {"matchLabels": {"app.kubernetes.io/name": "llm-gateway"}}}]
    ]
    targets = {
        rule["podSelector"]["matchLabels"]["app.kubernetes.io/name"]
        for entry in gateway["egress"]
        for rule in entry["to"]
    }
    assert targets == {"ray-serve", "redis", "litellm-proxy"}
    for entry in policy["egress"]:
        assert all("ipBlock" not in rule for rule in entry["to"])


@pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl is not installed")
def test_the_overlay_renders_with_ray_in_place_of_the_single_engine() -> None:
    rendered = subprocess.run(
        ["kubectl", "kustomize", str(OVERLAY)], capture_output=True, text=True, check=True
    )
    documents = [item for item in yaml.safe_load_all(rendered.stdout) if item]
    names = {(item["kind"], item["metadata"]["name"]) for item in documents}

    assert ("RayService", "llm-serve") in names
    assert ("PodMonitor", "ray-serve") in names
    assert not any(name == "vllm-serve" for _, name in names)
    gateway = next(
        item
        for item in documents
        if item["metadata"]["name"] == "llm-gateway" and item["kind"] == "Deployment"
    )
    environment = {
        item["name"]: item.get("value")
        for item in gateway["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert environment["ROUTER_VLLM_BASE_URL"] == (
        "http://llm-serve-serve-svc.llm-routing.svc.cluster.local:8000"
    )
    # Everything the base promised about the gateway is still there.
    assert environment["ROUTER_BACKEND"] == "vllm"
    assert "ROUTER_JWT_JWKS" in environment


def emitted_metrics() -> set[str]:
    """Metric families the gateway publishes, without their sample suffixes."""

    with TestClient(create_app(Settings(api_keys="k"))) as client:
        exposition = client.get("/metrics").text
    return {
        re.sub(r"_total$", "", line.split()[2])
        for line in exposition.splitlines()
        if line.startswith("# TYPE router_")
    }


def referenced_metrics(expression: str) -> set[str]:
    names = set(re.findall(r"\b(?:router_|DCGM_)[A-Za-z0-9_]+", expression))
    return {re.sub(r"_(total|bucket|sum|count)$", "", name) for name in names}


def dashboards() -> dict[str, dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((BASE / "dashboards").glob("*.json"))
    }


def alert_rules() -> list[dict[str, Any]]:
    documents = yaml.safe_load_all((BASE / "observability.yaml").read_text(encoding="utf-8"))
    rule = next(item for item in documents if item["kind"] == "PrometheusRule")
    return [alert for group in rule["spec"]["groups"] for alert in group["rules"]]


def test_dashboards_and_alerts_only_query_metrics_that_exist() -> None:
    emitted = emitted_metrics()
    expressions = [
        target["expr"]
        for board in dashboards().values()
        for panel in board["panels"]
        for target in panel["targets"]
    ] + [alert["expr"] for alert in alert_rules()]

    assert len(expressions) > 30
    for expression in expressions:
        used = referenced_metrics(expression)
        assert used, expression
        for name in used:
            if name.startswith("DCGM_"):
                assert name in {
                    "DCGM_FI_DEV_GPU_UTIL",
                    "DCGM_FI_DEV_FB_USED",
                    "DCGM_FI_DEV_FB_FREE",
                }
            else:
                assert name in emitted, f"{name} in {expression}"


def test_the_spec_inference_and_routing_metrics_are_all_on_a_dashboard() -> None:
    charted = {
        name
        for board in dashboards().values()
        for panel in board["panels"]
        for target in panel["targets"]
        for name in referenced_metrics(target["expr"])
    }

    assert {
        "router_time_to_first_token_seconds",
        "router_time_per_output_token_seconds",
        "router_request_latency_seconds",
        "router_requests",
        "router_tokens",
        "router_engine_batch_size",
        "router_queued_requests",
        "router_inflight_requests",
        "router_gpu_utilization_ratio",
        "router_gpu_memory_used_bytes",
        "router_engine_kv_cache_occupancy_ratio",
        "router_model_load_seconds",
        "router_routes",
        "router_fallbacks",
        "router_queue_delay_prediction_error_ms",
        "router_predicted_quality",
        "router_observed_quality",
        "router_cache_events",
        "router_rejections",
    } <= charted


def test_every_dashboard_is_shipped_to_grafana() -> None:
    kustomization = yaml.safe_load((BASE / "kustomization.yaml").read_text(encoding="utf-8"))
    generator = kustomization["configMapGenerator"][0]

    assert {Path(item).name for item in generator["files"]} == set(dashboards())
    assert generator["options"]["labels"] == {"grafana_dashboard": "1"}
    assert generator["options"]["disableNameSuffixHash"] is True
    for board in dashboards().values():
        assert board["uid"].startswith("llm-routing-") and board["panels"]


def test_alerts_filter_on_rejection_types_the_gateway_really_reports() -> None:
    source = Path("src/llm_router/app.py").read_text(encoding="utf-8")
    shedding = next(item for item in alert_rules() if item["alert"] == "GatewayRejectingRequests")

    for rejection in re.search(r'type=~"([^"]+)"', shedding["expr"]).group(1).split("|"):  # type: ignore[union-attr]
        assert f'record_rejection("{rejection}")' in source
    for alert in alert_rules():
        assert alert["labels"]["severity"] in {"warning", "critical"}
        assert alert["annotations"]["summary"]


def test_the_gpu_operator_supplies_what_the_serving_manifests_rely_on() -> None:
    values = yaml.safe_load(Path("deploy/gpu-operator/values.yaml").read_text(encoding="utf-8"))

    # The nvidia.com/gpu resource, the gpu.product label, and the GPU exporter.
    assert values["devicePlugin"]["enabled"] and values["driver"]["enabled"]
    assert values["nfd"]["enabled"] and values["gfd"]["enabled"]
    assert values["dcgmExporter"]["enabled"]
    assert values["dcgmExporter"]["serviceMonitor"]["enabled"]
    assert values["daemonsets"]["tolerations"][0]["key"] == "nvidia.com/gpu"
