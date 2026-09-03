"""Invariants for the deployment topology in section 17.

These assertions are the deployment contract: every manifest must parse, run
unprivileged with bounded resources and real probes, carry no secret material,
and keep stateless ingress scaling independently of GPU replicas.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

MANIFEST_DIR = Path("deploy/kubernetes")
MANIFESTS = sorted(MANIFEST_DIR.glob("*.yaml"))
SECRET_MARKERS = ("password:", "token:", "apiKey:", "api_key:", "BEGIN PRIVATE KEY")


def load_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in MANIFESTS:
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if document:
                documents.append(document)
    return documents


DOCUMENTS = load_documents()
WORKLOADS = [
    document for document in DOCUMENTS if document["kind"] in {"Deployment", "StatefulSet"}
]


def pod_spec(workload: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = workload["spec"]["template"]["spec"]
    return spec


def test_every_manifest_is_listed_in_the_kustomization() -> None:
    kustomization = yaml.safe_load((MANIFEST_DIR / "kustomization.yaml").read_text())

    listed = set(kustomization["resources"])
    on_disk = {path.name for path in MANIFESTS} - {"kustomization.yaml"}
    assert listed == on_disk


def test_all_resources_are_namespaced_to_the_platform() -> None:
    namespace = next(document for document in DOCUMENTS if document["kind"] == "Namespace")
    assert namespace["metadata"]["name"] == "llm-routing"

    for document in DOCUMENTS:
        if document["kind"] in {"Namespace", "Kustomization"}:
            continue
        assert document["metadata"]["namespace"] == "llm-routing", document["metadata"]["name"]


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: str(item["metadata"]["name"]))
def test_workloads_run_unprivileged(workload: dict[str, Any]) -> None:
    spec = pod_spec(workload)

    assert spec["securityContext"]["runAsNonRoot"] is True
    for container in spec["containers"]:
        security = container["securityContext"]
        assert security["allowPrivilegeEscalation"] is False
        assert security["readOnlyRootFilesystem"] is True
        assert security["capabilities"]["drop"] == ["ALL"]


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: str(item["metadata"]["name"]))
def test_workloads_declare_bounded_resources_and_probes(workload: dict[str, Any]) -> None:
    for container in pod_spec(workload)["containers"]:
        assert container["resources"]["requests"]
        assert container["resources"]["limits"]
        assert container["livenessProbe"]
        assert container["readinessProbe"]


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: str(item["metadata"]["name"]))
def test_images_are_pinned_by_digest(workload: dict[str, Any]) -> None:
    for container in pod_spec(workload)["containers"]:
        assert "@sha256:" in container["image"], container["image"]


def test_no_manifest_contains_secret_material() -> None:
    for path in MANIFESTS:
        content = path.read_text(encoding="utf-8")
        assert "kind: Secret\n" not in content
        for marker in SECRET_MARKERS:
            assert marker not in content, f"{path.name} contains {marker}"


def test_gateway_reads_credentials_from_the_secret_manager() -> None:
    external_secret = next(
        document for document in DOCUMENTS if document["kind"] == "ExternalSecret"
    )
    gateway = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "llm-gateway"
    )
    container = pod_spec(gateway)["containers"][0]
    api_keys = next(item for item in container["env"] if item["name"] == "ROUTER_API_KEYS")

    assert "value" not in api_keys
    assert (
        api_keys["valueFrom"]["secretKeyRef"]["name"] == external_secret["spec"]["target"]["name"]
    )


def test_gateway_probes_target_the_health_and_readiness_endpoints() -> None:
    gateway = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "llm-gateway"
    )
    container = pod_spec(gateway)["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["lifecycle"]["preStop"], "graceful shutdown drain is required"


def test_gpu_replicas_are_pinned_to_an_accelerator_pool() -> None:
    engine = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "vllm-serve"
    )
    spec = pod_spec(engine)
    container = spec["containers"][0]

    assert spec["nodeSelector"]["nvidia.com/gpu.product"]
    assert any(toleration["key"] == "nvidia.com/gpu" for toleration in spec["tolerations"])
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert spec["terminationGracePeriodSeconds"] >= 120


def test_stateless_ingress_scales_independently_of_gpu_replicas() -> None:
    scaled_object = next(document for document in DOCUMENTS if document["kind"] == "ScaledObject")

    assert scaled_object["spec"]["scaleTargetRef"]["name"] == "llm-gateway"
    assert scaled_object["spec"]["minReplicaCount"] >= 2
    assert scaled_object["spec"]["maxReplicaCount"] > scaled_object["spec"]["minReplicaCount"]
    metrics = {trigger["metadata"]["metricName"] for trigger in scaled_object["spec"]["triggers"]}
    assert "router_queued_requests" in metrics


def test_metrics_are_scraped_and_reachable_only_from_monitoring() -> None:
    monitor = next(document for document in DOCUMENTS if document["kind"] == "ServiceMonitor")
    policy = next(
        document
        for document in DOCUMENTS
        if document["kind"] == "NetworkPolicy" and document["metadata"]["name"] == "llm-gateway"
    )

    assert monitor["spec"]["endpoints"][0]["path"] == "/metrics"
    namespaces = {
        rule["namespaceSelector"]["matchLabels"]["kubernetes.io/metadata.name"]
        for entry in policy["spec"]["ingress"]
        for rule in entry["from"]
    }
    assert namespaces == {"applications", "monitoring"}


def test_engine_ingress_is_restricted_to_the_gateway() -> None:
    policy = next(
        document
        for document in DOCUMENTS
        if document["kind"] == "NetworkPolicy" and document["metadata"]["name"] == "vllm-serve"
    )

    sources = policy["spec"]["ingress"][0]["from"]
    assert sources == [{"podSelector": {"matchLabels": {"app.kubernetes.io/name": "llm-gateway"}}}]
