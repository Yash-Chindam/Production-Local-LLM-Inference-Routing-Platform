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


def secret_material(content: str) -> list[str]:
    """Lines that hold a secret, as opposed to pointing at where one is kept.

    A secret-shaped key may reference the environment, which the secret
    manager populates. It may never carry a value of its own.
    """

    found: list[str] = []
    for line in content.splitlines():
        for marker in SECRET_MARKERS:
            if marker in line and not line.split(marker, 1)[1].strip().startswith("os.environ/"):
                found.append(line.strip())
    return found


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


def external_secret_named(name: str) -> dict[str, Any]:
    return next(
        document
        for document in DOCUMENTS
        if document["kind"] == "ExternalSecret" and document["metadata"]["name"] == name
    )


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
        assert secret_material(content) == [], path.name


def test_gateway_reads_credentials_from_the_secret_manager() -> None:
    external_secret = external_secret_named("llm-gateway-credentials")
    gateway = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "llm-gateway"
    )
    container = pod_spec(gateway)["containers"][0]
    environment = {item["name"]: item for item in container["env"]}
    keys = environment["ROUTER_JWT_JWKS"]

    assert "value" not in keys
    assert keys["valueFrom"]["secretKeyRef"]["name"] == external_secret["spec"]["target"]["name"]
    # The cluster accepts short-lived tokens only; no static key is mounted.
    assert environment["ROUTER_REQUIRE_SHORT_LIVED_CREDENTIALS"]["value"] == "true"
    assert int(environment["ROUTER_JWT_MAX_LIFETIME_SECONDS"]["value"]) <= 3600
    assert "ROUTER_API_KEYS" not in environment
    assert "ROUTER_TENANT_KEYS" not in environment
    provided = {item["secretKey"] for item in external_secret["spec"]["data"]}
    for variable in container["env"]:
        if "valueFrom" in variable:
            assert variable["valueFrom"]["secretKeyRef"]["key"] in provided


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


def test_a_secret_shaped_key_may_reference_the_environment_but_not_hold_a_value() -> None:
    reference = "          api_key: os.environ/EXTERNAL_PROVIDER_API_KEY"
    literal = "          api_key: sk-live-not-a-reference"

    assert secret_material(reference) == []
    assert secret_material("\n".join((reference, literal))) == [literal.strip()]
    assert secret_material("-----BEGIN PRIVATE KEY-----") == ["-----BEGIN PRIVATE KEY-----"]


def test_proxy_configuration_matches_the_committed_litellm_config() -> None:
    config_map = next(document for document in DOCUMENTS if document["kind"] == "ConfigMap")

    embedded = yaml.safe_load(config_map["data"]["config.yaml"])
    committed = yaml.safe_load(Path("config/litellm.yaml").read_text(encoding="utf-8"))
    assert embedded == committed


def test_every_external_model_in_the_catalog_has_a_proxy_alias() -> None:
    catalog = yaml.safe_load(Path("config/registry.yaml").read_text(encoding="utf-8"))
    proxy = yaml.safe_load(Path("config/litellm.yaml").read_text(encoding="utf-8"))

    external = {model["id"] for model in catalog["models"] if model.get("local") is False}
    aliases = {entry["model_name"] for entry in proxy["model_list"]}
    assert external and external == aliases


def test_the_proxy_keeps_prompts_out_of_its_logs_and_leaves_retries_to_the_gateway() -> None:
    proxy = yaml.safe_load(Path("config/litellm.yaml").read_text(encoding="utf-8"))

    assert proxy["litellm_settings"]["turn_off_message_logging"] is True
    assert proxy["litellm_settings"]["num_retries"] == 0
    for entry in proxy["model_list"]:
        assert entry["litellm_params"]["api_key"].startswith("os.environ/")


def test_only_the_gateway_reaches_the_proxy_and_only_the_proxy_reaches_out() -> None:
    policies = {
        document["metadata"]["name"]: document["spec"]
        for document in DOCUMENTS
        if document["kind"] == "NetworkPolicy"
    }
    proxy = policies["litellm-proxy"]

    assert proxy["ingress"][0]["from"] == [
        {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "llm-gateway"}}}
    ]
    blocks = [
        rule["ipBlock"] for entry in proxy["egress"] for rule in entry["to"] if "ipBlock" in rule
    ]
    assert blocks == [
        {"cidr": "0.0.0.0/0", "except": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]}
    ]
    # No other workload has an egress rule that leaves the namespace by address.
    for name, spec in policies.items():
        if name == "litellm-proxy":
            continue
        for entry in spec.get("egress", []):
            assert all("ipBlock" not in rule for rule in entry["to"]), name
    gateway_targets = {
        rule["podSelector"]["matchLabels"]["app.kubernetes.io/name"]
        for entry in policies["llm-gateway"]["egress"]
        for rule in entry["to"]
    }
    assert gateway_targets == {"vllm-serve", "redis", "litellm-proxy"}


def test_the_proxy_reads_both_of_its_keys_from_the_secret_manager() -> None:
    external_secret = external_secret_named("llm-gateway-credentials")
    proxy = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "litellm-proxy"
    )
    provided = {item["secretKey"] for item in external_secret["spec"]["data"]}

    for variable in pod_spec(proxy)["containers"][0]["env"]:
        assert "value" not in variable
        assert variable["valueFrom"]["secretKeyRef"]["key"] in provided


def test_the_governance_store_is_reachable_only_by_the_delivery_pipeline() -> None:
    policy = next(
        document["spec"]
        for document in DOCUMENTS
        if document["kind"] == "NetworkPolicy" and document["metadata"]["name"] == "mlflow"
    )

    assert policy["ingress"] == [
        {
            "from": [
                {
                    "namespaceSelector": {
                        "matchLabels": {"kubernetes.io/metadata.name": "platform-delivery"}
                    }
                }
            ],
            "ports": [{"protocol": "TCP", "port": 5000}],
        }
    ]
    reachable = {
        rule["namespaceSelector"]["matchLabels"]["kubernetes.io/metadata.name"]
        for entry in policy["egress"]
        for rule in entry["to"]
    }
    assert reachable == {"kube-system", "storage"}


def test_the_governance_store_keeps_its_own_credentials_apart_from_the_gateway() -> None:
    mlflow = next(document for document in WORKLOADS if document["metadata"]["name"] == "mlflow")
    gateway = next(
        document for document in WORKLOADS if document["metadata"]["name"] == "llm-gateway"
    )
    provided = {
        item["secretKey"] for item in external_secret_named("mlflow-credentials")["spec"]["data"]
    }

    referenced = {
        variable["valueFrom"]["secretKeyRef"]["key"]
        for variable in pod_spec(mlflow)["containers"][0]["env"]
        if "valueFrom" in variable
    }
    assert referenced == provided
    # The database address embeds a password, so it may never be a literal.
    backend = next(
        variable
        for variable in pod_spec(mlflow)["containers"][0]["env"]
        if variable["name"] == "MLFLOW_BACKEND_STORE_URI"
    )
    assert "value" not in backend
    gateway_secrets = {
        variable["valueFrom"]["secretKeyRef"]["name"]
        for variable in pod_spec(gateway)["containers"][0]["env"]
        if "valueFrom" in variable
    }
    assert gateway_secrets == {"llm-gateway-credentials"}


def test_artifacts_are_proxied_so_clients_never_hold_storage_credentials() -> None:
    mlflow = next(document for document in WORKLOADS if document["metadata"]["name"] == "mlflow")
    arguments = pod_spec(mlflow)["containers"][0]["args"]

    assert "--serve-artifacts" in arguments
    assert any(item.startswith("--artifacts-destination=s3://") for item in arguments)
