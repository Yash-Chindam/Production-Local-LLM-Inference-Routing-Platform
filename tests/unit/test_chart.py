"""The Helm chart is the reviewed manifests with installation values lifted out."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_router.chart import (
    CHART,
    IMAGES,
    ChartError,
    _template,
    build_chart,
    main,
    stale_files,
    write_chart,
)

TOOLS_PRESENT = shutil.which("helm") is not None and shutil.which("kubectl") is not None
# Skipping is for a workstation without the tools. In CI their absence is a failure.
needs_tools = pytest.mark.skipif(
    not TOOLS_PRESENT and not os.environ.get("CI"), reason="helm and kubectl are not installed"
)

Key = tuple[str, str]


def run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def resources(rendered: str) -> dict[Key, dict[str, Any]]:
    found: dict[Key, dict[str, Any]] = {}
    for document in yaml.safe_load_all(rendered):
        if not document or document["kind"] == "Namespace":
            continue
        if document["kind"] == "Deployment":
            # Kustomize moves a patched variable to the front; order means nothing.
            for container in document["spec"]["template"]["spec"]["containers"]:
                container.get("env", []).sort(key=lambda item: item["name"])
        found[document["kind"], document["metadata"]["name"]] = document
    return found


def helm_template(*arguments: str, namespace: str = "llm-routing") -> dict[Key, dict[str, Any]]:
    return resources(run("helm", "template", "release", str(CHART), "-n", namespace, *arguments))


def test_the_committed_chart_is_what_the_manifests_generate() -> None:
    assert stale_files(CHART) == []
    assert main(["--check"]) == 0


def test_a_changed_missing_or_stray_chart_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--output", str(tmp_path)]) == 0
    assert stale_files(tmp_path) == []

    (tmp_path / "values.yaml").write_text("serving:\n  mode: ray\n", encoding="utf-8")
    (tmp_path / "templates" / "gateway.yaml").unlink()
    (tmp_path / "templates" / "hand-written.yaml").write_text("kind: Secret\n", encoding="utf-8")

    assert stale_files(tmp_path) == [
        "templates/gateway.yaml",
        "templates/hand-written.yaml",
        "values.yaml",
    ]
    assert main(["--output", str(tmp_path), "--check"]) == 1
    assert "stale:" in capsys.readouterr().out

    write_chart(tmp_path)
    assert stale_files(tmp_path) == ["templates/hand-written.yaml"]


def test_the_chart_holds_every_manifest_and_owns_no_namespace() -> None:
    chart = build_chart()

    for name in (
        "gateway",
        "state",
        "mlflow",
        "external-provider",
        "vllm-serve",
        "ray-ray-service",
    ):
        assert f"templates/{name}.yaml" in chart
    assert "templates/namespace.yaml" not in chart
    assert "templates/ray-kustomization.yaml" not in chart
    assert all("kind: Namespace" not in content for content in chart.values())
    assert set(yaml.safe_load(chart["values.yaml"])["images"]) == set(IMAGES)
    assert yaml.safe_load(chart["Chart.yaml"])["name"] == "llm-routing"


def test_every_image_and_namespace_reference_is_a_value() -> None:
    templates = {
        name: content
        for name, content in build_chart().items()
        if name.startswith("templates/") and name.endswith(".yaml")
    }

    for name, content in templates.items():
        assert "REPLACE_ME" not in content, name
        assert "namespace: llm-routing" not in content, name
        assert ".llm-routing.svc" not in content, name
    for image in yaml.safe_load(build_chart()["values.yaml"])["images"].values():
        assert "@sha256:" in image


def test_prometheus_templating_in_alerts_survives_helm() -> None:
    rules = build_chart()["templates/observability.yaml"]

    assert '{{ "{{" }} $labels.model }}' in rules
    assert "{{ $labels" not in rules


def test_a_manifest_that_lost_what_the_chart_parameterizes_is_refused() -> None:
    gateway = (
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: llm-gateway\n"
        "spec:\n  replicas: 2\n"
    )

    with pytest.raises(ChartError, match=r"Deployment/llm-gateway: expected to find 'http://"):
        _template(gateway, source="gateway.yaml")


@needs_tools
def test_the_chart_lints_cleanly() -> None:
    assert "0 chart(s) failed" in run("helm", "lint", str(CHART))


@needs_tools
def test_default_values_render_exactly_the_single_engine_base() -> None:
    assert helm_template() == resources(run("kubectl", "kustomize", "deploy/kubernetes"))


@needs_tools
def test_ray_mode_renders_exactly_the_ray_overlay() -> None:
    rendered = helm_template("--set", "serving.mode=ray")

    assert rendered == resources(run("kubectl", "kustomize", "deploy/overlays/ray"))
    assert ("RayService", "llm-serve") in rendered
    assert not any(name == "vllm-serve" for _, name in rendered)


@needs_tools
def test_values_reach_the_rendered_workloads() -> None:
    rendered = helm_template(
        "--set",
        "images.gateway=registry.example/router@sha256:abc",
        "--set",
        "gateway.replicas=5",
        "--set",
        "gateway.autoscaling.maxReplicas=40",
        namespace="inference",
    )
    gateway = rendered["Deployment", "llm-gateway"]
    container = gateway["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item.get("value") for item in container["env"]}

    assert container["image"] == "registry.example/router@sha256:abc"
    assert gateway["spec"]["replicas"] == 5
    assert rendered["ScaledObject", "llm-gateway"]["spec"]["maxReplicaCount"] == 40
    assert environment["ROUTER_VLLM_BASE_URL"] == (
        "http://vllm-serve.inference.svc.cluster.local:8000"
    )
    assert {item["metadata"]["namespace"] for item in rendered.values()} == {"inference"}


@needs_tools
def test_an_unknown_serving_mode_is_refused() -> None:
    with pytest.raises(subprocess.CalledProcessError) as raised:
        helm_template("--set", "serving.mode=both")
    assert "serving.mode must be vllm or ray" in raised.value.stderr
