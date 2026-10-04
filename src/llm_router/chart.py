"""Helm chart built from the Kubernetes manifests (section 17).

The manifests under ``deploy/kubernetes`` and the Ray overlay are the reviewed
source. The chart is those same documents with the values an installation
changes lifted out: image references, the namespace, gateway scale, and which
serving topology to run. Building it rather than writing it means the chart
cannot drift from the manifests the contract tests check.
"""

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

CHART_NAME = "llm-routing"
BASE = Path("deploy/kubernetes")
RAY_OVERLAY = Path("deploy/overlays/ray")
CHART = Path("deploy/helm") / CHART_NAME
# Applied by the overlay as patches rather than shipped as resources.
OVERLAY_ONLY = {"kustomization.yaml", "gateway-network-policy.patch.yaml"}
ENGINE_URL = "http://vllm-serve.llm-routing.svc.cluster.local:8000"
TOKEN_ISSUER = "https://identity.REPLACE_ME"
TOKEN_AUDIENCE = "llm-gateway"

IMAGES = {
    "gateway": "ghcr.io/REPLACE_ME/local-llm-router@sha256:REPLACE_ME",
    "engine": "docker.io/vllm/vllm-openai@sha256:REPLACE_ME",
    "ray": "docker.io/rayproject/ray-llm@sha256:REPLACE_ME",
    "redis": "docker.io/library/redis@sha256:REPLACE_ME",
    "externalProxy": "ghcr.io/berriai/litellm@sha256:REPLACE_ME",
    "mlflow": "ghcr.io/mlflow/mlflow@sha256:REPLACE_ME",
}

HELPERS = """{{/* The OpenAI-compatible endpoint the gateway sends inference to. */}}
{{- define "llm-routing.engineUrl" -}}
{{- if eq .Values.serving.mode "ray" -}}
http://llm-serve-serve-svc.{{ .Release.Namespace }}.svc.cluster.local:8000
{{- else -}}
http://vllm-serve.{{ .Release.Namespace }}.svc.cluster.local:8000
{{- end -}}
{{- end -}}

{{/* The pods that endpoint resolves to, for the gateway's network policy. */}}
{{- define "llm-routing.engineSelector" -}}
{{- if eq .Values.serving.mode "ray" -}}ray-serve{{- else -}}vllm-serve{{- end -}}
{{- end -}}

{{- define "llm-routing.validate" -}}
{{- if not (has .Values.serving.mode (list "vllm" "ray")) -}}
{{- fail "serving.mode must be vllm or ray" -}}
{{- end -}}
{{- end -}}
"""

DASHBOARDS = """{{- include "llm-routing.validate" . -}}
# Grafana's dashboard sidecar loads any ConfigMap carrying this label.
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-routing-dashboards
  namespace: {{ .Release.Namespace }}
  labels:
    grafana_dashboard: "1"
data:
{{- range $path, $_ := .Files.Glob "dashboards/*.json" }}
  {{ base $path }}: {{ $.Files.Get $path | quote }}
{{- end }}
"""


class ChartError(RuntimeError):
    """Raised when a manifest no longer has the shape the chart expects."""


def _documents(path: Path) -> list[str]:
    """Split a manifest into its documents, keeping comments where they sit."""

    text = path.read_text(encoding="utf-8")
    return [chunk.strip("\n") for chunk in re.split(r"(?m)^---\s*$", text) if chunk.strip()]


def _swap(text: str, old: str, new: str, *, where: str) -> str:
    if old not in text:
        raise ChartError(f"{where}: expected to find {old!r}")
    return text.replace(old, new)


def _template(document: str, *, source: str) -> str | None:
    """Turn one manifest document into a chart template, or drop it."""

    parsed: dict[str, Any] = yaml.safe_load(document)
    kind, name = parsed["kind"], parsed["metadata"]["name"]
    where = f"{source} {kind}/{name}"
    if kind == "Namespace":
        # Helm installs into a namespace it is given; it does not own one.
        return None

    # Alert annotations use Prometheus templating, which Helm must pass through.
    text = document.replace("{{", '{{ "{{" }}')
    text = text.replace("namespace: llm-routing", "namespace: {{ .Release.Namespace }}")
    text = text.replace('namespace="llm-routing"', 'namespace="{{ .Release.Namespace }}"')
    for key, image in IMAGES.items():
        text = text.replace(f"image: {image}", f"image: {{{{ .Values.images.{key} | quote }}}}")

    if kind == "Deployment" and name == "llm-gateway":
        text = _swap(text, ENGINE_URL, '{{ include "llm-routing.engineUrl" . }}', where=where)
        text = _swap(text, "replicas: 2", "replicas: {{ .Values.gateway.replicas }}", where=where)
        for literal, value in ((TOKEN_ISSUER, "issuer"), (TOKEN_AUDIENCE, "audience")):
            text = _swap(
                text,
                f"value: {literal}\n",
                f"value: {{{{ .Values.auth.{value} | quote }}}}\n",
                where=where,
            )
    if kind == "NetworkPolicy" and name == "llm-gateway":
        text = _swap(
            text,
            "app.kubernetes.io/name: vllm-serve",
            'app.kubernetes.io/name: {{ include "llm-routing.engineSelector" . }}',
            where=where,
        )
    if kind == "ScaledObject":
        for field, value in (
            ("minReplicaCount", "minReplicas"),
            ("maxReplicaCount", "maxReplicas"),
        ):
            current = parsed["spec"][field]
            text = _swap(
                text,
                f"{field}: {current}",
                f"{field}: {{{{ .Values.gateway.autoscaling.{value} }}}}",
                where=where,
            )
    text = text.replace(".llm-routing.svc", ".{{ .Release.Namespace }}.svc")

    if name == "vllm-serve":
        return f'{{{{- if eq .Values.serving.mode "vllm" }}}}\n{text}\n{{{{- end }}}}'
    if source.startswith("ray/"):
        return f'{{{{- if eq .Values.serving.mode "ray" }}}}\n{text}\n{{{{- end }}}}'
    return text


def _values(source_root: Path) -> str:
    scaled = next(
        yaml.safe_load(document)
        for document in _documents(source_root / BASE / "autoscaling.yaml")
        if yaml.safe_load(document)["kind"] == "ScaledObject"
    )
    lines = [
        "# Which serving topology to run.",
        "#   vllm: one vLLM engine, serving one model.",
        "#   ray:  a KubeRay RayService serving every local model in the catalog from",
        "#         worker pools split by accelerator type. Needs the KubeRay operator.",
        "serving:",
        "  mode: vllm",
        "",
        "# Every image is pinned by digest. Replace each placeholder with the digest",
        "# recorded for the release; a tag alone is refused by the contract tests.",
        "images:",
        *(f"  {key}: {image}" for key, image in IMAGES.items()),
        "",
        "# The identity provider whose short-lived tokens the gateway accepts. Its",
        "# public keys are read from the secret manager, not from this chart.",
        "auth:",
        f"  issuer: {TOKEN_ISSUER}",
        f"  audience: {TOKEN_AUDIENCE}",
        "",
        "gateway:",
        "  # Starting size; the KEDA ScaledObject takes over within the bounds below.",
        "  replicas: 2",
        "  autoscaling:",
        f"    minReplicas: {scaled['spec']['minReplicaCount']}",
        f"    maxReplicas: {scaled['spec']['maxReplicaCount']}",
    ]
    return "\n".join(lines) + "\n"


def _chart_metadata(source_root: Path) -> str:
    project = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    return (
        "apiVersion: v2\n"
        f"name: {CHART_NAME}\n"
        "description: OpenAI-compatible gateway, policy router and local LLM serving plane.\n"
        "type: application\n"
        f"version: {version}\n"
        f'appVersion: "{version}"\n'
        'kubeVersion: ">=1.27.0-0"\n'
    )


def build_chart(source_root: Path = Path(".")) -> dict[str, str]:
    """Every file of the chart, keyed by its path inside the chart directory."""

    files = {
        "Chart.yaml": _chart_metadata(source_root),
        "values.yaml": _values(source_root),
        "templates/_helpers.tpl": HELPERS,
        "templates/dashboards.yaml": DASHBOARDS,
    }
    sources = [
        (path, path.name)
        for path in sorted((source_root / BASE).glob("*.yaml"))
        if path.name != "kustomization.yaml"
    ] + [
        (path, f"ray/{path.name}")
        for path in sorted((source_root / RAY_OVERLAY).glob("*.yaml"))
        if path.name not in OVERLAY_ONLY
    ]
    for path, source in sources:
        templates = [
            template
            for document in _documents(path)
            if (template := _template(document, source=source)) is not None
        ]
        if templates:
            target = f"templates/{source.replace('/', '-')}"
            files[target] = "\n---\n".join(templates) + "\n"
    for path in sorted((source_root / BASE / "dashboards").glob("*.json")):
        files[f"dashboards/{path.name}"] = path.read_text(encoding="utf-8")
    return files


def write_chart(target: Path, source_root: Path = Path(".")) -> None:
    for relative, content in build_chart(source_root).items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def stale_files(target: Path, source_root: Path = Path(".")) -> list[str]:
    """Chart files that are missing, changed, or no longer generated."""

    expected = build_chart(source_root)
    on_disk = {
        path.relative_to(target).as_posix(): path.read_text(encoding="utf-8")
        for path in target.rglob("*")
        if path.is_file()
    }
    return sorted(
        name for name in expected.keys() | on_disk.keys() if expected.get(name) != on_disk.get(name)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the chart, or with --check exit 1 if the committed chart is stale."""

    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output", default=str(CHART))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    target = Path(arguments.output)
    if arguments.check:
        stale = stale_files(target)
        for name in stale:
            print(f"stale: {target.as_posix()}/{name}")
        return 1 if stale else 0
    write_chart(target)
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
