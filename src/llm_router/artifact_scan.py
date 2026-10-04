"""Model and adapter artifact scanning before release (section 14).

Loading model weights can run code. A pickle-based checkpoint executes
whatever it names when it is opened, and a repository that ships its own
Python asks the loader to import it. This module inspects an artifact
directory without loading anything from it and refuses both, along with
weights whose safetensors header does not describe the bytes that follow and
an artifact whose digest is not the one the catalog recorded.
"""

import hashlib
import io
import json
import pickletools
import struct
import zipfile
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Formats that deserialize through pickle, or can.
PICKLE_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib", ".npy", ".npz"}
)
CODE_SUFFIXES = frozenset({".py", ".pyc", ".sh", ".so", ".dll", ".dylib", ".exe"})
CONFIG_FILES = frozenset({"config.json", "adapter_config.json", "tokenizer_config.json"})
MAX_HEADER_BYTES = 100 * 1024 * 1024
# The serving configuration loads adapters up to this rank and no further.
MAX_LORA_RANK = 32
SAFETENSORS_DTYPES = frozenset(
    {"BOOL", "U8", "I8", "I16", "U16", "F16", "BF16", "I32", "U32", "F32", "F64", "I64", "U64"}
)


class ArtifactKind(StrEnum):
    MODEL = "model"
    ADAPTER = "adapter"


class Rule(StrEnum):
    PICKLE = "pickle-format"
    CODE = "executable-code"
    REMOTE_CODE = "remote-code"
    SAFETENSORS = "invalid-safetensors"
    NO_WEIGHTS = "no-weights"
    SYMLINK = "symlink"
    ADAPTER_CONFIG = "adapter-config"
    CHECKSUM = "checksum-mismatch"


class Finding(BaseModel):
    rule: Rule
    path: str
    detail: str


class ScanReport(BaseModel):
    root: str
    kind: ArtifactKind
    digest: str
    files: int
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


def _files(root: Path) -> Iterator[Path]:
    yield from sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())


def artifact_digest(root: Path) -> str:
    """One digest for a whole artifact: every file's path and content, in order."""

    outer = hashlib.sha256()
    for path in _files(root):
        if path.is_symlink():
            continue
        inner = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                inner.update(block)
        outer.update(f"{path.relative_to(root).as_posix()}\0{inner.hexdigest()}\n".encode())
    return f"sha256:{outer.hexdigest()}"


def pickle_imports(data: bytes) -> tuple[str, ...]:
    """The callables a pickle would import, read from its opcodes without running it."""

    imports: list[str] = []
    strings: list[str] = []
    try:
        for opcode, argument, _ in pickletools.genops(io.BytesIO(data)):
            if opcode.name == "GLOBAL" and isinstance(argument, str):
                imports.append(argument.replace(" ", "."))
            elif opcode.name == "STACK_GLOBAL" and len(strings) >= 2:
                imports.append(f"{strings[-2]}.{strings[-1]}")
            elif isinstance(argument, str):
                strings.append(argument)
    except Exception:
        # A truncated or hostile stream; what was read so far is still reported.
        pass
    return tuple(dict.fromkeys(imports))


def _pickle_payloads(path: Path) -> list[bytes]:
    """Pickle streams in a file, whatever it is named."""

    with path.open("rb") as handle:
        head = handle.read(4)
    if head[:2] == b"PK":
        try:
            with zipfile.ZipFile(path) as archive:
                return [archive.read(name) for name in archive.namelist() if name.endswith(".pkl")]
        except zipfile.BadZipFile:
            return []
    if len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        return [path.read_bytes()]
    return []


def _safetensors_problem(path: Path) -> str | None:
    """Why a safetensors file is not what its header claims, or None if it is sound."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            return "shorter than the 8-byte header length"
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length > MAX_HEADER_BYTES or header_length > size - 8:
            return f"header length {header_length} does not fit the file"
        raw = handle.read(header_length)
    try:
        header = json.loads(raw)
    except ValueError:
        return "header is not JSON"
    if not isinstance(header, dict):
        return "header is not an object"
    data_size = size - 8 - header_length
    spans: list[tuple[int, int]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in SAFETENSORS_DTYPES:
            return f"tensor {name} has no known dtype"
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            return f"tensor {name} points outside the file"
        spans.append((offsets[0], offsets[1]))
    position = 0
    for begin, end in sorted(spans):
        if begin != position:
            return "tensor data has a gap or an overlap"
        position = end
    if position != data_size:
        return "file holds bytes no tensor accounts for"
    return None


def _config_findings(path: Path, relative: str, kind: ArtifactKind) -> list[Finding]:
    try:
        config: Any = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return [Finding(rule=Rule.REMOTE_CODE, path=relative, detail="configuration is not JSON")]
    if not isinstance(config, dict):
        return [
            Finding(rule=Rule.REMOTE_CODE, path=relative, detail="configuration is not an object")
        ]
    findings: list[Finding] = []
    if "auto_map" in config or config.get("trust_remote_code"):
        findings.append(
            Finding(
                rule=Rule.REMOTE_CODE,
                path=relative,
                detail="asks the loader to import code shipped with the artifact",
            )
        )
    if path.name == "adapter_config.json" and kind is ArtifactKind.ADAPTER:
        if str(config.get("peft_type", "")).upper() != "LORA":
            findings.append(
                Finding(
                    rule=Rule.ADAPTER_CONFIG,
                    path=relative,
                    detail=f"peft_type is {config.get('peft_type')!r}, only LORA is served",
                )
            )
        rank = config.get("r")
        if not isinstance(rank, int) or not 1 <= rank <= MAX_LORA_RANK:
            findings.append(
                Finding(
                    rule=Rule.ADAPTER_CONFIG,
                    path=relative,
                    detail=f"rank {rank!r} is outside 1..{MAX_LORA_RANK}",
                )
            )
    return findings


def scan(
    root: Path, kind: ArtifactKind = ArtifactKind.MODEL, expected_digest: str | None = None
) -> ScanReport:
    """Inspect an artifact directory and report everything that blocks its release."""

    resolved = root.resolve()
    findings: list[Finding] = []
    weights = 0
    count = 0
    for path in _files(root):
        count += 1
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(resolved):
                findings.append(
                    Finding(rule=Rule.SYMLINK, path=relative, detail="links outside the artifact")
                )
            continue
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES:
            findings.append(
                Finding(rule=Rule.CODE, path=relative, detail="artifacts may not ship code")
            )
            continue
        payloads = _pickle_payloads(path)
        if payloads or suffix in PICKLE_SUFFIXES:
            imports = [name for payload in payloads for name in pickle_imports(payload)]
            detail = "weights must be safetensors; this format runs code when loaded"
            if imports:
                detail += f" (imports {', '.join(sorted(set(imports))[:8])})"
            findings.append(Finding(rule=Rule.PICKLE, path=relative, detail=detail))
            continue
        if suffix == ".safetensors":
            weights += 1
            problem = _safetensors_problem(path)
            if problem:
                findings.append(Finding(rule=Rule.SAFETENSORS, path=relative, detail=problem))
        elif path.name in CONFIG_FILES:
            findings.extend(_config_findings(path, relative, kind))
    if not weights:
        findings.append(Finding(rule=Rule.NO_WEIGHTS, path=".", detail="no safetensors weights"))
    if kind is ArtifactKind.ADAPTER and not (root / "adapter_config.json").is_file():
        findings.append(
            Finding(rule=Rule.ADAPTER_CONFIG, path=".", detail="adapter_config.json is missing")
        )
    digest = artifact_digest(root)
    if expected_digest and digest != expected_digest:
        findings.append(
            Finding(
                rule=Rule.CHECKSUM,
                path=".",
                detail=f"digest is {digest}, the catalog records {expected_digest}",
            )
        )
    return ScanReport(
        root=root.as_posix(), kind=kind, digest=digest, files=count, findings=tuple(findings)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Scan a model or adapter directory; exit 1 if anything blocks its release.

    With --subject the digest is checked against the checksum the catalog
    records for that model or adapter. With --expect it is checked against
    the digest given.
    """

    import argparse

    from llm_router.governance import SubjectKind, governed_versions
    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("path")
    parser.add_argument("--kind", choices=[item.value for item in ArtifactKind])
    parser.add_argument("--subject", help="catalog model or adapter this artifact is")
    parser.add_argument("--expect", help="digest the artifact must have")
    parser.add_argument("--catalog", default="config/registry.yaml")
    arguments = parser.parse_args(argv)

    root = Path(arguments.path)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")
    kind = ArtifactKind(arguments.kind) if arguments.kind else ArtifactKind.MODEL
    expected = arguments.expect
    if arguments.subject:
        subject = next(
            (
                item
                for item in governed_versions(load_registry(arguments.catalog))
                if item.name == arguments.subject
            ),
            None,
        )
        if subject is None:
            parser.error(f"{arguments.subject} is not in the catalog")
        if not arguments.kind and subject.kind is SubjectKind.ADAPTER:
            kind = ArtifactKind.ADAPTER
        expected = expected or subject.checksum

    report = scan(root, kind, expected)
    print(json.dumps({**report.model_dump(mode="json"), "passed": report.passed}, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
