import json
import os
import pickle
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest

from llm_router.artifact_scan import (
    ArtifactKind,
    Rule,
    artifact_digest,
    main,
    pickle_imports,
    scan,
)


def safetensors(tensors: dict[str, tuple[int, int]], data_size: int, **extra: Any) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}, **extra}
    for name, (begin, end) in tensors.items():
        header[name] = {"dtype": "F32", "shape": [(end - begin) // 4], "data_offsets": [begin, end]}
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + bytes(data_size)


def write_model(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(safetensors({"a": (0, 16), "b": (16, 48)}, 48))
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def write_adapter(root: Path, **config: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(safetensors({"lora_A": (0, 32)}, 32))
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 16, **config}), encoding="utf-8"
    )
    return root


class Hostile:
    """Pickles to a call of os.system. It is only ever serialized, never loaded."""

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.system, ("echo should-never-run",))


def rules(root: Path, kind: ArtifactKind = ArtifactKind.MODEL) -> list[Rule]:
    return [finding.rule for finding in scan(root, kind).findings]


def test_a_sound_model_and_a_sound_adapter_pass(tmp_path: Path) -> None:
    model = scan(write_model(tmp_path / "model"))
    adapter = scan(write_adapter(tmp_path / "adapter"), ArtifactKind.ADAPTER)

    assert model.passed and model.files == 3
    assert adapter.passed
    assert model.digest.startswith("sha256:") and model.digest != adapter.digest


def test_pickle_weights_are_refused_and_what_they_would_import_is_named(tmp_path: Path) -> None:
    root = write_model(tmp_path)
    (root / "pytorch_model.bin").write_bytes(pickle.dumps(Hostile()))

    report = scan(root)

    finding = next(item for item in report.findings if item.rule is Rule.PICKLE)
    assert finding.path == "pytorch_model.bin"
    assert "runs code when loaded" in finding.detail
    assert "system" in finding.detail


def test_a_pickle_is_found_whatever_the_file_is_called(tmp_path: Path) -> None:
    root = write_model(tmp_path)
    (root / "notes.txt").write_bytes(pickle.dumps(Hostile()))
    with zipfile.ZipFile(root / "weights.dat", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(Hostile(), protocol=2))
    (root / "broken.zip.txt").write_bytes(b"PK not really a zip")

    found = {item.path for item in scan(root).findings if item.rule is Rule.PICKLE}

    assert found == {"notes.txt", "weights.dat"}


def test_pickle_imports_are_read_without_running_the_pickle() -> None:
    for protocol in (0, 2, 4):
        imports = pickle_imports(pickle.dumps(Hostile(), protocol=protocol))
        assert any(name.endswith(".system") for name in imports), protocol
    assert pickle_imports(pickle.dumps({"weights": [1, 2, 3]})) == ()
    # A truncated stream still yields what came before the cut.
    assert pickle_imports(pickle.dumps(Hostile(), protocol=0)[:-3])


def test_an_artifact_may_not_ship_code_or_ask_for_it_to_be_imported(tmp_path: Path) -> None:
    root = write_model(tmp_path)
    (root / "modeling_custom.py").write_text("import os\n", encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_custom.Model"}}), encoding="utf-8"
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"trust_remote_code": True}), encoding="utf-8"
    )

    assert sorted(rules(root)) == [Rule.CODE, Rule.REMOTE_CODE, Rule.REMOTE_CODE]


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_an_unreadable_configuration_is_not_waved_through(tmp_path: Path, content: str) -> None:
    root = write_model(tmp_path)
    (root / "config.json").write_text(content, encoding="utf-8")

    assert rules(root) == [Rule.REMOTE_CODE]


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (b"\x01\x02", "shorter than the 8-byte header length"),
        (struct.pack("<Q", 10_000) + b"{}", "does not fit the file"),
        (struct.pack("<Q", 4) + b"nope", "header is not JSON"),
        (struct.pack("<Q", 2) + b"[]", "header is not an object"),
        (safetensors({"a": (0, 16)}, 64), "bytes no tensor accounts for"),
        (safetensors({"a": (0, 16), "b": (8, 32)}, 32), "a gap or an overlap"),
        (safetensors({"a": (0, 64)}, 16), "points outside the file"),
        (safetensors({}, 0, a={"dtype": "PICKLE", "data_offsets": [0, 0]}), "no known dtype"),
    ],
)
def test_weights_whose_header_does_not_describe_the_file_are_refused(
    tmp_path: Path, content: bytes, problem: str
) -> None:
    root = write_model(tmp_path)
    (root / "model.safetensors").write_bytes(content)

    findings = scan(root).findings

    assert [item.rule for item in findings] == [Rule.SAFETENSORS]
    assert problem in findings[0].detail


def test_an_artifact_with_no_weights_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert rules(tmp_path) == [Rule.NO_WEIGHTS]


@pytest.mark.parametrize(
    ("config", "detail"),
    [({"peft_type": "PROMPT_TUNING"}, "only LORA is served"), ({"r": 64}, "outside 1..32")],
)
def test_an_adapter_the_engine_could_not_serve_is_refused(
    tmp_path: Path, config: dict[str, Any], detail: str
) -> None:
    report = scan(write_adapter(tmp_path, **config), ArtifactKind.ADAPTER)

    assert [item.rule for item in report.findings] == [Rule.ADAPTER_CONFIG]
    assert detail in report.findings[0].detail


def test_an_adapter_without_its_configuration_is_refused(tmp_path: Path) -> None:
    write_adapter(tmp_path)
    (tmp_path / "adapter_config.json").unlink()

    assert rules(tmp_path, ArtifactKind.ADAPTER) == [Rule.ADAPTER_CONFIG]


def test_a_link_out_of_the_artifact_is_refused(tmp_path: Path) -> None:
    root = write_model(tmp_path / "model")
    outside = tmp_path / "secret.txt"
    outside.write_text("not part of the artifact", encoding="utf-8")
    try:
        (root / "leak.txt").symlink_to(outside)
        (root / "alias.json").symlink_to(root / "tokenizer.json")
    except OSError:
        pytest.skip("this platform does not allow creating symlinks")

    findings = scan(root).findings

    assert [(item.rule, item.path) for item in findings] == [(Rule.SYMLINK, "leak.txt")]


def test_the_digest_follows_content_and_paths_and_gates_the_release(tmp_path: Path) -> None:
    first = write_model(tmp_path / "first")
    second = write_model(tmp_path / "second")
    assert artifact_digest(first) == artifact_digest(second)

    recorded = artifact_digest(first)
    assert scan(first, expected_digest=recorded).passed

    (second / "tokenizer.json").write_text('{"changed": true}', encoding="utf-8")
    assert artifact_digest(second) != recorded
    tampered = scan(second, expected_digest=recorded)
    assert [item.rule for item in tampered.findings] == [Rule.CHECKSUM]

    (first / "tokenizer.json").rename(first / "renamed.json")
    assert artifact_digest(first) != recorded


def test_the_command_line_passes_a_sound_artifact_and_fails_an_unsafe_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_model(tmp_path / "model")
    assert main([str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True

    (root / "pytorch_model.bin").write_bytes(pickle.dumps({"weights": 1}))
    assert main([str(root), "--expect", artifact_digest(root)]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] is False
    assert [item["rule"] for item in printed["findings"]] == ["pickle-format"]


def test_a_catalog_subject_supplies_the_kind_and_the_recorded_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = write_adapter(tmp_path / "adapter")

    # The catalog records a placeholder checksum, which no real artifact has.
    assert main([str(adapter), "--subject", "claims-extraction-lora"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["kind"] == "adapter"
    assert [item["rule"] for item in printed["findings"]] == ["checksum-mismatch"]
    assert "sha256:mock-claims" in printed["findings"][0]["detail"]

    # A subject with no recorded checksum is scanned without a digest gate.
    assert main([str(adapter), "--subject", "support-classification-lora"]) == 0


@pytest.mark.parametrize("arguments", [["missing-directory"], [".", "--subject", "nobody"]])
def test_the_command_line_refuses_what_it_cannot_scan(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2
