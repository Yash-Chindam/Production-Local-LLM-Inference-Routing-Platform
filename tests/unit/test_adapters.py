import json
import struct
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llm_router.adapters import (
    AdapterError,
    AdapterRecipe,
    Framework,
    Method,
    catalog_entry,
    check_recipes,
    load_recipes,
    main,
    peft_arguments,
    plan,
    quantization_arguments,
    training_arguments,
    unsloth_arguments,
)
from llm_router.artifact_scan import artifact_digest
from llm_router.registry import AdapterProfile, LifecycleStage, load_registry

CATALOG = load_registry("config/registry.yaml")
RECIPES = load_recipes()
COMMIT = "a" * 40


def recipe(**changes: Any) -> AdapterRecipe:
    base = RECIPES["claims-extraction-lora"].model_dump(mode="json")
    return AdapterRecipe.model_validate({**base, **changes})


def write_adapter(root: Path, rank: int = 16) -> Path:
    header = json.dumps({"lora_A": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}})
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header.encode() + bytes(32)
    )
    (root / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": rank}), encoding="utf-8"
    )
    return root


def test_every_catalog_adapter_has_a_recipe_that_agrees_with_it() -> None:
    assert set(RECIPES) == {adapter.id for adapter in CATALOG.adapters}
    assert check_recipes(CATALOG, RECIPES) == ()


def test_a_catalog_adapter_with_no_recipe_cannot_be_reproduced() -> None:
    remaining = {
        key: value for key, value in RECIPES.items() if key != "support-classification-lora"
    }

    assert check_recipes(CATALOG, remaining) == (
        "support-classification-lora: in the catalog but has no recipe",
    )


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        (
            {"base_revision": "mock-small@sha256:old"},
            "the catalog says small-specialist@mock-small",
        ),
        (
            {"dataset": {"path": "x.jsonl", "version": "claims-2025-01"}},
            "uses dataset claims-2025-01",
        ),
        ({"method": "lora"}, "records quantized=true"),
        ({"domain": "billing"}, "domain or tasks differ"),
    ],
)
def test_a_recipe_that_describes_a_different_adapter_is_reported(
    changes: dict[str, Any], problem: str
) -> None:
    recipes = {**RECIPES, "claims-extraction-lora": recipe(**changes)}

    problems = check_recipes(CATALOG, recipes)

    assert any(problem in item for item in problems), problems


def test_a_recipe_for_a_base_the_catalog_does_not_hold_is_reported() -> None:
    orphan = recipe(id="new-adapter", base_model_id="retired-model")

    assert check_recipes(CATALOG, {**RECIPES, "new-adapter": orphan}) == (
        "new-adapter: base retired-model@mock-small@sha256:dev is not a revision in the catalog",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hf_revision": "main"}, "must be a 40-character commit"),
        ({"rank": 64}, "exceeds the 32 the engine loads"),
        ({"target_modules": []}, "at least one target module"),
    ],
)
def test_a_recipe_that_is_not_reproducible_or_servable_is_rejected(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        recipe(**changes)


def test_a_recipe_file_must_be_named_after_its_adapter(tmp_path: Path) -> None:
    (tmp_path / "misnamed.yaml").write_text(
        yaml.safe_dump(recipe().model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(AdapterError, match="name the file after it"):
        load_recipes(tmp_path)


def test_peft_gets_a_lora_config_and_four_bit_loading_only_for_qlora() -> None:
    qlora = recipe(hf_revision=COMMIT)
    lora = recipe(method="lora")

    assert peft_arguments(qlora) == {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
    }
    assert quantization_arguments(qlora) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert quantization_arguments(lora) is None


def test_unsloth_is_pinned_to_the_same_commit_and_seed() -> None:
    load, adapt = unsloth_arguments(recipe(hf_revision=COMMIT))

    assert load == {
        "model_name": "REPLACE_ME/small-specialist",
        "revision": COMMIT,
        "max_seq_length": 2048,
        "load_in_4bit": True,
    }
    assert adapt["r"] == 16 and adapt["random_state"] == 7
    assert unsloth_arguments(recipe(method="lora"))[0]["load_in_4bit"] is False


def test_training_never_writes_checkpoints_that_could_not_be_released() -> None:
    arguments = training_arguments(recipe())

    assert arguments["save_strategy"] == "no"
    assert arguments["seed"] == 7 and arguments["num_train_epochs"] == 3


def test_the_plan_shows_the_arguments_for_the_chosen_framework() -> None:
    unsloth = plan(recipe())
    peft = plan(recipe(framework="peft"))

    assert unsloth["framework"] == Framework.UNSLOTH.value and "peft" not in unsloth
    assert unsloth["unsloth"]["from_pretrained"]["load_in_4bit"] is True
    assert peft["peft"]["lora_config"]["r"] == 16 and "unsloth" not in peft
    assert peft["method"] == Method.QLORA.value
    assert peft["dataset"]["version"] == "claims-2026-05"


def test_a_finished_adapter_becomes_a_development_entry_named_by_its_digest(
    tmp_path: Path,
) -> None:
    output = write_adapter(tmp_path)

    entry = catalog_entry(recipe(), output)

    assert entry.adapter_revision == f"claims-extraction-lora@{artifact_digest(output)}"
    assert entry.stage is LifecycleStage.DEVELOPMENT
    assert entry.benchmark.quality_delta == 0.0
    assert entry.quantized is True
    assert (entry.base_model_id, entry.base_revision) == (
        "small-specialist",
        "mock-small@sha256:dev",
    )


def test_an_adapter_that_fails_its_scan_is_never_described(tmp_path: Path) -> None:
    output = write_adapter(tmp_path, rank=64)
    (output / "optimizer.pt").write_bytes(b"\x80\x04N.")

    with pytest.raises(AdapterError, match=r"pickle-format at optimizer\.pt"):
        catalog_entry(recipe(), output)


def test_the_command_line_checks_plans_and_registers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []

    assert main(["plan", "support-classification-lora"]) == 0
    assert json.loads(capsys.readouterr().out)["peft"]["lora_config"]["r"] == 8

    output = write_adapter(tmp_path / "adapter")
    assert main(["register", "claims-extraction-lora", "--output", str(output)]) == 0
    (printed,) = yaml.safe_load(capsys.readouterr().out)
    assert AdapterProfile.model_validate(printed).stage is LifecycleStage.DEVELOPMENT

    (output / "train.py").write_text("print('hello')\n", encoding="utf-8")
    assert main(["register", "claims-extraction-lora", "--output", str(output)]) == 1
    assert "executable-code" in json.loads(capsys.readouterr().out)["error"]


def test_the_command_line_fails_when_recipes_and_catalog_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "claims-extraction-lora.yaml").write_text(
        yaml.safe_dump(recipe().model_dump(mode="json")), encoding="utf-8"
    )

    assert main(["check", "--recipes", str(tmp_path)]) == 1
    assert len(json.loads(capsys.readouterr().out)["problems"]) == 2


@pytest.mark.parametrize(
    "arguments", [["plan"], ["plan", "nobody"], ["register", "claims-extraction-lora"]]
)
def test_the_command_line_says_what_it_is_missing(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2
