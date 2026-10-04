"""LoRA and QLoRA adapter recipes for PEFT and Unsloth (sections 6 and 7.5).

An adapter in the catalog is the result of a training run. The recipe is the
reviewed record of that run: which immutable base revision it started from,
which dataset version it saw, and the hyperparameters used. This module
validates recipes against the catalog, turns one into the arguments PEFT or
Unsloth take, and turns a finished run into the catalog entry that describes
it.

Training itself needs a GPU and the ``training`` extra. Everything else runs
anywhere, which is what lets a recipe be reviewed and checked in CI.
"""

import json
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from llm_router.artifact_scan import MAX_LORA_RANK, ArtifactKind, scan
from llm_router.models import TaskClass
from llm_router.registry import AdapterProfile, BenchmarkDelta, LifecycleStage, Registry

RECIPE_DIRECTORY = Path("config/adapters")
COMMIT = re.compile(r"[0-9a-f]{40}")


class AdapterError(RuntimeError):
    """Raised when a recipe or a finished adapter cannot be accepted."""


class Method(StrEnum):
    LORA = "lora"
    # LoRA trained over a base model loaded in 4-bit.
    QLORA = "qlora"


class Framework(StrEnum):
    PEFT = "peft"
    UNSLOTH = "unsloth"


class DatasetReference(BaseModel):
    path: str
    version: str
    text_field: str = "text"


class TrainingSettings(BaseModel):
    epochs: float = Field(default=3.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=2048, ge=16)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    seed: int = 0


class AdapterRecipe(BaseModel):
    """Everything needed to reproduce one adapter."""

    id: str
    base_model_id: str
    base_revision: str
    # Where the base weights come from. The revision is a commit, never a
    # branch or tag, so the base cannot move underneath the adapter.
    hf_repo: str
    hf_revision: str
    domain: str
    intended_tasks: frozenset[TaskClass]
    method: Method = Method.LORA
    framework: Framework = Framework.PEFT
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dataset: DatasetReference
    training: TrainingSettings = TrainingSettings()

    @model_validator(mode="after")
    def keep_the_recipe_servable_and_reproducible(self) -> "AdapterRecipe":
        if not COMMIT.fullmatch(self.hf_revision):
            raise ValueError(
                f"recipe {self.id}: hf_revision must be a 40-character commit, "
                "not a branch or tag that can move"
            )
        if self.rank > MAX_LORA_RANK:
            raise ValueError(
                f"recipe {self.id}: rank {self.rank} exceeds the {MAX_LORA_RANK} the engine loads"
            )
        if not self.target_modules:
            raise ValueError(f"recipe {self.id}: at least one target module is required")
        return self


def load_recipes(directory: Path = RECIPE_DIRECTORY) -> dict[str, AdapterRecipe]:
    recipes: dict[str, AdapterRecipe] = {}
    for path in sorted(directory.glob("*.yaml")):
        recipe = AdapterRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if recipe.id != path.stem:
            raise AdapterError(f"{path.name} holds recipe {recipe.id}; name the file after it")
        recipes[recipe.id] = recipe
    return recipes


def check_recipes(registry: Registry, recipes: dict[str, AdapterRecipe]) -> tuple[str, ...]:
    """Every way the recipes and the catalog disagree.

    A catalog adapter with no recipe cannot be reproduced. A recipe that names
    a different base, dataset or quantization than its catalog entry describes
    some other adapter.
    """

    problems: list[str] = []
    bases = {card.id: card.revision for card in registry.models}
    for adapter in registry.adapters:
        recipe = recipes.get(adapter.id)
        if recipe is None:
            problems.append(f"{adapter.id}: in the catalog but has no recipe")
            continue
        if (recipe.base_model_id, recipe.base_revision) != (
            adapter.base_model_id,
            adapter.base_revision,
        ):
            problems.append(
                f"{adapter.id}: recipe trains on {recipe.base_model_id}@{recipe.base_revision}, "
                f"the catalog says {adapter.base_model_id}@{adapter.base_revision}"
            )
        if recipe.dataset.version != adapter.dataset_version:
            problems.append(
                f"{adapter.id}: recipe uses dataset {recipe.dataset.version}, "
                f"the catalog says {adapter.dataset_version}"
            )
        if (recipe.method is Method.QLORA) != adapter.quantized:
            problems.append(
                f"{adapter.id}: recipe method is {recipe.method.value} but the catalog "
                f"records quantized={str(adapter.quantized).lower()}"
            )
        if (recipe.domain, recipe.intended_tasks) != (adapter.domain, adapter.intended_tasks):
            problems.append(f"{adapter.id}: recipe domain or tasks differ from the catalog")
    for recipe in recipes.values():
        if bases.get(recipe.base_model_id) != recipe.base_revision:
            problems.append(
                f"{recipe.id}: base {recipe.base_model_id}@{recipe.base_revision} "
                "is not a revision in the catalog"
            )
    return tuple(dict.fromkeys(problems))


def peft_arguments(recipe: AdapterRecipe) -> dict[str, Any]:
    """Keyword arguments for ``peft.LoraConfig``."""

    return {
        "task_type": "CAUSAL_LM",
        "r": recipe.rank,
        "lora_alpha": recipe.alpha,
        "lora_dropout": recipe.dropout,
        "target_modules": list(recipe.target_modules),
        "bias": "none",
    }


def quantization_arguments(recipe: AdapterRecipe) -> dict[str, Any] | None:
    """Keyword arguments for ``transformers.BitsAndBytesConfig``, for QLoRA only."""

    if recipe.method is not Method.QLORA:
        return None
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


def unsloth_arguments(recipe: AdapterRecipe) -> tuple[dict[str, Any], dict[str, Any]]:
    """Arguments for Unsloth's ``from_pretrained`` and ``get_peft_model``."""

    load = {
        "model_name": recipe.hf_repo,
        "revision": recipe.hf_revision,
        "max_seq_length": recipe.training.max_seq_length,
        "load_in_4bit": recipe.method is Method.QLORA,
    }
    adapt = {
        "r": recipe.rank,
        "lora_alpha": recipe.alpha,
        "lora_dropout": recipe.dropout,
        "target_modules": list(recipe.target_modules),
        "bias": "none",
        "random_state": recipe.training.seed,
    }
    return load, adapt


def training_arguments(recipe: AdapterRecipe) -> dict[str, Any]:
    """Keyword arguments for ``transformers.TrainingArguments``."""

    settings = recipe.training
    return {
        "num_train_epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "warmup_ratio": settings.warmup_ratio,
        "seed": settings.seed,
        # Checkpoints carry pickled optimizer state, which may not be released.
        "save_strategy": "no",
        "report_to": [],
    }


def plan(recipe: AdapterRecipe) -> dict[str, Any]:
    """What a training run for this recipe would be given, for review."""

    document: dict[str, Any] = {
        "recipe": recipe.id,
        "framework": recipe.framework.value,
        "method": recipe.method.value,
        "base": {"repo": recipe.hf_repo, "revision": recipe.hf_revision},
        "dataset": recipe.dataset.model_dump(),
        "training_arguments": training_arguments(recipe),
    }
    if recipe.framework is Framework.UNSLOTH:
        load, adapt = unsloth_arguments(recipe)
        document["unsloth"] = {"from_pretrained": load, "get_peft_model": adapt}
    else:
        document["peft"] = {
            "lora_config": peft_arguments(recipe),
            "quantization_config": quantization_arguments(recipe),
        }
    return document


def train(recipe: AdapterRecipe, output: Path) -> None:  # pragma: no cover - needs a GPU
    """Train the adapter and write it to ``output`` as safetensors.

    This has not been run: it needs a GPU and the training extra, neither of
    which the verification environment has.
    """

    import tempfile

    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if recipe.framework is Framework.UNSLOTH:
        from unsloth import FastLanguageModel

        load, adapt = unsloth_arguments(recipe)
        model, tokenizer = FastLanguageModel.from_pretrained(**load)
        model = FastLanguageModel.get_peft_model(model, **adapt)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization = quantization_arguments(recipe)
        model = AutoModelForCausalLM.from_pretrained(
            recipe.hf_repo,
            revision=recipe.hf_revision,
            quantization_config=BitsAndBytesConfig(**quantization) if quantization else None,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(recipe.hf_repo, revision=recipe.hf_revision)
        if quantization:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(**peft_arguments(recipe)))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=recipe.dataset.path, split="train")
    tokenized = dataset.map(
        lambda row: tokenizer(
            row[recipe.dataset.text_field],
            truncation=True,
            max_length=recipe.training.max_seq_length,
        ),
        remove_columns=dataset.column_names,
    )
    with tempfile.TemporaryDirectory() as scratch:
        Trainer(
            model=model,
            args=TrainingArguments(output_dir=scratch, **training_arguments(recipe)),
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        ).train()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)


def catalog_entry(recipe: AdapterRecipe, output: Path) -> AdapterProfile:
    """Scan a finished adapter and describe it as a catalog entry.

    The entry starts in development with no measured gain: promotion is a
    separate, reviewed change made once a benchmark exists. The revision is
    the artifact's own digest, so the entry cannot describe different bytes.
    """

    report = scan(output, ArtifactKind.ADAPTER)
    if not report.passed:
        reasons = "; ".join(f"{item.rule.value} at {item.path}" for item in report.findings)
        raise AdapterError(f"adapter {recipe.id} failed its artifact scan: {reasons}")
    return AdapterProfile(
        id=recipe.id,
        base_model_id=recipe.base_model_id,
        base_revision=recipe.base_revision,
        adapter_revision=f"{recipe.id}@{report.digest}",
        domain=recipe.domain,
        intended_tasks=recipe.intended_tasks,
        dataset_version=recipe.dataset.version,
        benchmark=BenchmarkDelta(quality_delta=0.0),
        stage=LifecycleStage.DEVELOPMENT,
        quantized=recipe.method is Method.QLORA,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Check recipes against the catalog, or plan, train, or register one adapter.

    ``check`` exits 1 if the recipes and the catalog disagree. ``plan`` prints
    what a run would be given. ``train`` runs it. ``register`` scans a
    finished adapter directory and prints its catalog entry.
    """

    import argparse

    from llm_router.registry import load_registry

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["check", "plan", "train", "register"])
    parser.add_argument("recipe", nargs="?")
    parser.add_argument("--recipes", default=str(RECIPE_DIRECTORY))
    parser.add_argument("--catalog", default="config/registry.yaml")
    parser.add_argument("--output", help="adapter directory to write or to register")
    arguments = parser.parse_args(argv)

    recipes = load_recipes(Path(arguments.recipes))
    if arguments.command == "check":
        problems = check_recipes(load_registry(arguments.catalog), recipes)
        print(json.dumps({"recipes": sorted(recipes), "problems": list(problems)}, indent=2))
        return 1 if problems else 0

    recipe = recipes.get(arguments.recipe or "")
    if recipe is None:
        parser.error(f"{arguments.command} needs a recipe, one of: {', '.join(sorted(recipes))}")
    if arguments.command == "plan":
        print(json.dumps(plan(recipe), indent=2))
        return 0
    if not arguments.output:
        parser.error(f"{arguments.command} needs --output")
    output = Path(arguments.output)
    if arguments.command == "train":  # pragma: no cover - needs a GPU
        train(recipe, output)
    try:
        entry = catalog_entry(recipe, output)
    except AdapterError as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 1
    print(yaml.safe_dump([entry.model_dump(mode="json")], sort_keys=False), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
