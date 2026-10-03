from pathlib import Path

import pytest

from llm_router.classifier import (
    DEFAULT_DATASET,
    TEMPERATURES,
    ClassifierError,
    Complexity,
    LabelledPrompt,
    NaiveBayes,
    expected_calibration_error,
    featurize,
    load_classifier,
    load_labelled_prompts,
    train_classifier,
)
from llm_router.models import TaskClass

# Prompts the classifier never trained or calibrated on.
EVALUATION = load_labelled_prompts("benchmarks/datasets/routing-tasks-v1.jsonl")
CLASSIFIER = load_classifier(DEFAULT_DATASET)


def test_features_include_bigrams_and_a_length_bucket() -> None:
    features = featurize("Extract the invoice total")

    assert {"extract", "invoice", "extract_the", "invoice_total", "__short__"} <= set(features)


def test_long_prompts_keep_the_instruction_at_either_end() -> None:
    context = " ".join(f"filler{index}" for index in range(400))

    features = set(featurize(f"Summarize the following. {context} Answer in one line."))

    # The instruction survives at both ends; the middle of the context does not.
    assert {"summarize", "answer", "__long__"} <= features
    assert "filler200" not in features


def test_committed_dataset_has_both_splits_and_every_task() -> None:
    rows = load_labelled_prompts(DEFAULT_DATASET)

    assert {row.split for row in rows} == {"train", "calibration"}
    assert {row.task for row in rows} == set(TaskClass)
    assert {row.complexity for row in rows} == set(Complexity)


def test_evaluation_prompts_are_disjoint_from_the_training_data() -> None:
    seen = {row.prompt for row in load_labelled_prompts(DEFAULT_DATASET)}

    assert not seen & {row.prompt for row in EVALUATION}


def test_task_accuracy_on_prompts_never_seen_in_training() -> None:
    correct = sum(CLASSIFIER.predict(row.prompt).task == row.task for row in EVALUATION)

    assert correct / len(EVALUATION) >= 0.9


def test_complexity_accuracy_on_prompts_never_seen_in_training() -> None:
    correct = sum(CLASSIFIER.predict(row.prompt).complexity == row.complexity for row in EVALUATION)

    assert correct / len(EVALUATION) >= 0.8


def test_stated_confidence_matches_observed_accuracy() -> None:
    pairs = [(featurize(row.prompt), row.task) for row in EVALUATION]

    assert expected_calibration_error(CLASSIFIER.task_model, pairs) <= 0.1


def test_calibration_chooses_a_temperature_from_the_grid() -> None:
    assert CLASSIFIER.task_model.temperature in TEMPERATURES
    assert CLASSIFIER.complexity_model.temperature in TEMPERATURES


def test_an_unrecognisable_prompt_abstains_to_the_general_task() -> None:
    prediction = CLASSIFIER.predict("zxqv flurble quonk")

    assert prediction.trusted is False
    assert prediction.task is TaskClass.GENERAL
    assert prediction.task_confidence < CLASSIFIER.confidence_threshold


def test_a_recognisable_prompt_is_trusted_with_a_probability() -> None:
    prediction = CLASSIFIER.predict("Classify this ticket as billing or technical")

    assert prediction.trusted is True
    assert prediction.task is TaskClass.CLASSIFICATION
    assert 0.5 <= prediction.task_confidence <= 1.0
    assert prediction.complexity is Complexity.LOW


def test_posterior_is_a_probability_distribution() -> None:
    prediction = CLASSIFIER.task_model.predict(featurize("Summarize the report"))

    assert sum(prediction.distribution.values()) == pytest.approx(1.0)
    assert prediction.confidence == max(prediction.distribution.values())


def test_a_higher_temperature_lowers_confidence() -> None:
    examples = [
        (featurize("extract the fields"), "a"),
        (featurize("extract the totals"), "a"),
        (featurize("summarize the notes"), "b"),
        (featurize("summarize the report"), "b"),
    ]
    model = NaiveBayes(examples)
    features = featurize("extract the report")

    model.temperature = 1.0
    sharp = model.predict(features).confidence
    model.temperature = 8.0
    soft = model.predict(features).confidence

    assert soft < sharp


def test_training_needs_data_and_both_splits() -> None:
    row = LabelledPrompt("Extract it", TaskClass.EXTRACTION, Complexity.LOW, "train")

    with pytest.raises(ClassifierError, match="empty dataset"):
        NaiveBayes([])
    with pytest.raises(ClassifierError, match="train and a calibration split"):
        train_classifier([row])
    with pytest.raises(ClassifierError, match="held-out split"):
        NaiveBayes([(featurize("x"), "a")]).calibrate([])
    with pytest.raises(ClassifierError, match="empty sample"):
        expected_calibration_error(NaiveBayes([(featurize("x"), "a")]), [])


def test_loader_reports_unusable_and_empty_files(tmp_path: Path) -> None:
    unusable = tmp_path / "bad.jsonl"
    unusable.write_text('{"prompt": "x", "task": "not-a-task", "complexity": "low"}\n', "utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", "utf-8")

    with pytest.raises(ClassifierError, match="not a usable example"):
        load_labelled_prompts(unusable)
    with pytest.raises(ClassifierError, match="no examples"):
        load_labelled_prompts(empty)
