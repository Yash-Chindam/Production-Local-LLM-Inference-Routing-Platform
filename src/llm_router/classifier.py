"""Calibrated task and complexity prediction for routing (section 7.2).

Section 7.2 keeps privacy and hard capability restrictions deterministic and
asks for a lightweight classifier or calibrated model for task and complexity.
This is a multinomial naive Bayes model over word unigrams and bigrams, trained
at start-up from a committed dataset. It needs no accelerator and no extra
dependency, and trains in milliseconds, so the routing decision never waits on
the models it is choosing between.

Naive Bayes posteriors are overconfident, so they are temperature-scaled
against a held-out calibration split. A confidence can then be read as a
probability, and a prediction below the threshold falls back to the general
task rather than being trusted.
"""

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Generic, TypeVar

from llm_router.models import TaskClass

DEFAULT_DATASET = "config/routing/task-classifier-v1.jsonl"
# The instruction is nearly always at the start or the end of a prompt. Long
# retrieved context in between says what the prompt is about, not what is being
# asked, and would otherwise drown the instruction out.
HEAD_TOKENS = 48
TAIL_TOKENS = 48
SMOOTHING = 0.5
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

_TOKEN = re.compile(r"[a-z0-9]+")
Label = TypeVar("Label")


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClassifierError(RuntimeError):
    """Raised when the training data cannot produce a usable model."""


def featurize(text: str) -> list[str]:
    """Turn a prompt into unigram, bigram, and coarse length features."""

    tokens = _TOKEN.findall(text.lower())
    if len(tokens) > HEAD_TOKENS + TAIL_TOKENS:
        tokens = tokens[:HEAD_TOKENS] + tokens[-TAIL_TOKENS:]
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in pairwise(tokens))
    # Length carries signal for complexity that word identity alone does not.
    words = len(text.split())
    features.append("__short__" if words <= 12 else "__medium__" if words <= 60 else "__long__")
    return features


@dataclass(frozen=True)
class Prediction(Generic[Label]):
    label: Label
    confidence: float
    distribution: dict[Label, float]


class NaiveBayes(Generic[Label]):
    """Multinomial naive Bayes with additive smoothing and temperature scaling."""

    def __init__(self, examples: Sequence[tuple[list[str], Label]]) -> None:
        if not examples:
            raise ClassifierError("cannot train on an empty dataset")
        self._label_counts: Counter[Label] = Counter(label for _, label in examples)
        self._feature_counts: dict[Label, Counter[str]] = defaultdict(Counter)
        for features, label in examples:
            self._feature_counts[label].update(features)
        self._vocabulary = {
            feature for counts in self._feature_counts.values() for feature in counts
        }
        self._totals = {
            label: sum(counts.values()) for label, counts in self._feature_counts.items()
        }
        self._examples = len(examples)
        self.temperature = 1.0

    @property
    def labels(self) -> tuple[Label, ...]:
        return tuple(self._label_counts)

    def _log_joint(self, features: Iterable[str]) -> dict[Label, float]:
        known = [feature for feature in features if feature in self._vocabulary]
        vocabulary = len(self._vocabulary)
        scores: dict[Label, float] = {}
        for label, count in self._label_counts.items():
            score = math.log(count / self._examples)
            denominator = self._totals[label] + SMOOTHING * vocabulary
            counts = self._feature_counts[label]
            for feature in known:
                score += math.log((counts[feature] + SMOOTHING) / denominator)
            scores[label] = score
        return scores

    def predict(self, features: Iterable[str]) -> Prediction[Label]:
        scores = self._log_joint(features)
        scaled = {label: score / self.temperature for label, score in scores.items()}
        peak = max(scaled.values())
        exponentials = {label: math.exp(score - peak) for label, score in scaled.items()}
        total = sum(exponentials.values())
        distribution = {label: value / total for label, value in exponentials.items()}
        label = max(distribution, key=lambda candidate: distribution[candidate])
        return Prediction(label=label, confidence=distribution[label], distribution=distribution)

    def calibrate(self, examples: Sequence[tuple[list[str], Label]]) -> float:
        """Pick the temperature that minimizes held-out negative log-likelihood."""

        if not examples:
            raise ClassifierError("cannot calibrate without a held-out split")

        def loss(temperature: float) -> float:
            self.temperature = temperature
            return -sum(
                math.log(max(self.predict(features).distribution.get(label, 0.0), 1e-12))
                for features, label in examples
            )

        self.temperature = min(TEMPERATURES, key=loss)
        return self.temperature


def expected_calibration_error(
    model: NaiveBayes[Label], examples: Sequence[tuple[list[str], Label]], *, bins: int = 5
) -> float:
    """Mean gap between stated confidence and observed accuracy, weighted by bin size."""

    if not examples:
        raise ClassifierError("cannot measure calibration on an empty sample")
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for features, label in examples:
        prediction = model.predict(features)
        index = min(int(prediction.confidence * bins), bins - 1)
        buckets[index].append((prediction.confidence, prediction.label == label))
    error = 0.0
    for members in buckets.values():
        confidence = sum(value for value, _ in members) / len(members)
        accuracy = sum(correct for _, correct in members) / len(members)
        error += abs(confidence - accuracy) * len(members) / len(examples)
    return error


@dataclass(frozen=True)
class LabelledPrompt:
    prompt: str
    task: TaskClass
    complexity: Complexity
    split: str = "train"


def load_labelled_prompts(path: str | Path) -> tuple[LabelledPrompt, ...]:
    """Read a JSON Lines file of prompts labelled with task and complexity."""

    rows: list[LabelledPrompt] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
            rows.append(
                LabelledPrompt(
                    prompt=str(document["prompt"]),
                    task=TaskClass(document["task"]),
                    complexity=Complexity(document["complexity"]),
                    split=str(document.get("split", "train")),
                )
            )
        except (ValueError, KeyError) as error:
            raise ClassifierError(f"{path}:{number} is not a usable example: {error}") from error
    if not rows:
        raise ClassifierError(f"{path} contains no examples")
    return tuple(rows)


@dataclass(frozen=True)
class RoutingPrediction:
    task: TaskClass
    task_confidence: float
    complexity: Complexity
    complexity_confidence: float
    # False when the task fell back to general because confidence was too low.
    trusted: bool


class TaskClassifier:
    """Predicts task and complexity, abstaining to the general task when unsure."""

    def __init__(
        self,
        task_model: NaiveBayes[TaskClass],
        complexity_model: NaiveBayes[Complexity],
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.task_model = task_model
        self.complexity_model = complexity_model
        self.confidence_threshold = confidence_threshold

    def predict(self, prompt: str) -> RoutingPrediction:
        features = featurize(prompt)
        task = self.task_model.predict(features)
        complexity = self.complexity_model.predict(features)
        trusted = task.confidence >= self.confidence_threshold
        return RoutingPrediction(
            # The general task is served by the broadest models, so abstaining
            # to it is the conservative choice when the model is unsure.
            task=task.label if trusted else TaskClass.GENERAL,
            task_confidence=task.confidence,
            complexity=complexity.label,
            complexity_confidence=complexity.confidence,
            trusted=trusted,
        )


def train_classifier(
    rows: Sequence[LabelledPrompt],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> TaskClassifier:
    """Fit both heads on the training split and calibrate on the held-out one."""

    train = [row for row in rows if row.split == "train"]
    held_out = [row for row in rows if row.split == "calibration"]
    if not train or not held_out:
        raise ClassifierError("the dataset needs both a train and a calibration split")

    task_model = NaiveBayes([(featurize(row.prompt), row.task) for row in train])
    task_model.calibrate([(featurize(row.prompt), row.task) for row in held_out])
    complexity_model = NaiveBayes([(featurize(row.prompt), row.complexity) for row in train])
    complexity_model.calibrate([(featurize(row.prompt), row.complexity) for row in held_out])
    return TaskClassifier(task_model, complexity_model, confidence_threshold=confidence_threshold)


@lru_cache(maxsize=4)
def load_classifier(path: str = DEFAULT_DATASET) -> TaskClassifier:
    """Train once per dataset path; the model is immutable after calibration."""

    return train_classifier(load_labelled_prompts(path))
