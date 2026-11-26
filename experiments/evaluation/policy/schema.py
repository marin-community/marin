# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An eval policy: which catalog models run which named evaluations, at which window, with which exclusions."""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from evalrecords.records import EvaluationSelection
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import requires_judge
from experiments.evaluation.models import models

HF_COMMIT = re.compile(r"[0-9a-f]{40}")
RESERVED_VERSION_PREFIX = "eval-policy-"
"""The generic launcher rejects unknown versions with this prefix; policy tags use it."""


@dataclass(frozen=True)
class Policy:
    """One version of an eval policy. Every evaluation runs for every model unless the pair is excluded.

    ``evaluations`` are named catalog selections, normally ``load_evaluations`` of the version's
    ``evaluations.yaml``. ``serving_window`` is the context window every model is served at, or the model's own
    window if smaller. ``exclusions`` maps a (model, evaluation) pair to the reason it does not run.
    """

    version: str
    models: tuple[str, ...]
    evaluations: tuple[EvaluationSelection, ...]
    serving_window: int
    exclusions: Mapping[tuple[str, str], str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.version.startswith(RESERVED_VERSION_PREFIX):
            raise ValueError(f"{self.version}: {RESERVED_VERSION_PREFIX!r} names policy tags, not versions")
        _require_unique("model", list(self.models))
        _require_unique("evaluation", [evaluation.name for evaluation in self.evaluations])
        catalog = models()
        for name in self.models:
            if name not in catalog:
                raise ValueError(f"{name} is not in the model catalog")
            _require_pinned(catalog[name])
        names = {evaluation.name for evaluation in self.evaluations}
        for model, evaluation in self.exclusions:
            if model not in self.models or evaluation not in names:
                raise ValueError(f"exclusion of {model} on {evaluation} names no model or evaluation of this policy")
        for evaluation in self.evaluations:
            _check_evaluation(evaluation)

    def model(self, name: str) -> ModelConfig:
        """The catalog entry of one of the policy's models."""
        if name not in self.models:
            raise ValueError(f"{name} is not a model of {self.version}; known: {list(self.models)}")
        return models()[name]


def _check_evaluation(evaluation: EvaluationSelection) -> None:
    """Reject a selection the catalog does not declare, a judge where none is needed or missing, and an item limit."""
    catalog = evaluation_benchmarks()
    if evaluation.benchmark not in catalog:
        raise ValueError(f"{evaluation.name}: no benchmark {evaluation.benchmark!r} in the benchmark catalog")
    catalog[evaluation.benchmark].resolve(evaluation.protocol, evaluation.num_fewshot, evaluation.prompt_format)
    if (evaluation.run.judge is not None) != requires_judge(evaluation):
        needs = "needs a judge" if evaluation.run.judge is None else "takes no judge"
        raise ValueError(f"{evaluation.name}: the benchmark {needs}")
    if evaluation.run.limit is not None:
        raise ValueError(f"{evaluation.name}: a policy evaluation runs every item; --limit makes a smoke run")


def _require_pinned(model: ModelConfig) -> None:
    """A Hugging Face model pins commits for its weights and tokenizer, so the policy names one checkpoint."""
    if "://" in model.location:
        return
    if model.revision is None or HF_COMMIT.fullmatch(model.revision) is None:
        raise ValueError(f"{model.name}: pin {model.location} to a Hugging Face commit SHA")
    tokenizer = model.tokenizer or model.location
    tokenizer_revision = model.effective_tokenizer_revision
    if "://" not in tokenizer and (tokenizer_revision is None or HF_COMMIT.fullmatch(tokenizer_revision) is None):
        raise ValueError(f"{model.name}: pin its tokenizer {tokenizer} to a Hugging Face commit SHA")


def _require_unique(kind: str, names: list[str]) -> None:
    if duplicates := sorted(name for name, count in Counter(names).items() if count > 1):
        raise ValueError(f"more than one {kind} {duplicates}")
