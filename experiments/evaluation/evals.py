# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation suites available to the shared launcher."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import yaml
from evalrecords.records import EvaluationSelection
from marin.evaluation.task_worker import DAYTONA_API_KEY
from rigging.secrets import SecretSpec

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import requires_judge

# Evals run only in the Daytona eval org; RL and data keys never reach this launcher.
DAYTONA_SECRET_ENV: Mapping[str, SecretSpec] = MappingProxyType(
    {
        DAYTONA_API_KEY: (
            "env:DAYTONA_EVAL_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }
)
# The RL evaluation adapter imports this selection type.
EvaluationDefinition = EvaluationSelection


_CONFIG = yaml.safe_load((Path(__file__).with_name("configs") / "evaluations.yaml").read_text())
EVALS: dict[str, EvaluationSelection] = {
    name: EvaluationSelection.model_validate({"label": name, **entry}) for name, entry in _CONFIG["evaluations"].items()
}


def evaluation_for(name: str) -> EvaluationSelection:
    if name in EVALS:
        return EVALS[name]
    if name in evaluation_benchmarks():
        return EvaluationSelection(benchmark=name)
    raise ValueError(f"unknown evaluation: {name}")


SUITES: dict[str, tuple[str, ...]] = {name: tuple(entries) for name, entries in _CONFIG["suites"].items()}
SUITES["standard-local"] = tuple(name for name in SUITES["standard"] if not requires_judge(evaluation_for(name)))


def resolve_evaluations(selection: str) -> tuple[EvaluationSelection, ...]:
    """Resolve suite and CLI labels once, at the textual input boundary."""
    keys = SUITES.get(selection) or tuple(part.strip() for part in selection.split(",") if part.strip())
    if not keys:
        raise ValueError("no evals selected")
    return tuple(evaluation_for(key) for key in keys)
