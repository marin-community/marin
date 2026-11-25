# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read an eval policy version from its YAML file."""

from pathlib import Path

import yaml
from evalrecords.records import PromptFormat, TaskMachineSettings

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.models import models
from experiments.evaluation.policy.schema import (
    EvalPolicy,
    PolicyCandidate,
    PolicyEvaluation,
    PolicyExclusion,
    PolicyOverride,
)

_POLICY_KEYS = frozenset(
    {"version", "released", "default_context", "candidates", "evaluations", "exclusions", "overrides"}
)


def _evaluation(entry: str | dict) -> PolicyEvaluation:
    """An evaluation from a benchmark name, or from a mapping of its fields."""
    if isinstance(entry, str):
        return PolicyEvaluation(entry)
    fields = dict(entry)
    if "prompt_format" in fields:
        fields["prompt_format"] = PromptFormat(fields["prompt_format"])
    if "task_machine" in fields:
        fields["task_machine"] = TaskMachineSettings.model_validate(fields["task_machine"])
    return PolicyEvaluation(**fields)


def load_policy(path: Path) -> EvalPolicy:
    """Read a policy, naming its candidates by model catalog key and its evaluations by catalog benchmark.

    Exclusions and overrides name a candidate and an evaluation. Unknown keys and names fail here, before any
    command uses the policy.
    """
    document = yaml.safe_load(path.read_text())
    if unknown := set(document) - _POLICY_KEYS:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    catalog = models()
    if unknown := set(document["candidates"]) - catalog.keys():
        raise ValueError(f"{path}: no models {sorted(unknown)} in the model catalog")
    candidates = {
        name: PolicyCandidate(model=catalog[name], **settings) for name, settings in document["candidates"].items()
    }
    in_distribution = tuple(map(_evaluation, document["evaluations"]))
    evaluations = {evaluation.name: evaluation for evaluation in in_distribution}
    if unknown := {evaluation.benchmark for evaluation in in_distribution} - evaluation_benchmarks().keys():
        raise ValueError(f"{path}: no benchmarks {sorted(unknown)} in the benchmark catalog")

    def named(entry: dict) -> dict:
        if entry["candidate"] not in candidates or entry["evaluation"] not in evaluations:
            raise ValueError(f"{path}: {entry['candidate']} on {entry['evaluation']} names no candidate or evaluation")
        return {**entry, "candidate": candidates[entry["candidate"]], "evaluation": evaluations[entry["evaluation"]]}

    return EvalPolicy(
        version=document["version"],
        released=document.get("released", False),
        default_context=document["default_context"],
        candidates=tuple(candidates.values()),
        in_distribution=in_distribution,
        exclusions=tuple(PolicyExclusion(**named(entry)) for entry in document.get("exclusions", ())),
        overrides=tuple(PolicyOverride(**named(entry)) for entry in document.get("overrides", ())),
    )
