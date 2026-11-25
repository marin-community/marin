# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read an eval policy version from its YAML file."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import requires_judge
from experiments.evaluation.models import models
from experiments.evaluation.policy.schema import (
    EvalPolicy,
    PolicyCandidate,
    PolicyEvaluation,
    PolicyExclusion,
    PolicyOverride,
)

_EVALUATION_FIELDS = frozenset(field.name for field in fields(PolicyEvaluation))
_EVALUATION = TypeAdapter(PolicyEvaluation)


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Candidate(_Entry):
    max_context: PositiveInt


class _Exclusion(_Entry):
    candidate: str
    evaluation: str
    reason: str


class _Override(_Exclusion):
    context: PositiveInt | None = None
    max_output: PositiveInt | None = None


class _PolicyFile(_Entry):
    """The shape of a policy file; names are resolved against the catalogs afterwards."""

    version: str
    released: bool = False
    default_context: PositiveInt
    candidates: dict[str, _Candidate]
    evaluations: list[str | dict[str, Any]]
    exclusions: list[_Exclusion] = []
    overrides: list[_Override] = []


class _UniqueKeyLoader(yaml.SafeLoader):
    """A YAML loader that rejects a mapping with the same key twice instead of keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"duplicate key {key!r} on line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _evaluation(entry: str | dict[str, Any]) -> PolicyEvaluation:
    """An evaluation from a benchmark name, or from a mapping of its fields."""
    if isinstance(entry, str):
        return PolicyEvaluation(entry)
    if unknown := set(entry) - _EVALUATION_FIELDS:
        raise ValueError(f"{entry.get('benchmark')}: unknown evaluation fields {sorted(unknown)}")
    return _EVALUATION.validate_python(entry)


def _check_catalog_choice(evaluation: PolicyEvaluation) -> None:
    """Fail unless the catalog declares the evaluation's variant, and a judged variant names its judge."""
    catalog = evaluation_benchmarks()
    if evaluation.benchmark not in catalog:
        raise ValueError(f"{evaluation.name}: no benchmark {evaluation.benchmark!r} in the benchmark catalog")
    catalog[evaluation.benchmark].resolve(evaluation.protocol, evaluation.num_fewshot, evaluation.prompt_format)
    if evaluation.judge is None and requires_judge(evaluation.selection):
        raise ValueError(f"{evaluation.name}: the benchmark needs a judge; name one with judge:")


def load_policy(path: Path) -> EvalPolicy:
    """Read a policy, naming its candidates by model catalog key and its evaluations by catalog benchmark.

    Exclusions and overrides name a candidate and an evaluation. Unknown keys and names, values of the wrong
    type, variants the catalog does not declare, and judged benchmarks without a judge fail here, before any
    command uses the policy.
    """
    document = _PolicyFile.model_validate(yaml.load(path.read_text(), Loader=_UniqueKeyLoader))
    catalog = models()
    if unknown := document.candidates.keys() - catalog.keys():
        raise ValueError(f"{path}: no models {sorted(unknown)} in the model catalog")
    candidates = {
        name: PolicyCandidate(model=catalog[name], max_context=entry.max_context)
        for name, entry in document.candidates.items()
    }
    in_distribution = tuple(map(_evaluation, document.evaluations))
    for evaluation in in_distribution:
        _check_catalog_choice(evaluation)
    evaluations = {evaluation.name: evaluation for evaluation in in_distribution}

    def named(entry: _Exclusion) -> dict[str, Any]:
        if entry.candidate not in candidates or entry.evaluation not in evaluations:
            raise ValueError(f"{path}: {entry.candidate} on {entry.evaluation} names no candidate or evaluation")
        return {
            **entry.model_dump(),
            "candidate": candidates[entry.candidate],
            "evaluation": evaluations[entry.evaluation],
        }

    return EvalPolicy(
        version=document.version,
        released=document.released,
        default_context=document.default_context,
        candidates=tuple(candidates.values()),
        in_distribution=in_distribution,
        exclusions=tuple(PolicyExclusion(**named(entry)) for entry in document.exclusions),
        overrides=tuple(PolicyOverride(**named(entry)) for entry in document.overrides),
    )
