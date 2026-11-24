# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line view of eval policies.

``uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import click
from evalrecords.records import EvaluationSelection, PromptFormat, ResolvedEvaluationSelection
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.curated import resolve_selection, selected_sources
from experiments.evaluation.policies import POLICIES
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate
from experiments.evaluation.preparation import EvaluationCorpusRecipe

_COLUMNS = ("evaluation", "context", "output", "seed", "runs as", "think", "trials", "cap", "judge", "metric")


@dataclass(frozen=True)
class _Row:
    evaluation: str
    context: str
    output: str
    seed: str
    runs_as: str
    think: str = "-"
    trials: str = "-"
    cap: str = "-"
    judge: str = "-"
    metric: str = "-"


def _thinking(kwargs: Mapping[str, object]) -> str:
    if "enable_thinking" not in kwargs:
        return "default"
    return "on" if kwargs["enable_thinking"] else "off"


def resolved_selection(selection: EvaluationSelection, model: ModelConfig) -> ResolvedEvaluationSelection:
    """The catalog protocol a selection runs as for this model."""
    return resolve_selection(selection, PromptFormat.CHAT if model.apply_chat_template else PromptFormat.COMPLETION)


def selection_dataset(selection: EvaluationSelection, model: ModelConfig) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that a selection runs as for this model."""
    config = selected_sources(resolved_selection(selection, model))[0].config
    assert config is not None
    return config


def _row(item: ResolvedPolicyEvaluation) -> _Row:
    selection = item.evaluation.definition
    config = selection_dataset(selection, item.candidate.model)
    protocol = config.protocol
    sampling = protocol.sampling
    judge = selection.run.judge
    return _Row(
        evaluation=item.evaluation.name,
        context=str(item.context),
        output=str(item.max_output or "-"),
        seed=str(item.seed) if item.seed is not None else "protocol",
        runs_as=f"{selection.benchmark}/{config.protocol_name} {protocol.operation.value}"
        + (f" {config.agent.harness.kind}" if config.agent is not None else ""),
        think=_thinking(sampling.chat_template_kwargs) if sampling is not None else "-",
        trials=str(protocol.trials.count),
        cap=str(sampling.max_tokens) if sampling is not None and sampling.max_tokens is not None else "-",
        judge=f"{judge.deployment}:{judge.default_model}" if judge is not None else "-",
        metric=protocol.primary_metric,
    )


def policy_rows(policy: EvalPolicy, candidate: PolicyCandidate) -> tuple[_Row, ...]:
    """One row per evaluation the candidate runs, with its resolved settings and benchmark-owned facts."""
    return tuple(_row(item) for item in resolve(policy, candidate))


def _print_table(rows: tuple[_Row, ...]) -> None:
    cells = [_COLUMNS, *((*vars(row).values(),) for row in rows)]
    widths = [max(len(row[index]) for row in cells) for index in range(len(_COLUMNS))]
    for row in cells:
        click.echo("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip())


@click.group()
def cli() -> None:
    """Inspect eval policies."""


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.option("--candidate", "candidate_name", help="Show one candidate; default all.")
def show(version: str, candidate_name: str | None) -> None:
    """Print each candidate's evaluations with resolved settings, exclusions and overrides."""
    policy = POLICIES[version]
    candidates = {candidate.name: candidate for candidate in policy.candidates}
    if candidate_name is not None and candidate_name not in candidates:
        raise click.BadParameter(f"not a candidate of {version}; known: {sorted(candidates)}", param_hint="--candidate")
    for name, candidate in candidates.items():
        if candidate_name is not None and name != candidate_name:
            continue
        model = candidate.model
        click.echo(
            f"\n{policy.version}  candidate {name} ({model.location}@{model.revision})  "
            f"max_context={candidate.max_context}  served max_model_len={model.serve.max_model_len}"
        )
        _print_table(policy_rows(policy, candidate))
        for exclusion in policy.exclusions:
            if exclusion.candidate == candidate:
                click.echo(f"excluded: {exclusion.evaluation.name} ({exclusion.reason})")
        for override in policy.overrides:
            if override.candidate == candidate:
                click.echo(f"override: {override.evaluation.name} ({override.reason})")


if __name__ == "__main__":
    cli()
