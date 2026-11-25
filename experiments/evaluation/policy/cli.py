# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line view of eval policies.

``uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial``.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import click
from marin.rollouts.evaluation import CHAT_OPERATIONS, served_chat_template_kwargs

from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, catalog_variant, resolve
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate
from experiments.evaluation.policy.versions import POLICIES

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
    """``on`` or ``off`` for the standard switch, else the template arguments the request sends, else ``default``."""
    if "enable_thinking" in kwargs:
        return "on" if kwargs["enable_thinking"] else "off"
    return ",".join(f"{key}={value}" for key, value in kwargs.items()) or "default"


def _seed(policy_seed: int | None, protocol_seed: int | None) -> str:
    """The base seed the run starts from: the policy's, else the protocol's, else none."""
    seed = policy_seed if policy_seed is not None else protocol_seed
    return "unseeded" if seed is None else str(seed)


def _row(item: ResolvedPolicyEvaluation) -> _Row:
    selection = item.selection
    config = catalog_variant(item.evaluation, item.candidate)
    protocol = config.protocol
    sampling = protocol.sampling
    generation = item.candidate.model.generation
    judge = selection.run.judge
    return _Row(
        evaluation=item.evaluation.name,
        context=str(item.context),
        output=str(item.max_output or "-"),
        seed=_seed(item.seed, protocol.trials.seed),
        runs_as=f"{selection.benchmark}/{config.protocol_name} {protocol.operation.value}"
        + (f" {config.agent.harness.kind}" if config.agent is not None else ""),
        think=(
            _thinking(
                served_chat_template_kwargs(
                    sampling.chat_template_kwargs,
                    generation.chat_template_kwargs,
                    generation.thinking_off_template_kwargs,
                )
            )
            if sampling is not None and protocol.operation in CHAT_OPERATIONS
            else "-"
        ),
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
