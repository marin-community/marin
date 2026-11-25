# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Show, launch and check eval policies."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import click
from iris.cli.connect import open_iris_client
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value
from marin.evaluation.runner import EvaluationBatch
from marin.rollouts.evaluation import CHAT_OPERATIONS, served_chat_template_kwargs
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import git_output

from experiments.evaluation.cli import require_coreweave_credentials
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, launch_group, prepare_evaluation_batch
from experiments.evaluation.policy.launch import evaluation_dataset, launch_specs
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.runs import (
    ExpectedRun,
    RunCheck,
    RunState,
    check_run,
    expected_record,
    expected_runs,
    find_record,
)
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate
from experiments.evaluation.policy.versions import POLICIES

logger = logging.getLogger(__name__)

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
    selection = item.evaluation.selection
    config = evaluation_dataset(item)
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


def _print_table(columns: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> None:
    cells = [columns, *rows]
    widths = [max(len(row[index]) for row in cells) for index in range(len(columns))]
    for row in cells:
        click.echo("  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip())


def _run_summary(check: RunCheck) -> str:
    """State, primary score, items scored, commit, and the first problem or warning of a checked run."""
    record = check.record
    parts = [check.state.value + (" ⚠" if check.warnings else "")]
    for task in record.evaluation.tasks:
        if task.benchmark is not None and (
            (score := record.metrics.get(task.name, {}).get(task.benchmark.primary_metric)) is not None
        ):
            parts.append(f"{task.benchmark.primary_metric}={score:.3f}")
    if record.coverage:
        scored = sum(coverage.n_scored for coverage in record.coverage.values())
        total = sum(coverage.n_benchmark or 0 for coverage in record.coverage.values())
        parts.append(f"{scored}/{total}")
    parts.append(f"@{record.provenance.git_sha[:7]}")
    if notes := (*check.problems, *check.warnings):
        more = f" (+{len(notes) - 1})" if len(notes) > 1 else ""
        parts.append(f"[{notes[0]}{more}]")
    return " ".join(parts)


def _expected_summary(run: ExpectedRun) -> str:
    record = expected_record(run)
    return "not run" if record is None else _run_summary(check_run(record, run))


@click.group()
def cli() -> None:
    """Show, launch and check eval policies."""


def _candidate(policy: EvalPolicy, name: str) -> PolicyCandidate:
    candidates = {candidate.name: candidate for candidate in policy.candidates}
    if name not in candidates:
        raise click.BadParameter(
            f"not a candidate of {policy.version}; known: {sorted(candidates)}", param_hint="--candidate"
        )
    return candidates[name]


def _print_launch(policy: EvalPolicy, candidate: PolicyCandidate, batches: list[EvaluationBatch]) -> None:
    model = candidate.model
    click.echo(f"policy {policy.version} @ {batches[0].provenance.git_sha[:10]}  candidate {candidate.name}")
    click.echo(f"  model {model.location}@{model.revision}")
    for index, batch in enumerate(batches, start=1):
        location = batch.accelerator.target_cluster or batch.accelerator.region
        click.echo(
            f"server {index}: {batch.accelerator.label} on {location}  "
            f"max_model_len={batch.model.serve.max_model_len}  "
            f"priority={priority_band_name(batch.priority_band)}"
        )
        for evaluation in batch.evaluations:
            reference = evaluation.identity.eval_ref
            assert reference.curated is not None
            settings = reference.curated.settings
            agent = reference.curated.agent
            budget = (
                f"  agent={agent.harness} input={agent.max_input_tokens} reply={agent.max_output_tokens}"
                if agent
                else ""
            )
            click.echo(
                f"  {reference.name:<20} limit={settings.source_limit}  max_tokens={settings.max_tokens}  "
                f"seed={settings.seed}{budget}  "
                f"results={evaluation.identity.output_dir}"
            )


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.option("--candidate", "candidate_name", help="Show one candidate; default all.")
def show(version: str, candidate_name: str | None) -> None:
    """Print each candidate's evaluations with resolved settings, exclusions and overrides."""
    policy = POLICIES[version]
    candidates = {candidate.name: candidate for candidate in policy.candidates}
    if candidate_name is not None:
        _candidate(policy, candidate_name)
    for name, candidate in candidates.items():
        if candidate_name is not None and name != candidate_name:
            continue
        model = candidate.model
        click.echo(
            f"\n{policy.version}  candidate {name} ({model.location}@{model.revision})  "
            f"max_context={candidate.max_context}  served max_model_len={model.serve.max_model_len}"
        )
        _print_table(_COLUMNS, tuple(tuple(vars(row).values()) for row in policy_rows(policy, candidate)))
        for exclusion in policy.exclusions:
            if exclusion.candidate == candidate:
                click.echo(f"excluded: {exclusion.evaluation.name} ({exclusion.reason})")
        for override in policy.overrides:
            if override.candidate == candidate:
                click.echo(f"override: {override.evaluation.name} ({override.reason})")


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.option("--candidate", "candidate_name", required=True, help="Candidate to evaluate.")
@click.option(
    "--priority",
    required=True,
    type=click.Choice(PRIORITY_BAND_NAMES, case_sensitive=False),
    help="Iris priority band for the orchestrator and serve jobs.",
)
@click.option("--evaluations", default=None, help="Comma-separated evaluation names; default every one.")
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Cap items per task; marks a smoke run.")
@click.option("--federated-cluster", default=None, help="GPU cluster to serve on; defaults to the fleet profile.")
@click.option("--dry-run", is_flag=True, help="Resolve and print each server's launch without submitting.")
def launch(
    version: str,
    candidate_name: str,
    priority: str,
    evaluations: str | None,
    limit: int | None,
    federated_cluster: str | None,
    dry_run: bool,
) -> None:
    """Submit one candidate's policy evaluations, one job per server, without waiting for results.

    A full run (no --limit) must launch from a clean checkout so its records name the code that ran.
    """
    policy = POLICIES[version]
    candidate = _candidate(policy, candidate_name)
    if not dry_run and (changes := git_output(["status", "--porcelain"], find_project_root())):
        if limit is None:
            raise click.ClickException(f"commit or remove uncommitted changes before a full launch:\n{changes}")
        logger.warning("smoke run from a checkout with uncommitted changes; its records name only HEAD")
    names = None if evaluations is None else {name.strip() for name in evaluations.split(",") if name.strip()}
    try:
        specs = launch_specs(
            policy,
            candidate,
            evaluations=names,
            limit=limit,
            priority_band=priority_band_value(priority),
            federated_cluster=federated_cluster,
        )
        batches = [prepare_evaluation_batch(spec) for spec in specs]
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    recorded, head = batches[0].provenance.git_sha, git_output(["rev-parse", "HEAD"], find_project_root())
    if not dry_run and limit is None and recorded != head:
        raise click.ClickException(
            f"records would name commit {recorded}, not HEAD {head}; unset MARIN_GIT_SHA and GIT_COMMIT"
        )
    for batch in batches:
        require_coreweave_credentials(batch)
    _print_launch(policy, candidate, batches)
    if dry_run:
        return
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        for batch in batches:
            group = launch_group(batch, client)
            click.echo(f"submitted {group.group_id}  job {group.job.job_id}  records {group.records_prefix}")


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.option("--candidate", "candidate_name", help="One candidate; default all.")
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True, help="Item limit of smoke runs.")
def status(version: str, candidate_name: str | None, limit: int) -> None:
    """Print each evaluation's full and smoke run, found where a launch from this checkout puts them.

    Runs at other paths, such as ones launched another way or with other inputs, are not shown; verify checks those.
    """
    policy = POLICIES[version]
    candidates = policy.candidates if candidate_name is None else (_candidate(policy, candidate_name),)
    configure_coreweave_s3()
    for candidate in candidates:
        smoke = {run.name: run for run in expected_runs(policy, candidate, limit=limit)}
        rows = tuple(
            (run.name, _expected_summary(run), _expected_summary(smoke[run.name]))
            for run in expected_runs(policy, candidate, limit=None)
        )
        click.echo(f"\n{policy.version}  candidate {candidate.name}")
        _print_table(("evaluation", "full run", f"smoke run (limit {limit})"), rows)


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.argument("run_ids", nargs=-1, required=True)
def verify(version: str, run_ids: tuple[str, ...]) -> None:
    """Check runs against the policy as this checkout resolves it, and exit non-zero if any is invalid.

    Warnings flag results far outside expectations without making a run invalid.
    """
    policy = POLICIES[version]
    configure_coreweave_s3()
    invalid = False
    for run_id in run_ids:
        record = find_record(run_id)
        # Records launched outside the policy carry ad-hoc model names, so match the location and use the
        # name only to tell apart candidates that share one.
        matches = [candidate for candidate in policy.candidates if candidate.model.location == record.model.location]
        if len(matches) > 1:
            matches = [candidate for candidate in matches if candidate.name == record.model.name]
        if len(matches) != 1 or record.evaluation.curated is None:
            raise click.ClickException(
                f"{run_id}: {record.model.location} / {record.evaluation.name} is not one curated {version} run"
            )
        (candidate,) = matches
        try:
            (expected,) = expected_runs(
                policy,
                candidate,
                limit=record.evaluation.curated.settings.source_limit,
                evaluations={record.evaluation.name},
            )
        except ValueError as error:
            raise click.ClickException(f"{run_id}: {error}") from error
        check = check_run(record, expected)
        click.echo(f"{run_id}  {_run_summary(check)}")
        for problem in check.problems:
            click.echo(f"  {problem}")
        for warning in check.warnings:
            click.echo(f"  warning: {warning}")
        invalid |= check.state is RunState.INVALID
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
