# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Show, launch and check eval policies."""

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import click
from evalrecords.records import EvalRunRecord
from iris.cli.connect import open_iris_client
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value
from marin.evaluation.model_config import load_model_config
from marin.evaluation.runner import EvaluationBatch
from marin.rollouts.evaluation import CHAT_OPERATIONS, served_chat_template_kwargs
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.provenance import git_output

from experiments.evaluation.cli import require_coreweave_credentials
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, launch_group
from experiments.evaluation.models import models
from experiments.evaluation.policy.launch import evaluation_dataset, model_candidate
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.runs import (
    ExpectedRun,
    RunCheck,
    RunState,
    check_run,
    expected_record,
    expected_runs,
    find_record,
    record_candidate,
    succeeded_runs,
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


def _expected_summary(run: ExpectedRun, release_commit: str | None) -> str:
    record = expected_record(run)
    return "not run" if record is None else _run_summary(check_run(record, run, release_commit=release_commit))


def _release_commit(policy: EvalPolicy) -> str | None:
    """The commit a released policy's tag names, or None for a policy not released."""
    if not policy.released:
        return None
    try:
        return git_output(["rev-parse", f"{policy.release_tag}^{{commit}}"], find_project_root())
    except RuntimeError as error:
        raise click.ClickException(f"{policy.version} is released as {policy.release_tag}; fetch its tag") from error


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


@dataclass(frozen=True)
class _LaunchPlan:
    """One candidate's servers to submit, and the evaluations left out because their runs already succeeded."""

    candidate: PolicyCandidate
    git_sha: str
    batches: tuple[EvaluationBatch, ...]
    succeeded: Mapping[str, EvalRunRecord]


def _launch_plan(
    policy: EvalPolicy, candidate: PolicyCandidate, evaluations: Collection[str] | None, **placement
) -> _LaunchPlan:
    """Resolve a candidate's launch, leaving out evaluations whose run at its path already succeeded.

    Run IDs depend only on each evaluation, so the evaluations left still group onto the same servers
    under the same IDs. ``placement`` is the priority band and cluster ``expected_runs`` takes.
    """
    try:
        runs = expected_runs(policy, candidate, evaluations=evaluations, **placement)
        git_sha = runs[0].batch.provenance.git_sha
        succeeded = succeeded_runs(runs)
        if succeeded:
            remaining = [run.name for run in runs if run.name not in succeeded]
            runs = expected_runs(policy, candidate, evaluations=remaining, **placement) if remaining else ()
    except ValueError as error:
        raise click.ClickException(f"{candidate.name}: {error}") from error
    batches = tuple({id(run.batch): run.batch for run in runs}.values())
    return _LaunchPlan(candidate=candidate, git_sha=git_sha, batches=batches, succeeded=succeeded)


def _print_launch(policy: EvalPolicy, plan: _LaunchPlan) -> None:
    model = plan.candidate.model
    click.echo(f"policy {policy.version} @ {plan.git_sha[:10]}  candidate {plan.candidate.name}")
    click.echo(f"  model {model.location}@{model.revision}")
    for name, record in sorted(plan.succeeded.items()):
        click.echo(f"  skip {name}: succeeded @{record.provenance.git_sha[:7]}")
    if not plan.batches:
        click.echo("  nothing to submit")
    for index, batch in enumerate(plan.batches, start=1):
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


def _target_options(command: Callable) -> Callable:
    """The options naming policy candidates or one supplied model."""
    options = (
        click.option("--candidate", "candidate_names", multiple=True, help="A candidate; repeat for several."),
        click.option("--all-candidates", is_flag=True, help="Every candidate of the policy."),
        click.option(
            "--model", "model_key", default=None, help="A model catalog key to run in place of the candidates."
        ),
        click.option(
            "--model-config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="A model YAML, such as a training checkpoint's, to run in place of the candidates.",
        ),
    )
    for option in reversed(options):
        command = option(command)
    return command


def _targets(
    policy: EvalPolicy,
    candidate_names: tuple[str, ...],
    all_candidates: bool,
    model_key: str | None,
    model_config: Path | None,
) -> tuple[PolicyCandidate, ...] | None:
    """The candidates a command runs: named ones, every one, or a supplied model; None if none chosen."""
    if model_key is not None and model_config is not None:
        raise click.UsageError("--model and --model-config are mutually exclusive")
    model_given = model_key is not None or model_config is not None
    chosen = bool(candidate_names) + all_candidates + model_given
    if chosen > 1:
        raise click.UsageError("name only one of --candidate, --all-candidates, --model and --model-config")
    if chosen == 0:
        return None
    if model_given:
        if model_config is not None:
            model = load_model_config(model_config)
        elif model_key in (catalog := models()):
            model = catalog[model_key]
        else:
            raise click.BadParameter(f"not in the model catalog: {model_key}", param_hint="--model")
        try:
            return (model_candidate(model),)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
    if candidate_names:
        return tuple(_candidate(policy, name) for name in dict.fromkeys(candidate_names))
    return policy.candidates


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


def _check_checkout(policy: EvalPolicy, *, recorded: str, smoke: bool) -> None:
    """Refuse a launch whose records would not name the code that ran, or a release launched off its tag.

    A smoke run of a policy not released may launch from a dirty checkout or another recorded commit.
    """
    root = find_project_root()
    release = _release_commit(policy)
    strict = not smoke or release is not None
    if changes := git_output(["status", "--porcelain"], root):
        if strict:
            raise click.ClickException(f"commit or remove uncommitted changes before this launch:\n{changes}")
        logger.warning("smoke run from a checkout with uncommitted changes; its records name only HEAD")
    head = git_output(["rev-parse", "HEAD"], root)
    if strict and recorded != head:
        raise click.ClickException(
            f"records would name commit {recorded}, not HEAD {head}; unset MARIN_GIT_SHA and GIT_COMMIT"
        )
    if release is not None and head != release:
        raise click.ClickException(f"{policy.version} launches from its tag: git checkout {policy.release_tag}")


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@_target_options
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
    candidate_names: tuple[str, ...],
    all_candidates: bool,
    model_key: str | None,
    model_config: Path | None,
    priority: str,
    evaluations: str | None,
    limit: int | None,
    federated_cluster: str | None,
    dry_run: bool,
) -> None:
    """Submit the policy's evaluations for each chosen model, one job per server, without waiting for results.

    Every launch resolves before any job is submitted. Evaluations whose run already succeeded are left out,
    and a server with none left is not started. A full run (no --limit), and any run of a released policy,
    must launch from a clean checkout so its records name the code that ran; a released policy launches only
    from its tag.
    """
    policy = POLICIES[version]
    candidates = _targets(policy, candidate_names, all_candidates, model_key, model_config)
    if candidates is None:
        raise click.UsageError("name --candidate, --all-candidates, --model or --model-config")
    names = None if evaluations is None else {name.strip() for name in evaluations.split(",") if name.strip()}
    configure_coreweave_s3()
    plans = [
        _launch_plan(
            policy,
            candidate,
            names,
            limit=limit,
            priority_band=priority_band_value(priority),
            federated_cluster=federated_cluster,
        )
        for candidate in candidates
    ]
    if not dry_run:
        _check_checkout(policy, recorded=plans[0].git_sha, smoke=limit is not None)
    for plan in plans:
        for batch in plan.batches:
            require_coreweave_credentials(batch)
        _print_launch(policy, plan)
    if dry_run or not any(plan.batches for plan in plans):
        return
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        for plan in plans:
            for batch in plan.batches:
                group = launch_group(batch, client)
                click.echo(f"submitted {group.group_id}  job {group.job.job_id}  records {group.records_prefix}")


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@_target_options
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True, help="Item limit of smoke runs.")
def status(
    version: str,
    candidate_names: tuple[str, ...],
    all_candidates: bool,
    model_key: str | None,
    model_config: Path | None,
    limit: int,
) -> None:
    """Print each evaluation's full and smoke run, found where a launch from this checkout puts them.

    Shows every candidate unless others are chosen. Runs at other paths, such as ones launched another way
    or with other inputs, are not shown; verify checks those.
    """
    policy = POLICIES[version]
    candidates = _targets(policy, candidate_names, all_candidates, model_key, model_config) or policy.candidates
    release = _release_commit(policy)
    configure_coreweave_s3()
    for candidate in candidates:
        smoke = {run.name: run for run in expected_runs(policy, candidate, limit=limit)}
        rows = tuple(
            (run.name, _expected_summary(run, release), _expected_summary(smoke[run.name], release))
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
    release = _release_commit(policy)
    configure_coreweave_s3()
    invalid = False
    for run_id in run_ids:
        record = find_record(run_id)
        if record.evaluation.curated is None:
            raise click.ClickException(f"{run_id}: {record.evaluation.name} is not a curated evaluation")
        try:
            candidate = record_candidate(policy, record)
            (expected,) = expected_runs(
                policy,
                candidate,
                limit=record.evaluation.curated.settings.source_limit,
                evaluations={record.evaluation.name},
            )
        except ValueError as error:
            raise click.ClickException(f"{run_id}: {error}") from error
        check = check_run(record, expected, release_commit=release)
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
