# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Show, launch and check eval policies."""

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import click
from evalrecords.records import EvalRunRecord, EvaluationSelection, RunStatus
from iris.cli.connect import open_iris_client
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value
from marin.evaluation.model_config import load_model_config, prompt_format
from marin.evaluation.runner import EvaluationBatch
from marin.execution.step_runner import check_cache
from marin.rollouts.evaluation import CHAT_OPERATIONS, served_chat_template_kwargs
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.provenance import git_output

from experiments.evaluation.cli import require_coreweave_credentials
from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, launch_group
from experiments.evaluation.models import models
from experiments.evaluation.policy.launch import (
    PolicyModel,
    evaluation_context,
    model_evaluations,
    policy_model,
    source_checkout,
)
from experiments.evaluation.policy.runs import (
    ExpectedRun,
    FoundRun,
    RunCheck,
    RunState,
    check_found,
    check_run,
    compliant_run,
    expected_runs,
    find_record,
    found_runs,
    record_model,
    run_suffix,
    split_run_id,
)
from experiments.evaluation.policy.schema import Policy
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


def _row(model: PolicyModel, evaluation: EvaluationSelection) -> _Row:
    config = selection_dataset(evaluation, prompt_format(model.model))
    protocol = config.protocol
    sampling = protocol.sampling
    generation = model.model.generation
    judge = evaluation.run.judge
    seed = evaluation.run.seed if evaluation.run.seed is not None else protocol.trials.seed
    return _Row(
        evaluation=evaluation.name,
        context=str(evaluation_context(model, evaluation)),
        output=str(evaluation.run.max_tokens or "-"),
        seed="unseeded" if seed is None else str(seed),
        runs_as=f"{evaluation.benchmark}/{config.protocol_name} {protocol.operation.value}"
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


def _evaluation_names(_context: click.Context, _parameter: click.Parameter, value: str | None) -> frozenset[str] | None:
    """Parse ``--evaluations``: None when it is absent, else the evaluations it names, at least one."""
    if value is None:
        return None
    names = frozenset(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise click.BadParameter("name at least one evaluation")
    return names


@click.group()
def cli() -> None:
    """Show, launch and check eval policies."""


def _target_options(command: Callable) -> Callable:
    """The options naming the policy's baselines, other catalog models, or one supplied model."""
    options = (
        click.option(
            "--model",
            "model_names",
            multiple=True,
            help="A model catalog key: one of the policy's baselines, or another to run its evaluations on. Repeatable.",
        ),
        click.option("--baselines", "all_baselines", is_flag=True, help="Every baseline of the policy."),
        click.option(
            "--model-config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="A model YAML, such as a training checkpoint's, to run in place of the policy's baselines.",
        ),
    )
    for option in reversed(options):
        command = option(command)
    return command


def _targets(
    policy: Policy, model_names: tuple[str, ...], all_baselines: bool, model_config: Path | None
) -> tuple[PolicyModel, ...] | None:
    """The models a command runs: named ones, every baseline, or a supplied model; None if none chosen."""
    chosen = bool(model_names) + all_baselines + (model_config is not None)
    if chosen > 1:
        raise click.UsageError("name only one of --model, --baselines and --model-config")
    if chosen == 0:
        return None
    catalog = models()
    try:
        if model_config is not None:
            model = load_model_config(model_config)
            if model.name in catalog:
                raise click.UsageError(f"{model.name} names a catalog model; give the supplied model its own name")
            return (policy_model(policy, model, baseline=False),)
        if unknown := [name for name in model_names if name not in catalog]:
            raise click.BadParameter(f"not in the model catalog: {unknown}", param_hint="--model")
        names = dict.fromkeys(model_names) if model_names else policy.baselines
        return tuple(policy_model(policy, catalog[name], baseline=name in policy.baselines) for name in names)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


@dataclass(frozen=True)
class _ModelPlan:
    """One model's servers to submit, and what the launch leaves out and why."""

    model: PolicyModel
    batches: tuple[EvaluationBatch, ...]
    excluded: Mapping[str, str]
    done: Mapping[str, FoundRun]
    """Evaluations with a compliant run of the same inputs, at any commit."""
    invalid: Mapping[str, FoundRun]
    """Evaluations whose run at this checkout finished but doesn't comply; ``--rerun`` replaces them."""
    reruns: frozenset[str]


@dataclass(frozen=True)
class _Decision:
    """What a launch does with one evaluation."""

    attempt: int | None
    """The attempt to submit at this checkout; None submits nothing."""
    run: FoundRun | None = None
    """The compliant run that makes it done, or the finished, invalid run at this checkout that waits for --rerun."""
    compliant: bool = False
    replaces: str | None = None
    """The run ID named with --rerun that this attempt runs again."""


def _decide(runs: tuple[FoundRun, ...], expected: ExpectedRun, checkout: str, rerun: Collection[str]) -> _Decision:
    """Decide one evaluation of a launch.

    A run named in ``rerun`` gets the next attempt at this checkout if it finished and is invalid or flagged.
    Otherwise a compliant run at any commit leaves it out; a run at this checkout whose record succeeded but
    doesn't comply waits for ``--rerun``, since running its attempt again would overwrite it; an unfinished or
    failed run resumes; else attempt 1 starts.
    """
    here = [run for run in runs if run.checkout == checkout]
    if named := [run for run in runs if run.run_id in rerun]:
        (target,) = named
        check = check_found(target, expected)
        if check is None:
            raise ValueError(f"{target.run_id} has not finished")
        if check.state is RunState.VALID and not check.warnings:
            raise ValueError(f"{target.run_id} is valid; only an invalid or flagged run is run again")
        return _Decision(attempt=here[-1].attempt + 1 if here else 1, replaces=target.run_id)
    if (hit := compliant_run(runs, expected)) is not None:
        return _Decision(attempt=None, run=hit[0], compliant=True)
    if not here:
        return _Decision(attempt=1)
    newest = here[-1]
    if newest.record is not None and newest.record.status is RunStatus.SUCCEEDED:
        return _Decision(attempt=None, run=newest)
    return _Decision(attempt=newest.attempt)


def _launch_plan(
    policy: Policy,
    model: PolicyModel,
    evaluations: Collection[str] | None,
    rerun: Collection[str],
    *,
    limit: int | None,
    checkout: str,
    **placement,
) -> _ModelPlan:
    """Resolve one model's launch: its evaluations, which of them already have a compliant run, and the attempt
    each of the rest runs as. ``placement`` is the priority band and cluster ``expected_runs`` takes."""
    try:
        plan = model_evaluations(policy, model, evaluations)
        first = {evaluation.name: run_suffix(checkout, 1) for evaluation in plan.evaluations}
        initial = expected_runs(policy, plan, limit=limit, run_suffixes=first, **placement)
        decisions = {expected.name: _decide(found_runs(expected), expected, checkout, rerun) for expected in initial}
        suffixes = {
            name: run_suffix(checkout, decision.attempt)
            for name, decision in decisions.items()
            if decision.attempt is not None
        }
        remaining = tuple(evaluation for evaluation in plan.evaluations if evaluation.name in suffixes)
        if suffixes == first:
            runs = initial
        elif remaining:
            runs = expected_runs(
                policy, replace(plan, evaluations=remaining), limit=limit, run_suffixes=suffixes, **placement
            )
        else:
            runs = ()
    except ValueError as error:
        raise click.ClickException(f"{model.name}: {error}") from error
    return _ModelPlan(
        model=model,
        batches=tuple({id(run.batch): run.batch for run in runs}.values()),
        excluded=plan.excluded,
        done={name: decision.run for name, decision in decisions.items() if decision.run and decision.compliant},
        invalid={name: decision.run for name, decision in decisions.items() if decision.run and not decision.compliant},
        reruns=frozenset(decision.replaces for decision in decisions.values() if decision.replaces is not None),
    )


def _print_launch(policy: Policy, plan: _ModelPlan, checkout: str) -> None:
    model = plan.model.model
    click.echo(f"policy {policy.version} @ {checkout}  model {model.name}")
    click.echo(f"  {model.location}@{model.revision}  served at {plan.model.window}")
    for name, reason in sorted(plan.excluded.items()):
        click.echo(f"  excluded {name}: {reason}")
    for name, run in sorted(plan.done.items()):
        click.echo(f"  skip {name}: compliant run {run.run_id}")
    for name, run in sorted(plan.invalid.items()):
        click.echo(f"  invalid {name}: {run.run_id}; run it again with --rerun")
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
                f"seed={settings.seed}{budget}  run={evaluation.identity.run_id}"
            )


def _check_checkout(*, recorded: str, smoke: bool) -> None:
    """Refuse a full run from uncommitted code, and records that would name another commit than HEAD.

    A smoke run may launch from a dirty checkout; its run IDs say so, and it never complies.
    """
    root = find_project_root()
    if not smoke and (changes := git_output(["status", "--porcelain"], root)):
        raise click.ClickException(f"commit or remove uncommitted changes before a full run:\n{changes}")
    head = git_output(["rev-parse", "HEAD"], root)
    if recorded != head:
        raise click.ClickException(
            f"records would name commit {recorded}, not HEAD {head}; unset MARIN_GIT_SHA and GIT_COMMIT"
        )


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.option("--model", "model_name", help="Show one of the policy's baselines; default all.")
def show(version: str, model_name: str | None) -> None:
    """Print each model's evaluations with their resolved settings and exclusions."""
    policy = POLICIES[version]
    names = (model_name,) if model_name is not None else policy.baselines
    for name in names:
        try:
            model = policy_model(policy, policy.baseline(name), baseline=True)
            plan = model_evaluations(policy, model)
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        click.echo(
            f"\n{policy.version}  model {name} ({model.model.location}@{model.model.revision})  served at {model.window}"
        )
        _print_table(_COLUMNS, tuple(tuple(vars(_row(model, evaluation)).values()) for evaluation in plan.evaluations))
        for evaluation, reason in sorted(plan.excluded.items()):
            click.echo(f"excluded: {evaluation} ({reason})")


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@_target_options
@click.option(
    "--priority",
    required=True,
    type=click.Choice(PRIORITY_BAND_NAMES, case_sensitive=False),
    help="Iris priority band for the orchestrator and serve jobs.",
)
@click.option(
    "--evaluations",
    default=None,
    callback=_evaluation_names,
    help="Comma-separated evaluation names; default every one.",
)
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Cap items per task; marks a smoke run.")
@click.option("--federated-cluster", default=None, help="GPU cluster to serve on; defaults to the fleet profile.")
@click.option("--dry-run", is_flag=True, help="Resolve and print each server's launch without submitting.")
@click.option(
    "--rerun",
    "rerun",
    multiple=True,
    help="Run ID of an invalid or flagged run to run again as a new attempt; repeat for several.",
)
def launch(
    version: str,
    model_names: tuple[str, ...],
    all_baselines: bool,
    model_config: Path | None,
    priority: str,
    evaluations: frozenset[str] | None,
    limit: int | None,
    federated_cluster: str | None,
    dry_run: bool,
    rerun: tuple[str, ...],
) -> None:
    """Submit the policy's evaluations for each chosen model, one job per server, without waiting for results.

    Every launch resolves before any job is submitted. An evaluation with a compliant run of the same inputs, at
    any commit, is left out, and a server with none left is not started. Runs are never replaced: ``--rerun``
    writes a new attempt beside the old one. A full run (no --limit) must launch from a clean checkout.
    """
    policy = POLICIES[version]
    # A supplied checkpoint's config is read from object storage while its model is resolved.
    configure_coreweave_s3()
    targets = _targets(policy, model_names, all_baselines, model_config)
    if targets is None:
        raise click.UsageError("name --model, --baselines or --model-config")
    checkout = source_checkout(find_project_root())
    plans = [
        _launch_plan(
            policy,
            model,
            evaluations,
            frozenset(rerun),
            limit=limit,
            checkout=checkout,
            priority_band=priority_band_value(priority),
            federated_cluster=federated_cluster,
        )
        for model in targets
    ]
    if unknown := set(rerun) - {run_id for plan in plans for run_id in plan.reruns}:
        raise click.UsageError(f"--rerun names runs this launch does not make: {sorted(unknown)}")
    batches = [batch for plan in plans for batch in plan.batches]
    if not dry_run and batches:
        _check_checkout(recorded=batches[0].provenance.git_sha, smoke=limit is not None)
    for plan in plans:
        for batch in plan.batches:
            require_coreweave_credentials(batch)
        _print_launch(policy, plan, checkout)
    if dry_run or not batches:
        return
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        for batch in batches:
            group = launch_group(batch, client)
            click.echo(f"submitted {group.group_id}  job {group.job.job_id}  records {group.records_prefix}")


def _status(runs: tuple[FoundRun, ...], expected: ExpectedRun, checkout: str) -> str:
    """The compliant run of an evaluation, else its newest run's standing, else ``not run``."""
    if (hit := compliant_run(runs, expected)) is not None:
        run, check = hit
        where = "this commit" if run.checkout == checkout else f"commit {run.checkout[:7]}"
        return f"{_run_summary(check)} ({where})"
    if not runs:
        return "not run"
    newest = max(runs, key=lambda run: (run.record.created_at if run.record else "~", run.attempt))
    check = check_found(newest, expected)
    if check is None:
        return f"running @{newest.checkout[:7]}"
    return _run_summary(check)


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@_target_options
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True, help="Item limit of smoke runs.")
def status(
    version: str, model_names: tuple[str, ...], all_baselines: bool, model_config: Path | None, limit: int
) -> None:
    """Print each evaluation's full and smoke run: its compliant run at any commit, else its newest run.

    Shows every model unless others are chosen.
    """
    policy = POLICIES[version]
    configure_coreweave_s3()
    targets = _targets(policy, model_names, all_baselines, model_config) or _targets(policy, (), True, None)
    assert targets is not None
    checkout = source_checkout(find_project_root())
    for model in targets:
        try:
            plan = model_evaluations(policy, model)
            suffixes = {evaluation.name: run_suffix(checkout, 1) for evaluation in plan.evaluations}
            full = expected_runs(policy, plan, limit=None, run_suffixes=suffixes)
            smoke = {run.name: run for run in expected_runs(policy, plan, limit=limit, run_suffixes=suffixes)}
        except ValueError as error:
            raise click.ClickException(f"{model.name}: {error}") from error
        rows = tuple(
            (
                run.name,
                _status(found_runs(run), run, checkout),
                _status(found_runs(smoke[run.name]), smoke[run.name], checkout),
            )
            for run in full
        )
        click.echo(f"\n{policy.version}  model {model.name}")
        _print_table(("evaluation", "full run", f"smoke run (limit {limit})"), rows)
        for name, reason in sorted(plan.excluded.items()):
            click.echo(f"excluded: {name} ({reason})")


def _verified(policy: Policy, record: EvalRunRecord, records_prefix: str) -> RunCheck:
    """Check a record against the run the policy expects, built by this checkout, which must be the run's own."""
    root = find_project_root()
    name = split_run_id(record.run_id)
    head = git_output(["rev-parse", "HEAD"], root)
    if head != record.provenance.git_sha or name.checkout != head[:12] or git_output(["status", "--porcelain"], root):
        raise ValueError(f"check it from a clean checkout of its commit: git checkout {record.provenance.git_sha}")
    assert record.evaluation.curated is not None
    plan = model_evaluations(policy, record_model(policy, record), {record.evaluation.name})
    if record.evaluation.name in plan.excluded:
        raise ValueError(f"{policy.version} leaves it out: {plan.excluded[record.evaluation.name]}")
    (expected,) = expected_runs(
        policy,
        plan,
        limit=record.evaluation.curated.settings.source_limit,
        run_suffixes={record.evaluation.name: run_suffix(name.checkout, name.attempt)},
    )
    return check_run(record, expected, step_succeeded=check_cache(prefix_join(records_prefix, record.run_id)))


@cli.command()
@click.option("--policy", "version", required=True, type=click.Choice(sorted(POLICIES)), help="Policy version.")
@click.argument("run_ids", nargs=-1, required=True)
def verify(version: str, run_ids: tuple[str, ...]) -> None:
    """Check runs against the policy from a clean checkout of their commit, and exit non-zero unless every one
    complies. Warnings flag results far outside expectations without making a run invalid."""
    policy = POLICIES[version]
    configure_coreweave_s3()
    invalid = False
    for run_id in run_ids:
        record, records_prefix = find_record(run_id)
        if record.evaluation.curated is None:
            raise click.ClickException(f"{run_id}: {record.evaluation.name} is not a curated evaluation")
        try:
            check = _verified(policy, record, records_prefix)
        except ValueError as error:
            raise click.ClickException(f"{run_id}: {error}") from error
        click.echo(f"{run_id}  {_run_summary(check)}")
        for problem in check.problems:
            click.echo(f"  {problem}")
        for warning in check.warnings:
            click.echo(f"  warning: {warning}")
        invalid |= check.state is not RunState.VALID
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
