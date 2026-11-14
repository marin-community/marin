# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data Belay command line: `uv run data-belay <command>`."""

import logging
import math
import os
import shlex
import statistics
import sys

import click
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import load_config
from levanter.tokenizers import load_tokenizer
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from experiments.data_belay.datakit import MIN_TOP_RUNG, datakit_intake
from experiments.data_belay.intake import MAX_EPOCHS, DatasetRequest, IntakeReport, LicenseVerdict, run_intake
from experiments.data_belay.judge import RungResult, harm_limits, judge, ladder_step
from experiments.data_belay.prepare import (
    CACHE_PATH,
    CachePlan,
    CacheReading,
    PreparedCache,
    build_cache,
    hf_files,
    plan_cache,
)
from experiments.data_belay.results import (
    CHOICE_DIR,
    CONTROL_SEEDS,
    DILUTION_REFERENCE_SOURCES,
    REFERENCE_SEEDS,
    arm_metrics,
    arm_mix,
    arm_share,
    control_run_id,
    dilution_arm,
    dilution_references,
    run_exists,
    run_scored,
    treatment_run_id,
)
from experiments.grug.fast_track.launch import (
    SEQ_LEN,
    V16384_TOKENIZER,
    baseline_steps,
    h100_ladder_rung,
)

# Runs train the fast_track dense model (`launch.py --dense`) at its baseline budget, 20 tokens per parameter.
# d512 screens a candidate; d768 and d1024 confirm a d512 result.
DATA_BELAY_RUNGS = ("d512", "d768", "d1024")
# The rung whose control seeds set the harm limits at every rung: it has the most seeds.
HARM_LIMIT_RUNG = "d512"
# The cluster and Iris band of the runs. prepare's tokenize job runs there too, beside the caches it writes.
CLUSTER = "cw-us-east-02a"
BAND = "batch"
# The launcher's --version, which only the run step takes: the validation caches pin their own.
LAUNCH_VERSION = "2026.09.25"
# Runs write their outputs here, under the cluster's MARIN_PREFIX, with a 30-day TTL.
RUNS_ROOT = "tmp/ttl=30d/data_belay/runs"
# One H100 scores a run's final checkpoint on every task of experiments.data_belay.accuracy in a few minutes.
SCORE_RESOURCES = ("--gpu", "H100x1", "--extra", "gpu", "--cpu", "16", "--memory", "128GB", "--disk", "64GB")


def rung_tokens(rung: str) -> int:
    """Training tokens of the dense baseline at ``rung``: its steps times its batch of 4,096-token sequences."""
    return baseline_steps(rung, dense=True) * h100_ladder_rung(rung).baseline_batch * SEQ_LEN


def train_command(rung: str, run_id: str, flags: list[str], seed: int) -> list[str]:
    """The launch of one run of the dense baseline at ``rung`` plus ``flags``, writing under RUNS_ROOT."""
    return [
        *("uv", "run", "python", "-m", "experiments.grug.fast_track.launch"),
        *("--size", rung, "--dense", "--save-checkpoints", "--version", LAUNCH_VERSION, *flags),
        *("--seed", str(seed), "--run-id", run_id, "--output-path", f"{RUNS_ROOT}/{run_id}"),
        *("--submit", "--target-cluster", CLUSTER, "--priority", BAND),
    ]


def _iris_job(job_name: str, resources: tuple[str, ...], module_args: list[str]) -> list[str]:
    """Iris command that runs ``python -m <module_args>`` on CLUSTER, passing on ``HF_TOKEN``."""
    env = ["-e", "HF_TOKEN", os.environ["HF_TOKEN"]] if os.environ.get("HF_TOKEN") else []
    return [
        *("uv", "run", "iris", "--cluster", "marin", "job", "run", "--no-wait", "--enable-extra-resources"),
        *("--target-cluster", CLUSTER, "--priority", BAND, *resources),
        *("--job-name", job_name, *env),
        *("--", "python", "-m", *module_args),
    ]


def _shown(cmd: list[str]) -> str:
    secret = os.environ.get("HF_TOKEN")
    return " ".join(shlex.quote("$HF_TOKEN" if c == secret else c) for c in cmd)


def _submit_cpu_job(job_name: str, cpu: int, memory: str) -> None:
    """Re-run this command, without ``--submit``, as an Iris CPU job. Never returns."""
    args = [a for a in sys.argv[1:] if a != "--submit"]
    cmd = _iris_job(
        job_name, ("--cpu", str(cpu), "--memory", memory, "--disk", "32GB"), ["experiments.data_belay", *args]
    )
    click.echo(f"submitting: {_shown(cmd)}", err=True)
    os.execvp(cmd[0], cmd)


def _whole_percent(ctx: click.Context, param: click.Parameter, share: float) -> float:
    """A run's label names its share in whole percent, so a share must be one for the label to be exact."""
    if not math.isclose(share * 100, round(share * 100)):
        raise click.BadParameter(f"{share:g} is not a whole percent; run labels name the share in whole percent")
    return round(share * 100) / 100


def _dataset_options(command):
    for option in reversed(
        [
            click.option("--dataset", "repo", required=True, help="Hugging Face dataset id."),
            click.option("--config", "config_name", default=None, help="Dataset config (subset)."),
            click.option("--split", default="train", show_default=True),
            click.option("--text-field", default="text", show_default=True),
            click.option(
                "--share",
                type=float,
                default=0.2,
                show_default=True,
                callback=_whole_percent,
                help="Share of the rung's tokens, a whole percent.",
            ),
            click.option(
                "--license",
                "license_id",
                default=None,
                help="For a card that lists no license: the one a person found in its README or terms.",
            ),
        ]
    ):
        command = option(command)
    return command


def _training_prefix() -> str:
    """``MARIN_PREFIX`` that the training cluster sets for its tasks: the caches and evals that its runs read."""
    config = load_config(resolve_cluster_config(CLUSTER, dirs=IRIS_CLUSTER_CONFIG_DIRS))
    prefix = config.defaults.task_env.get("MARIN_PREFIX")
    if prefix is None:
        raise click.ClickException(f"cluster {CLUSTER} sets no MARIN_PREFIX in its task_env")
    return prefix


def _choice_root() -> str:
    return prefix_join(_training_prefix(), CHOICE_DIR)


def _intake(repo, config_name, split, text_field, share, license_id) -> IntakeReport:
    configure_coreweave_s3()
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    needs = {rung: rung_tokens(rung) for rung in DATA_BELAY_RUNGS}
    return run_intake(DatasetRequest(repo, config_name, split, text_field, license_id), share, needs, tokenizer)


@click.group()
def cli() -> None:
    pass


@cli.command()
@_dataset_options
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
def check(
    repo: str,
    config_name: str | None,
    split: str,
    text_field: str,
    share: float,
    license_id: str | None,
    as_json: bool,
) -> None:
    """Pin the head commit, check the license and estimate the dataset's size at each rung."""
    report = _intake(repo, config_name, split, text_field, share, license_id)
    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(f"dataset: {report.repo}@{report.revision} config={report.config} split={report.split}")
        source = "card" if report.license_from_card else "given by a person"
        click.echo(f"license ({source}): {', '.join(report.licenses) or 'none on the card'} -> {report.license_verdict}")
        if report.gated:
            click.echo("gated: yes (the terms of access must allow training)")
        click.echo(
            f"size: {report.num_rows:,} rows, about {report.estimated_tokens:,} tokens "
            f"({report.sample.tokens_per_row:,.0f} tokens per row over {report.sample.rows:,} sampled rows)"
        )
        for need in report.needs:
            click.echo(
                f"{need.rung}: {need.new_tokens:,} new tokens of {need.rung_tokens:,} at share {report.share:g} "
                f"= {need.epochs:.3f} epochs"
            )
    if report.license_verdict is not LicenseVerdict.ALLOWED:
        click.echo(f"refusing: license {report.license_verdict}; a person must approve it", err=True)
        sys.exit(2)
    if not report.passed:
        click.echo(f"refusing: more than {MAX_EPOCHS:g} epochs at some rung; lower --share", err=True)
        sys.exit(2)
    click.echo("passed")


@cli.command()
@_dataset_options
@click.option("--submit", is_flag=True, help=f"Run as an Iris CPU job on {CLUSTER}.")
def prepare(
    repo: str,
    config_name: str | None,
    split: str,
    text_field: str,
    share: float,
    license_id: str | None,
    submit: bool,
) -> None:
    """Tokenize a seeded file sample that covers the largest rung's new-data tokens into a cache."""
    if submit:
        name = "-".join(part for part in (repo.replace("/", "-"), config_name, f"p{round(share * 100)}") if part)
        _submit_cpu_job(f"data-belay-prepare-{name.lower()}", cpu=4, memory="16GB")
    logging.basicConfig(level=logging.INFO)
    report = _intake(repo, config_name, split, text_field, share, license_id)
    if not report.passed:
        raise click.ClickException(f"intake did not pass: license {report.license_verdict}, needs {report.needs}")
    need_tokens = max(need.new_tokens for need in report.needs)
    cache_root = prefix_join(_training_prefix(), CACHE_PATH)
    plan = plan_cache(report, hf_files(report), need_tokens, V16384_TOKENIZER, cache_root)
    _build(report, plan, allow_test_in_train=False)


def _build(report: IntakeReport, plan: CachePlan, *, allow_test_in_train: bool) -> None:
    click.echo(f"plan: {len(plan.files)} files, about {plan.estimated_tokens:,} tokens -> {plan.cache_path}")
    manifest = build_cache(report, plan, V16384_TOKENIZER, allow_test_in_train=allow_test_in_train)
    cache = PreparedCache.read(plan.cache_path)
    click.echo(f"cache: {cache.root} ({manifest.stats.total_tokens:,} tokens), runs named {cache.candidate}")


@cli.command("prepare-datakit")
@click.option("--label", required=True, help="The candidate's name, e.g. dk-cp-arxiv_papers; runs add -p<share>-<hash>.")
@click.option(
    "--source", "sources", multiple=True, required=True, help="A Datakit registry source; repeat it for a family."
)
@click.option(
    "--share",
    type=float,
    default=0.2,
    show_default=True,
    callback=_whole_percent,
    help="Share of each rung's tokens, a whole percent.",
)
@click.option("--submit", is_flag=True, help=f"Run as an Iris CPU job on {CLUSTER}.")
def prepare_datakit(label: str, sources: tuple[str, ...], share: float, submit: bool) -> None:
    """Tokenize a seeded file sample of Datakit sources' normalized output, sized for the largest rung they supply."""
    if submit:
        _submit_cpu_job(f"data-belay-prepare-{label.lower()}-p{round(share * 100)}", cpu=4, memory="16GB")
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    rung_need = {rung: rung_tokens(rung) for rung in DATA_BELAY_RUNGS}
    intake = datakit_intake(label, sources, share, rung_need, tokenizer, _training_prefix())
    report = intake.report
    licenses = ", ".join(report.licenses) or "none on the cards"
    click.echo(f"license: {licenses} -> {report.license_verdict} (recorded: the sources are in the Datakit registry)")
    click.echo(f"size: about {report.estimated_tokens:,} tokens; top rung {intake.top_rung}")
    if intake.top_rung is None or DATA_BELAY_RUNGS.index(intake.top_rung) < DATA_BELAY_RUNGS.index(MIN_TOP_RUNG):
        raise click.ClickException(
            f"{label} reads more than {MAX_EPOCHS:g} epochs at {MIN_TOP_RUNG}; test it with the rest of its family"
        )
    need_tokens = next(need.new_tokens for need in report.needs if need.rung == intake.top_rung)
    cache_root = prefix_join(_training_prefix(), CACHE_PATH)
    # Registry sources are training data, and some are named for test generation (methods2test).
    plan = plan_cache(report, list(intake.files), need_tokens, V16384_TOKENIZER, cache_root)
    _build(report, plan, allow_test_in_train=True)


@cli.command()
@click.option("--rung", type=click.Choice(DATA_BELAY_RUNGS), required=True)
@click.option("--seed", type=int, required=True)
@click.option(
    "--mix-cache",
    default=None,
    help="The cache path that `prepare` printed; omit for the control. It names the runs and sets the share.",
)
@click.option("--num-steps", type=int, default=None, help="Smoke run of this many steps; the run id gets -smoke<N>.")
@click.option("--submit", is_flag=True, help=f"Submit to {CLUSTER}; without it, print the command.")
def train(rung: str, seed: int, mix_cache: str | None, num_steps: int | None, submit: bool) -> None:
    """Train one arm of the dense baseline: the control, or the base cache mixed with a prepared cache."""
    flags = ["--eval-policy-answer-bpb"]
    if mix_cache is None:
        run_id = control_run_id(rung, seed)
    else:
        configure_coreweave_s3()
        cache = PreparedCache.read(mix_cache)
        try:
            cache.check_rung(rung, rung_tokens(rung))
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        run_id = treatment_run_id(rung, cache.candidate, seed)
        flags += ["--mix-cache", cache.train_cache, "--mix-share", str(cache.share)]
    if num_steps is not None:
        run_id += f"-smoke{num_steps}"
        flags += ["--num-steps", str(num_steps)]
    command = train_command(rung, run_id, flags, seed)
    if not submit:
        click.echo(shlex.join(command))
        return
    if run_exists(run_id):
        raise click.ClickException(f"{run_id} is already in W&B, and report reads runs by id; a rerun needs a new id")
    click.echo(f"submitting {run_id}", err=True)
    os.execvp(command[0], command)


@cli.command()
@click.option("--rung", type=click.Choice(DATA_BELAY_RUNGS), required=True)
@click.option("--submit", is_flag=True, help=f"Submit to {CLUSTER} on one H100; without it, print the command.")
@click.argument("run_ids", nargs=-1, required=True)
def score(rung: str, submit: bool, run_ids: tuple[str, ...]) -> None:
    """Choice loss and accuracy of finished runs at ``rung``, which `report` reads for the knowledge group."""
    runs_root = prefix_join(_training_prefix(), RUNS_ROOT)
    module_args = [
        *("experiments.data_belay.accuracy", "--rung", rung),
        *("--runs-root", runs_root, "--choice-root", _choice_root(), *run_ids),
    ]
    cmd = _iris_job(f"data-belay-score-{run_ids[0]}", SCORE_RESOURCES, module_args)
    if not submit:
        click.echo(_shown(cmd))
        return
    click.echo(f"submitting: {_shown(cmd)}", err=True)
    os.execvp(cmd[0], cmd)


@cli.command()
@click.option("--rung", type=click.Choice(DATA_BELAY_RUNGS), required=True)
@click.option("--candidate", required=True, help="The candidate's run-name label, as `prepare` prints it.")
@click.option("--seeds", default="0,1,2", show_default=True, help="Treatment seeds.")
@click.option(
    "--against",
    default=None,
    help="Compare with this candidate's runs (same seeds) instead of the control: e.g. filtered against unfiltered.",
)
def report(rung: str, candidate: str, seeds: str, against: str | None) -> None:
    """Verdict and per-group change of a candidate against the shared control or another candidate, how much of each
    arm's data it read, and the ladder's next step after the verdicts at this rung and the rungs below it."""
    seed_list = [int(s) for s in seeds.split(",")]
    baseline_name = against or "control"
    configure_coreweave_s3()
    choice_root = _choice_root()
    limits = harm_limits(_control_values(choice_root, HARM_LIMIT_RUNG))
    result, against_dilution = _rung_result(choice_root, rung, candidate, seed_list, against, limits)
    harm_name = "dilution" if against_dilution else baseline_name
    click.echo(f"{candidate} against {baseline_name} at {rung}: {result.verdict}")
    arms = (candidate,) if against is None else (candidate, against)
    for name in arms:
        click.echo(_data_line(rung, name, seed_list))
    click.echo(
        f"| group | Δ against {baseline_name} | interval | Δ against {harm_name} | interval | harm limit | call |"
    )
    click.echo("|---|---:|---|---:|---|---:|---|")
    for d in result.groups:
        click.echo(
            f"| {d.group} | {d.gain.delta:+.2%} | [{d.gain.low:+.2%}, {d.gain.high:+.2%}] | {d.harm.delta:+.2%} | "
            f"[{d.harm.low:+.2%}, {d.harm.high:+.2%}] | {d.harm_limit:.2%} | {d.call} |"
        )
    results = []
    for lower in DATA_BELAY_RUNGS[: DATA_BELAY_RUNGS.index(rung)]:
        lower_baseline, lower_treatment = _arm_run_ids(lower, candidate, seed_list, against)
        compared = lower_treatment if against is None else [*lower_baseline, *lower_treatment]
        if not all(run_exists(run_id) and run_scored(choice_root, run_id) for run_id in compared):
            click.echo(f"ladder: no scored {lower} runs for {candidate}; score them, or apply the README's rule by hand")
            return
        results.append(_rung_result(choice_root, lower, candidate, seed_list, against, limits)[0])
    results.append(result)
    rungs = ", ".join(
        f"{r} {x.verdict}{_epochs_note(r, arms, seed_list)}" for r, x in zip(DATA_BELAY_RUNGS, results, strict=False)
    )
    click.echo(f"ladder: {rungs} -> {ladder_step(results, len(DATA_BELAY_RUNGS))}")


def _arm_reading(rung: str, candidate: str, seeds: list[int]) -> tuple[float, PreparedCache, CacheReading] | None:
    """The share, cache and reading of the candidate's runs at ``rung``; None for runs on the base mix alone."""
    mix = arm_mix([treatment_run_id(rung, candidate, s) for s in seeds])
    if mix is None:
        return None
    cache = PreparedCache.read(mix.cache_root)
    return mix.share, cache, cache.reading(mix.share, mix.train_tokens)


def _data_line(rung: str, candidate: str, seeds: list[int]) -> str:
    arm = _arm_reading(rung, candidate, seeds)
    if arm is None:
        return f"data: {candidate} trains on the base mix alone"
    share, cache, reading = arm
    return (
        f"data: {candidate} trained at a share of {share * 100:g}%: read {_tokens(reading.tokens)} tokens, "
        f"{reading.epochs:.2f} epochs of its {_tokens(cache.tokens)}-token cache, {reading.dataset_seen * 100:.2g}% of "
        f"the dataset's about {_tokens(cache.dataset_tokens)} tokens"
    )


def _epochs_note(rung: str, arms: tuple[str, ...], seeds: list[int]) -> str:
    """The epochs of each arm's cache at ``rung``, leaving out an arm on the base mix alone."""
    epochs = [f"{arm[2].epochs:.2f}" for name in arms if (arm := _arm_reading(rung, name, seeds)) is not None]
    return f" ({' and '.join(epochs)} epochs)" if epochs else ""


def _tokens(tokens: int) -> str:
    return f"{tokens / 1e9:,.2f}B" if tokens >= 1e9 else f"{tokens / 1e6:,.0f}M"


def _rung_result(
    choice_root: str, rung: str, candidate: str, seeds: list[int], against: str | None, limits: dict[str, float]
) -> tuple[RungResult, bool]:
    """The candidate's result at ``rung``, and whether its harm is against the dilution references."""
    baseline_ids, treatment_ids = _arm_run_ids(rung, candidate, seeds, against)
    share = arm_share(treatment_ids)
    # Against another candidate, both arms give up the same share of the base mix, so the harm is against it too.
    if against is not None and (against_share := arm_share(baseline_ids)) != share:
        raise click.ClickException(
            f"{candidate} trained at a share of {share:g} and {against} at {against_share:g}; "
            "compare candidates at one share"
        )
    if against is None:
        require_dilution_references(candidate, share, rung)
    dilution = dilution_arm(choice_root, rung, candidate, share) if against is None else None
    baseline, treatment = arm_metrics(choice_root, rung, baseline_ids), arm_metrics(choice_root, rung, treatment_ids)
    return judge(baseline, treatment, limits, dilution), dilution is not None


def require_dilution_references(candidate: str, share: float, rung: str) -> None:
    """Refuse to judge a candidate against the control at a share without dilution references, where the harm would
    include what giving up the share costs; the control gives up none."""
    if share == 0 or dilution_references(candidate, share):
        return
    prepare = "\n".join(
        f"  uv run data-belay prepare-datakit --label {label} --source {source} --share {share:g} --submit"
        for label, source in DILUTION_REFERENCE_SOURCES.items()
    )
    raise click.ClickException(
        f"{candidate} trained at a share of {share * 100:g}%, which has no dilution references, so its harm would "
        "include what giving up the share costs. Compare it with another candidate at that share (--against), or "
        f"build the references:\n{prepare}\nthen train seeds {', '.join(map(str, REFERENCE_SEEDS))} of each at "
        f"{rung}, score them, and add their labels to results.DILUTION_REFERENCES[{share:g}]."
    )


def _arm_run_ids(rung: str, candidate: str, seeds: list[int], against: str | None) -> tuple[list[str], list[str]]:
    """Run ids of the baseline arm (the shared control, or ``against``) and of the candidate at ``rung``."""
    if against is None:
        baseline = [control_run_id(rung, s) for s in CONTROL_SEEDS[rung]]
    else:
        baseline = [treatment_run_id(rung, against, s) for s in seeds]
    return baseline, [treatment_run_id(rung, candidate, s) for s in seeds]


@cli.command()
@click.option("--rung", type=click.Choice(DATA_BELAY_RUNGS), required=True)
def noise(rung: str) -> None:
    """Seed spread of the control's group values, and the harm limits it sets; `report` uses those of the d512 rung."""
    configure_coreweave_s3()
    control = _control_values(_choice_root(), rung)
    limits = harm_limits(control)
    click.echo("| group | mean | seed SD | SD / mean | harm limit |")
    click.echo("|---|---:|---:|---:|---:|")
    for group, values in control.items():
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        click.echo(f"| {group} | {mean:.4f} | {sd:.4f} | {sd / mean:.3%} | {limits[group]:.2%} |")


def _control_values(choice_root: str, rung: str) -> dict[str, list[float]]:
    """Each group's value on each of the shared control's seeds at ``rung``."""
    return arm_metrics(choice_root, rung, [control_run_id(rung, s) for s in CONTROL_SEEDS[rung]])


if __name__ == "__main__":
    cli()
