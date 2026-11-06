# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data Belay command line: `uv run data-belay <command>`."""

import logging
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

from experiments.belay.config import TRACKS, Track
from experiments.belay.submit import track_command
from experiments.data_belay.answer_bpb import ANSWER_BPB_PATH, build_answer_bpb
from experiments.data_belay.answer_bpb import MANIFEST_NAME as ANSWER_BPB_MANIFEST
from experiments.data_belay.intake import MAX_EPOCHS, DatasetRequest, IntakeReport, LicenseVerdict, run_intake
from experiments.data_belay.judge import HARM_LIMITS, judge
from experiments.data_belay.prepare import CACHE_PATH, PreparedCache, build_cache, plan_cache
from experiments.data_belay.results import CONTROL_SEEDS, arm_metrics, control_run_id, run_exists, treatment_run_id
from experiments.grug.fast_track import belay_tracks
from experiments.grug.fast_track.launch import SEQ_LEN, V16384_TOKENIZER, h100_ladder_rung

# The Belay `vanilla` Track at each rung. d512 screens a candidate; d768 and d1024 confirm a d512 result.
VANILLA_TRACKS = {"d512": Track.VANILLA_D512, "d768": Track.VANILLA_D768, "d1024": Track.VANILLA_D1024}
DATA_BELAY_RUNGS = tuple(VANILLA_TRACKS)
# The Track's cluster and Iris band. prepare's tokenize job runs there too, beside the caches it writes.
CLUSTER = TRACKS[Track.VANILLA_D512].cluster
BAND = TRACKS[Track.VANILLA_D512].band


def rung_tokens(rung: str) -> int:
    """Training tokens of the `vanilla` Track at ``rung``: its steps times its batch of 4,096-token sequences."""
    _, vanilla_steps, _ = belay_tracks.RUNGS[rung]
    return vanilla_steps * h100_ladder_rung(rung).baseline_batch * SEQ_LEN


def _submit_cpu_job(job_name: str, cpu: int, memory: str) -> None:
    """Re-run this command, without ``--submit``, as an Iris CPU job. Never returns."""
    args = [a for a in sys.argv[1:] if a != "--submit"]
    env = ["-e", "HF_TOKEN", os.environ["HF_TOKEN"]] if os.environ.get("HF_TOKEN") else []
    cmd = [
        *("uv", "run", "iris", "--cluster", "marin", "job", "run", "--no-wait", "--enable-extra-resources"),
        *("--target-cluster", CLUSTER, "--priority", BAND, "--cpu", str(cpu), "--memory", memory),
        *("--disk", "32GB", "--job-name", job_name, *env),
        *("--", "python", "-m", "experiments.data_belay", *args),
    ]
    secret = os.environ.get("HF_TOKEN")
    click.echo("submitting: " + " ".join(shlex.quote("$HF_TOKEN" if c == secret else c) for c in cmd), err=True)
    os.execvp(cmd[0], cmd)


def _dataset_options(command):
    for option in reversed(
        [
            click.option("--dataset", "repo", required=True, help="Hugging Face dataset id."),
            click.option("--config", "config_name", default=None, help="Dataset config (subset)."),
            click.option("--split", default="train", show_default=True),
            click.option("--text-field", default="text", show_default=True),
            click.option("--share", type=float, default=0.2, show_default=True, help="Share of the rung's tokens."),
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


def _intake(repo, config_name, split, text_field, share) -> IntakeReport:
    configure_coreweave_s3()
    tokenizer = load_tokenizer(V16384_TOKENIZER)
    needs = {rung: rung_tokens(rung) for rung in DATA_BELAY_RUNGS}
    return run_intake(DatasetRequest(repo, config_name, split, text_field), share, needs, tokenizer)


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
    as_json: bool,
) -> None:
    """Pin the head commit, check the license and estimate the dataset's size at each rung."""
    report = _intake(repo, config_name, split, text_field, share)
    if as_json:
        click.echo(report.to_json())
    else:
        click.echo(f"dataset: {report.repo}@{report.revision} config={report.config} split={report.split}")
        click.echo(f"license: {', '.join(report.licenses) or 'none on the card'} -> {report.license_verdict}")
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
    submit: bool,
) -> None:
    """Tokenize a seeded file sample that covers the largest rung's new-data tokens into a cache."""
    if submit:
        name = "-".join(part for part in (repo.replace("/", "-"), config_name, f"p{round(share * 100)}") if part)
        _submit_cpu_job(f"data-belay-prepare-{name.lower()}", cpu=4, memory="16GB")
    logging.basicConfig(level=logging.INFO)
    report = _intake(repo, config_name, split, text_field, share)
    if not report.passed:
        raise click.ClickException(f"intake did not pass: license {report.license_verdict}, needs {report.needs}")
    need_tokens = max(need.new_tokens for need in report.needs)
    plan = plan_cache(report, need_tokens, V16384_TOKENIZER, prefix_join(_training_prefix(), CACHE_PATH))
    click.echo(f"plan: {len(plan.files)} files, about {plan.estimated_tokens:,} tokens -> {plan.cache_path}")
    manifest = build_cache(report, plan, V16384_TOKENIZER)
    cache = PreparedCache.read(plan.cache_path)
    click.echo(f"cache: {cache.root} ({manifest.stats.total_tokens:,} tokens), runs named {cache.candidate}")


@cli.command("build-answer-bpb")
def build_answer_bpb_command() -> None:
    """Write the Eval Policy gold-answer BPB caches and their manifest under the training cluster's prefix."""
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    root = prefix_join(_training_prefix(), ANSWER_BPB_PATH)
    for cache in build_answer_bpb(load_tokenizer(V16384_TOKENIZER), root):
        click.echo(
            f"{cache.name}: {cache.examples} examples, {cache.dropped} dropped, "
            f"{cache.answer_tokens:,} answer tokens"
        )


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
    """Train one arm on the Belay `vanilla` Track: the control, or the base cache mixed with a prepared cache."""
    configure_coreweave_s3()
    flags = ["--eval-manifest", prefix_join(_training_prefix(), f"{ANSWER_BPB_PATH}/{ANSWER_BPB_MANIFEST}")]
    if mix_cache is None:
        run_id = control_run_id(rung, seed)
    else:
        cache = PreparedCache.read(mix_cache)
        run_id = treatment_run_id(rung, cache.candidate, seed)
        flags += ["--mix-cache", cache.train_cache, "--mix-share", str(cache.share)]
    if num_steps is not None:
        run_id += f"-smoke{num_steps}"
        flags += ["--num-steps", str(num_steps)]
    command = track_command(VANILLA_TRACKS[rung], run_id, tuple(flags), seed=seed)
    if not submit:
        click.echo(shlex.join(command))
        return
    if run_exists(run_id):
        raise click.ClickException(f"{run_id} is already in W&B, and report reads runs by id; a rerun needs a new id")
    click.echo(f"submitting {run_id}", err=True)
    os.execvp(command[0], command)


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
    """Verdict and per-group BPB change of a candidate against the shared control or another candidate."""
    seed_list = [int(s) for s in seeds.split(",")]
    if against is None:
        baseline_name = "control"
        baseline = arm_metrics([control_run_id(rung, s) for s in CONTROL_SEEDS[rung]])
    else:
        baseline_name = against
        baseline = arm_metrics([treatment_run_id(rung, against, s) for s in seed_list])
    treatment = arm_metrics([treatment_run_id(rung, candidate, s) for s in seed_list])
    verdict, deltas = judge(baseline, treatment, HARM_LIMITS)
    click.echo(f"{candidate} against {baseline_name} at {rung}: {verdict}")
    click.echo(f"| group | {baseline_name} BPB | {candidate} BPB | Δ | 95% interval | harm limit | call |")
    click.echo("|---|---:|---:|---:|---|---:|---|")
    for d in deltas:
        b, t = baseline[d.group], treatment[d.group]
        click.echo(
            f"| {d.group} | {sum(b) / len(b):.4f} | {sum(t) / len(t):.4f} | {d.delta:+.2%} | "
            f"[{d.low:+.2%}, {d.high:+.2%}] | {d.harm_limit:.1%} | {d.call} |"
        )


@cli.command()
@click.option("--rung", type=click.Choice(DATA_BELAY_RUNGS), required=True)
def noise(rung: str) -> None:
    """Seed spread of the control's group BPB: the input to the harm limits."""
    control = arm_metrics([control_run_id(rung, s) for s in CONTROL_SEEDS[rung]])
    click.echo("| group | mean BPB | seed SD | SD / mean |")
    click.echo("|---|---:|---:|---:|")
    for group, values in control.items():
        mean = statistics.fmean(values)
        sd = statistics.stdev(values)
        click.echo(f"| {group} | {mean:.4f} | {sd:.4f} | {sd / mean:.3%} |")


if __name__ == "__main__":
    cli()
