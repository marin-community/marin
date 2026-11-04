# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mine the thoughtcrime rows: double-think prefixes from Marin pi evaluation trajectories.

A double-think reply closes its reasoning span a second time or opens a second span (#10400): the
``stray_end`` and ``second_span`` reasoning-span violations. Each row is the conversation before such
a reply, with the tools pi advertised. Rows are independent samples:

- a trajectory contributes only its first malformed reply, and only when that reply is a double-think,
  so no row's history holds an earlier markup failure;
- a benchmark task contributes one row, the lowest digest among the trajectories that qualify.

Source runs are the bixbench-pi, bfclparity-pi and tau3-pi runs of Marin models (Grug, Snowball and
quick-loop checkpoints) under the evals root that started before ``RUNS_BEFORE``. The rows are the
pinned source input of the ``Marin:thoughtcrime`` task-curation dataset.

    uv run python -m experiments.post_training.thoughtcrime.mine_prefixes --version 2026.11.04
    uv run python -m experiments.post_training.thoughtcrime.mine_prefixes --version 2026.11.04 --run

Prefixes stop at pi's first context compaction and at the prompt-length cap, which keeps them inside
the quick RL loop's input window. ``report.json`` records the counts per benchmark, model and
violation, and how closely the Snowball tokenizer's rendering of each prefix matches the prompt
length pi observed.
"""

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import pyarrow as pa
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.convert.pi_trajectory import (
    PiReply,
    PiTranscript,
    PrefixRow,
    UnsupportedTranscript,
    parse_pi_transcript,
    prefix_row,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from verifyit.reasoning_markup import ReasoningViolation
from zephyr.writers import write_parquet_file

from experiments.post_training.inventory.train_quick_loop import MODEL_ID, MODEL_REVISION
from experiments.post_training.task_curation.datasets.thoughtcrime import ROWS_FILE

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "rl-data/inputs/thoughtcrime"
REPORT_FILENAME = "report.json"
ROW_SCHEMA = pa.schema(
    [
        pa.field("digest", pa.string(), nullable=False),
        pa.field("messages", pa.string(), nullable=False),
        pa.field("tools", pa.string(), nullable=False),
        pa.field("recorded_violation", pa.string(), nullable=False),
        pa.field("prompt_tokens", pa.int64(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("task", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("run", pa.string(), nullable=False),
        pa.field("trial", pa.string(), nullable=False),
        pa.field("position", pa.int64(), nullable=False),
        pa.field("reply", pa.string(), nullable=False),
    ]
)
EVALS_ROOT = "s3://marin-us-east-02a/marin/evals"
TRANSCRIPT_PATH = "agent/pi.txt"
TRIAL_CONFIG = "config.json"
DOUBLE_THINK = frozenset({ReasoningViolation.STRAY_END, ReasoningViolation.SECOND_SPAN})
# Evaluation run directories are named <YYYYMMDD-HHMMSS>-<model>-<benchmark>-<suffix>.
PI_BENCHMARK_RUN = re.compile(r"-(bixbench|bfclparity|tau3)-pi-")
MARIN_MODEL_RUN = re.compile(r"grug|snowball|marin|^\d{8}-\d{6}-dml-", re.IGNORECASE)
# Runs started before this timestamp; a fixed cutoff keeps the source set reproducible.
RUNS_BEFORE = "20261104-000000"
# Harbor names a trial directory <task>__<suffix>.
TRIAL_TASK_SEPARATOR = "__"
# The quick loop's input window is 16,384 tokens; leave room for the tools and the final answer.
MAX_PROMPT_TOKENS = 12_288
READ_THREADS = 64


@dataclass(frozen=True)
class MiningConfig:
    output_path: str
    evals_root: str = EVALS_ROOT
    runs_before: str = RUNS_BEFORE
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    check_prompt_tokens: bool = True
    """Compare the Snowball tokenizer's rendering of each prefix with the prompt length pi observed."""


@dataclass(frozen=True)
class Trajectory:
    """One pi trial's transcript and where it came from."""

    run: str
    trial: str
    benchmark: str
    task: str
    model: str
    transcript: PiTranscript


@dataclass
class MiningReport:
    runs: int = 0
    runs_without_trajectories: int = 0
    trajectories: int = 0
    compacted_trajectories: int = 0
    truncated_trajectories: int = 0
    first_violations: dict[str, int] = field(default_factory=dict)
    """Trajectories by the kind of their first malformed reply; ``none`` when every reply was well formed."""
    first_double_think_over_budget: int = 0
    double_think_trajectories: int = 0
    rows: int = 0
    rows_by_benchmark: dict[str, int] = field(default_factory=dict)
    rows_by_model: dict[str, int] = field(default_factory=dict)
    rows_by_violation: dict[str, int] = field(default_factory=dict)
    prompt_tokens_checked: int = 0
    prompt_tokens_matching_tokenizer: int = 0
    prompt_tokens_within_one_percent: int = 0
    largest_prompt_token_gap: int = 0


def source_runs(run_names: Iterable[str], runs_before: str) -> tuple[str, ...]:
    """The Marin-model pi runs among evaluation run directory names, started before the cutoff."""
    return tuple(
        sorted(
            name
            for name in run_names
            if name < runs_before and PI_BENCHMARK_RUN.search(name) and MARIN_MODEL_RUN.search(name)
        )
    )


def first_malformed(transcript: PiTranscript) -> PiReply | None:
    return next((reply for reply in transcript.replies if reply.violation is not None), None)


def fits_prompt_budget(reply: PiReply, max_prompt_tokens: int) -> bool:
    return reply.position > 0 and reply.prompt_tokens is not None and reply.prompt_tokens <= max_prompt_tokens


def _trial_source(fs, trial_dir: str) -> tuple[str, str]:
    with fs.open(prefix_join(trial_dir, TRIAL_CONFIG), "rt") as handle:
        config = json.load(handle)
    agent = config["agent"]
    # Campaign runs before the alias field name the model by its served path.
    model = agent.get("model_alias") or agent["model_name"]
    return config["task"]["source"], model.rsplit("/", 1)[-1]


def _read_trajectory(evals_root: str, run: str, trial_dir: str) -> Trajectory | None:
    fs, _ = filesystem_for(evals_root)
    transcript_paths = sorted(fs.glob(prefix_join(trial_dir, f"attempts/*/{TRANSCRIPT_PATH}")))
    trial = trial_dir.rsplit("/", 1)[-1]
    if not transcript_paths:
        return None
    if not fs.exists(prefix_join(trial_dir, TRIAL_CONFIG)):
        # An interrupted upload can leave a transcript without the config that names its benchmark and model.
        logger.info("Skipping %s/%s: no trial config", run, trial)
        return None
    benchmark, model = _trial_source(fs, trial_dir)
    with fs.open(transcript_paths[-1], "rt") as handle:
        try:
            transcript = parse_pi_transcript(handle)
        except UnsupportedTranscript as error:
            logger.info("Skipping %s/%s: %s", run, trial, error)
            return None
    return Trajectory(
        run=run,
        trial=trial,
        benchmark=benchmark,
        task=trial.split(TRIAL_TASK_SEPARATOR, 1)[0],
        model=model,
        transcript=transcript,
    )


def _trial_dirs(evals_root: str, run: str) -> list[str]:
    """Trial directories of one run under its Harbor jobs; empty when the run wrote none."""
    fs, jobs_root = filesystem_for(prefix_join(evals_root, f"{run}/results/harbor_jobs"))
    if not fs.exists(jobs_root):
        return []
    return sorted(
        entry["name"]
        for job in fs.ls(jobs_root, detail=False)
        for entry in fs.ls(job, detail=True)
        if entry["type"] == "directory"
    )


def read_trajectories(
    evals_root: str, runs: Iterable[str], report: MiningReport, *, threads: int = READ_THREADS
) -> Iterator[Trajectory]:
    """Yield the final attempt's transcript of every trial in the given evaluation runs."""
    with ThreadPoolExecutor(threads) as pool:
        trials = []
        for run, run_trials in zip(runs, pool.map(partial(_trial_dirs, evals_root), runs), strict=True):
            report.runs_without_trajectories += not run_trials
            trials.extend((run, trial) for trial in run_trials)
        logger.info("Reading %d trials", len(trials))
        read = partial(_read_trajectory, evals_root)
        for trajectory in pool.map(lambda item: read(*item), trials):
            if trajectory is not None:
                yield trajectory


def _rendered_prompt_tokens(tokenizer: PreTrainedTokenizerBase, row: PrefixRow) -> int:
    tools = [{"type": "function", "function": tool} for tool in json.loads(row.tools)]
    rendered = tokenizer.apply_chat_template(
        json.loads(row.messages), tools=tools, add_generation_prompt=True, tokenize=False
    )
    # The template writes the begin-of-text marker itself.
    return len(tokenizer(rendered, add_special_tokens=False).input_ids)


def _audit_prompt_tokens(report: MiningReport, tokenizer: PreTrainedTokenizerBase, row: PrefixRow) -> None:
    gap = abs(_rendered_prompt_tokens(tokenizer, row) - row.prompt_tokens)
    report.prompt_tokens_checked += 1
    report.prompt_tokens_matching_tokenizer += gap == 0
    report.prompt_tokens_within_one_percent += gap * 100 <= row.prompt_tokens
    report.largest_prompt_token_gap = max(report.largest_prompt_token_gap, gap)


def _record_transcript(report: MiningReport, transcript: PiTranscript, first: PiReply | None) -> None:
    report.trajectories += 1
    report.compacted_trajectories += transcript.compacted
    report.truncated_trajectories += transcript.truncated
    name = "none" if first is None or first.violation is None else first.violation.value
    report.first_violations[name] = report.first_violations.get(name, 0) + 1


def mine_rows(
    trajectories: Iterable[Trajectory],
    config: MiningConfig,
    tokenizer: PreTrainedTokenizerBase | None,
    report: MiningReport,
) -> list[PrefixRow]:
    """Keep one double-think prefix per benchmark task and record the mining statistics."""
    rows: dict[tuple[str, str], PrefixRow] = {}
    for trajectory in trajectories:
        reply = first_malformed(trajectory.transcript)
        _record_transcript(report, trajectory.transcript, reply)
        if reply is None or reply.violation not in DOUBLE_THINK:
            continue
        if not fits_prompt_budget(reply, config.max_prompt_tokens):
            report.first_double_think_over_budget += 1
            continue
        report.double_think_trajectories += 1
        row = prefix_row(
            trajectory.transcript,
            reply,
            benchmark=trajectory.benchmark,
            task=trajectory.task,
            model=trajectory.model,
            run=trajectory.run,
            trial=trajectory.trial,
        )
        key = (row.benchmark, row.task)
        if key not in rows or row.digest < rows[key].digest:
            rows[key] = row
    kept = sorted(rows.values(), key=lambda row: row.digest)
    report.rows = len(kept)
    report.rows_by_benchmark = dict(Counter(row.benchmark for row in kept))
    report.rows_by_model = dict(Counter(row.model for row in kept))
    report.rows_by_violation = dict(Counter(row.recorded_violation for row in kept))
    if tokenizer is not None:
        for row in kept:
            _audit_prompt_tokens(report, tokenizer, row)
    return kept


def mine_prefixes(config: MiningConfig) -> None:
    """Mine the source runs and publish the double-think rows with their report."""
    tokenizer = None
    if config.check_prompt_tokens:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    fs, evals_root = filesystem_for(config.evals_root)
    runs = source_runs((path.rsplit("/", 1)[-1] for path in fs.ls(evals_root, detail=False)), config.runs_before)
    report = MiningReport(runs=len(runs))
    rows = mine_rows(read_trajectories(config.evals_root, runs, report), config, tokenizer, report)
    if not rows:
        raise ValueError("No double-think prefixes were selected from the source runs")
    destination = StoragePath(config.output_path)
    destination.mkdirs()
    with TemporaryDirectory(prefix="thoughtcrime-") as temporary:
        local = Path(temporary) / ROWS_FILE
        write_parquet_file((asdict(row) for row in rows), str(local), schema=ROW_SCHEMA)
        (destination / ROWS_FILE).upload_from(str(local))
    summary = {"config": asdict(config), "source_runs": list(runs), **asdict(report)}
    (destination / REPORT_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Published %d double-think rows from %d runs to %s", report.rows, len(runs), destination)


def mining_step(version: str, *, check_prompt_tokens: bool) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> MiningConfig:
        return MiningConfig(output_path=ctx.output_path, check_prompt_tokens=check_prompt_tokens)

    return ArtifactStep(
        name=ARTIFACT_NAME,
        version=version,
        artifact_type=Artifact,
        run=mine_prefixes,
        build_config=build_config,
    )


@click.command(help=__doc__)
@click.option(
    "--skip-prompt-check",
    is_flag=True,
    help="Do not compare the Snowball tokenizer's prefix rendering with pi's observed prompt lengths.",
)
@build_options
def main(skip_prompt_check: bool) -> ArtifactStep[Artifact]:
    return mining_step(resolve_version(ARTIFACT_NAME, None), check_prompt_tokens=not skip_prompt_check)


if __name__ == "__main__":
    main()
