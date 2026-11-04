# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mine reasoning-span prefix rows from Marin pi evaluation trajectories.

Each row is a conversation prefix from a bixbench-pi or bfclparity-pi trajectory, cut before an
assistant reply, with the tools pi advertised. ``hazard.parquet`` holds prefixes whose recorded reply
had a malformed reasoning span (#10400's double-think and its relatives); ``control.parquet`` holds
the nearest well-formed replies of the same trajectories. The rows are the pinned source input of the
``Marin:reasoning-span-*`` task-curation datasets, which convert each row into a one-turn task.

    uv run python -m experiments.post_training.reasoning_spans.mine_pi_prefixes --version 2026.11.03
    uv run python -m experiments.post_training.reasoning_spans.mine_pi_prefixes --version 2026.11.03 --run

Prefixes stop at pi's first context compaction and at the prompt-length cap, which keeps them inside
the quick RL loop's input window. ``report.json`` records the counts per run and violation, and how
closely the Snowball tokenizer's rendering of each prefix matches the prompt length pi observed.
"""

import json
import logging
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
    CONTROL_SPLIT,
    HAZARD_SPLIT,
    PiReply,
    PiTranscript,
    PrefixRow,
    UnsupportedTranscript,
    parse_pi_transcript,
    prefix_row,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from zephyr.writers import write_parquet_file

from experiments.post_training.inventory.train_quick_loop import MODEL_ID, MODEL_REVISION

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "rl-data/inputs/reasoning-span-prefixes"
HAZARD_FILENAME = "hazard.parquet"
CONTROL_FILENAME = "control.parquet"
REPORT_FILENAME = "report.json"
ROW_SCHEMA = pa.schema(
    [
        pa.field("digest", pa.string(), nullable=False),
        pa.field("messages", pa.string(), nullable=False),
        pa.field("tools", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("violation", pa.string()),
        pa.field("prompt_tokens", pa.int64(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("run", pa.string(), nullable=False),
        pa.field("trial", pa.string(), nullable=False),
        pa.field("position", pa.int64(), nullable=False),
        pa.field("reply", pa.string(), nullable=False),
    ]
)
EVALS_ROOT = "s3://marin-us-east-02a/marin/evals"
TRANSCRIPT_PATH = "agent/pi.txt"
# The quick loop's input window is 16,384 tokens; leave room for the tools and the final answer.
MAX_PROMPT_TOKENS = 12_288
MAX_HAZARDS_PER_TRAJECTORY = 4
READ_THREADS = 16
# Evaluation runs under EVALS_ROOT whose pi transcripts hold replies with malformed reasoning spans:
# Grug and Snowball checkpoints from the September-October 2026 evaluation campaign (issue #10400
# measured their rates), the Datakit SFT checkpoints of 2026.09.21 and 2026.10.05 they trained from,
# and 2026-11-03 checkpoints from 25-step quick-loop runs and 8-step RLVR1 runs on those SFT models.
DEFAULT_RUNS = (
    "20261003-114254-open-athena-Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02-bixbench-pi-ec3e",
    "20261004-053226-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom-bixbench-pi-4208",
    "20261011-111055-open-athena-Grug-67B-A2B-Merge-Campaign2-DARE-TIES-02-2026.10.10-bixbench-pi-cd7f",
    "20261017-022041-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-bfclparity-pi-db16e60f0fe7406bac9c3589f6625338",
    "20261017-022041-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-bixbench-pi-9e72e12402284053a8e6b07c89a7232e",
    "20261019-104831-open-athena-Snowball-Smartie-RL-2026.10.15.32-step14-bixbench-pi-72d5",
    "20261017-021715-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.10.05-bfclparity-pi-e7fbc623fdc94f6ab1061ae854b18b2f",
    "20261017-021715-open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.10.05-bixbench-pi-749f38c421c545cd99715682c0115b60",
    "20261103-204830-dml-quickloop-d25-step25-59d7eeb6f2b64b6594bd210af183cd63-bixbench-pi-6cf8963dc14a57a7becd34ec9dca9151",
    "20261103-204830-dml-quickloop-d25-step25-59d7eeb6f2b64b6594bd210af183cd63-bfclparity-pi-bfedb1b86d995a00aedb59ed77d8181e",
    "20261103-204913-dml-quickloop-rollout-step25-7a3dc14147a54fd0b823114ca7a1f729-bixbench-pi-3bc665526e855ae7a5b6314206df2d32",
    "20261103-204913-dml-quickloop-rollout-step25-7a3dc14147a54fd0b823114ca7a1f729-bfclparity-pi-b7160228f48158a18d2ab145eaf7f524",
    "20261103-204934-dml-quickloop-marker-step25-d081d0b1a3dc4be38efd62f3dd82e3b2-bixbench-pi-8aec5d07c2fb5e86b3c2ab3ffc0bd435",
    "20261103-204934-dml-quickloop-marker-step25-d081d0b1a3dc4be38efd62f3dd82e3b2-bfclparity-pi-6eebe1214ee9563ba7e8170c036ccce2",
    "20261103-204852-dml-armA-baseline-step8-a0e116d53af543e28cd591d694110353-bixbench-pi-87bde601febc5c648fd26951570c828c",
    "20261103-204955-dml-armA-10520-hardzero-step8-0dc1b12b2b1641cd8fa96f1c1de87497-bixbench-pi-9783eaebed5f5907b5a86e9338d50921",
)


@dataclass(frozen=True)
class MiningConfig:
    output_path: str
    runs: tuple[str, ...]
    evals_root: str = EVALS_ROOT
    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    max_hazards_per_trajectory: int = MAX_HAZARDS_PER_TRAJECTORY
    check_prompt_tokens: bool = True
    """Compare the Snowball tokenizer's rendering of each prefix with the prompt length pi observed."""


@dataclass(frozen=True)
class Trajectory:
    """One pi trial's transcript and where it came from."""

    run: str
    trial: str
    benchmark: str
    model: str
    transcript: PiTranscript


@dataclass(frozen=True)
class Selection:
    """The replies chosen from one trajectory: hazards and their matched controls."""

    hazards: tuple[PiReply, ...]
    controls: tuple[PiReply, ...]


@dataclass
class MiningReport:
    runs: int = 0
    trajectories: int = 0
    compacted_trajectories: int = 0
    truncated_trajectories: int = 0
    replies: int = 0
    replies_over_budget: int = 0
    hazard_replies_by_violation: dict[str, int] = field(default_factory=dict)
    hazard_rows: int = 0
    control_rows: int = 0
    duplicate_rows: int = 0
    controls_promoted_to_hazard: int = 0
    """Prompts first seen before a well-formed reply that another run answered with a malformed span."""
    hazard_rows_by_run: dict[str, int] = field(default_factory=dict)
    prompt_tokens_checked: int = 0
    prompt_tokens_matching_tokenizer: int = 0
    prompt_tokens_within_one_percent: int = 0
    largest_prompt_token_gap: int = 0


def select_replies(transcript: PiTranscript, *, max_prompt_tokens: int, max_hazards: int) -> Selection:
    """Keep hazards within the prompt budget and pair each with its nearest unused well-formed reply."""
    eligible = [reply for reply in transcript.replies if reply.position > 0 and reply.prompt_tokens is not None]
    eligible = [reply for reply in eligible if reply.prompt_tokens <= max_prompt_tokens]
    hazards = [reply for reply in eligible if reply.violation is not None][:max_hazards]
    available = [reply for reply in eligible if reply.violation is None]
    controls = []
    for hazard in hazards:
        if not available:
            break
        nearest = min(available, key=lambda reply: (abs(reply.position - hazard.position), reply.position))
        available.remove(nearest)
        controls.append(nearest)
    return Selection(tuple(hazards), tuple(controls))


def _trial_source(fs, trial_dir: str) -> tuple[str, str]:
    with fs.open(f"{trial_dir}/config.json", "rt") as handle:
        config = json.load(handle)
    agent = config["agent"]
    # Campaign runs before the alias field name the model by its served path.
    model = agent.get("model_alias") or agent["model_name"]
    return config["task"]["source"], model.rsplit("/", 1)[-1]


def _read_trajectory(evals_root: str, run: str, trial_dir: str) -> Trajectory | None:
    fs, _ = filesystem_for(evals_root)
    attempts = sorted(fs.ls(f"{trial_dir}/attempts", detail=False))
    transcript_paths = [f"{attempt}/{TRANSCRIPT_PATH}" for attempt in attempts]
    transcript_paths = [path for path in transcript_paths if fs.exists(path)]
    if not transcript_paths:
        return None
    benchmark, model = _trial_source(fs, trial_dir)
    with fs.open(transcript_paths[-1], "rt") as handle:
        try:
            transcript = parse_pi_transcript(handle)
        except UnsupportedTranscript as error:
            logger.info("Skipping %s/%s: %s", run, trial_dir.rsplit("/", 1)[-1], error)
            return None
    return Trajectory(run, trial_dir.rsplit("/", 1)[-1], benchmark, model, transcript)


def _trial_dirs(evals_root: str, run: str) -> list[str]:
    """Trial directories of one run, as protocol-stripped listing entries under the run's Harbor jobs."""
    fs, jobs_root = filesystem_for(prefix_join(evals_root, f"{run}/results/harbor_jobs"))
    jobs = fs.ls(jobs_root, detail=False)
    return sorted(trial for job in jobs for trial in fs.ls(job, detail=False) if fs.isdir(trial))


def read_trajectories(evals_root: str, runs: Iterable[str], *, threads: int = READ_THREADS) -> Iterator[Trajectory]:
    """Yield the final attempt's transcript of every trial in the given evaluation runs."""
    with ThreadPoolExecutor(threads) as pool:
        for run in runs:
            trials = _trial_dirs(evals_root, run)
            for trajectory in pool.map(partial(_read_trajectory, evals_root, run), trials):
                if trajectory is not None:
                    yield trajectory


def _rendered_prompt_tokens(tokenizer: PreTrainedTokenizerBase, row: PrefixRow) -> int:
    tools = [{"type": "function", "function": tool} for tool in json.loads(row.tools)]
    rendered = tokenizer.apply_chat_template(
        json.loads(row.messages), tools=tools, add_generation_prompt=True, tokenize=False
    )
    # The template writes the begin-of-text marker itself.
    return len(tokenizer(rendered, add_special_tokens=False).input_ids)


def _record_transcript(report: MiningReport, transcript: PiTranscript, max_prompt_tokens: int) -> None:
    report.trajectories += 1
    report.compacted_trajectories += transcript.compacted
    report.truncated_trajectories += transcript.truncated
    report.replies += len(transcript.replies)
    report.replies_over_budget += sum(
        reply.prompt_tokens is None or reply.prompt_tokens > max_prompt_tokens for reply in transcript.replies
    )
    for reply in transcript.replies:
        if reply.violation is not None:
            name = reply.violation.value
            report.hazard_replies_by_violation[name] = report.hazard_replies_by_violation.get(name, 0) + 1


def _admit_row(rows: dict[str, PrefixRow], row: PrefixRow, report: MiningReport) -> bool:
    """Keep the first row per prompt, except that a hazard replaces a control for the same prompt."""
    existing = rows.get(row.digest)
    if existing is not None and (row.split == CONTROL_SPLIT or existing.split == HAZARD_SPLIT):
        report.duplicate_rows += 1
        return False
    if existing is not None:
        # A prompt one checkpoint answered well and another answered with a malformed span is a hazard.
        report.controls_promoted_to_hazard += 1
        report.control_rows -= 1
    rows[row.digest] = row
    if row.split == HAZARD_SPLIT:
        report.hazard_rows += 1
        report.hazard_rows_by_run[row.run] = report.hazard_rows_by_run.get(row.run, 0) + 1
    else:
        report.control_rows += 1
    return True


def _audit_prompt_tokens(report: MiningReport, tokenizer: PreTrainedTokenizerBase, row: PrefixRow) -> None:
    gap = abs(_rendered_prompt_tokens(tokenizer, row) - row.prompt_tokens)
    report.prompt_tokens_checked += 1
    report.prompt_tokens_matching_tokenizer += gap == 0
    report.prompt_tokens_within_one_percent += gap * 100 <= row.prompt_tokens
    report.largest_prompt_token_gap = max(report.largest_prompt_token_gap, gap)


def mine_rows(
    trajectories: Iterable[Trajectory],
    config: MiningConfig,
    tokenizer: PreTrainedTokenizerBase | None,
    report: MiningReport,
) -> list[PrefixRow]:
    """Select hazard and control prefixes, drop duplicate prompts, and record the mining statistics."""
    rows: dict[str, PrefixRow] = {}
    for trajectory in trajectories:
        transcript = trajectory.transcript
        _record_transcript(report, transcript, config.max_prompt_tokens)
        selection = select_replies(
            transcript, max_prompt_tokens=config.max_prompt_tokens, max_hazards=config.max_hazards_per_trajectory
        )
        for split, replies in ((HAZARD_SPLIT, selection.hazards), (CONTROL_SPLIT, selection.controls)):
            for reply in replies:
                row = prefix_row(
                    transcript,
                    reply,
                    split=split,
                    benchmark=trajectory.benchmark,
                    model=trajectory.model,
                    run=trajectory.run,
                    trial=trajectory.trial,
                )
                if _admit_row(rows, row, report) and tokenizer is not None:
                    _audit_prompt_tokens(report, tokenizer, row)
    return list(rows.values())


def mine_prefixes(config: MiningConfig) -> None:
    """Mine the configured runs and publish the hazard and control rows with their report."""
    tokenizer = None
    if config.check_prompt_tokens:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    report = MiningReport(runs=len(config.runs))
    rows = mine_rows(read_trajectories(config.evals_root, config.runs), config, tokenizer, report)
    if not rows:
        raise ValueError("No prefix rows were selected from the configured runs")
    destination = StoragePath(config.output_path)
    destination.mkdirs()
    with TemporaryDirectory(prefix="reasoning-span-prefixes-") as temporary:
        for split, filename in ((HAZARD_SPLIT, HAZARD_FILENAME), (CONTROL_SPLIT, CONTROL_FILENAME)):
            local = Path(temporary) / filename
            write_parquet_file((asdict(row) for row in rows if row.split == split), str(local), schema=ROW_SCHEMA)
            (destination / filename).upload_from(str(local))
    summary = {"config": asdict(config), **asdict(report)}
    (destination / REPORT_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info("Published %d hazard and %d control rows to %s", report.hazard_rows, report.control_rows, destination)


def mining_step(version: str, runs: tuple[str, ...], *, check_prompt_tokens: bool) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> MiningConfig:
        return MiningConfig(output_path=ctx.output_path, runs=runs, check_prompt_tokens=check_prompt_tokens)

    return ArtifactStep(
        name=ARTIFACT_NAME,
        version=version,
        artifact_type=Artifact,
        run=mine_prefixes,
        build_config=build_config,
    )


@click.command(help=__doc__)
@click.option(
    "--run-id",
    "runs",
    multiple=True,
    help="Evaluation run directory under the evals root. Repeatable; defaults to the runs listed in DEFAULT_RUNS.",
)
@click.option(
    "--skip-prompt-check",
    is_flag=True,
    help="Do not compare the Snowball tokenizer's prefix rendering with pi's observed prompt lengths.",
)
@build_options
def main(runs: tuple[str, ...], skip_prompt_check: bool) -> ArtifactStep[Artifact]:
    return mining_step(
        resolve_version(ARTIFACT_NAME, None), runs or DEFAULT_RUNS, check_prompt_tokens=not skip_prompt_check
    )


if __name__ == "__main__":
    main()
