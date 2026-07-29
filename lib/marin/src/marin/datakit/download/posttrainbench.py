# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""aisa-group/PostTrainBench-Trajectories download and transform.

PostTrainBench (https://posttrainbench.com/) hands a CLI agent a pre-trained base
model, an evaluation script, and ten hours on one H100, then scores how far the
agent moves the model on a target benchmark. Each task directory holds the raw
harness log the agent produced (``solve_out.txt``) next to the benchmark's two
anti-cheating judgements.

Only runs driven by an open-weights agent model are ingested, so a model trained
on this corpus imitates agents whose weights are public. The benchmark's Claude,
GPT/Codex, Gemini, and Qwen3-Max runs are left out. Traces the judges flagged for
benchmark contamination or for calling a disallowed model are dropped: those
trajectories succeed by cheating, which is the opposite of what we want imitated.

Two harness log formats appear across the open-weights runs, both line-delimited
JSON over a short plain-text GPU-check preamble: the Claude Code stream-json
format (the ``glmx`` runs) and opencode's event stream. Both are parsed into
role-tagged turns. Session init, task-tracker updates, and per-step token
accounting carry no training signal and are skipped.

Every document opens with the grader's verdict on the trajectory (see
:func:`outcome_prefix`), so training conditions on how well the attempt went and a
good result can be asked for at generation time.
"""

import json
import posixpath
import re
from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import NamedTuple

from fray.types import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import render_tool_message, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "aisa-group/PostTrainBench-Trajectories"
HF_REVISION = "46b3fec"


class TraceFormat(StrEnum):
    """Harness log format a run directory's ``solve_out.txt`` is written in."""

    CLAUDE_STREAM_JSON = "claude_stream_json"
    OPENCODE = "opencode"


OPEN_WEIGHT_RUNS: dict[str, TraceFormat] = {
    "glmx_glm-5.2-preview_1m__10h_run1": TraceFormat.CLAUDE_STREAM_JSON,
    "glmx_glm-5.2-preview_1m__10h_run2": TraceFormat.CLAUDE_STREAM_JSON,
    "glmx_glm-5.2-preview_1m__10h_run3": TraceFormat.CLAUDE_STREAM_JSON,
    "opencode_opencode_glm-4.7-free_10h": TraceFormat.OPENCODE,
    "opencode_opencode_kimi-k2-thinking_10h": TraceFormat.OPENCODE,
    "opencode_opencode_kimi-k2.5_10h_run2": TraceFormat.OPENCODE,
    "opencode_opencode_minimax-m2.1-free_10h": TraceFormat.OPENCODE,
    "opencode_opencode_minimax-m2.5-free_10h_run2": TraceFormat.OPENCODE,
    "opencode_zai_glm-5_10h_run2": TraceFormat.OPENCODE,
}
"""Run directories whose agent model has published weights, and the log format of each.

Keyed by the directory name the benchmark publishes, ``{agent}_{model}_{hours}h_{run}``.
GLM is MIT-licensed; Kimi K2 is Modified MIT; MiniMax M2 ships weights on the Hub. The
``kimi-k2.5`` and ``zai_glm-5`` ``run1`` directories are absent because they hold
judgements but no trace.
"""


class Judgement(NamedTuple):
    """One of the benchmark's anti-cheating verdict files and the verdict that disqualifies a trace."""

    filename: str
    flagged: str
    counter: str


# The judge writes one of two sentences per file, and the clean verdict ("no
# contamination detected") contains the flagged one as a substring, so a verdict has
# to be compared for equality rather than searched for.
JUDGEMENTS = (
    Judgement("contamination_judgement.txt", "contamination detected", "dropped_contaminated"),
    Judgement("disallowed_model_judgement.txt", "disallowed use detected", "dropped_disallowed_model"),
)

BENCHMARK_LABELS = {
    "aime2025": "AIME 2025",
    "arenahardwriting": "ArenaHardWriting",
    "bfcl": "BFCL",
    "gpqamain": "GPQA (Main)",
    "gsm8k": "GSM8K",
    "healthbench": "HealthBench",
    "humaneval": "HumanEval",
}
"""Benchmark each task targets, keyed by the task directory's name prefix.

Values are the names the dataset's README gives them.
"""

BASE_MODELS = frozenset(
    {
        "HuggingFaceTB/SmolLM3-3B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Base",
        "google/gemma-3-4b-pt",
    }
)
"""The four pre-trained models the benchmark asks agents to post-train."""

TASK_FILES = ("solve_out.txt", "metrics.json", *(judgement.filename for judgement in JUDGEMENTS))

OPEN_WEIGHT_GLOBS: tuple[str, ...] = tuple(f"{run}/*/{filename}" for run in OPEN_WEIGHT_RUNS for filename in TASK_FILES)
"""Each open-weights run's traces plus the score and judgement files ``trace_to_doc`` reads."""

_TIMESTAMP_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2}T[\d:]+Z\]\s+")


def _events(lines: Iterable[str]) -> Iterator[dict]:
    """Yield the JSON events of a harness log.

    Skips the plain-text GPU-check preamble every trace opens with, and strips the
    ISO-timestamp prefix the ``glmx`` runs put on each line. A log truncated when the
    ten-hour budget expired can end mid-object; that last line is counted and dropped.
    """
    for line in lines:
        event = _TIMESTAMP_PREFIX.sub("", line).strip()
        if not event.startswith("{"):
            continue
        try:
            yield json.loads(event)
        except json.JSONDecodeError:
            counters.pipeline.update_counter("posttrainbench/malformed_event", 1)


def _open_assistant_turn(messages: list[dict]) -> dict:
    """Return the trailing assistant turn, opening a new one when the last turn is not the agent's.

    The harnesses emit one content block per event, so a single agent turn arrives as a
    run of events; merging them keeps reasoning, prose, and tool calls in one turn.
    """
    if not messages or messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
    return messages[-1]


def _append_text(message: dict, text: str) -> None:
    if not text:
        return
    message["content"] = f"{message['content']}\n\n{text}" if message["content"] else text


def _tool_call(name: str, arguments: object) -> dict:
    """Wrap a harness tool call in the OpenAI shape ``render_tool_message`` renders."""
    return {"function": {"name": name, "arguments": arguments}}


def _tool_result_text(block: dict) -> str:
    """Flatten a ``tool_result`` block, whose content is either a string or a list of text blocks."""
    content = block.get("content")
    if isinstance(content, list):
        return "\n".join(part.get("text") or "" for part in content if part.get("type") == "text")
    return content or ""


def _append_assistant_blocks(messages: list[dict], blocks: list[dict]) -> None:
    """Fold one assistant event's content blocks into the open agent turn.

    Reasoning arrives as its own ``thinking`` block but is appended as plain text, the
    way the other rollout sources keep inline reasoning untagged.
    """
    turn = _open_assistant_turn(messages)
    for block in blocks:
        match block.get("type"):
            case "thinking":
                _append_text(turn, block.get("thinking") or "")
            case "text":
                _append_text(turn, block.get("text") or "")
            case "tool_use":
                turn["tool_calls"].append(_tool_call(block.get("name"), block.get("input")))


def _append_user_blocks(messages: list[dict], blocks: list[dict]) -> None:
    for block in blocks:
        match block.get("type"):
            case "tool_result":
                messages.append({"role": "tool", "content": _tool_result_text(block)})
            case "text":
                messages.append({"role": "user", "content": block.get("text") or ""})


def parse_claude_stream_json(lines: Iterable[str]) -> list[dict]:
    """Parse a Claude Code stream-json log into role-tagged turns."""
    messages: list[dict] = []
    for event in _events(lines):
        kind = event.get("type")
        content = (event.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        if kind == "assistant":
            _append_assistant_blocks(messages, content)
        elif kind == "user":
            _append_user_blocks(messages, content)
    return messages


def parse_opencode_events(lines: Iterable[str]) -> list[dict]:
    """Parse an opencode event stream into role-tagged turns.

    Tool output is taken from ``state.output``; ``state.metadata.output`` repeats it
    verbatim. ``step_start``/``step_finish`` events only carry token accounting.
    """
    messages: list[dict] = []
    for event in _events(lines):
        part = event.get("part") or {}
        match part.get("type"):
            case "text":
                _append_text(_open_assistant_turn(messages), part.get("text") or "")
            case "tool":
                state = part.get("state") or {}
                _open_assistant_turn(messages)["tool_calls"].append(_tool_call(part.get("tool"), state.get("input")))
                output = state.get("output") or ""
                if output:
                    messages.append({"role": "tool", "content": output})
    return messages


_PARSERS = {
    TraceFormat.CLAUDE_STREAM_JSON: parse_claude_stream_json,
    TraceFormat.OPENCODE: parse_opencode_events,
}


def _replace_lone_surrogates(text: str) -> str:
    """Substitute the unpaired surrogates the harnesses log for ``?``.

    Terminal output that gets cut mid-emoji reaches the log as an escape like ``\\ud83d``
    with no low surrogate after it. Python decodes that happily but Parquet cannot encode
    it, so a single truncated emoji would otherwise fail the whole shard.
    """
    return text.encode("utf-8", "replace").decode("utf-8")


def _read_sidecar(task_dir: str, filename: str) -> str | None:
    """Read one of a task's sidecar files, or None when the harness never wrote it."""
    path = posixpath.join(task_dir, filename)
    if not StoragePath(path).exists():
        return None
    with open_url(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def disqualifying_judgement(task_dir: str) -> str | None:
    """Name the counter for the judgement that disqualifies this trace, or None if it passes.

    A handful of traces carry no judgement files because the benchmark's judge never ran
    on them. A missing verdict is not a flag, so those traces are kept, counted under
    ``unjudged`` so the gap stays visible in the pipeline stats.
    """
    verdicts = [_read_sidecar(task_dir, judgement.filename) for judgement in JUDGEMENTS]
    if any(verdict is None for verdict in verdicts):
        counters.pipeline.update_counter("posttrainbench/unjudged", 1)
    for judgement, verdict in zip(JUDGEMENTS, verdicts, strict=True):
        if verdict is not None and verdict.strip() == judgement.flagged:
            return judgement.counter
    return None


def task_target(task_dir: str) -> tuple[str, str]:
    """Split a task directory name into the benchmark it targets and the model it post-trained.

    The benchmark publishes these as ``{benchmark}_{hf_org}_{base_model}_{job_id}``.
    """
    benchmark, *model_parts, _job_id = posixpath.basename(task_dir).split("_")
    base_model = "/".join(model_parts)
    if base_model not in BASE_MODELS:
        raise ValueError(f"{task_dir}: {base_model!r} is not one of the benchmark's base models")
    return BENCHMARK_LABELS[benchmark], base_model


def _accuracy(task_dir: str) -> float | None:
    """Read the grader's accuracy for a task, or None when it never scored a model.

    ``metrics.json`` is missing for a trajectory that produced nothing the grader could
    load. That is a real outcome and not a gap in the dataset, so it is reported rather
    than treated as an error.
    """
    metrics = _read_sidecar(task_dir, "metrics.json")
    if metrics is None:
        return None
    return json.loads(metrics)["accuracy"]


def outcome_prefix(task_dir: str) -> str:
    """The sentence prepended to a trajectory so training conditions on how well it did.

    Names the benchmark and base model alongside the score because the scores are not
    comparable without them: the median attempt scores 0% on AIME 2025 and 77% on BFCL.
    """
    benchmark, base_model = task_target(task_dir)
    accuracy = _accuracy(task_dir)
    if accuracy is None:
        return f"This trajectory post-trained {base_model} for {benchmark} but produced no model the grader could score."
    return f"This trajectory post-trained {base_model} for {benchmark} and scored {accuracy:.1%}."


def trace_to_doc(path: str) -> list[dict]:
    """Render one ``solve_out.txt`` into an outcome-prefixed document, dropping judge-flagged traces."""
    task_dir = posixpath.dirname(path)
    run = posixpath.basename(posixpath.dirname(task_dir))
    parse = _PARSERS[OPEN_WEIGHT_RUNS[run]]

    disqualified = disqualifying_judgement(task_dir)
    if disqualified:
        counters.pipeline.update_counter(f"posttrainbench/{disqualified}", 1)
        return []

    with open_url(path, "rt", encoding="utf-8", errors="replace") as f:
        messages = parse(f)

    if not messages:
        counters.pipeline.update_counter("posttrainbench/dropped_empty", 1)
        return []

    rendered = "\n\n".join(render_tool_message(message) for message in messages)
    text = _replace_lone_surrogates(f"{outcome_prefix(task_dir)}\n\n{rendered}")
    counters.pipeline.update_counter("posttrainbench/kept", 1)
    return [text_document(text, HF_DATASET_ID)]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/solve_out.txt")
        .flat_map(trace_to_doc)
        # One shard per trace would leave 252 files of a few hundred KB each.
        .reshard(16)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="posttrainbench-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_posttrainbench_step() -> StepSpec:
    """Download the open-weights PostTrainBench runs and render each trace into a Parquet document."""
    dl = download_hf_step(
        "raw/posttrainbench-trajectories",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=list(OPEN_WEIGHT_GLOBS),
    )

    return StepSpec(
        name="processed/posttrainbench-open-weights",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v2"},
    )


def posttrainbench_open_weights_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for posttrainbench/open-weights."""
    processed = download_posttrainbench_step()
    return (
        processed,
        normalize_step(name="normalized/posttrainbench-open-weights", download=processed),
    )
