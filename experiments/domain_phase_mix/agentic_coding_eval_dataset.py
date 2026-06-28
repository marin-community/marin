# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pyarrow", "pandas"]
# ///
"""Materialize an assistant-action BPB eval bundle from agentic coding traces."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import fsspec
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.rl.placement import marin_prefix_for_region

logger = logging.getLogger(__name__)

RAW_CODERFORGE_PREFIX = "gs://marin-us-central1/raw/coderforge-preview_ad26b119/trajectories"
RAW_OPENHANDS_PATH = "gs://marin-us-central1/raw/swe-rebench-openhands-trajectories_e1e457c7/trajectories.parquet"
DEFAULT_OUTPUT_URI = "gs://marin-us-east5/raw/eval-datasets/agentic-coding-v1"

CODERFORGE_SPLITS = ("SWE_Rebench", "SWE_Smith", "R2E_Gym")
OUTCOMES = ("success", "fail")
SAMPLE_SIZE_PER_SLICE = 512
MAX_TEXT_BYTES = 32_768
SAMPLE_SEED = 20260503
ASSISTANT_ACTION_VIEW = "assistant_action"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class AgenticEvalSlice:
    """One materialized eval slice."""

    source_dataset: str
    source_split: str
    outcome: str
    output_name: str
    path: str


@dataclass(frozen=True)
class AgenticEvalBundleConfig:
    """Config for deterministic agentic-coding eval bundle materialization."""

    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    sample_size_per_slice: int = SAMPLE_SIZE_PER_SLICE
    max_text_bytes: int = MAX_TEXT_BYTES
    sample_seed: int = SAMPLE_SEED
    coderforge_prefix: str = RAW_CODERFORGE_PREFIX
    openhands_path: str = RAW_OPENHANDS_PATH


@dataclass(frozen=True)
class RenderedTrace:
    """Rendered trace selected for the eval bundle."""

    source_dataset: str
    source_split: str
    outcome: str
    source_trajectory_id: str
    source_path: str
    view: str
    text: str


@dataclass(frozen=True)
class RankedTrace:
    """Trace plus deterministic hash rank for sampling."""

    rank: int
    seq: int
    trace: RenderedTrace


def coderforge_outcome(reward: object) -> str:
    """Return the CoderForge outcome label from reward."""
    try:
        value = float(reward)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CoderForge reward: {reward!r}") from exc
    return "success" if value >= 1.0 else "fail"


def openhands_outcome(resolved: object) -> str:
    """Return the OpenHands outcome label from resolved."""
    try:
        value = float(resolved)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid OpenHands resolved value: {resolved!r}") from exc
    return "success" if value >= 1.0 else "fail"


def parse_messages(value: object) -> list[Mapping[str, Any]]:
    """Parse raw message payloads from either JSON strings or nested objects."""
    if value is None:
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        return []
    return [message for message in parsed if isinstance(message, Mapping)]


def _content_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, Mapping):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return str(content)


def _jsonish(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_tool_call(tool_call: object) -> str:
    """Render one assistant tool call without tool/environment observations."""
    if isinstance(tool_call, str):
        return f"<tool_call>\n{tool_call}\n</tool_call>"
    if not isinstance(tool_call, Mapping):
        return f"<tool_call>\n{_jsonish(tool_call)}\n</tool_call>"

    function = tool_call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name") or tool_call.get("name") or tool_call.get("tool_name") or "unknown"
        arguments = function.get("arguments")
    else:
        name = tool_call.get("name") or tool_call.get("tool_name") or tool_call.get("type") or "unknown"
        arguments = tool_call.get("arguments") or tool_call.get("input") or tool_call.get("parameters")
    return f"<tool_call name={json.dumps(str(name), ensure_ascii=False)}>\n{_jsonish(arguments)}\n</tool_call>"


def render_assistant_action_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    """Render assistant content and tool calls, excluding user/system/tool observations."""
    chunks: list[str] = []
    for message in messages:
        if str(message.get("role", "")).lower() != "assistant":
            continue
        content = _content_text(message.get("content")).strip()
        tool_calls = message.get("tool_calls") or message.get("toolCalls") or []
        if isinstance(tool_calls, Mapping) or isinstance(tool_calls, str):
            tool_call_items: list[object] = [tool_calls]
        elif isinstance(tool_calls, list):
            tool_call_items = list(tool_calls)
        else:
            tool_call_items = []

        parts: list[str] = []
        if content:
            parts.append(content)
        parts.extend(render_tool_call(tool_call) for tool_call in tool_call_items)
        if parts:
            chunks.append("<assistant>\n" + "\n".join(parts).strip() + "\n</assistant>")
    return "\n\n".join(chunks).strip()


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate text to at most max_bytes while preserving valid UTF-8."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _rank_for_trace(*, seed: int, source_dataset: str, source_split: str, outcome: str, trajectory_id: str) -> int:
    key = f"{seed}:{source_dataset}:{source_split}:{outcome}:{trajectory_id}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], byteorder="big")


def _trajectory_id(row: Mapping[str, Any], fallback: str) -> str:
    for key in ("trajectory_id", "instance_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return fallback


def _candidate_from_row(
    *,
    row: Mapping[str, Any],
    source_dataset: str,
    source_split: str,
    source_path: str,
    outcome: str,
    messages_key: str,
    fallback_id: str,
    max_text_bytes: int,
) -> RenderedTrace | None:
    trajectory_id = _trajectory_id(row, fallback_id)
    text = render_assistant_action_messages(parse_messages(row.get(messages_key)))
    text = truncate_utf8(text, max_text_bytes).strip()
    if not text:
        return None
    return RenderedTrace(
        source_dataset=source_dataset,
        source_split=source_split,
        outcome=outcome,
        source_trajectory_id=trajectory_id,
        source_path=source_path,
        view=ASSISTANT_ACTION_VIEW,
        text=text,
    )


def _push_top_ranked(
    heap: list[tuple[int, int, RankedTrace]],
    *,
    ranked: RankedTrace,
    sample_size: int,
) -> None:
    entry = (-ranked.rank, -ranked.seq, ranked)
    if len(heap) < sample_size:
        heapq.heappush(heap, entry)
        return
    if entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _parquet_rows(path: str, columns: list[str]):
    with fsspec.open(path, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        for batch in parquet_file.iter_batches(batch_size=64, columns=columns):
            rows = batch.to_pylist()
            yield from rows


def _glob_gcs(pattern: str) -> list[str]:
    fs, _, paths = fsspec.get_fs_token_paths(pattern)
    if paths:
        return sorted(f"gs://{path}" if not str(path).startswith("gs://") else str(path) for path in paths)
    return sorted(f"gs://{path}" if not str(path).startswith("gs://") else str(path) for path in fs.glob(pattern))


def _collect_coderforge_slice(
    *,
    config: AgenticEvalBundleConfig,
    split: str,
    outcome: str,
) -> list[RenderedTrace]:
    pattern = f"{config.coderforge_prefix.rstrip('/')}/{split}-*.parquet"
    paths = _glob_gcs(pattern)
    if not paths:
        raise ValueError(f"No CoderForge parquet files matched {pattern}")

    heap: list[tuple[int, int, RankedTrace]] = []
    seq = 0
    for path in paths:
        for row in _parquet_rows(path, columns=["trajectory_id", "reward", "messages"]):
            try:
                row_outcome = coderforge_outcome(row.get("reward"))
            except ValueError:
                continue
            if row_outcome != outcome:
                continue
            trace = _candidate_from_row(
                row=row,
                source_dataset="coderforge",
                source_split=split,
                source_path=path,
                outcome=outcome,
                messages_key="messages",
                fallback_id=f"{path}:{seq}",
                max_text_bytes=config.max_text_bytes,
            )
            if trace is None:
                continue
            rank = _rank_for_trace(
                seed=config.sample_seed,
                source_dataset=trace.source_dataset,
                source_split=trace.source_split,
                outcome=trace.outcome,
                trajectory_id=trace.source_trajectory_id,
            )
            _push_top_ranked(
                heap,
                ranked=RankedTrace(rank=rank, seq=seq, trace=trace),
                sample_size=config.sample_size_per_slice,
            )
            seq += 1
    return _sorted_heap_traces(heap)


def _collect_openhands_slice(*, config: AgenticEvalBundleConfig, outcome: str) -> list[RenderedTrace]:
    heap: list[tuple[int, int, RankedTrace]] = []
    seq = 0
    for row in _parquet_rows(config.openhands_path, columns=["trajectory_id", "resolved", "trajectory"]):
        try:
            row_outcome = openhands_outcome(row.get("resolved"))
        except ValueError:
            continue
        if row_outcome != outcome:
            continue
        trace = _candidate_from_row(
            row=row,
            source_dataset="openhands",
            source_split="swe_rebench",
            source_path=config.openhands_path,
            outcome=outcome,
            messages_key="trajectory",
            fallback_id=f"{config.openhands_path}:{seq}",
            max_text_bytes=config.max_text_bytes,
        )
        if trace is None:
            continue
        rank = _rank_for_trace(
            seed=config.sample_seed,
            source_dataset=trace.source_dataset,
            source_split=trace.source_split,
            outcome=trace.outcome,
            trajectory_id=trace.source_trajectory_id,
        )
        _push_top_ranked(
            heap,
            ranked=RankedTrace(rank=rank, seq=seq, trace=trace),
            sample_size=config.sample_size_per_slice,
        )
        seq += 1
    return _sorted_heap_traces(heap)


def _sorted_heap_traces(heap: list[tuple[int, int, RankedTrace]]) -> list[RenderedTrace]:
    ranked = [entry[2] for entry in heap]
    return [item.trace for item in sorted(ranked, key=lambda item: (item.rank, item.seq))]


def _output_name(source_dataset: str, source_split: str, outcome: str) -> str:
    split_slug = source_split.lower().replace("-", "_")
    return f"{source_dataset}_{split_slug}_{outcome}.jsonl"


def _slice_specs(output_path: str) -> list[AgenticEvalSlice]:
    specs: list[AgenticEvalSlice] = []
    for split in CODERFORGE_SPLITS:
        for outcome in OUTCOMES:
            name = _output_name("coderforge", split, outcome)
            specs.append(
                AgenticEvalSlice(
                    source_dataset="coderforge",
                    source_split=split,
                    outcome=outcome,
                    output_name=name,
                    path=f"{output_path.rstrip('/')}/{name}",
                )
            )
    for outcome in OUTCOMES:
        name = _output_name("openhands", "swe_rebench", outcome)
        specs.append(
            AgenticEvalSlice(
                source_dataset="openhands",
                source_split="swe_rebench",
                outcome=outcome,
                output_name=name,
                path=f"{output_path.rstrip('/')}/{name}",
            )
        )
    return specs


def agentic_eval_slices(output_path: str = DEFAULT_OUTPUT_URI) -> list[AgenticEvalSlice]:
    """Return the fixed slice layout for an output bundle."""
    return _slice_specs(output_path)


def _records_for_slice(config: AgenticEvalBundleConfig, spec: AgenticEvalSlice) -> list[RenderedTrace]:
    if spec.source_dataset == "coderforge":
        return _collect_coderforge_slice(config=config, split=spec.source_split, outcome=spec.outcome)
    if spec.source_dataset == "openhands":
        return _collect_openhands_slice(config=config, outcome=spec.outcome)
    raise ValueError(f"Unknown source dataset: {spec.source_dataset}")


def _write_jsonl(path: str, records: list[RenderedTrace]) -> None:
    directory = path.rsplit("/", maxsplit=1)[0]
    fs, _, _ = fsspec.get_fs_token_paths(directory)
    fs.makedirs(directory, exist_ok=True)
    with fsspec.open(path, "wt") as handle:
        for index, record in enumerate(records):
            row = asdict(record)
            row["id"] = (
                f"{record.source_dataset}:{record.source_split}:{record.outcome}:" f"{record.source_trajectory_id}"
            )
            row["document_index"] = index
            row["text_bytes"] = len(record.text.encode("utf-8"))
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def materialize_agentic_coding_eval_bundle(config: AgenticEvalBundleConfig) -> None:
    """Materialize the deterministic agentic-coding eval bundle."""
    output_path = config.output_path.rstrip("/")
    specs = _slice_specs(output_path)
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        records = _records_for_slice(config, spec)
        if len(records) != config.sample_size_per_slice:
            raise ValueError(
                f"{spec.source_dataset}/{spec.source_split}/{spec.outcome} has {len(records)} records; "
                f"expected {config.sample_size_per_slice}"
            )
        _write_jsonl(spec.path, records)
        manifest_rows.append(
            {
                **asdict(spec),
                "rows": len(records),
                "sample_seed": config.sample_seed,
                "max_text_bytes": config.max_text_bytes,
                "view": ASSISTANT_ACTION_VIEW,
            }
        )
        logger.info("Wrote %d rows to %s", len(records), spec.path)

    manifest = {
        "bundle": "agentic-coding-v1",
        "view": ASSISTANT_ACTION_VIEW,
        "rows": sum(row["rows"] for row in manifest_rows),
        "sample_size_per_slice": config.sample_size_per_slice,
        "max_text_bytes": config.max_text_bytes,
        "sample_seed": config.sample_seed,
        "slices": manifest_rows,
    }
    with fsspec.open(f"{output_path}/{MANIFEST_FILENAME}", "wt") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("Wrote manifest to %s/%s", output_path, MANIFEST_FILENAME)


def agentic_coding_eval_bundle_step(output_uri: str = DEFAULT_OUTPUT_URI) -> ExecutorStep:
    """Build the materialization step with a stable raw-eval output path."""
    step = ExecutorStep(
        name="eval_datasets/agentic_coding_v1",
        description="Materialize agentic-coding assistant-action BPB eval bundle",
        fn=remote(
            materialize_agentic_coding_eval_bundle,
            resources=ResourceConfig(cpu=2, ram="32g", disk="10g", regions=["us-central1"]),
            pip_dependency_groups=["cpu"],
        ),
        config=AgenticEvalBundleConfig(),
    )
    return step.with_output_path(output_uri.rstrip("/"))


def _executor_prefix_for_output(output_uri: str) -> str | None:
    if output_uri.startswith("gs://marin-us-central1/"):
        return marin_prefix_for_region("us-central1")
    if output_uri.startswith("gs://marin-us-east5/"):
        return marin_prefix_for_region("us-east5")
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    parser.add_argument("--sample-size-per-slice", type=int, default=SAMPLE_SIZE_PER_SLICE)
    parser.add_argument("--max-text-bytes", type=int, default=MAX_TEXT_BYTES)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.submit:
        step = agentic_coding_eval_bundle_step(args.output_uri)
        executor_main(
            ExecutorMainConfig(
                prefix=_executor_prefix_for_output(args.output_uri),
                max_concurrent=args.max_concurrent,
            ),
            steps=[step],
            description="Materialize agentic-coding assistant-action BPB eval bundle",
        )
        return
    materialize_agentic_coding_eval_bundle(
        AgenticEvalBundleConfig(
            output_path=args.output_uri,
            sample_size_per_slice=args.sample_size_per_slice,
            max_text_bytes=args.max_text_bytes,
            sample_seed=args.sample_seed,
        )
    )


if __name__ == "__main__":
    main()
