# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for datakit download pipelines."""

import hashlib
import json
import logging
from collections.abc import Callable, Iterator

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import open_url, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

logger = logging.getLogger(__name__)

DOCUMENT_SHARD_PATTERN = "data-{shard:05d}-of-{total:05d}.parquet"
"""Filename pattern for the parquet shards a datakit download step produces."""


def load_parquet_batched(path: str) -> Iterator[dict]:
    """Read parquet via iter_batches to avoid OOM on large nested-struct columns."""
    with open_url(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            try:
                rows = batch.to_pydict()
            except UnicodeDecodeError as e:
                counters.pipeline.update_counter("load_parquet_batched/utf8_skip_batch", 1)
                logger.warning("Skipping batch from %s due to invalid UTF-8: %s", path, e)
                continue
            n = len(next(iter(rows.values())))
            for i in range(n):
                yield {k: rows[k][i] for k in rows}


def write_document_shards(
    input_glob: str,
    output_path: str,
    *,
    name: str,
    row_to_doc: Callable[[dict], list[dict]],
    resources: ResourceConfig,
    loader: Callable[[str], Iterator[dict]] = load_parquet,
    num_shards: int | None = None,
) -> None:
    """Render downloaded rows into documents and write them as parquet shards.

    This is the shape shared by most datakit download steps: read every file matched by
    ``input_glob``, expand each row into zero or more documents, and write the result under
    ``output_path`` as ``DOCUMENT_SHARD_PATTERN`` shards. Writes skip shards that already
    exist so an interrupted step resumes rather than redoing finished work.

    Args:
        input_glob: Glob matching the downloaded source files.
        output_path: Directory the parquet shards are written to.
        name: Zephyr context name for the transform job.
        row_to_doc: Renders one source row into zero or more documents.
        resources: Per-worker resources. Sources with large nested-struct columns need
            considerably more RAM than flat text sources.
        loader: Reads one input file into rows. Pass :func:`load_parquet_batched` for
            sources whose rows are too large to materialize a whole file at once.
        num_shards: Reshard the documents to this many output shards before writing. Use
            it when the input file count is a poor fit for the output size.
    """
    pipeline = Dataset.from_files(input_glob).flat_map(loader).flat_map(row_to_doc)
    if num_shards is not None:
        pipeline = pipeline.reshard(num_shards)
    ZephyrContext(name=name, resources=resources).execute(
        pipeline.write_parquet(prefix_join(output_path, DOCUMENT_SHARD_PATTERN), skip_existing=True)
    )


def strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def text_document(text: str, source: str) -> dict:
    """Build a datakit document with a content-addressed ``id`` derived from ``text``.

    The ``id`` is the SHA-256 hex digest of the UTF-8-encoded text, so byte-identical
    documents share an id and collapse during exact-dedup normalization.
    """
    return {
        "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": source,
    }


def render_role_message(msg: dict) -> str:
    """Render a single chat message as ``<role>\\ncontent\\n</role>``.

    Missing or null content renders as an empty body.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    return f"<{role}>\n{content}\n</{role}>"


# Outcome tags prepended to agent-rollout transcripts so the model can
# distinguish successful, failed, and unverified attempts. Agent trajectory
# sources derive the outcome differently but render the same tag text.
TRAJECTORY_SOLVED_TAG = "This trajectory solved the task successfully."
TRAJECTORY_FAILED_TAG = "This trajectory failed to solve the task."
TRAJECTORY_UNVERIFIED_TAG = "This trajectory ended before the task was verified."


def render_tool_call(tool_call: dict) -> str:
    """Render an OpenAI-style tool call as a ``<tool_call:name>`` … ``</tool_call:name>`` block.

    ``arguments`` may be a JSON string or an already-decoded value; a mapping renders one
    indented ``key: value`` line per argument, and any other non-null value renders on a
    single indented line. Malformed JSON arguments are kept as their raw string rather than
    raising, so a single bad tool call does not abort the whole transform.
    """
    func = tool_call.get("function") or {}
    name = func.get("name") or "unknown"
    args = func.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    parts = [f"<tool_call:{name}>"]
    if isinstance(args, dict):
        for key, value in args.items():
            parts.append(f"  {key}: {value}")
    elif args is not None:
        parts.append(f"  {args}")
    parts.append(f"</tool_call:{name}>")
    return "\n".join(parts)


def render_tool_message(msg: dict) -> str:
    """Render a chat message that may carry ``tool_calls`` as a role-tagged block.

    Like :func:`render_role_message`, but appends any ``tool_calls`` after the text content
    and omits the content line entirely when the message has no text.
    """
    role = msg.get("role") or "unknown"
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    parts = [f"<{role}>"]
    if content:
        parts.append(content)
    if tool_calls:
        parts.extend(render_tool_call(tc) for tc in tool_calls)
    parts.append(f"</{role}>")
    return "\n".join(parts)
