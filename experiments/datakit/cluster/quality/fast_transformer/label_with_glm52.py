# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label a document set against the oracle rubric with a self-hosted GLM-5.2.

Brings up GLM-5.2-FP8 (:mod:`experiments.datakit.cluster.quality.glm52_vllm`) as a
child job, waits for its endpoint, scores every row of the labeling set, and writes
the parquet :mod:`experiments.datakit.cluster.quality.fast_transformer.train`
consumes. The server is stopped on the way out, including on failure — an idle
two-node GPU gang is expensive.

One document per request, not a batch. The rubric emits JSONL keyed by document
index so a batch is possible, but then a single skipped or renumbered index
silently misaligns every label after it, and a mislabeled training set is far more
costly than the extra requests. One document per request makes the contract
checkable: exactly one object, or the row is dropped and counted.

Scores are written as ``score_normalized`` on the same ``(quality - 1) / 4`` scale
``train.py`` reads, alongside the raw ``quality`` and the ``content_type`` the
oracle assigned, so per-type parity can be audited before the model is trained.

Labeling is checkpointed per chunk under ``<out>.chunks/`` and resumes by id, so a
preempted run costs one chunk rather than the whole set. That is not a nicety at
this scale: a 11.5k-document run is hours of a two-node GPU gang on a cluster that
stays near capacity, and holding every result in memory until the end means a
preemption in the final hour loses all of it.
"""

import argparse
import json
import logging
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from iris.client import iris_ctx
from openai import OpenAI
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES, SYSTEM_PROMPT
from experiments.datakit.cluster.quality.glm52_vllm import (
    GB200_FLEET,
    H100_FLEET,
    MODEL,
    Glm52LaunchConfig,
    ServerConfig,
    submit_glm52,
    wait_for_endpoint_url,
)

logger = logging.getLogger(__name__)

QUALITY_LEVELS = (1, 2, 3, 4, 5)
# Documents are capped at 12k *characters* upstream, which is not a token budget:
# CJK and code tokenize far denser than English, so a 12k-character document can
# exceed 12k tokens. With MAX_OUTPUT_TOKENS reserved on top, a 16384 context
# rejected those documents outright with a 400 — silently dropping long documents
# and reintroducing the very length bias the excerpting fix removed. Sized so the
# cap plus the reserved answer always fits.
DEFAULT_MAX_MODEL_LEN = 32_768
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_CONCURRENCY = 48
# Documents per checkpoint. Small enough that a preemption loses minutes of GPU
# time, large enough that the write is amortized over a full concurrency sweep.
DEFAULT_CHUNK_SIZE = 500
# The answer itself is one short JSON object, but GLM-5.2 reasons before emitting it
# and that reasoning counts against the same budget. At 512 it ran out mid-thought
# and never reached the JSON, dropping 93% of a probe batch; the survivors were
# whichever documents it happened to think briefly about.
MAX_OUTPUT_TOKENS = 4_096
# Reasoning traces arrive in-band. Everything up to the final close tag is thinking,
# not answer, so the verdict is parsed from what follows it.
THINK_CLOSE_TAG = "</think>"
# How many unusable replies to show before falling back to counting them: enough to
# tell a truncation from a format drift, few enough not to flood a 20k-row run.
MAX_LOGGED_REJECTS = 3
# Minimum share of a chunk that must label successfully. Below this the run aborts
# rather than treating a dead server as a very unlucky batch of documents.
MIN_CHUNK_SUCCESS = 0.5
# GB200 is the shape upstream serves GLM-5.2 on: the weights fit in 8 GPUs rather
# than 16, and the gang binds hard to one NVLink domain.
FLEETS = {"gb200": GB200_FLEET, "h100": H100_FLEET}
VLLM_ENDPOINT = "glm52-vllm"
RAY_ENDPOINT = "glm52-ray"


@dataclass(frozen=True)
class LabelStats:
    """What the oracle actually returned, for auditing before training."""

    labeled: int
    dropped: int

    def log(self) -> None:
        total = self.labeled + self.dropped
        share = self.dropped / total if total else 0.0
        logger.info("label_with_glm52: %d labeled, %d dropped (%.2f%%)", self.labeled, self.dropped, share * 100)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


def _parse_verdict(content: str) -> dict | None:
    """The rubric's one-object-per-line reply for a single document, or ``None``.

    Returns ``None`` rather than raising for anything malformed: an oracle that
    occasionally emits prose should cost a row, not the run.
    """
    if THINK_CLOSE_TAG in content:
        content = content.rsplit(THINK_CLOSE_TAG, 1)[-1]
    for line in _strip_code_fence(content).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            verdict = json.loads(line)
        except json.JSONDecodeError:
            continue
        quality = verdict.get("quality")
        content_type = verdict.get("content_type")
        if quality not in QUALITY_LEVELS or content_type not in CONTENT_TYPES:
            continue
        return {"quality": int(quality), "content_type": content_type, "valid": bool(verdict.get("valid", True))}
    return None


def label_document(client: OpenAI, text: str, rejects: list[str]) -> dict | None:
    """Ask the oracle for one document's verdict; ``None`` if the reply is unusable.

    Unusable replies are sampled into ``rejects`` (with the finish reason, which
    distinguishes a budget truncation from a format drift) so a high drop rate is
    diagnosable from the run's own logs.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'<document index="0">\n{text}\n</document>'},
            ],
        )
    except Exception as e:  # one bad request must not abort a 20k-row run
        logger.warning("label_with_glm52: request failed: %r", e)
        return None
    choice = response.choices[0]
    content = choice.message.content or ""
    verdict = _parse_verdict(content)
    if verdict is None and len(rejects) < MAX_LOGGED_REJECTS:
        rejects.append(f"finish_reason={choice.finish_reason} reply={content[-400:]!r}")
    return verdict


def label_rows(client: OpenAI, rows: Sequence[dict], concurrency: int, rejects: list[str]) -> list[dict]:
    """Label every row, dropping the ones the oracle did not answer cleanly."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        verdicts = list(pool.map(lambda r: label_document(client, r["text"], rejects), rows))
    return [
        {
            "source": row["source"],
            "id": row["id"],
            "text": row["text"],
            "quality": verdict["quality"],
            "score_normalized": (verdict["quality"] - 1) / 4.0,
            "content_type": verdict["content_type"],
            "valid": verdict["valid"],
            "v0_score": row.get("v0_score"),
            "label_batch": "glm52_rubric_v2",
        }
        for row, verdict in zip(rows, verdicts, strict=True)
        if verdict is not None
    ]


def _read_label_set(label_set: str) -> list[dict]:
    """Rows of the labeling set, whether it is one parquet file or a directory of shards.

    A large draw is written as shards rather than a single file, so accept both: a
    path ending in ``.parquet`` is read directly, anything else is treated as a
    directory and its shards concatenated.
    """
    if label_set.endswith(".parquet"):
        with StoragePath(label_set).open("rb") as handle:
            return pq.read_table(handle).to_pylist()
    rows: list[dict] = []
    shards = sorted(str(m) for m in StoragePath(f"{label_set.rstrip('/')}/*.parquet").glob())
    if not shards:
        raise ValueError(f"no parquet shards under {label_set}")
    for shard in shards:
        with StoragePath(shard).open("rb") as handle:
            rows.extend(pq.read_table(handle).to_pylist())
    return rows


def _chunk_dir(out: str) -> str:
    return f"{out.rstrip('/')}.chunks"


def _labeled_ids(chunk_dir: str) -> set[str]:
    """Ids already written to a checkpoint chunk, so a resume relabels nothing."""
    done: set[str] = set()
    for path in StoragePath(f"{chunk_dir}/*.parquet").glob():
        with StoragePath(str(path)).open("rb") as handle:
            done.update(pq.ParquetFile(handle).read(columns=["id"]).column("id").to_pylist())
    return done


def label_with_checkpoints(
    client: OpenAI, rows: Sequence[dict], *, concurrency: int, chunk_size: int, out: str
) -> tuple[pa.Table, LabelStats]:
    """Label ``rows`` in chunks, checkpointing each chunk before starting the next.

    Labeling a large set is hours of GPU time on a preemptible gang, so nothing is
    held only in memory: each chunk is written under ``<out>.chunks/`` as it
    finishes and a rerun skips ids already present there. Losing a preempted run
    then costs one chunk, not the whole set.

    Ids are the resume key, so a row whose verdict was unusable is retried on the
    next run rather than being silently treated as done.
    """
    chunk_dir = _chunk_dir(out)
    already = _labeled_ids(chunk_dir)
    if already:
        logger.info("label_with_glm52: resuming, %d ids already labeled", len(already))
    pending = [r for r in rows if r["id"] not in already]

    rejects: list[str] = []
    written = len(already)
    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        labeled = label_rows(client, batch, concurrency, rejects)
        # A dropped row normally means one unusable reply. A chunk that mostly fails
        # means the server is gone, and continuing would quietly discard the rest of
        # the set and still report success: a 512-concurrency run collapsed this way
        # and wrote 11k of 88k labels with an exit code of zero.
        if len(labeled) < len(batch) * MIN_CHUNK_SUCCESS:
            raise RuntimeError(
                f"label_with_glm52: only {len(labeled)}/{len(batch)} of a chunk succeeded — "
                "treating this as an unhealthy server rather than dropping the remaining rows. "
                f"Completed chunks are checkpointed under {chunk_dir}; rerun to resume."
            )
        if labeled:
            path = f"{chunk_dir}/part-{start // chunk_size:05d}-{uuid.uuid4().hex[:8]}.parquet"
            with StoragePath(path).open("wb") as handle:
                pq.write_table(pa.Table.from_pylist(labeled), handle)
            written += len(labeled)
        logger.info("label_with_glm52: %d/%d attempted, %d labeled so far", start + len(batch), len(pending), written)
    for sample in rejects[:MAX_LOGGED_REJECTS]:
        logger.warning("label_with_glm52: unusable reply: %s", sample)

    tables = []
    for path in StoragePath(f"{chunk_dir}/*.parquet").glob():
        with StoragePath(str(path)).open("rb") as handle:
            tables.append(pq.read_table(handle))
    table = pa.concat_tables(tables) if tables else pa.Table.from_pylist([])
    stats = LabelStats(labeled=table.num_rows, dropped=len(rows) - table.num_rows)
    stats.log()
    return table, stats


def run(
    *,
    label_set: str,
    out: str,
    server: ServerConfig,
    concurrency: int,
    run_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    object_store_endpoint: str | None = None,
    fleet_name: str = "h100",
) -> None:
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("labeling must run inside an Iris job so it can submit the GLM-5.2 server")

    rows = _read_label_set(label_set)
    logger.info("label_with_glm52: %d documents to label", len(rows))

    # Endpoint names are cluster-wide, so a repeat run must not collide with a
    # server another run already registered.
    vllm_endpoint = f"{VLLM_ENDPOINT}-{run_id}"
    ray_endpoint = f"{RAY_ENDPOINT}-{run_id}"
    launch = Glm52LaunchConfig(
        vllm_endpoint,
        ray_endpoint,
        server,
        fleet=FLEETS[fleet_name],
        object_store_endpoint=object_store_endpoint,
    )
    vllm_job = submit_glm52(ctx, launch)
    try:
        base_url = wait_for_endpoint_url(vllm_endpoint, vllm_job)
        logger.info("label_with_glm52: endpoint ready at %s", base_url)
        client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")
        table, _ = label_with_checkpoints(client, rows, concurrency=concurrency, chunk_size=chunk_size, out=out)
    finally:
        # Never let a teardown failure mask the real error, but never leave a
        # two-node GPU gang running either.
        try:
            vllm_job.terminate()
        except Exception:
            logger.warning("label_with_glm52: failed to terminate the GLM-5.2 server job", exc_info=True)

    with StoragePath(out).open("wb") as handle:
        pq.write_table(table, handle)
    logger.info("label_with_glm52: wrote %d labels to %s", table.num_rows, out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-set", required=True, help="parquet of documents to label (source/id/text)")
    parser.add_argument("--out", required=True, help="parquet path for the oracle labels")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="documents per checkpoint")
    parser.add_argument("--fleet", choices=sorted(FLEETS), default="h100", help="GPU shape to serve on")
    parser.add_argument("--run-id", required=True, help="unique tag for this run's server endpoints")
    parser.add_argument(
        "--object-store-endpoint",
        default=None,
        help=(
            "S3 endpoint for the serving task (e.g. https://cwobject.com). Required when the "
            "weight cache is in a different region than the GPUs, which makes the pod's "
            "node-local LOTA endpoint unable to read it."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    run(
        label_set=args.label_set,
        out=args.out,
        server=ServerConfig(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs),
        concurrency=args.concurrency,
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        object_store_endpoint=args.object_store_endpoint,
        fleet_name=args.fleet,
    )


if __name__ == "__main__":
    main()
