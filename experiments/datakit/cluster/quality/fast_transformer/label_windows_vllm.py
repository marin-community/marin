# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label bme grading windows with a self-hosted GLM-5.2 vLLM gang.

The grading contract is :mod:`label_windows_openrouter`'s exactly — one window
per request as ``<document index="0">``, the v2 rubric system prompt, the
bracketed position notice on middle/end windows, temperature 0, one-JSON-object
verdict — with the serving path swapped from OpenRouter to
:mod:`label_with_glm52`'s self-hosting pattern: this driver runs as an Iris job,
submits the two-node GLM-5.2-FP8 server as a child, resolves its endpoint by
name, and terminates it on the way out.

The campaign fans the window set across independent driver jobs, each with its
own server gang. ``--partition``/``--num-partitions`` assign rows by a
deterministic hash of the document id, so partitions are disjoint, stable across
resubmissions, and keep a document's windows together. Chunked checkpoints under
``<out>.chunks/`` resume by ``(id, window)``; a partition whose windows are all
checkpointed already skips bringing up a server at all.
"""

import argparse
import hashlib
import json
import logging
import os
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from iris.client import iris_ctx
from iris.rpc import job_pb2
from openai import OpenAI
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3, configure_fsspec_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.label_windows_openrouter import (
    DEFAULT_LABEL_BATCH,
    _chunk_dir,
    _labeled_keys,
    _read_windows,
    consolidate_chunks,
    label_with_checkpoints,
    window_key,
    window_user_content,
)
from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import (
    FLEETS,
    MAX_LOGGED_REJECTS,
    MAX_OUTPUT_TOKENS,
    RAY_ENDPOINT,
    VLLM_ENDPOINT,
    _parse_verdict,
)
from experiments.datakit.cluster.quality.fast_transformer.rubric import SYSTEM_PROMPT
from experiments.datakit.cluster.quality.glm52_vllm import (
    MODEL,
    Glm52LaunchConfig,
    ServerConfig,
    submit_glm52,
    wait_for_endpoint_url,
)

logger = logging.getLogger(__name__)

# A 512-gemma-token window plus the system prompt stays well under 4k GLM tokens,
# and the 4k reasoning budget sits on top: 8k holds both with headroom while
# giving each sequence half the KV cache of the whole-document labeler's 16k.
DEFAULT_MAX_MODEL_LEN = 8_192
# 192 sequences measured 3.96 docs/s on this fleet at 10.5k-char prompts; windows
# are ~5x shorter, so 192 keeps the queue full without starving KV cache.
DEFAULT_MAX_NUM_SEQS = 192
DEFAULT_CONCURRENCY = 160
DEFAULT_CHUNK_SIZE = 1_000


def partition_index(doc_id: str, num_partitions: int) -> int:
    """Stable partition of a document id (Python's ``hash`` is salted per process)."""
    digest = hashlib.sha256(doc_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_partitions


def _expand_globs(patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        if "*" not in pattern:
            paths.append(pattern)
            continue
        matches = sorted(str(m) for m in StoragePath(pattern).glob())
        if not matches:
            raise ValueError(f"no parquet files match {pattern}")
        paths.extend(matches)
    return paths


def label_window(client: OpenAI, row: dict, rejects: list[str]) -> dict | None:
    """One window's verdict plus its token usage, or ``None`` if unusable."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": window_user_content(row)},
            ],
        )
    except Exception as failure:  # one bad request must not abort the partition
        logger.warning("label_windows_vllm: request failed for %s: %r", window_key(row), failure)
        return None
    choice = response.choices[0]
    content = choice.message.content or ""
    verdict = _parse_verdict(content)
    if verdict is None:
        if len(rejects) < MAX_LOGGED_REJECTS:
            rejects.append(f"finish_reason={choice.finish_reason} reply={content[-400:]!r}")
        return None
    usage = response.usage
    return verdict | {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


def label_rows(
    client: OpenAI, rows: Sequence[dict], concurrency: int, label_batch: str, rejects: list[str]
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        verdicts = list(pool.map(lambda r: label_window(client, r, rejects), rows))
    return [
        {
            "id": row["id"],
            "source": row["source"],
            "window": row["window"],
            "token_start": row["token_start"],
            "token_end": row["token_end"],
            "text": row["text"],
            "content_type": verdict["content_type"],
            "valid": verdict["valid"],
            "quality": verdict["quality"],
            "score_normalized": (verdict["quality"] - 1) / 4.0,
            "why": verdict["why"],
            "label_batch": label_batch,
            "prompt_tokens": verdict["prompt_tokens"],
            "completion_tokens": verdict["completion_tokens"],
        }
        for row, verdict in zip(rows, verdicts, strict=True)
        if verdict is not None
    ]


def run(
    *,
    windows: list[str],
    out: str,
    server: ServerConfig,
    run_id: str,
    partition: int,
    num_partitions: int,
    concurrency: int,
    chunk_size: int,
    label_batch: str,
    limit: int | None,
    fleet_name: str,
    object_store_endpoint: str | None,
) -> None:
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("window labeling must run inside an Iris job so it can submit the GLM-5.2 server")

    rows = [r for r in _read_windows(_expand_globs(windows)) if partition_index(r["id"], num_partitions) == partition]
    # A stable shuffle so --limit draws a representative smoke rather than one
    # shard's head, and so a full run's chunks mix sources evenly.
    random.Random(0).shuffle(rows)
    if limit:
        rows = rows[:limit]
    logger.info("label_windows_vllm: partition %d/%d holds %d windows", partition, num_partitions, len(rows))

    pending = [r for r in rows if window_key(r) not in _labeled_keys(_chunk_dir(out))]
    if not pending:
        logger.info("label_windows_vllm: nothing pending, consolidating existing checkpoints")
        table = consolidate_chunks(_chunk_dir(out))
    else:
        vllm_endpoint = f"{VLLM_ENDPOINT}-{run_id}"
        launch = Glm52LaunchConfig(
            vllm_endpoint,
            f"{RAY_ENDPOINT}-{run_id}",
            server,
            fleet=FLEETS[fleet_name],
            object_store_endpoint=object_store_endpoint,
            # The campaign runs both driver and server at interactive priority:
            # a batch-band server behind an interactive driver would hold the
            # driver's slot while its own gang waits at the back of the queue.
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        )
        vllm_job = submit_glm52(ctx, launch)
        try:
            base_url = wait_for_endpoint_url(vllm_endpoint, vllm_job)
            logger.info("label_windows_vllm: endpoint ready at %s", base_url)
            client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")

            def label_chunk(batch: Sequence[dict], rejects: list[str]) -> list[dict]:
                return label_rows(client, batch, concurrency, label_batch, rejects)

            table = label_with_checkpoints(label_chunk, rows, chunk_size=chunk_size, out=out)
        finally:
            # Never let a teardown failure mask the real error, but never leave
            # a two-node GPU gang running either.
            try:
                vllm_job.terminate()
            except Exception:
                logger.warning("label_windows_vllm: failed to terminate the GLM-5.2 server job", exc_info=True)

    with StoragePath(out).open("wb") as fh:
        pq.write_table(table, fh)
    stats = {
        "attempted": len(rows),
        "labeled": table.num_rows,
        "dropped": len(rows) - table.num_rows,
        "prompt_tokens": sum(table.column("prompt_tokens").to_pylist()) if table.num_rows else 0,
        "completion_tokens": sum(table.column("completion_tokens").to_pylist()) if table.num_rows else 0,
    }
    with StoragePath(f"{out.removesuffix('.parquet')}_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("label_windows_vllm: %s", json.dumps(stats))
    logger.info("label_windows_vllm: wrote %d labels to %s", table.num_rows, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, nargs="+", help="windows parquet path(s) or glob(s)")
    parser.add_argument("--out", required=True, help="parquet path for this partition's window labels")
    parser.add_argument("--partition", type=int, required=True, help="which id-hash partition this driver labels")
    parser.add_argument("--num-partitions", type=int, default=8)
    parser.add_argument("--run-id", required=True, help="unique tag for this run's server endpoints")
    parser.add_argument("--label-batch", default=DEFAULT_LABEL_BATCH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="windows per checkpoint")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N windows (smoke runs)")
    # gb200 is the shape this stack is verified on; the H100 path currently fails
    # in FlashInfer's SM90 JIT link.
    parser.add_argument("--fleet", choices=sorted(FLEETS), default="gb200", help="GPU shape to serve on")
    parser.add_argument(
        "--object-store-endpoint",
        default=None,
        help=(
            "S3 endpoint for both this driver and the serving task (e.g. https://cwobject.com). "
            "Required when the windows, labels, or weight cache live in a different region than "
            "the GPUs: the pod's node-local LOTA endpoint cannot read cross-region buckets."
        ),
    )
    args = parser.parse_args()
    configure_logging(logging.INFO)
    if not 0 <= args.partition < args.num_partitions:
        raise ValueError(f"--partition {args.partition} outside [0, {args.num_partitions})")
    if args.object_store_endpoint:
        # The pod arrives with FSSPEC_S3/AWS_ENDPOINT_URL pointing at its
        # node-local LOTA endpoint; rebuild the process-wide config against the
        # external endpoint so cross-region reads and writes work.
        os.environ.pop("FSSPEC_S3", None)
        os.environ.pop("AWS_ENDPOINT_URL", None)
        configure_fsspec_s3(args.object_store_endpoint)
    else:
        configure_coreweave_s3()
    run(
        windows=args.windows,
        out=args.out,
        server=ServerConfig(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs),
        run_id=args.run_id,
        partition=args.partition,
        num_partitions=args.num_partitions,
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        label_batch=args.label_batch,
        limit=args.limit,
        fleet_name=args.fleet,
        object_store_endpoint=args.object_store_endpoint,
    )


if __name__ == "__main__":
    main()
