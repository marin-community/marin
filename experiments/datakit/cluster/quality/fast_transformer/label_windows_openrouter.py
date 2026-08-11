# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label bme grading windows with GLM-5.2 over OpenRouter.

The scale-up's grading contract is identical to the deployed labeler's
(:mod:`label_with_glm52`): one window per request, ``<document index="0">``, the
v2 rubric system prompt, temperature 0, and the same one-JSON-object verdict
parser — only the serving path changes, from a self-hosted vLLM gang to
``z-ai/glm-5.2`` on OpenRouter (the hosted equivalent of the original
``zai-org/GLM-5.2-FP8``). Single-window requests keep the contract checkable:
exactly one object, or the row is dropped and counted.

Reliability machinery is carried over: chunked sweeps checkpointed under
``<out>.chunks/`` with resume by ``(id, window)``, and a chunk that mostly fails
aborts the run rather than quietly dropping rows. Each row records the
OpenRouter-reported token counts and dollar cost, so a pilot measures realized
$/1k windows from the run's own output.
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import (
    MAX_LOGGED_REJECTS,
    MAX_OUTPUT_TOKENS,
    MIN_CHUNK_SUCCESS,
    _parse_verdict,
)
from experiments.datakit.cluster.quality.fast_transformer.rubric import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

OPENROUTER_MODEL = "z-ai/glm-5.2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ORACLE_KEY_VAR = "OR_KEY_SCALE_UP"
DEFAULT_LABEL_BATCH = "glm52_rubric_v2_bme_scaleup"
DEFAULT_CONCURRENCY = 128
DEFAULT_CHUNK_SIZE = 1_000
REQUEST_TIMEOUT = 300.0
MAX_ATTEMPTS = 6
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 520, 524)

WINDOW_COLUMNS = ["id", "source", "window", "token_start", "token_end", "text"]

# Framing notices for windows cut from inside a document, phrased like the
# rubric's excerpt marker: harness slicing, not corpus damage. Without them the
# grader reads an abrupt start as truncation and marks genuine mid-document text
# invalid. Begin windows and short documents keep the original framing — their
# window starts where the document does.
WINDOW_NOTICES = {
    "middle": (
        "[This is a window from the MIDDLE of a longer document; it may begin and end "
        "mid-sentence. Judge the text shown on its own merits and never mark it invalid, "
        "or lower its quality, merely for starting or ending abruptly.]"
    ),
    "end": (
        "[This is a window from the END of a longer document; it may begin mid-sentence. "
        "Judge the text shown on its own merits and never mark it invalid, or lower its "
        "quality, merely for starting abruptly.]"
    ),
}


def window_key(row: dict) -> str:
    return f"{row['id']}\t{row['window']}"


def window_user_content(row: dict) -> str:
    """The request's user message: the window in the document tag, with the
    position notice above it for middle/end windows."""
    document = f'<document index="0">\n{row["text"]}\n</document>'
    notice = WINDOW_NOTICES.get(row["window"])
    return f"{notice}\n{document}" if notice else document


def ask_oracle(client: httpx.Client, model: str, row: dict) -> dict:
    """One window's raw completion JSON. Raises on anything worth another attempt."""
    response = client.post(
        "/chat/completions",
        json={
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "usage": {"include": True},
            # Cheapest provider first: the same 300-window batch measured $0.69/1k
            # on the default routing's cheap provider and $1.03/1k (list price) on
            # another, a 1.5x swing the campaign budget should not float on.
            "provider": {"sort": "price"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": window_user_content(row)},
            ],
        },
    )
    if response.status_code in RETRYABLE_STATUS:
        raise httpx.HTTPStatusError(f"retryable {response.status_code}", request=response.request, response=response)
    response.raise_for_status()
    return response.json()


def label_window(client: httpx.Client, model: str, row: dict, rejects: list[str]) -> dict | None:
    """The oracle's verdict on one window plus its usage, or ``None`` if unusable."""
    reply = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            reply = ask_oracle(client, model, row)
            break
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as failure:
            if attempt == MAX_ATTEMPTS - 1:
                logger.warning("label_windows: giving up on %s: %r", window_key(row), failure)
                return None
            # Jittered so a throttled fleet does not resynchronize on the next try.
            time.sleep(2**attempt + random.random())
    content = (reply.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    verdict = _parse_verdict(content)
    if verdict is None:
        if len(rejects) < MAX_LOGGED_REJECTS:
            finish = (reply.get("choices") or [{}])[0].get("finish_reason")
            rejects.append(f"finish_reason={finish} reply={content[-400:]!r}")
        return None
    usage = reply.get("usage") or {}
    return verdict | {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost": float(usage.get("cost") or 0.0),
        "provider": str(reply.get("provider") or ""),
    }


def label_rows(
    client: httpx.Client, model: str, rows: Sequence[dict], concurrency: int, label_batch: str, rejects: list[str]
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        verdicts = list(pool.map(lambda r: label_window(client, model, r, rejects), rows))
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
            "cost": verdict["cost"],
            "provider": verdict["provider"],
        }
        for row, verdict in zip(rows, verdicts, strict=True)
        if verdict is not None
    ]


def _read_windows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with StoragePath(path).open("rb") as fh:
            rows.extend(pq.ParquetFile(fh).read(columns=WINDOW_COLUMNS).to_pylist())
    return rows


def _chunk_dir(out: str) -> str:
    return f"{out.rstrip('/')}.chunks"


def _labeled_keys(chunk_dir: str) -> set[str]:
    done: set[str] = set()
    for path in StoragePath(f"{chunk_dir}/*.parquet").glob():
        with StoragePath(str(path)).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "window"])
        ids, windows = table.column("id").to_pylist(), table.column("window").to_pylist()
        done.update(f"{i}\t{w}" for i, w in zip(ids, windows, strict=True))
    return done


def label_with_checkpoints(
    client: httpx.Client,
    model: str,
    rows: Sequence[dict],
    *,
    concurrency: int,
    chunk_size: int,
    label_batch: str,
    out: str,
) -> pa.Table:
    """Label every window in chunks, checkpointing each chunk before the next."""
    chunk_dir = _chunk_dir(out)
    already = _labeled_keys(chunk_dir)
    if already:
        logger.info("label_windows: resuming, %d windows already labeled", len(already))
    pending = [r for r in rows if window_key(r) not in already]

    rejects: list[str] = []
    written = len(already)
    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        labeled = label_rows(client, model, batch, concurrency, label_batch, rejects)
        if len(labeled) < len(batch) * MIN_CHUNK_SUCCESS:
            raise RuntimeError(
                f"label_windows: only {len(labeled)}/{len(batch)} of a chunk succeeded — treating this as an "
                f"unhealthy upstream rather than dropping rows. Chunks are checkpointed under {chunk_dir}; "
                "rerun to resume."
            )
        if labeled:
            path = f"{chunk_dir}/part-{start // chunk_size:05d}-{uuid.uuid4().hex[:8]}.parquet"
            with StoragePath(path).open("wb") as fh:
                pq.write_table(pa.Table.from_pylist(labeled), fh)
            written += len(labeled)
        cost = sum(r["cost"] for r in labeled)
        logger.info(
            "label_windows: %d/%d attempted, %d labeled so far, chunk cost $%.4f",
            start + len(batch),
            len(pending),
            written,
            cost,
        )
    for sample in rejects[:MAX_LOGGED_REJECTS]:
        logger.warning("label_windows: unusable reply: %s", sample)

    tables = []
    for path in StoragePath(f"{chunk_dir}/*.parquet").glob():
        with StoragePath(str(path)).open("rb") as fh:
            tables.append(pq.read_table(fh))
    return pa.concat_tables(tables) if tables else pa.Table.from_pylist([])


def run_stats(table: pa.Table, attempted: int) -> dict:
    labeled = table.num_rows
    cost = sum(table.column("cost").to_pylist()) if labeled else 0.0
    return {
        "attempted": attempted,
        "labeled": labeled,
        "dropped": attempted - labeled,
        "total_cost": cost,
        "cost_per_1k_windows": 1000.0 * cost / labeled if labeled else None,
        "prompt_tokens": sum(table.column("prompt_tokens").to_pylist()) if labeled else 0,
        "completion_tokens": sum(table.column("completion_tokens").to_pylist()) if labeled else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, nargs="+", help="windows parquet path(s)")
    parser.add_argument("--out", required=True, help="parquet path for the window labels")
    parser.add_argument("--model", default=OPENROUTER_MODEL)
    parser.add_argument("--label-batch", default=DEFAULT_LABEL_BATCH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N windows (pilot runs)")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed applied before --limit")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    key = os.environ.get(ORACLE_KEY_VAR)
    if not key:
        raise ValueError(f"{ORACLE_KEY_VAR} is not set")

    rows = _read_windows(args.windows)
    # A stable shuffle so --limit draws a representative pilot rather than one shard's head.
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    logger.info("label_windows: %d windows to label with %s", len(rows), args.model)

    limits = httpx.Limits(max_connections=args.concurrency + 16, max_keepalive_connections=args.concurrency)
    with httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        limits=limits,
    ) as client:
        table = label_with_checkpoints(
            client,
            args.model,
            rows,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size,
            label_batch=args.label_batch,
            out=args.out,
        )

    with StoragePath(args.out).open("wb") as fh:
        pq.write_table(table, fh)
    stats = run_stats(table, len(rows))
    with StoragePath(f"{args.out.removesuffix('.parquet')}_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("label_windows: %s", json.dumps(stats, indent=2))
    logger.info("label_windows: wrote %d labels to %s", table.num_rows, args.out)


if __name__ == "__main__":
    main()
