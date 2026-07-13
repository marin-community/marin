# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label sampled documents with the Claude quality oracle.

Reads the parquet emitted by :mod:`sample`, dispatches each row to the
oracle (claude-sonnet-4-6 by default) in parallel, and writes a labeled
parquet:

    source             : string  -- registry name
    id                 : string  -- canonical document id
    partition_id       : int64
    text               : string  -- input text (truncated to sample's MAX_DOC_CHARS)
    score_raw          : int64   -- 1..5 from rubric; -1 if unparseable
    score_normalized   : float64 -- (raw - 1) / 4 in [0, 1]; NaN if unparseable
    rationale          : string  -- one-line model justification (or error)
    input_tokens       : int64   -- billed non-cached input
    cache_read_tokens  : int64   -- billed cache-read input
    cache_write_tokens : int64   -- billed cache-creation input
    output_tokens      : int64
    cost_usd           : float64
    oracle_model       : string

Resume-safe: every successful (or unparseable) call is appended as one
JSON line to ``<output>.partial.jsonl``. Restarting skips already-scored
ids and resumes from where the previous run stopped.

Budget-safe: each completed call's cost is added to a running tally; once
the tally exceeds ``--budget-usd``, no further requests are dispatched.
The system prompt is cache-marked so the bulk of requests cost only
cached-input + output rates after warmup.

Submit:

    uv run iris --cluster=marin job run --no-wait --cpu=2 --memory=4G \\
        --extra=cpu --priority production --region europe-west4 \\
        --job-name "llm-quality-score-$(date +%Y%m%d-%H%M%S)" \\
        --env-file .marin.yaml -- \\
        python -m experiments.datakit.cluster.quality.v0.score \\
          --input  gs://marin-eu-west4/datakit/llm-quality-classifier/samples/train-n7000-seed42.parquet \\
          --output gs://marin-eu-west4/datakit/llm-quality-classifier/scored/train-n7000-seed42-sonnet46.parquet \\
          --budget-usd 45  # leave $5 for the held-out eval sample
"""

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.v0.rubric import (
    DEFAULT_ORACLE_MODEL,
    PRICING_PER_MTOK,
    SYSTEM_PROMPT,
    build_user_message,
    parse_response,
    read_anthropic_key,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_WORKERS = 12
DEFAULT_BUDGET_USD = 50.0
DEFAULT_MAX_OUTPUT_TOKENS = 200

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("partition_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("score_raw", pa.int64()),
        pa.field("score_normalized", pa.float64()),
        pa.field("rationale", pa.string()),
        pa.field("input_tokens", pa.int64()),
        pa.field("cache_read_tokens", pa.int64()),
        pa.field("cache_write_tokens", pa.int64()),
        pa.field("output_tokens", pa.int64()),
        pa.field("cost_usd", pa.float64()),
        pa.field("oracle_model", pa.string()),
    ]
)


@dataclass
class ScoredRow:
    source: str
    id: str
    partition_id: int
    text: str
    score_raw: int
    score_normalized: float
    rationale: str
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: float
    oracle_model: str


def _exact_cost(model: str, input_tokens: int, cache_read: int, cache_write: int, output_tokens: int) -> float:
    """Anthropic pricing including cache surcharges.

    Cache writes are billed at 1.25x the input rate (cache_creation_input_tokens);
    cache reads at 0.1x (the table's ``input_cached``); plain input and output
    at their normal rates.
    """
    price = PRICING_PER_MTOK[model]
    return (
        input_tokens * price["input"] / 1_000_000
        + cache_write * price["input"] * 1.25 / 1_000_000
        + cache_read * price["input_cached"] / 1_000_000
        + output_tokens * price["output"] / 1_000_000
    )


def _score_one(
    client, model: str, max_output_tokens: int, source: str, doc_id: str, partition_id: int, text: str
) -> ScoredRow:
    """One blocking Anthropic call. Returns a ScoredRow even on parse failure."""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": build_user_message(text)}],
        )
    except Exception as exc:
        return ScoredRow(
            source=source,
            id=doc_id,
            partition_id=partition_id,
            text=text,
            score_raw=-1,
            score_normalized=float("nan"),
            rationale=f"api error: {type(exc).__name__}: {exc}",
            input_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            oracle_model=model,
        )

    body = resp.content[0].text if resp.content else ""
    parsed = parse_response(body)
    usage = resp.usage
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cost = _exact_cost(model, input_tokens, cache_read, cache_write, output_tokens)
    return ScoredRow(
        source=source,
        id=doc_id,
        partition_id=partition_id,
        text=text,
        score_raw=parsed.score_raw,
        score_normalized=parsed.score_normalized,
        rationale=parsed.rationale,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=output_tokens,
        cost_usd=cost,
        oracle_model=model,
    )


def _load_partial(partial_path: str) -> dict[str, ScoredRow]:
    """Load already-scored rows keyed by ``(source, id)``-equivalent string id."""
    path = StoragePath(partial_path)
    if not path.exists():
        return {}
    out: dict[str, ScoredRow] = {}
    with path.open("rb") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = ScoredRow(**obj)
            out[f"{row.source}\t{row.id}"] = row
    logger.info("resume: loaded %d already-scored rows from %s", len(out), partial_path)
    return out


def _append_partial(partial_path: str, row: ScoredRow, lock: threading.Lock) -> None:
    line = json.dumps(asdict(row)) + "\n"
    with lock:
        with open_url(partial_path, "ab") as fh:
            fh.write(line.encode("utf-8"))


def _read_sample(input_path: str) -> list[dict[str, str | int]]:
    with StoragePath(input_path).open("rb") as fh:
        table = pq.read_table(fh)
    rows: list[dict[str, str | int]] = []
    sources = table.column("source").to_pylist()
    ids = table.column("id").to_pylist()
    pids = table.column("partition_id").to_pylist()
    texts = table.column("text").to_pylist()
    for s, i, p, t in zip(sources, ids, pids, texts, strict=True):
        rows.append({"source": str(s), "id": str(i), "partition_id": int(p), "text": str(t)})
    return rows


def _write_final_parquet(rows: list[ScoredRow], output_path: str) -> None:
    table = pa.Table.from_pylist([asdict(r) for r in rows], schema=_OUTPUT_SCHEMA)
    output = StoragePath(output_path)
    if output.parent.segments:
        output.parent.mkdirs()
    with output.open("wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def score(
    *,
    input_path: str,
    output_path: str,
    oracle_model: str,
    budget_usd: float,
    max_workers: int,
    max_output_tokens: int,
) -> None:
    if oracle_model not in PRICING_PER_MTOK:
        raise ValueError(f"unknown oracle model {oracle_model!r}; add it to rubric.PRICING_PER_MTOK")

    from anthropic import Anthropic  # noqa: PLC0415  # optional dep: anthropic

    client = Anthropic(api_key=read_anthropic_key())

    sample_rows = _read_sample(input_path)
    logger.info("input: %d sampled docs from %s", len(sample_rows), input_path)

    partial_path = output_path + ".partial.jsonl"
    already = _load_partial(partial_path)

    pending = [r for r in sample_rows if f"{r['source']}\t{r['id']}" not in already]
    logger.info("pending: %d (skipping %d already-scored)", len(pending), len(already))

    total_cost = sum(r.cost_usd for r in already.values())
    cost_lock = threading.Lock()
    file_lock = threading.Lock()
    budget_hit = threading.Event()
    n_done = 0
    n_parse_fail = 0
    n_api_fail = 0

    t0 = time.monotonic()

    def _submit(r):
        if budget_hit.is_set():
            return None
        return _score_one(
            client=client,
            model=oracle_model,
            max_output_tokens=max_output_tokens,
            source=r["source"],
            doc_id=r["id"],
            partition_id=r["partition_id"],
            text=r["text"],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_submit, r): r for r in pending}
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            _append_partial(partial_path, row, file_lock)
            already[f"{row.source}\t{row.id}"] = row
            n_done += 1
            if row.score_raw < 0:
                if "api error" in row.rationale:
                    n_api_fail += 1
                else:
                    n_parse_fail += 1
            with cost_lock:
                total_cost += row.cost_usd
                if total_cost >= budget_usd and not budget_hit.is_set():
                    budget_hit.set()
                    logger.warning(
                        "BUDGET HIT: spent $%.4f >= $%.2f; not dispatching further calls",
                        total_cost,
                        budget_usd,
                    )

            if n_done % 50 == 0 or n_done == len(pending):
                elapsed = time.monotonic() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "progress: %d/%d (api_fail=%d parse_fail=%d) cost=$%.4f rate=%.2f/s",
                    n_done,
                    len(pending),
                    n_api_fail,
                    n_parse_fail,
                    total_cost,
                    rate,
                )

    final_rows = list(already.values())
    logger.info(
        "writing %d scored rows -> %s (total cost $%.4f, budget $%.2f)",
        len(final_rows),
        output_path,
        total_cost,
        budget_usd,
    )
    _write_final_parquet(final_rows, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Sample parquet (from sample.py)")
    parser.add_argument("--output", required=True, help="Output scored parquet path")
    parser.add_argument("--oracle-model", default=DEFAULT_ORACLE_MODEL, choices=sorted(PRICING_PER_MTOK))
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    score(
        input_path=args.input,
        output_path=args.output,
        oracle_model=args.oracle_model,
        budget_usd=args.budget_usd,
        max_workers=args.max_workers,
        max_output_tokens=args.max_output_tokens,
    )


if __name__ == "__main__":
    main()
