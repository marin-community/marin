# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Can a cross-provider panel detect quality-bucket structure in the 88k join at all?

Three one-pool intruder detection runs over one shared subsample of the
``glm52_labels_88k`` x harrier-50M join, one per quality-label source:

* ``v3`` — scores from the deployed ``pooled_glm52_v3`` fast-transformer,
  banded into per-source quintiles;
* ``glm52`` — the GLM-5.2 oracle labels already in the join (``glm52_quality``);
* ``luna`` — fresh ``openai/gpt-5.6-luna`` labels generated here, on the
  subsample only, under the same v2 rubric that produced the GLM-5.2 labels.

Unlike :mod:`~experiments.datakit.cluster.quality.fast_transformer.intruder_ab`,
this is not an A/B between two bucketings: each run buckets the *same* documents
by one source's quality signal and asks whether the panel's detection rate
excludes the 1/5 chance floor, with a single anytime-valid Robbins confidence
sequence per run. Trials are stratified on ``glm52_source`` so the in-group and
the intruder always share a source and differ only in assigned quality.

Four stages, each skipped when its output already exists on storage, so the
pipeline is driven as independent Iris jobs (``luna`` and ``score`` can run
concurrently once ``subsample`` is done; the three ``panel`` runs can run
concurrently once their labels exist):

``subsample``
    Draws one deterministic ~2.5k-document subsample: sources with the fullest
    coverage across the five GLM quality levels, up to ``PER_BUCKET`` documents
    per (source, quality) cell, within ``DOC_BUDGET`` total.
``luna``
    Labels the subsample with gpt-5.6-luna over OpenRouter, one document per
    request under the v2 rubric — the contract that made the GLM-5.2 labels
    checkable — with checkpointed resume and a hard spend guard.
``score``
    Scores the subsample with ``pooled_glm52_v3`` (``score_bme``, the deployed
    begin/middle/end mean pooling). CPU-only.
``panel --label-source {v3,glm52,luna}``
    Builds that source's bucket pool and runs the detection test with the
    terra / sonnet / flash cross-provider panel, writing a results JSON.

All stages read the OpenRouter key from the same ``OR_INTRUDER_key`` env var the
panel uses.
"""

import argparse
import asyncio
import json
import logging
import math
import random
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from os import environ

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.intruder import (
    CHANCE_LEVEL,
    IN_GROUP_COUNT,
    OPENROUTER_KEY_ENV,
    Bucket,
    BucketPool,
    ConfidenceSequence,
    Panelist,
    _score_round,
    openrouter_panel,
)
from experiments.datakit.cluster.quality.fast_transformer.intruder_ab import _domain_quantile_buckets
from experiments.datakit.cluster.quality.fast_transformer.label_with_glm52 import (
    PROMPT_TEXT_CHARS,
    _parse_verdict,
)
from experiments.datakit.cluster.quality.fast_transformer.rubric import SYSTEM_PROMPT
from experiments.datakit.cluster.quality.fast_transformer.sample_labels import excerpt
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

JOINED_ROOT = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
V3_MODEL_DIR = "s3://marin-us-east-02a/marin/user/rav/quality_v2/models/pooled_glm52_v3"
OUT_PREFIX = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/intruder_88k_detect"
SUBSAMPLE_PATH = f"{OUT_PREFIX}/subsample.parquet"
LUNA_LABELS_PATH = f"{OUT_PREFIX}/luna_labels.parquet"
LUNA_SPEND_PATH = f"{OUT_PREFIX}/luna_spend.json"
V3_SCORES_PATH = f"{OUT_PREFIX}/v3_scores.parquet"
RESULTS_PREFIX = f"{OUT_PREFIX}/results"

LABEL_SOURCES = ("v3", "glm52", "luna")
SEED = 42
# Documents per (source, quality) cell, matching the intruder_ab runs. Four is the
# floor a trial needs; 64 leaves room for distinct trials.
PER_BUCKET = 64
N_QUALITY_BANDS = 5
# A cell must hold at least this many documents to count toward a source's
# coverage: twice the trial floor, so the pool never scrapes the minimum.
MIN_CELL_DOCS = 8
# Total subsample budget: 8 full sources (5 cells x 64 docs). Every document here
# is one Luna labeling request, so the budget is a spend cap, not a nicety.
DOC_BUDGET = 2_560

LUNA_MODEL = "openai/gpt-5.6-luna"
LUNA_CONCURRENCY = 64
LUNA_MAX_ATTEMPTS = 6
LUNA_TIMEOUT = 300.0
# Reasoning counts against the completion budget; a tight cap truncates the reply
# before the verdict JSON (the same failure MAX_OUTPUT_TOKENS=512 caused GLM-5.2).
LUNA_MAX_COMPLETION_TOKENS = 8_000
# Hard stop on labeling spend, from per-response OpenRouter cost accounting. The
# projection is ~$2 at current pricing; hitting this means the estimate was wrong
# by 5x and the run must stop rather than eat the panel budget.
LUNA_BUDGET_USD = 10.0
LUNA_CHECKPOINT_EVERY = 500

PANEL_MODELS = ("openai/gpt-5.6-terra", "anthropic/claude-sonnet-5", "google/gemini-3.5-flash")
ALPHA = 0.05
MIN_TRIALS = 40
DEFAULT_MAX_TRIALS = 150
BATCH_SIZE = 8
TARGET_TRIALS = 150
MAX_DOC_CHARS = 8_000
MAX_WORKERS = 16

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"


# ---------------------------------------------------------------------------
# Shared I/O
# ---------------------------------------------------------------------------


def _walk_parquet(root: str, max_depth: int = 5) -> list[str]:
    """Every ``*.parquet`` under ``root`` via single-level globs (a recursive glob
    HeadObjects the prefix, which the CW store answers with a 400). Copied from
    ``embed_exp`` rather than imported: that module drags in the training loop,
    which none of these stages needs."""
    shards: list[str] = []
    dirs = [root.rstrip("/")]
    for _ in range(max_depth):
        next_dirs: list[str] = []
        for d in dirs:
            for entry in sorted(str(m) for m in StoragePath(f"{d}/*").glob()):
                if entry.endswith(".parquet"):
                    shards.append(entry)
                else:
                    next_dirs.append(entry)
        dirs = next_dirs
        if not dirs:
            break
    return shards


def _read_join_columns(columns: list[str]) -> dict[str, list]:
    """The requested columns of every join row, deduplicated by id (first wins).

    The join holds 80,897 rows over 79,335 distinct ids; shards are visited in
    sorted order, so which duplicate wins is deterministic.
    """
    shards = _walk_parquet(f"{JOINED_ROOT}/outputs")
    if not shards:
        raise ValueError(f"no parquet shards under {JOINED_ROOT}/outputs")
    out: dict[str, list] = {c: [] for c in columns}
    seen: set[str] = set()
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=columns)
        rows = {c: table.column(c).to_pylist() for c in columns}
        for i, doc_id in enumerate(rows["id"]):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            for c in columns:
                out[c].append(rows[c][i])
    logger.info("read %d distinct rows from %d shards", len(out["id"]), len(shards))
    return out


def _read_parquet(path: str, columns: list[str] | None = None) -> dict[str, list]:
    with StoragePath(path).open("rb") as fh:
        table = pq.read_table(fh, columns=columns)
    return {c: table.column(c).to_pylist() for c in table.column_names}


def _write_parquet(path: str, table: pa.Table) -> None:
    with StoragePath(path).open("wb") as fh:
        pq.write_table(table, fh)


def _write_json(path: str, payload: dict) -> None:
    with StoragePath(path).open("w") as fh:
        fh.write(json.dumps(payload, indent=2))


def _openrouter_usage() -> float | None:
    """Lifetime usage in USD on the configured key, or ``None`` if unreadable.

    Telemetry only: a credits-endpoint hiccup after a finished (paid-for) run must
    not discard the run, so this is the one place a network error is absorbed.
    """
    try:
        response = requests.get(
            OPENROUTER_CREDITS_URL,
            headers={"Authorization": f"Bearer {environ[OPENROUTER_KEY_ENV]}"},
            timeout=30,
        )
        response.raise_for_status()
        return float(response.json()["data"]["total_usage"])
    except requests.RequestException:
        logger.warning("could not read OpenRouter usage", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Stage: subsample
# ---------------------------------------------------------------------------


def _select_sources(cell_sizes: dict[tuple[str, int], int]) -> list[str]:
    """Sources whose (source, quality) cells the subsample will draw from.

    Prefers sources covering more of the five quality levels (each eligible cell
    holds >= ``MIN_CELL_DOCS`` documents), then larger ones, and adds greedily
    while the total draw stays within ``DOC_BUDGET``. A source needs >= 2 eligible
    levels — a lone bucket has no intruder to draw against.
    """
    by_source: dict[str, dict[int, int]] = {}
    for (source, quality), n in cell_sizes.items():
        by_source.setdefault(source, {})[quality] = n

    ranked = []
    for source, cells in by_source.items():
        eligible = {q: n for q, n in cells.items() if n >= MIN_CELL_DOCS}
        if len(eligible) < 2:
            continue
        contribution = sum(min(PER_BUCKET, n) for n in eligible.values())
        ranked.append((-len(eligible), -contribution, source, contribution))
    ranked.sort()

    chosen: list[str] = []
    total = 0
    for _neg_levels, _neg_contribution, source, contribution in ranked:
        if total + contribution > DOC_BUDGET:
            continue
        chosen.append(source)
        total += contribution
    if not chosen:
        raise ValueError("no source has >= 2 quality levels with enough documents")
    logger.info("subsample: %d sources, %d documents planned", len(chosen), total)
    return chosen


def build_subsample() -> None:
    """Draw the shared subsample and write it to ``SUBSAMPLE_PATH``."""
    if StoragePath(SUBSAMPLE_PATH).exists():
        logger.info("subsample already exists at %s", SUBSAMPLE_PATH)
        return
    meta = _read_join_columns(["id", "glm52_source", "glm52_quality"])
    cells: dict[tuple[str, int], list[str]] = {}
    for doc_id, source, quality in zip(meta["id"], meta["glm52_source"], meta["glm52_quality"], strict=True):
        cells.setdefault((str(source), int(quality)), []).append(doc_id)

    sources = set(_select_sources({cell: len(ids) for cell, ids in cells.items()}))
    rng = np.random.default_rng(SEED)
    chosen: dict[str, tuple[str, int]] = {}  # id -> (source, quality)
    for (source, quality), ids in sorted(cells.items()):
        if source not in sources or len(ids) < MIN_CELL_DOCS:
            continue
        take = rng.permutation(len(ids))[: min(PER_BUCKET, len(ids))]
        for i in take:
            chosen[ids[i]] = (source, quality)

    # Second pass for text only: holding every join row's text in memory to then
    # keep 3% of it is what the two-pass read avoids.
    texts: dict[str, str] = {}
    for shard in _walk_parquet(f"{JOINED_ROOT}/outputs"):
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "text"])
        for doc_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True):
            if doc_id in chosen and doc_id not in texts:
                texts[doc_id] = text
        if len(texts) == len(chosen):
            break

    ids = sorted(chosen)
    table = pa.table(
        {
            "id": ids,
            "text": [texts[i] for i in ids],
            "glm52_source": [chosen[i][0] for i in ids],
            "glm52_quality": [chosen[i][1] for i in ids],
        }
    )
    _write_parquet(SUBSAMPLE_PATH, table)
    logger.info("subsample: wrote %d documents to %s", table.num_rows, SUBSAMPLE_PATH)


# ---------------------------------------------------------------------------
# Stage: luna
# ---------------------------------------------------------------------------

LUNA_RETRYABLE_STATUS = (408, 429, 500, 502, 503, 520, 524)
LUNA_FAILURES = (httpx.HTTPError, KeyError, IndexError, ValueError)


async def _luna_ask(client: httpx.AsyncClient, text: str) -> tuple[str, float]:
    """One rubric request; returns (reply content, cost in USD). Raises on anything
    worth another attempt."""
    response = await client.post(
        "/chat/completions",
        json={
            "model": LUNA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'<document index="0">\n{excerpt(text, PROMPT_TEXT_CHARS)}\n</document>'},
            ],
            "temperature": 0.0,
            "max_tokens": LUNA_MAX_COMPLETION_TOKENS,
            # The effort the 100k PDF oracle sample ran gpt-5.6-luna at.
            "reasoning": {"effort": "medium"},
            "usage": {"include": True},
        },
    )
    if response.status_code in LUNA_RETRYABLE_STATUS:
        raise httpx.HTTPStatusError(f"retryable {response.status_code}", request=response.request, response=response)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise ValueError(f"OpenRouter error: {json.dumps(body['error'])[:200]}")
    content = body["choices"][0]["message"]["content"] or ""
    return content, float(body.get("usage", {}).get("cost") or 0.0)


async def _luna_label_one(client: httpx.AsyncClient, row: dict) -> tuple[dict | None, float]:
    """One document's verdict row (or ``None`` for an unusable reply) plus its cost.

    Retries are jittered exponential backoff, mirroring the 100k oracle sample run;
    a document that never answers is dropped and retried by the next run, which
    resumes from the checkpoint chunks.
    """
    cost = 0.0
    for attempt in range(LUNA_MAX_ATTEMPTS):
        if attempt:
            await asyncio.sleep(2 ** (attempt - 1) + random.random())
        try:
            content, call_cost = await _luna_ask(client, row["text"])
        except LUNA_FAILURES as failure:
            if attempt == LUNA_MAX_ATTEMPTS - 1:
                logger.warning("luna: giving up on %s: %r", row["id"], failure)
                return None, cost
            continue
        cost += call_cost
        verdict = _parse_verdict(content)
        if verdict is None:
            logger.warning("luna: unusable reply for %s: %r", row["id"], content[-200:])
            return None, cost
        return {
            "id": row["id"],
            "quality": verdict["quality"],
            "content_type": verdict["content_type"],
            "valid": verdict["valid"],
            "label_batch": "luna_rubric_v2",
        }, cost
    raise AssertionError("unreachable: the loop either returns or continues to the last attempt")


async def _luna_label_all(rows: Sequence[dict], chunk_dir: str) -> float:
    """Label every row, ``LUNA_CONCURRENCY`` in flight, checkpointing as verdicts
    land. Returns total spend; raises once spend crosses ``LUNA_BUDGET_USD``."""
    done: list[dict] = []
    written = 0
    total_cost = 0.0

    def flush() -> None:
        nonlocal written
        if len(done) == written:
            return
        path = f"{chunk_dir}/part-{uuid.uuid4().hex[:12]}.parquet"
        _write_parquet(path, pa.Table.from_pylist(done[written:]))
        written = len(done)

    queue = iter(rows)
    limits = httpx.Limits(max_connections=LUNA_CONCURRENCY + 8, max_keepalive_connections=LUNA_CONCURRENCY)
    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {environ[OPENROUTER_KEY_ENV]}"},
        timeout=httpx.Timeout(LUNA_TIMEOUT),
        limits=limits,
    ) as client:

        async def worker() -> None:
            nonlocal total_cost
            for row in queue:
                verdict, cost = await _luna_label_one(client, row)
                total_cost += cost
                if total_cost > LUNA_BUDGET_USD:
                    flush()
                    raise RuntimeError(
                        f"luna: spend ${total_cost:.2f} crossed the ${LUNA_BUDGET_USD} guard; "
                        f"completed labels are checkpointed under {chunk_dir}"
                    )
                if verdict is not None:
                    done.append(verdict)
                if len(done) - written >= LUNA_CHECKPOINT_EVERY:
                    flush()
                    logger.info("luna: %d labeled, ~$%.2f spent", written, total_cost)

        await asyncio.gather(*[worker() for _ in range(min(LUNA_CONCURRENCY, len(rows)))])
    flush()
    return total_cost


def label_with_luna() -> None:
    """Label the subsample with gpt-5.6-luna and write ``LUNA_LABELS_PATH``."""
    if StoragePath(LUNA_LABELS_PATH).exists():
        logger.info("luna labels already exist at %s", LUNA_LABELS_PATH)
        return
    sub = _read_parquet(SUBSAMPLE_PATH, ["id", "text"])
    rows = [{"id": i, "text": t} for i, t in zip(sub["id"], sub["text"], strict=True)]

    chunk_dir = f"{LUNA_LABELS_PATH}.chunks"
    already: set[str] = set()
    for chunk in StoragePath(f"{chunk_dir}/*.parquet").glob():
        already.update(_read_parquet(str(chunk), ["id"])["id"])
    if already:
        logger.info("luna: resuming, %d ids already labeled", len(already))
    pending = [r for r in rows if r["id"] not in already]

    cost = asyncio.run(_luna_label_all(pending, chunk_dir)) if pending else 0.0

    tables = [pa.Table.from_pydict(_read_parquet(str(c))) for c in StoragePath(f"{chunk_dir}/*.parquet").glob()]
    table = pa.concat_tables(tables).combine_chunks()
    labeled = table.num_rows
    if labeled < len(rows) * 0.9:
        raise RuntimeError(f"luna: only {labeled}/{len(rows)} documents labeled; rerun to retry the rest")
    _write_parquet(LUNA_LABELS_PATH, table)
    _write_json(LUNA_SPEND_PATH, {"model": LUNA_MODEL, "labeled": labeled, "requested": len(rows), "cost_usd": cost})
    logger.info("luna: wrote %d labels (%d dropped) to %s, ~$%.2f", labeled, len(rows) - labeled, LUNA_LABELS_PATH, cost)


# ---------------------------------------------------------------------------
# Stage: score
# ---------------------------------------------------------------------------


def score_with_v3() -> None:
    """Score the subsample with the deployed ``pooled_glm52_v3`` model."""
    if StoragePath(V3_SCORES_PATH).exists():
        logger.info("v3 scores already exist at %s", V3_SCORES_PATH)
        return
    sub = _read_parquet(SUBSAMPLE_PATH, ["id", "text"])
    scorer = load_pooled_scorer(V3_MODEL_DIR)
    scores = score_bme(scorer, sub["text"])
    _write_parquet(V3_SCORES_PATH, pa.table({"id": sub["id"], "v3_score": scores.astype(np.float32)}))
    logger.info("v3: wrote %d scores to %s", len(scores), V3_SCORES_PATH)


# ---------------------------------------------------------------------------
# Stage: panel
# ---------------------------------------------------------------------------


def _bucket_labels(label_source: str, sub: dict[str, list]) -> list[int]:
    """Each subsample document's quality bucket under ``label_source``.

    ``glm52`` and ``luna`` use the oracle's absolute 1-5 quality; a document the
    Luna run failed to label gets ``-1`` and is excluded. ``v3`` bands the model's
    continuous score into per-source quintiles (equal-count bands within each
    source), the same construction intruder_ab's domain-quantile bucketing uses.
    """
    if label_source == "glm52":
        return [int(q) for q in sub["glm52_quality"]]
    if label_source == "luna":
        labels = _read_parquet(LUNA_LABELS_PATH, ["id", "quality"])
        by_id = dict(zip(labels["id"], labels["quality"], strict=True))
        return [int(by_id.get(i, -1)) for i in sub["id"]]
    if label_source == "v3":
        scores = _read_parquet(V3_SCORES_PATH, ["id", "v3_score"])
        by_id = dict(zip(scores["id"], scores["v3_score"], strict=True))
        missing = [i for i in sub["id"] if i not in by_id]
        if missing:
            raise ValueError(f"v3 scores missing for {len(missing)} subsample ids (e.g. {missing[:3]})")
        return _domain_quantile_buckets([by_id[i] for i in sub["id"]], sub["glm52_source"], N_QUALITY_BANDS)
    raise ValueError(f"unknown label source {label_source!r}")


def build_pool(label_source: str) -> BucketPool:
    """The ``source|q<bucket>`` bucket pool for one label source, stratified on
    source so every trial holds domain fixed and differs only in quality."""
    sub = _read_parquet(SUBSAMPLE_PATH, ["id", "text", "glm52_source", "glm52_quality"])
    labels = _bucket_labels(label_source, sub)

    members: dict[str, list[str]] = {}
    dropped = 0
    for text, source, label in zip(sub["text"], sub["glm52_source"], labels, strict=True):
        if label < 0:
            dropped += 1
            continue
        members.setdefault(f"{source}|q{label}", []).append(text)
    if dropped:
        logger.info("%s: %d documents have no label and are excluded", label_source, dropped)

    rng = random.Random(SEED)
    buckets = []
    for key, docs in sorted(members.items()):
        if len(docs) < IN_GROUP_COUNT:
            continue
        rng.shuffle(docs)  # BucketPool's head-as-uniform-sample contract
        buckets.append(Bucket(key, docs[:PER_BUCKET]))
    logger.info("%s: %d (source, bucket) cells over %d documents", label_source, len(buckets), len(sub["id"]))
    return BucketPool(label_source, buckets, stratum_of=lambda key: key.rsplit("|q", 1)[0])


def run_detection(
    pool: BucketPool,
    panel: Sequence[Panelist],
    *,
    max_trials: int,
    prior: dict | None = None,
) -> dict:
    """One-pool sequential detection test against the ``CHANCE_LEVEL`` floor.

    A single Robbins confidence sequence at level ``ALPHA`` tracks the panel
    detection rate; the run stops once at least ``MIN_TRIALS`` trials are in and
    the interval excludes the chance floor, or after ``max_trials`` attempted
    trials. Attempted (not completed) trials bound the loop, so a panel that
    abstains on everything cannot loop forever issuing paid calls.

    ``prior`` resumes an unresolved run from its results JSON instead of paying
    for its trials again: the confidence sequence and tallies restart from the
    stored counts, which the sequence's anytime validity licenses — it covers the
    truth at every n, so extending with further independent draws is just moving
    to a later n. The extension samples with a seed offset by the prior attempt
    count so it draws a fresh trial stream rather than replaying (and double
    counting) the exact trials already scored.
    """
    prior = prior or {}
    attempted = prior.get("n_attempted", 0)
    abstained = prior.get("n_abstained", 0)
    rng = np.random.default_rng(SEED + attempted)
    cs = ConfidenceSequence(
        alpha=ALPHA,
        rho=1.0 / math.sqrt(TARGET_TRIALS),
        n=prior.get("n_trials", 0),
        total=prior.get("detection_rate", 0.0) * prior.get("n_trials", 0),
    )
    stored = prior.get("per_model", {})
    per_model: dict[str, dict[str, int]] = {
        p.name: {
            "correct": round((stored.get(p.name, {}).get("accuracy") or 0.0) * stored.get(p.name, {}).get("votes", 0)),
            "votes": stored.get(p.name, {}).get("votes", 0),
        }
        for p in panel
    }
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while attempted < max_trials:
            batch = [pool.sample_trial(rng) for _ in range(min(BATCH_SIZE, max_trials - attempted))]
            attempted += len(batch)
            scores = _score_round(batch, panel, executor, MAX_DOC_CHARS)
            abstained += scores.n_abstained
            for rate in scores.detection_rates:
                cs.update(rate)
            for name, hits in scores.model_hits.items():
                per_model[name]["votes"] += len(hits)
                per_model[name]["correct"] += sum(hits)
            lo, hi = cs.interval()
            logger.info("%s: n=%d rate=%.3f [%.3f, %.3f]", pool.name, cs.n, cs.mean, lo, hi)
            if cs.n >= MIN_TRIALS and (lo > CHANCE_LEVEL or hi < CHANCE_LEVEL):
                break

    lo, hi = cs.interval()
    decision = "above_chance" if lo > CHANCE_LEVEL else "below_chance" if hi < CHANCE_LEVEL else "unresolved"
    return {
        "label_source": pool.name,
        "decision": decision,
        "detection_rate": cs.mean,
        "interval": [lo, hi],
        "chance": CHANCE_LEVEL,
        "n_trials": cs.n,
        "n_attempted": attempted,
        "n_abstained": abstained,
        "per_model": {
            name: {"accuracy": t["correct"] / t["votes"] if t["votes"] else None, "votes": t["votes"]}
            for name, t in per_model.items()
        },
    }


def run_panel(label_source: str, max_trials: int) -> None:
    """Run one label source's detection test and write its results JSON.

    A resolved result is final. An unresolved one is *resumed* — its scored
    trials carry over and only the extension up to ``max_trials`` attempted
    trials is paid for — so re-running with a larger ``--max-trials`` is the way
    to buy an unresolved run more evidence.
    """
    result_path = f"{RESULTS_PREFIX}/{label_source}.json"
    prior = None
    if StoragePath(result_path).exists():
        with StoragePath(result_path).open("r") as fh:
            prior = json.loads(fh.read())
        if prior["decision"] != "unresolved":
            logger.info("results at %s are already resolved (%s)", result_path, prior["decision"])
            return
        if prior["n_attempted"] >= max_trials:
            logger.info("results at %s already attempted %d >= %d trials", result_path, prior["n_attempted"], max_trials)
            return
        logger.info(
            "resuming %s from %d scored trials (%d attempted)", label_source, prior["n_trials"], prior["n_attempted"]
        )
    pool = build_pool(label_source)
    panel = openrouter_panel(list(PANEL_MODELS))
    usage_start = _openrouter_usage()
    result = run_detection(pool, panel, max_trials=max_trials, prior=prior)
    result.update(
        {
            "panel_models": list(PANEL_MODELS),
            "alpha": ALPHA,
            "per_bucket": PER_BUCKET,
            "max_doc_chars": MAX_DOC_CHARS,
            "seed": SEED,
            "openrouter_usage_start": usage_start,
            "openrouter_usage_end": _openrouter_usage(),
        }
    )
    _write_json(result_path, result)
    logger.info("%s: %s", label_source, json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("subsample", "luna", "score", "panel"))
    parser.add_argument("--label-source", choices=LABEL_SOURCES, help="which quality signal the panel stage tests")
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    args = parser.parse_args(argv)
    if args.stage == "panel" and not args.label_source:
        parser.error("--stage panel requires --label-source")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    start = time.monotonic()
    if args.stage == "subsample":
        build_subsample()
    elif args.stage == "luna":
        label_with_luna()
    elif args.stage == "score":
        score_with_v3()
    else:
        run_panel(args.label_source, args.max_trials)
    logger.info("stage %s finished in %.0fs", args.stage, time.monotonic() - start)


if __name__ == "__main__":
    main()
