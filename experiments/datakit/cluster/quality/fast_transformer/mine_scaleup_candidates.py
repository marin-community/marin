# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mine new labeling candidates from the 50M corpus with the embedding domain MLP.

The scale-up needs fresh documents for the content types the 88k labels under-
cover. Every row of the harrier 50M sample carries a stored 1024-d int8
embedding, and the trained :mod:`domain_mlp` types it at ~86% held-out accuracy,
so mining is one pass over the embedding columns: score every unlabeled row,
then draw ``deficit x oversample`` documents per needed type at random from the
rows predicted as that type (oversampling covers the MLP's error rate — the
oracle's own content_type is what counts toward the per-type target).

The ``cluster_5000`` majority-type prefilter was considered and skipped: the MLP
forward is negligible next to the embedding-column read, which any prefilter
would still have to do.

Three stages, each skipped when its output exists (Zephyr for the shard fans):

``score``    one task per corpus shard — read id+embedding, exclude already-
             labeled ids, write predicted type + confidence per row.
``select``   draw the per-type candidate sample with a fixed seed.
``extract``  re-read the selected rows' full records, cut their bme grading
             windows with the parity-gated gemma tokenizer, and write both the
             full rows (co-partitioned by shard, the later join's layout) and
             the windows parquet the labeling driver consumes.

Task parameters travel inside the mapped elements rather than module globals:
zephyr workers import this module fresh, so driver-set globals never reach them.
"""

import argparse
import json
import logging
import os
from io import BytesIO

import fsspec
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    check_gigatoken_parity,
    doc_windows,
    encode_documents,
)
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import EMBED_DIM, embedding_matrix
from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES

logger = logging.getLogger(__name__)

BASE_URL = "s3://marin-us-east-02a/marin/datakit/samples/harrier-oss-v1-0.6b-50m-text-v1"
DEFAULT_MLP = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/domain_mlp/domain_mlp.npz"
DEFAULT_LABELED_IDS = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"

MAX_WORKERS = 48
SCORE_RESOURCES = ResourceConfig(cpu=4, ram="16g")
EXTRACT_RESOURCES = ResourceConfig(cpu=4, ram="16g")
PREDICT_BATCH = 16_384

# Per-process caches keyed by path, so a worker loads each side input once.
_MLP_CACHE: dict[str, dict] = {}
_ID_CACHE: dict[str, set[str]] = {}


def _mlp(path: str) -> dict:
    if path not in _MLP_CACHE:
        with fsspec.filesystem("s3").open(path, "rb") as fh:
            data = np.load(BytesIO(fh.read()), allow_pickle=False)
            _MLP_CACHE[path] = {k: np.asarray(data[k]) for k in ("w1", "b1", "w2", "b2", "w3", "b3")} | {
                "labels": [str(x) for x in data["labels"]]
            }
    return _MLP_CACHE[path]


def _labeled_ids(path: str) -> set[str]:
    if path not in _ID_CACHE:
        with fsspec.filesystem("s3").open(path, "rb") as fh:
            _ID_CACHE[path] = set(pq.ParquetFile(fh).read(columns=["id"]).column("id").to_pylist())
    return _ID_CACHE[path]


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """jax.nn.gelu's default tanh approximation, so scores match the trained model."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def mlp_predict(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(argmax class index, softmax confidence) per row, batched."""
    # A fully-labeled shard yields zero rows to type; concatenate needs the guard.
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    idx_out, conf_out = [], []
    for start in range(0, len(x), PREDICT_BATCH):
        h = _gelu_tanh(x[start : start + PREDICT_BATCH] @ params["w1"] + params["b1"])
        h = _gelu_tanh(h @ params["w2"] + params["b2"])
        logits = h @ params["w3"] + params["b3"]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        idx_out.append(probs.argmax(axis=1))
        conf_out.append(probs.max(axis=1))
    return np.concatenate(idx_out), np.concatenate(conf_out)


def _write_parquet(fs: fsspec.AbstractFileSystem, frame: pl.DataFrame, path: str) -> None:
    buf = BytesIO()
    frame.write_parquet(buf)
    fs.pipe_file(path, buf.getvalue())


def score_shard(task: dict) -> dict:
    """Type every unlabeled row of one corpus shard from its stored embedding."""
    relpath = task["relpath"]
    fs = fsspec.filesystem("s3")
    out_path = f"{task['out']}/candidates/{relpath}"
    if fs.exists(out_path):
        return {"shard": relpath, "skipped": True}
    with fs.open(f"{BASE_URL}/{relpath}", "rb", cache_type="none") as fh:
        table = pq.ParquetFile(fh).read(columns=["id", "embedding"])
    ids = table.column("id").to_pylist()
    labeled = _labeled_ids(task["labeled_ids"])
    keep = np.array([doc_id not in labeled for doc_id in ids], dtype=bool)
    flat = table.column("embedding").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    emb = flat.reshape(len(ids), EMBED_DIM)[keep]
    params = _mlp(task["mlp"])
    idx, conf = mlp_predict(params, embedding_matrix(emb))
    frame = pl.DataFrame(
        {
            "id": [doc_id for doc_id, k in zip(ids, keep.tolist(), strict=True) if k],
            "relpath": [relpath] * int(keep.sum()),
            "pred_type": [params["labels"][i] for i in idx],
            "pred_conf": conf.astype(np.float32),
        }
    )
    _write_parquet(fs, frame, out_path)
    return {"shard": relpath, "rows": len(ids), "unlabeled": int(keep.sum())}


def extract_shard(task: dict) -> dict:
    """Write the selected rows of one shard, plus their bme grading windows."""
    relpath = task["relpath"]
    out = task["out"]
    fs = fsspec.filesystem("s3")
    docs_path = f"{out}/docs/{relpath}"
    windows_path = f"{out}/windows/{relpath}"
    if fs.exists(docs_path) and fs.exists(windows_path):
        return {"shard": relpath, "skipped": True}
    with fs.open(f"{out}/selected.parquet", "rb") as fh:
        selected = pl.read_parquet(fh).filter(pl.col("relpath") == relpath)
    wanted = set(selected.get_column("id").to_list())
    with fs.open(f"{BASE_URL}/{relpath}", "rb", cache_type="none") as fh:
        pf = pq.ParquetFile(fh)
        parts = []
        for rg in range(pf.metadata.num_row_groups):
            df = pl.from_arrow(pf.read_row_group(rg))
            sub = df.filter(pl.col("id").is_in(wanted))
            if sub.height:
                parts.append(sub)
    if not parts:
        return {"shard": relpath, "rows": 0, "windows": 0}
    rows = pl.concat(parts).join(selected.select("id", "pred_type", "pred_conf"), on="id", how="inner")

    token_ids = encode_documents(rows.get_column("text").to_list())
    source_dir = os.path.dirname(relpath)
    windows = []
    for doc_id, ids in zip(rows.get_column("id").to_list(), token_ids, strict=True):
        for w in doc_windows(ids):
            windows.append(
                {
                    "id": doc_id,
                    "source": source_dir,
                    "window": w.position,
                    "token_start": w.token_start,
                    "token_end": w.token_end,
                    "text": w.text,
                    "doc_tokens": len(ids),
                }
            )
    _write_parquet(fs, rows, docs_path)
    _write_parquet(fs, pl.DataFrame(windows), windows_path)
    return {"shard": relpath, "rows": rows.height, "windows": len(windows)}


def select(
    fs: fsspec.AbstractFileSystem,
    out: str,
    candidates_dir: str,
    deficits: dict[str, int],
    oversample: float,
    seed: int,
    exclude_selected: list[str],
) -> None:
    """Draw the per-type candidate sample from the scored shards.

    ``candidates_dir`` may belong to an earlier round's output prefix so a
    follow-up draw reuses its scores, and ``exclude_selected`` drops the ids
    earlier rounds already took — a later round's pool is what is left.
    """
    selected_path = f"{out}/selected.parquet"
    if fs.exists(selected_path):
        logger.info("select: reusing %s", selected_path)
        return
    shards = sorted("s3://" + p for p in fs.glob(f"{candidates_dir.removeprefix('s3://')}/**/*.parquet"))
    frames = []
    for path in shards:
        with fs.open(path, "rb") as fh:
            frames.append(pl.read_parquet(fh))
    candidates = pl.concat(frames)
    logger.info("select: %d unlabeled candidates over %d shards", candidates.height, len(shards))
    for prior in exclude_selected:
        with fs.open(prior, "rb") as fh:
            taken = pl.read_parquet(fh).get_column("id")
        candidates = candidates.filter(~pl.col("id").is_in(taken))
        logger.info("select: %d candidates after excluding %d from %s", candidates.height, taken.len(), prior)

    picks = []
    for ctype, deficit in sorted(deficits.items()):
        pool = candidates.filter(pl.col("pred_type") == ctype)
        want = int(deficit * oversample)
        take = min(want, pool.height)
        picks.append(pool.sample(n=take, seed=seed, shuffle=True))
        logger.info("select: %s deficit=%d want=%d pool=%d took=%d", ctype, deficit, want, pool.height, take)
    chosen = pl.concat(picks)
    _write_parquet(fs, chosen, selected_path)
    logger.info("select: wrote %d selected candidates -> %s", chosen.height, selected_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output prefix for candidates/selected/docs/windows")
    parser.add_argument("--deficits", required=True, help='JSON per-type new-doc deficits, e.g. {"math": 15100}')
    parser.add_argument("--oversample", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp", default=DEFAULT_MLP)
    parser.add_argument("--labeled-ids", default=DEFAULT_LABELED_IDS)
    parser.add_argument("--limit-shards", type=int, default=None, help="score only the first N shards (smoke runs)")
    parser.add_argument(
        "--candidates-dir",
        default=None,
        help="reuse an earlier round's scored candidates instead of scoring (skips the score stage)",
    )
    parser.add_argument(
        "--exclude-selected",
        nargs="*",
        default=[],
        help="earlier rounds' selected.parquet paths whose ids are excluded from the pools",
    )
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    deficits = json.loads(args.deficits)
    unknown = sorted(set(deficits) - set(CONTENT_TYPES))
    if unknown:
        raise ValueError(f"deficit types outside the rubric: {unknown}")

    out = args.out.rstrip("/")
    fs = fsspec.filesystem("s3")
    base_key = BASE_URL.removeprefix("s3://")
    shard_paths = sorted(fs.glob(f"{base_key}/**/*.parquet"))
    relpaths = [p[len(base_key) + 1 :] for p in shard_paths]
    if args.limit_shards:
        relpaths = relpaths[: args.limit_shards]
    logger.info("mine: %d corpus shards", len(relpaths))

    # Parity-gate the fast tokenizer once, on real corpus text, before any worker relies on it.
    with fs.open(f"{BASE_URL}/{relpaths[0]}", "rb", cache_type="none") as fh:
        sample_texts = pq.ParquetFile(fh).read_row_group(0, columns=["text"]).column("text").to_pylist()
    check_gigatoken_parity(sample_texts[:256])

    if args.candidates_dir is None:
        score_tasks = [{"relpath": r, "out": out, "mlp": args.mlp, "labeled_ids": args.labeled_ids} for r in relpaths]
        outcome = ZephyrContext(
            name="mine-scaleup-score",
            resources=SCORE_RESOURCES,
            max_workers=MAX_WORKERS,
        ).execute(Dataset.from_list(score_tasks).map(score_shard), verbose=True)
        logger.info("score: done, %d shards", len(outcome.results))
    candidates_dir = args.candidates_dir or f"{out}/candidates"

    select(fs, out, candidates_dir, deficits, args.oversample, args.seed, args.exclude_selected)

    with fs.open(f"{out}/selected.parquet", "rb") as fh:
        chosen_shards = sorted(set(pl.read_parquet(fh).get_column("relpath").to_list()))
    extract_tasks = [{"relpath": r, "out": out} for r in chosen_shards]
    outcome = ZephyrContext(
        name="mine-scaleup-extract",
        resources=EXTRACT_RESOURCES,
        max_workers=MAX_WORKERS,
    ).execute(Dataset.from_list(extract_tasks).map(extract_shard), verbose=True)
    docs = sum(r.get("rows", 0) for r in outcome.results)
    windows = sum(r.get("windows", 0) for r in outcome.results)
    logger.info("extract: %d docs, %d windows across %d shards -> %s", docs, windows, len(chosen_shards), out)


if __name__ == "__main__":
    main()
