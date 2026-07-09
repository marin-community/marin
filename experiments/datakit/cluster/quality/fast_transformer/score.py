# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production batch-scoring of a normalized corpus with the pooled fast-transformer.

Mirrors the v0 fasttext classify step (a zephyr ``Dataset`` map, one job per source),
but scores with the pooled FT and applies a monotonic calibration so the output
score's fixed 0.2-bucket quantization is quality-coherent across content types. Emits
``id``, ``score`` (calibrated, in ``[0, 1]``) and ``quality_bucket`` (0..4).

The model dir holds the four scorer artifacts (``*.eqx`` + ``*_remap.json`` +
``*_meta.json``) plus ``calib_*.json`` (piecewise-linear cutpoint remap). ``.eqx``
deserialisation needs a local path, so each worker streams it down once (cached).

Run over the CoreWeave sample (bme whole-doc scoring, single-node ``--direct``). The
model dir, calibration and baseline all live on CoreWeave — CW pods cannot reach R2/GCS::

    uv run iris --controller-url http://localhost:10000 job run --no-wait \\
        --cpu 8 --memory 24G --enable-extra-resources --priority production \\
        --job-name ft-quality-score -- \\
        python -m experiments.datakit.cluster.quality.fast_transformer.score \\
          --direct --score-mode bme --per-source 300 \\
          --sample-prefix  s3://marin-us-east-02a/marin/datakit/sample_1t_733c8c5c \\
          --model-dir      s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2 \\
          --calib-file     calib_bme.json \\
          --baseline-model s3://marin-us-east-02a/marin/user/rav/quality/oldft/model.bin \\
          --output-prefix  s3://marin-us-east-02a/marin/user/rav/quality/scored_1t_bme \\
          --sources cp/arxiv_abstracts cp/wikiteam starcoder2/ir_python
"""

import argparse
import functools
import json
import logging
import os
import tempfile
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer
from experiments.datakit.cluster.quality.v0.ops.eval_holdout import predict_p_high
from experiments.datakit.fasttext import _load_fasttext_model

logger = logging.getLogger(__name__)

BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)
BATCH_SIZE = 512
# The FT truncates to ~512 tokens; cap text before scoring to bound transfer/tokenize.
MAX_TEXT_CHARS = 8_000
# bme mode scores begin/middle/end ~512-token windows of the WHOLE doc and mean-pools,
# so a shared boilerplate prefix (agent/tool trajectories) no longer blinds the score.
CHUNK_CHARS = 2_000
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Model-dir file names (the junk-gate deployable).
MODEL_EQX = "pooled_junkgate2.eqx"
MODEL_REMAP = "pooled_junkgate2_remap.json"
MODEL_META = "pooled_junkgate2_meta.json"
MODEL_CALIB = "calib_pooled_junkgate2.json"


def _score_bme(scorer: PooledScorer, texts: list[str]) -> np.ndarray:
    """Mean-pool the FT score over begin/middle/end ~512-token windows of each doc.
    Short docs (<= one chunk) reduce to a single scored window."""
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        if len(t) <= CHUNK_CHARS:
            cs = [t]
        else:
            m = len(t) // 2
            cs = [t[:CHUNK_CHARS], t[max(0, m - CHUNK_CHARS // 2) : m + CHUNK_CHARS // 2], t[-CHUNK_CHARS:]]
        spans.append((len(flat), len(flat) + len(cs)))
        flat.extend(cs)
    s = scorer.score(flat)
    return np.array([s[a:b].mean() for a, b in spans])


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process (streams the .eqx local)."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    scorer = PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")
    with open_url(f"{model_dir}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _predict_batch(
    records: list[dict], *, model_dir: str, score_mode: str = "first", calib_file: str = MODEL_CALIB
) -> Iterator[dict]:
    """Score a batch of records; annotate each with calibrated ``score`` + ``quality_bucket``.
    ``score_mode='bme'`` mean-pools begin/middle/end windows of the whole doc."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    if score_mode == "bme":
        raw = _score_bme(scorer, [r.get("text") or "" for r in records])
    else:
        raw = scorer.score([(r.get("text") or "")[:MAX_TEXT_CHARS] for r in records])
    cal = np.interp(raw, xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {**r, "score": float(c), "quality_bucket": int(b)}


def get_ft_batch_predict(*, model_dir: str, score_mode: str = "first", calib_file: str = MODEL_CALIB):
    """Bind the model dir + scoring config and return a ``flat_map`` batch-predict callable."""
    return functools.partial(_predict_batch, model_dir=model_dir, score_mode=score_mode, calib_file=calib_file)


def run_one_source(
    *,
    input_dir: str,
    output_path: str,
    source_name: str,
    model_dir: str,
    max_workers: int | None = None,
    score_mode: str = "first",
    calib_file: str = MODEL_CALIB,
):
    """Score one source's normalized parquet shards on the cluster, writing id/score/bucket."""
    files = sorted(str(m) for m in StoragePath(f"{input_dir.rstrip('/')}/**/*.parquet").glob())
    if not files:
        raise FileNotFoundError(f"{source_name}: no .parquet under {input_dir}")
    pattern = f"{output_path.rstrip('/')}/data-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_file)
        .window(BATCH_SIZE)
        .flat_map(get_ft_batch_predict(model_dir=model_dir, score_mode=score_mode, calib_file=calib_file))
        .select("id", "score", "quality_bucket")
        .write_parquet(pattern, skip_existing=True)
    )
    # InlineRunner: keep the per-process cached model alive across shards in a worker.
    kwargs: dict = {
        "name": f"ft-quality-{source_name.replace('/', '__')}",
        "resources": WORKER_RESOURCES,
        "stage_runner_factory": InlineRunner,
    }
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    return ZephyrContext(**kwargs).execute(pipeline)


def _read_bounded(input_glob: str, per_source: int) -> list[dict]:
    """Bounded-memory read: iterate row-group batches, stop at ``per_source`` docs
    (avoids OOM on multi-GB source shards, e.g. massive_function_calling)."""
    files = sorted(str(m) for m in StoragePath(input_glob).glob())
    got: list[dict] = []
    for f in files:
        with StoragePath(f).open("rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=BATCH_SIZE, columns=["id", "text"]):
                ids = batch.column("id").to_pylist()
                txts = batch.column("text").to_pylist()
                for i, t in zip(ids, txts, strict=True):
                    if t:
                        got.append({"id": str(i), "text": t})
                    if len(got) >= per_source:
                        return got
    return got


def run_direct(
    *,
    sample_prefix: str,
    output_path: str,
    model_dir: str,
    sources: list[str],
    per_source: int,
    score_mode: str = "first",
    calib_file: str = MODEL_CALIB,
    baseline_model: str | None = None,
) -> None:
    """Score a bounded sample per source in one process (no zephyr) and write a single
    parquet with source/id/ft_raw/ft_score/ft_bucket/old_score/old_bucket/text. Keeps text
    + baseline for downstream red-teaming / the quality-score debugging report.

    ``score_mode='bme'`` mean-pools begin/middle/end windows of the whole doc (fixes
    shared-prefix degeneracy); ``baseline_model`` (fasttext .bin) adds the old-scorer column."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    oldm = _load_fasttext_model(baseline_model) if baseline_model else None
    cols = ["source", "id", "ft_raw", "ft_score", "ft_bucket", "old_score", "old_bucket", "text"]
    acc: dict[str, list] = {c: [] for c in cols}
    for src in sources:
        try:
            got = _read_bounded(f"{sample_prefix.rstrip('/')}/{src}/outputs/main/**/*.parquet", per_source)
        except Exception as e:  # one pathological source must not kill the whole job
            logger.warning("%s: READ ERR %s", src, repr(e)[:160])
            continue
        if not got:
            logger.warning("%s: no docs", src)
            continue
        texts = [g["text"] for g in got]
        raw = _score_bme(scorer, texts) if score_mode == "bme" else scorer.score([t[:MAX_TEXT_CHARS] for t in texts])
        cal = np.interp(raw, xk, yk)
        buckets = np.digitize(cal, BUCKET_EDGES)
        if oldm is not None:
            old = np.array([predict_p_high(oldm, t, MAX_TEXT_CHARS) for t in texts])
            oldb = np.digitize(old, BUCKET_EDGES)
        else:
            old = np.full(len(texts), float("nan"))
            oldb = np.full(len(texts), -1)
        for g, rw, c, b, os_, ob in zip(got, raw, cal, buckets, old, oldb, strict=True):
            acc["source"].append(src)
            acc["id"].append(g["id"])
            acc["ft_raw"].append(float(rw))
            acc["ft_score"].append(float(c))
            acc["ft_bucket"].append(int(b))
            acc["old_score"].append(float(os_))
            acc["old_bucket"].append(int(ob))
            acc["text"].append((g["text"] or "")[:4000])
        logger.info("scored %s (%s): %d docs ft_mean=%.2f std=%.03f", src, score_mode, len(got), cal.mean(), cal.std())
    with StoragePath(output_path).open("wb") as fh:
        pq.write_table(pa.table(acc), fh, compression="zstd")
    logger.info("wrote %d scored rows -> %s", len(acc["id"]), output_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-prefix", required=True, help="e.g. s3://.../datakit/sample_100b_<hash>")
    p.add_argument("--model-dir", required=True, help="dir with the 4 scorer/calib artifacts")
    p.add_argument("--output-prefix", required=True, help="scored output prefix (per-source subdirs)")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--direct", action="store_true", help="single-process bounded scoring (no zephyr), keeps text")
    p.add_argument("--per-source", type=int, default=3000, help="docs/source in --direct mode")
    p.add_argument(
        "--score-mode",
        choices=["first", "bme"],
        default="first",
        help="first=score text[:8000]; bme=mean-pool begin/middle/end windows (fixes prefix degeneracy)",
    )
    p.add_argument("--calib-file", default=MODEL_CALIB, help="calibration json name in --model-dir")
    p.add_argument("--baseline-model", default=None, help="fasttext .bin for the old-scorer column (--direct)")
    args = p.parse_args()
    configure_logging(logging.INFO)
    if args.direct:
        run_direct(
            sample_prefix=args.sample_prefix,
            output_path=f"{args.output_prefix.rstrip('/')}/scored.parquet",
            model_dir=args.model_dir,
            sources=args.sources,
            per_source=args.per_source,
            score_mode=args.score_mode,
            calib_file=args.calib_file,
            baseline_model=args.baseline_model,
        )
        return
    for src in args.sources:
        input_dir = f"{args.sample_prefix.rstrip('/')}/{src}/outputs/main"
        output_path = f"{args.output_prefix.rstrip('/')}/{src}"
        logger.info("scoring %s -> %s", src, output_path)
        outcome = run_one_source(
            input_dir=input_dir,
            output_path=output_path,
            source_name=src,
            model_dir=args.model_dir,
            max_workers=args.max_workers,
            score_mode=args.score_mode,
            calib_file=args.calib_file,
        )
        logger.info("done %s: counters=%s", src, dict(outcome.counters))


if __name__ == "__main__":
    main()
