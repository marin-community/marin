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

Run over the CoreWeave sample on cw-us-east-02a::

    uv run iris --cluster=cw-us-east-02a job run --no-wait --cpu 2 --memory 16G \\
        --extra=cpu --priority production --region US-EAST-02A \\
        --job-name ft-quality-score -- \\
        python -m experiments.datakit.cluster.quality.fast_transformer.score \\
          --sample-prefix s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f \\
          --model-dir    s3://marin-na/marin/user/rav/quality/pooled_junkgate2 \\
          --output-prefix s3://marin-us-east-02a/marin/user/rav/quality/scored_sample_100b \\
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
from rigging.filesystem import StoragePath, open_url, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer

logger = logging.getLogger(__name__)

BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)
BATCH_SIZE = 512
# The FT truncates to ~512 tokens; cap text before scoring to bound transfer/tokenize.
MAX_TEXT_CHARS = 8_000
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Model-dir file names (the junk-gate deployable).
MODEL_EQX = "pooled_junkgate2.eqx"
MODEL_REMAP = "pooled_junkgate2_remap.json"
MODEL_META = "pooled_junkgate2_meta.json"
MODEL_CALIB = "calib_pooled_junkgate2.json"


@functools.cache
def _load_scorer(model_dir: str) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process (streams the .eqx local)."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    scorer = PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")
    with open_url(f"{model_dir}/{MODEL_CALIB}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration from %s", model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _predict_batch(records: list[dict], *, model_dir: str) -> Iterator[dict]:
    """Score a batch of records; annotate each with calibrated ``score`` + ``quality_bucket``."""
    scorer, xk, yk = _load_scorer(model_dir)
    texts = [(r.get("text") or "")[:MAX_TEXT_CHARS] for r in records]
    cal = np.interp(scorer.score(texts), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {**r, "score": float(c), "quality_bucket": int(b)}


def get_ft_batch_predict(*, model_dir: str):
    """Bind the model dir and return a ``flat_map`` batch-predict callable."""
    return functools.partial(_predict_batch, model_dir=model_dir)


def run_one_source(
    *, input_dir: str, output_path: str, source_name: str, model_dir: str, max_workers: int | None = None
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
        .flat_map(get_ft_batch_predict(model_dir=model_dir))
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


def run_direct(*, sample_prefix: str, output_path: str, model_dir: str, sources: list[str], per_source: int) -> None:
    """Score a bounded sample per source in one process (no zephyr) and write a single
    parquet with id/source/score/quality_bucket/text. Suitable for a single GPU job and
    for downstream red-teaming (text kept, truncated)."""
    rows: list[dict] = []
    for src in sources:
        files = sorted(
            str(m) for m in StoragePath(f"{sample_prefix.rstrip('/')}/{src}/outputs/main/**/*.parquet").glob()
        )
        got: list[dict] = []
        for f in files:
            fs, res = url_to_fs(f)
            with fs.open(res, "rb") as fh:
                tbl = pq.read_table(fh, columns=["id", "text"])
            for i, t in zip(tbl.column("id").to_pylist(), tbl.column("text").to_pylist(), strict=True):
                got.append({"id": str(i), "source": src, "text": t or ""})
                if len(got) >= per_source:
                    break
            if len(got) >= per_source:
                break
        rows.extend(_predict_batch(got, model_dir=model_dir))
        logger.info("scored %s: %d docs", src, len(got))
    out = pa.table(
        {
            "id": [r["id"] for r in rows],
            "source": [r["source"] for r in rows],
            "score": [r["score"] for r in rows],
            "quality_bucket": [r["quality_bucket"] for r in rows],
            "text": [(r.get("text") or "")[:4000] for r in rows],
        }
    )
    fs, res = url_to_fs(output_path)
    parent = os.path.dirname(res)
    if parent:
        fs.mkdirs(parent, exist_ok=True)
    with fs.open(res, "wb") as fh:
        pq.write_table(out, fh, compression="zstd")
    logger.info("wrote %d scored rows -> %s", len(rows), output_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-prefix", required=True, help="e.g. s3://.../datakit/sample_100b_<hash>")
    p.add_argument("--model-dir", required=True, help="dir with the 4 scorer/calib artifacts")
    p.add_argument("--output-prefix", required=True, help="scored output prefix (per-source subdirs)")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--direct", action="store_true", help="single-process bounded scoring (no zephyr), keeps text")
    p.add_argument("--per-source", type=int, default=3000, help="docs/source in --direct mode")
    args = p.parse_args()
    configure_logging(logging.INFO)
    if args.direct:
        run_direct(
            sample_prefix=args.sample_prefix,
            output_path=f"{args.output_prefix.rstrip('/')}/scored.parquet",
            model_dir=args.model_dir,
            sources=args.sources,
            per_source=args.per_source,
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
        )
        logger.info("done %s: counters=%s", src, dict(outcome.counters))


if __name__ == "__main__":
    main()
