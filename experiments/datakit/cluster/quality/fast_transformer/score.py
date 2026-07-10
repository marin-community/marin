# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production batch-scoring of a normalized corpus with the pooled fast-transformer.

Mirrors the v0 fasttext classify step (a zephyr ``Dataset`` map, one job per source
on iris), but scores with the pooled FT and applies a monotonic calibration so the
output score's fixed 0.2-bucket quantization is quality-coherent across content types.
Emits ``source``, ``id``, ``score`` (calibrated, in ``[0, 1]``) and ``quality_bucket``
(0..4).

Scoring is whole-doc (bme): the score is the mean over begin/middle/end ~512-token
windows of each doc, so a shared boilerplate prefix (agent/tool trajectories) can't
blind the score by filling the single 512-token window.

The model dir holds the four scorer artifacts (``*.eqx`` + ``*_remap.json`` +
``*_meta.json``) plus the calibration json (piecewise-linear cutpoint remap; ``bme``
cutpoints by default). ``.eqx`` deserialisation needs a local path, so each worker
streams it down once (cached).

Run over a normalized corpus on iris (one zephyr job per source)::

    uv run iris --controller-url http://localhost:10000 job run --no-wait \\
        --cpu 8 --memory 24G --enable-extra-resources --priority production \\
        --job-name ft-quality-score -- \\
        python -m experiments.datakit.cluster.quality.fast_transformer.score \\
          --sample-prefix s3://marin-us-east-02a/marin/datakit/sample_1t_733c8c5c \\
          --model-dir     s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2 \\
          --output-prefix s3://marin-us-east-02a/marin/user/rav/quality/scored_1t \\
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
from fray.cluster import ResourceConfig
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer

logger = logging.getLogger(__name__)

BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)
BATCH_SIZE = 512
# bme scores begin/middle/end ~512-token (~2000-char) windows of the whole doc and
# mean-pools them, so a shared boilerplate prefix no longer dominates the score.
CHUNK_CHARS = 2_000
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")

# Model-dir file names (the junk-gate deployable + bme calibration).
MODEL_EQX = "pooled_junkgate2.eqx"
MODEL_REMAP = "pooled_junkgate2_remap.json"
MODEL_META = "pooled_junkgate2_meta.json"
MODEL_CALIB = "calib_bme.json"


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


def load_pooled_scorer(model_dir: str) -> PooledScorer:
    """Load just the `PooledScorer` from a model dir (streams the .eqx to a local path,
    which eqx deserialisation requires). Used by scoring and by calibration fitting."""
    model_dir = model_dir.rstrip("/")
    fd, local_eqx = tempfile.mkstemp(suffix=".eqx")
    with os.fdopen(fd, "wb") as out, open_url(f"{model_dir}/{MODEL_EQX}", "rb") as fh:
        out.write(fh.read())
    return PooledScorer.load(local_eqx, f"{model_dir}/{MODEL_REMAP}", f"{model_dir}/{MODEL_META}")


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(f"{model_dir.rstrip('/')}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _predict_batch(records: list[dict], *, model_dir: str, source: str, calib_file: str = MODEL_CALIB) -> Iterator[dict]:
    """Score a batch of records with bme; emit source/id/score/quality_bucket."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    cal = np.interp(_score_bme(scorer, [r.get("text") or "" for r in records]), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {"source": source, "id": r["id"], "score": float(c), "quality_bucket": int(b)}


def get_ft_batch_predict(*, model_dir: str, source: str, calib_file: str = MODEL_CALIB):
    """Bind the model dir + source and return a ``flat_map`` batch-predict callable."""
    return functools.partial(_predict_batch, model_dir=model_dir, source=source, calib_file=calib_file)


def run_one_source(
    *,
    input_dir: str,
    output_path: str,
    source_name: str,
    model_dir: str,
    max_workers: int | None = None,
    calib_file: str = MODEL_CALIB,
):
    """Score one source's normalized parquet shards on iris, writing source/id/score/bucket."""
    files = sorted(str(m) for m in StoragePath(f"{input_dir.rstrip('/')}/**/*.parquet").glob())
    if not files:
        raise FileNotFoundError(f"{source_name}: no .parquet under {input_dir}")
    pattern = f"{output_path.rstrip('/')}/data-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_file)
        .window(BATCH_SIZE)
        .flat_map(get_ft_batch_predict(model_dir=model_dir, source=source_name, calib_file=calib_file))
        .select("source", "id", "score", "quality_bucket")
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-prefix", required=True, help="e.g. s3://.../datakit/sample_100b_<hash>")
    p.add_argument("--model-dir", required=True, help="dir with the scorer artifacts + calibration json")
    p.add_argument("--output-prefix", required=True, help="scored output prefix (per-source subdirs)")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--calib-file", default=MODEL_CALIB, help="calibration json name in --model-dir")
    args = p.parse_args()
    configure_logging(logging.INFO)
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
            calib_file=args.calib_file,
        )
        logger.info("done %s: counters=%s", src, dict(outcome.counters))


if __name__ == "__main__":
    main()
