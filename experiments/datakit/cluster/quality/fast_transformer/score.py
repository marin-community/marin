# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production batch-scoring of a normalized corpus with the pooled fast-transformer.

Mirrors the v0 fasttext classify step (a zephyr ``Dataset`` map, one job per source
on iris), but scores with the pooled FT and applies a monotonic calibration so the
output score's fixed 0.2-bucket quantization is quality-coherent across content types.

Writes two per-source outputs via a split-writer (like normalize's main/dups): the
lean scored records (``source``, ``id``, ``score`` calibrated in ``[0, 1]``,
``quality_bucket`` 0..4) to ``outputs/main/``, and a ~``--sample-pct`` systematic
sample *with text* to ``outputs/samples/`` that the debugging report reads directly
(no separate text fetch).

Scoring is whole-doc (bme): the score is the mean over begin/middle/end ~512-token
windows of each doc, so a shared boilerplate prefix (agent/tool trajectories) can't
blind the score by filling the single 512-token window.

The model dir holds the four scorer artifacts (``*.eqx`` + ``*_remap.json`` +
``*_meta.json``) plus the calibration json (piecewise-linear cutpoint remap; ``bme``
cutpoints by default). ``.eqx`` deserialisation needs a local path, so each worker
streams it down once (cached).

Run over a normalized corpus on iris (one zephyr job per source). Invoke via
``-c "... import main; main()"`` rather than ``-m``: zephyr pickles the pipeline
callables by module, and ``-m`` would make this module ``__main__`` so workers
can't resolve them::

    uv run iris --controller-url http://localhost:10000 job run --no-wait \\
        --cpu 8 --memory 24G --enable-extra-resources --priority production \\
        --job-name ft-quality-score -- \\
        python -c "from experiments.datakit.cluster.quality.fast_transformer.score import main; main()" \\
          --sample-prefix s3://marin-us-east-02a/marin/datakit/sample_1t_733c8c5c \\
          --model-dir     s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2 \\
          --output-prefix s3://marin-us-east-02a/marin/user/rav/quality/scored_1t \\
          --sources cp/arxiv_abstracts cp/wikiteam starcoder2/ir_python
"""

import argparse
import functools
import json
import logging
from collections.abc import Iterator

import numpy as np
from fray.cluster import ResourceConfig
from marin.datakit import partition_filename
from rigging.filesystem import StoragePath, open_url, prefix_join
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file
from zephyr.runners import InlineRunner
from zephyr.writers import ThreadedBatchWriter, write_parquet_file

from experiments.datakit.cluster.quality.fast_transformer.scorer import (
    BUCKET_EDGES,
    PooledScorer,
    load_pooled_scorer,
    score_bme,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 512
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g")
MODEL_CALIB = "calib_bme.json"  # calibration json name in the model dir
SAMPLE_TEXT_CHARS = 4_000  # text kept per sampled doc for the report spot-check
DEFAULT_SAMPLE_PCT = 0.02  # fraction of each shard emitted (with text) as the samples side output


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(f"{model_dir.rstrip('/')}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _predict_batch(records: list[dict], *, model_dir: str, source: str, calib_file: str = MODEL_CALIB) -> Iterator[dict]:
    """Score a batch of records with bme; carry source/id/score/quality_bucket + text.
    ``text`` is dropped for the lean main output and kept for the samples side output."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    cal = np.interp(score_bme(scorer, [r.get("text") or "" for r in records]), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        yield {
            "source": source,
            "id": r["id"],
            "score": float(c),
            "quality_bucket": int(b),
            "text": (r.get("text") or "")[:SAMPLE_TEXT_CHARS],
        }


def get_ft_batch_predict(*, model_dir: str, source: str, calib_file: str = MODEL_CALIB):
    """Bind the model dir + source and return a ``flat_map`` batch-predict callable."""
    return functools.partial(_predict_batch, model_dir=model_dir, source=source, calib_file=calib_file)


def _systematic_take(index: int, pct: float) -> bool:
    """Whether to keep record ``index`` (0-based) in a ~``pct`` sample.

    Deterministic and non-hashing: a systematic rule that keeps every ~1/pct-th record
    by position. No RNG and no id-hashing, so a given shard (records arrive in a stable
    order from the sorted input files) always yields exactly the same sample."""
    return int((index + 1) * pct) > int(index * pct)


def _make_scored_writer(output_path: str, sample_pct: float, skip_existing: bool):
    """A ``map_shard`` split-writer that fans each shard to two Parquet outputs (like
    normalize's main/dups): the lean scored records to ``outputs/main/`` and a
    ~``sample_pct`` systematic sample *with text* to ``outputs/samples/`` for the report."""
    out = output_path.rstrip("/")

    def scored_writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        shard_filename = partition_filename(shard.shard_idx, shard.total_shards)
        main_path = prefix_join(out, f"outputs/main/{shard_filename}")
        sample_path = prefix_join(out, f"outputs/samples/{shard_filename}")
        # skip_existing: return without consuming `records`, so the upstream scoring is
        # skipped too (the pipeline is pull-based), not just the write.
        if skip_existing and StoragePath(main_path).exists() and StoragePath(sample_path).exists():
            yield {"main": {"path": main_path, "skipped": True}, "samples": {"path": sample_path, "skipped": True}}
            return

        results: dict[str, dict] = {}

        def write_to(path: str, key: str):
            def _fn(items):
                results[key] = write_parquet_file(items, output_path=path)

            return _fn

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(sample_path, "samples")) as sample_writer,
        ):
            for i, r in enumerate(records):
                main_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket")})
                counters.pipeline.update_counter("ft_quality/scored", 1)
                if _systematic_take(i, sample_pct):
                    sample_writer.submit(r)  # full record incl. text
                    counters.pipeline.update_counter("ft_quality/sampled", 1)
        yield results

    return scored_writer


def run_one_source(
    *,
    input_dir: str,
    output_path: str,
    source_name: str,
    model_dir: str,
    max_workers: int | None = None,
    calib_file: str = MODEL_CALIB,
    sample_pct: float = DEFAULT_SAMPLE_PCT,
    skip_existing: bool = True,
):
    """Score one source's normalized parquet shards on iris. Writes the lean scored
    records to ``outputs/main/`` and a ~``sample_pct`` sample (with text) to
    ``outputs/samples/`` for the debugging report."""
    files = sorted(str(m) for m in StoragePath(f"{input_dir.rstrip('/')}/**/*.parquet").glob())
    if not files:
        raise FileNotFoundError(f"{source_name}: no .parquet under {input_dir}")
    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_file)
        .window(BATCH_SIZE)
        .flat_map(get_ft_batch_predict(model_dir=model_dir, source=source_name, calib_file=calib_file))
        .map_shard(_make_scored_writer(output_path, sample_pct, skip_existing))
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
    p.add_argument(
        "--sample-pct",
        type=float,
        default=DEFAULT_SAMPLE_PCT,
        help="fraction of each shard written (with text) to outputs/samples for the report",
    )
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
            sample_pct=args.sample_pct,
        )
        logger.info("done %s: counters=%s", src, dict(outcome.counters))


if __name__ == "__main__":
    main()
