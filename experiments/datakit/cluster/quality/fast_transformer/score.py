# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production batch-scoring of a normalized corpus with the pooled fast-transformer.

Scores the whole corpus in a *single* zephyr pipeline over every input file (one
shard per file), so ``--max-workers`` is the one knob that sets the fleet size:
the files fan out across up to that many workers at once, independent of how the
docs are split across sources. This is the difference from the v0 fasttext step,
which ran one iris job per source; here a single driver drives the whole run and a
big ``--max-workers`` (e.g. 512) saturates a large CPU cluster in ~one wave.

Each doc is scored with the pooled FT and a monotonic calibration is applied so the
output score's fixed 0.2-bucket quantization is quality-coherent across content
types. The source (e.g. ``cp/foodista``) is recovered from each input file's path
(injected by ``load_file`` as ``__file_path``), so output stays partitioned by
source even though one pipeline handles them all.

Writes two per-source outputs via a split-writer (like normalize's main/dups): the
lean scored records (``source``, ``id``, ``score`` calibrated in ``[0, 1]``,
``quality_bucket`` 0..4) to ``<source>/outputs/main/``, and a ~``--sample-pct``
systematic sample *with text* to ``<source>/outputs/samples/`` that the debugging
report reads directly (no separate text fetch). Each input file maps 1:1 to one
output file (named after the input), so ``skip_existing`` is a plain driver-side
check: input files whose output already exists are dropped before the pipeline runs.

Scoring is whole-doc (bme): the score is the mean over begin/middle/end ~512-token
windows of each doc, so a shared boilerplate prefix (agent/tool trajectories) can't
blind the score by filling the single 512-token window.

The model dir holds the four scorer artifacts (``*.eqx`` + ``*_remap.json`` +
``*_meta.json``) plus the calibration json (piecewise-linear cutpoint remap; ``bme``
cutpoints by default). ``.eqx`` deserialisation needs a local path, so each worker
streams it down once (cached).

Run over a normalized corpus on iris. Invoke via ``-c "... import main; main()"``
rather than ``-m``: zephyr pickles the pipeline callables by module, and ``-m``
would make this module ``__main__`` so workers can't resolve them::

    uv run iris --controller-url http://localhost:10000 job run --no-wait \\
        --cpu 8 --memory 24G --enable-extra-resources --priority production \\
        --job-name ft-quality-score -- \\
        python -c "from experiments.datakit.cluster.quality.fast_transformer.score import main; main()" \\
          --data-prefix s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f \\
          --model-dir     s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2 \\
          --output-prefix s3://marin-us-east-02a/marin/user/rav/quality/scored_100b \\
          --max-workers   512 --sources cp/arxiv_abstracts cp/wikiteam starcoder2/ir_python
"""

import argparse
import functools
import json
import logging
import posixpath
from collections.abc import Iterator

import numpy as np
from fray.cluster import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from rigging.filesystem import StoragePath, open_url
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import DEFAULT_FILE_PATH_COLUMN, load_file
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
# Scoring is I/O-bound (workers sit ~25% CPU streaming parquet). A worker sits ~3 GiB
# resident (model + a batch's JAX activations + per-seq-len compiled caches); a heavier
# shard can spike transiently above that -- 4g OOM-killed workers on the 100B corpus.
# 8g covers the spike with margin; packing is CPU-bound (cpu=2), so the extra RAM is free.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
MODEL_CALIB = "calib_bme.json"  # calibration json name in the model dir
SAMPLE_TEXT_CHARS = 4_000  # text kept per sampled doc for the report spot-check
DEFAULT_SAMPLE_PCT = 0.02  # fraction of each shard emitted (with text) as the samples side output
_SHARD_FILE = "__shard_file"  # internal: input basename carried to the writer to name the output


@functools.cache
def _load_scorer(model_dir: str, calib_file: str = MODEL_CALIB) -> tuple[PooledScorer, np.ndarray, np.ndarray]:
    """Load the scorer + calibration once per worker process."""
    scorer = load_pooled_scorer(model_dir)
    with open_url(f"{model_dir.rstrip('/')}/{calib_file}", "r") as fh:
        calib = json.loads(fh.read())
    logger.info("loaded FT scorer + calibration (%s) from %s", calib_file, model_dir)
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def _source_of(file_path: str, data_prefix: str) -> str:
    """Recover the datakit source (e.g. ``cp/foodista``) from an input file path.

    Input files live at ``<data_prefix>/<source>/outputs/main/<name>.parquet``."""
    return file_path.split(data_prefix.rstrip("/") + "/", 1)[1].split("/outputs/main/", 1)[0]


def _predict_batch(
    records: list[dict], *, model_dir: str, data_prefix: str, calib_file: str = MODEL_CALIB
) -> Iterator[dict]:
    """Score a batch of records with bme; carry source (from the file path)/id/score/
    quality_bucket + text. ``text`` is dropped for the lean main output and kept for
    the samples side output; ``_SHARD_FILE`` names the per-source output file."""
    scorer, xk, yk = _load_scorer(model_dir, calib_file)
    cal = np.interp(score_bme(scorer, [r.get("text") or "" for r in records]), xk, yk)
    buckets = np.digitize(cal, BUCKET_EDGES)
    for r, c, b in zip(records, cal, buckets, strict=True):
        path = r[DEFAULT_FILE_PATH_COLUMN]
        yield {
            "source": _source_of(path, data_prefix),
            "id": r["id"],
            "score": float(c),
            "quality_bucket": int(b),
            "text": (r.get("text") or "")[:SAMPLE_TEXT_CHARS],
            _SHARD_FILE: posixpath.basename(path),
        }


def get_ft_batch_predict(*, model_dir: str, data_prefix: str, calib_file: str = MODEL_CALIB):
    """Bind the model dir + data prefix and return a ``flat_map`` batch-predict callable."""
    return functools.partial(_predict_batch, model_dir=model_dir, data_prefix=data_prefix, calib_file=calib_file)


def _systematic_take(index: int, pct: float) -> bool:
    """Whether to keep record ``index`` (0-based) in a ~``pct`` sample.

    Deterministic and non-hashing: a systematic rule that keeps every ~1/pct-th record
    by position. No RNG and no id-hashing, so a given shard (records arrive in a stable
    order from the sorted input files) always yields exactly the same sample."""
    return int((index + 1) * pct) > int(index * pct)


def _source_out_base(output_prefix: str, source: str, nest_by_source: bool) -> str:
    """Output base for ``source``: ``<output_prefix>/<source>`` when several sources share
    the prefix, else ``<output_prefix>`` itself -- a single-source run's prefix is already
    that source's directory (e.g. a per-source step), so nesting ``<source>/`` again is
    redundant."""
    out = output_prefix.rstrip("/")
    return f"{out}/{source}" if nest_by_source else out


def _output_paths(output_prefix: str, source: str, shard_file: str, nest_by_source: bool) -> tuple[str, str]:
    """(main, samples) output paths for one input file's scored records."""
    base = _source_out_base(output_prefix, source, nest_by_source)
    return f"{base}/outputs/main/{shard_file}", f"{base}/outputs/samples/{shard_file}"


def _make_corpus_writer(output_prefix: str, sample_pct: float, nest_by_source: bool):
    """A ``map_shard`` split-writer. One input file per shard, so all its records share
    a source + output name: fan them to ``outputs/main/`` (lean) and a ~``sample_pct``
    systematic sample *with text* to ``outputs/samples/``. When several sources share
    ``output_prefix`` the paths nest under ``<source>/`` to keep them apart."""

    def scored_writer(records: Iterator[dict], shard: ShardInfo) -> Iterator[dict]:
        records = iter(records)
        first = next(records, None)
        if first is None:
            return  # empty shard (e.g. all inputs skipped) -> nothing to write
        main_path, sample_path = _output_paths(output_prefix, first["source"], first[_SHARD_FILE], nest_by_source)

        results: dict[str, dict] = {}

        def write_to(path: str, key: str):
            def _fn(items):
                results[key] = write_parquet_file(items, output_path=path)

            return _fn

        with (
            ThreadedBatchWriter(write_to(main_path, "main")) as main_writer,
            ThreadedBatchWriter(write_to(sample_path, "samples")) as sample_writer,
        ):
            for i, r in enumerate((first, *records)):
                main_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket")})
                counters.pipeline.update_counter("ft_quality/scored", 1)
                if _systematic_take(i, sample_pct):
                    sample_writer.submit({k: r[k] for k in ("source", "id", "score", "quality_bucket", "text")})
                    counters.pipeline.update_counter("ft_quality/sampled", 1)
        yield results

    return scored_writer


def _pending_input_files(
    data_prefix: str, source: str, output_prefix: str, skip_existing: bool, nest_by_source: bool
) -> list[str]:
    """The input parquet files of ``source`` still to score.

    The input dir comes from the source's persisted ``NormalizedData`` artifact
    (``.main_output_dir``) rather than a hand-built ``outputs/main`` path. Both the
    input list and the existing-output check are single-level ``*.parquet`` listings
    (no ``**``): a recursive glob makes s3fs ``HeadObject`` the prefix, which the CW
    object store answers with a 400. skip_existing drops files whose lean main output
    (written last per shard, so its presence means fully scored) already exists."""
    main_dir = read_artifact(f"{data_prefix}/{source}", NormalizedData).main_output_dir.rstrip("/")
    inputs = sorted(str(m) for m in StoragePath(f"{main_dir}/*.parquet").glob())
    if not skip_existing:
        return inputs
    out_main = f"{_source_out_base(output_prefix, source, nest_by_source)}/outputs/main"
    done = {posixpath.basename(str(m)) for m in StoragePath(f"{out_main}/*.parquet").glob()}
    return [f for f in inputs if posixpath.basename(f) not in done]


def run_corpus(
    *,
    data_prefix: str,
    sources: list[str],
    output_prefix: str,
    model_dir: str,
    max_workers: int | None = None,
    calib_file: str = MODEL_CALIB,
    sample_pct: float = DEFAULT_SAMPLE_PCT,
    skip_existing: bool = True,
    file_list: str | None = None,
):
    """Score every file of the given sources in one pipeline (one shard per file).

    ``max_workers`` sets the concurrent fleet size; with N input files the run uses
    ``min(N, max_workers)`` workers at a time. Writes lean scored records to
    ``outputs/main/`` and a ~``sample_pct`` sample (with text) to ``outputs/samples/``
    for the debugging report. Output nests under ``<source>/`` only when several sources
    share ``output_prefix``; a single-source run writes straight under ``output_prefix``
    (its prefix is already that source's dir, e.g. a per-source step).

    ``file_list`` is a newline-delimited text file of the exact input paths to score
    (see ``write_file_list``). Use it when the object store answers the driver's
    ``HeadObject`` listing probes with a 400 (a known CW s3fs gotcha): the list is
    built once where listing works, and the driver/workers then do only GET/PUT."""
    dp = data_prefix.rstrip("/")
    # A single-source run's output_prefix is already that source's dir, so don't re-nest
    # <source>/; a file_list spans the whole corpus, so keep it nested there.
    nest_by_source = file_list is not None or len(sources) != 1
    if file_list:
        with open_url(file_list, "r") as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
        logger.info("scoring %d files from --file-list %s with max_workers=%s", len(files), file_list, max_workers)
    else:
        files = [
            f for src in sources for f in _pending_input_files(dp, src, output_prefix, skip_existing, nest_by_source)
        ]
        logger.info("scoring %d files across %d sources with max_workers=%s", len(files), len(sources), max_workers)
    if not files:
        logger.info("nothing to score (all outputs exist, or no inputs)")
        return None
    pipeline = (
        Dataset.from_list(files)
        .flat_map(functools.partial(load_file, include_file_paths=True))
        .window(BATCH_SIZE)
        .flat_map(get_ft_batch_predict(model_dir=model_dir, data_prefix=dp, calib_file=calib_file))
        .map_shard(_make_corpus_writer(output_prefix, sample_pct, nest_by_source))
    )
    # InlineRunner: keep the per-process cached model alive across shards in a worker.
    kwargs: dict = {
        "name": "ft-quality-corpus",
        "resources": WORKER_RESOURCES,
        "stage_runner_factory": InlineRunner,
    }
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    return ZephyrContext(**kwargs).execute(pipeline)


def write_file_list(
    *, data_prefix: str, sources: list[str], output_prefix: str, out: str, skip_existing: bool = True
) -> int:
    """Write the pending input paths (one per line) to ``out`` for ``--file-list``.

    Run this where object-store listing works (e.g. a laptop) so the driver never
    has to list. Returns the number of files written. ``--file-list`` runs span the
    whole corpus, so the pending check uses the nested (``<source>/``) output layout."""
    dp = data_prefix.rstrip("/")
    files = [
        f for src in sources for f in _pending_input_files(dp, src, output_prefix, skip_existing, nest_by_source=True)
    ]
    with StoragePath(out).open("w") as fh:
        fh.write("\n".join(files) + "\n")
    logger.info("wrote %d input paths -> %s", len(files), out)
    return len(files)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-prefix",
        required=True,
        help="normalized-corpus prefix (per-source <source>/outputs/main); a sample corpus or the full corpus",
    )
    p.add_argument("--model-dir", required=True, help="dir with the scorer artifacts + calibration json")
    p.add_argument("--output-prefix", required=True, help="scored output prefix (per-source subdirs)")
    p.add_argument("--sources", nargs="+", help="sources to score (omit when using --file-list)")
    p.add_argument(
        "--file-list",
        help="text file of exact input paths to score, one per line (bypasses driver-side listing)",
    )
    p.add_argument("--max-workers", type=int, default=None, help="concurrent worker fleet size (one shard per file)")
    p.add_argument("--calib-file", default=MODEL_CALIB, help="calibration json name in --model-dir")
    p.add_argument(
        "--sample-pct",
        type=float,
        default=DEFAULT_SAMPLE_PCT,
        help="fraction of each shard written (with text) to outputs/samples for the report",
    )
    args = p.parse_args()
    if not args.file_list and not args.sources:
        p.error("one of --sources or --file-list is required")
    configure_logging(logging.INFO)
    outcome = run_corpus(
        data_prefix=args.data_prefix,
        sources=args.sources or [],
        output_prefix=args.output_prefix,
        model_dir=args.model_dir,
        max_workers=args.max_workers,
        calib_file=args.calib_file,
        sample_pct=args.sample_pct,
        file_list=args.file_list,
    )
    if outcome is not None:
        logger.info("done: counters=%s", dict(outcome.counters))


if __name__ == "__main__":
    main()
