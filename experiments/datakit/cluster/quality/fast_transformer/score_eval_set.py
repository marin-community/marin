# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a flat evaluation set with one candidate quality model.

Runs the deployed scoring path
(:func:`~experiments.datakit.cluster.quality.fast_transformer.score.score_normalized`)
over a single directory of parquet shards rather than a per-source tree, so two
candidate models can be compared on byte-identical documents. The evaluation set is
built by :mod:`experiments.datakit.cluster.quality.fast_transformer.domain_eval_set`.

The scorer reads a :class:`~marin.datakit.normalize.NormalizedData` artifact, so the
directory is wrapped in one rather than being re-materialized: it only needs ``id``
and ``text``, which the evaluation shards already carry.

``--model-version`` names the scorer in the output path. Comparing models means
scoring the same documents twice, so the two runs must not collide.
"""

import argparse
import logging

from marin.datakit.normalize import NormalizedData
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score import score_normalized
from experiments.datakit.reference_pipeline import DEFAULT_SCALE, quality_model_path

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", required=True, help="directory of evaluation shards (id + text)")
    parser.add_argument("--out", required=True, help="output root for this model's scores")
    parser.add_argument("--quality-model", default=None, help="scorer + calib dir (default: the deployed model)")
    parser.add_argument("--model-version", required=True, help="identity tag for the scorer being evaluated")
    parser.add_argument("--pool-workers", type=int, default=None, help="Zephyr worker count (override scale)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    model_dir = args.quality_model or quality_model_path()
    max_workers = args.pool_workers if args.pool_workers is not None else DEFAULT_SCALE.pool.n_workers
    logger.info("score_eval_set: scoring %s with %s (%s)", args.docs_dir, args.model_version, model_dir)

    result = score_normalized(
        output_path=args.out,
        normalized=NormalizedData(main_output_dir=args.docs_dir, dup_output_dir="", counters={}),
        source=args.model_version,
        model_dir=model_dir,
        max_workers=max_workers,
        worker_resources=DEFAULT_SCALE.pool.worker,
    )
    logger.info("score_eval_set: wrote %s (counters: %s)", result.main_output_dir, result.counters)


if __name__ == "__main__":
    main()
