# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke the reference Datakit DAG end-to-end on a few small sources.

Same code paths as :mod:`experiments.datakit.reference_pipeline` (embed →
train centroids → assign → quality → decontam → minhash → dedup → store),
but pinned to three tiny sources and ``SMOKE_SCALE`` (K=64, modest workers)
so the whole graph finishes quickly and cheaply. Centroids are trained
inline (no pre-staged ``--domain-centroids`` needed); only the quality
``.bin`` is an external input, because v0 quality training is not yet wrapped
as a StepSpec.

Per-source steps (embed / tokenize / decontam / minhash / quality) are keyed
by name + content, NOT by scale, so this smoke *shares* those caches with the
full run -- it does not produce throwaway copies. Only the K-dependent steps
(sample / train / assign / dedup / store) get smoke-specific output paths.

Run in eu-west4 so the one-time decontam-bloom build reads the eval corpus
(``EVAL_ROOT``) in-region::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority interactive --cpu 2 --memory 8GB \\
        -- python -m experiments.datakit.reference_pipeline_smoke \\
            --quality-model gs://marin-eu-west4/datakit/llm-quality-classifier/model/sonnet46-thr05/model.bin
"""

import argparse
import logging
from dataclasses import replace

from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    ClusterConfig,
    reference_datakit_steps,
    select_sources,
)

logger = logging.getLogger(__name__)

# Three of the smallest registry entries (~0.003B / 0.02B / 0.17B tokens),
# each a simple single-family source -- cheap to normalize and embed e2e.
SMOKE_SOURCES = ("cp/peps", "cp/foodista", "nsf_awards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-model",
        required=True,
        help="GCS path to a trained fasttext quality model.bin (cluster/quality/v0).",
    )
    parser.add_argument(
        "--domain-centroids",
        default=None,
        help="Optional pre-staged centroids dir. Omit to train K=64 centroids inline.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SMOKE_SOURCES),
        help=f"Comma-separated source names. Default: {','.join(SMOKE_SOURCES)}",
    )
    parser.add_argument("--max-concurrent", type=int, default=None, metavar="N")
    args = parser.parse_args()

    configure_logging(logging.INFO)

    names = [s.strip() for s in args.sources.split(",") if s.strip()]

    # Reusing pre-staged centroids means the cluster K (and view filenames) must
    # match what those centroids were trained at -- the production
    # ClusterConfig() (K=5000, views 40/1000). SMOKE_SCALE's K=64 only applies
    # when we train inline. Keep the small resources either way.
    if args.domain_centroids is not None:
        scale = replace(SMOKE_SCALE, cluster=ClusterConfig())
        logger.info("reusing centroids %s -> production cluster K=%d", args.domain_centroids, scale.cluster.k_train)
    else:
        scale = SMOKE_SCALE
        logger.info("no centroids given -> inline-train K=%d", scale.cluster.k_train)

    result = reference_datakit_steps(
        select_sources(names),
        domain_centroids=args.domain_centroids,
        quality_model=args.quality_model,
        scale=scale,
    )
    StepRunner().run(result.all_steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
