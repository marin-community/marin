# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce the hero quality data: fusion scores, then store-ready buckets.

Two stages, one step per registered source each, resolved through
:mod:`experiments.datakit.hero_data` so the steps land at the paths it registers:

* ``score`` runs :func:`score_fusion.fusion_score_step` over a source's normalized
  text and its Harrier leaf on GPU workers. It is the producer of record for
  :func:`hero_data.fusion_scores`; that accessor is pinned to a completed run, so
  this stage only needs to run again to score under a new pin.
* ``bucket`` runs :func:`bucket.quality_step` over the pinned fusion scores and
  content types on CPU workers, writing the :class:`QualityScores` leaves that
  :func:`hero_data.quality` resolves and the store reads.

Submit through the hub; the workers place on the CoreWeave peer with the data::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a --job-name hero-quality-bucket \\
        --no-wait -- python -m experiments.datakit.cluster.quality.fast_transformer.run --stage bucket
"""

import argparse
import logging
from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.datakit import hero_data
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import fusion_score_step

logger = logging.getLogger(__name__)

# Each step drives one Zephyr pipeline from a dedicated coordinator and blocks, so
# it needs almost nothing itself. The score stage's driver declares the ``gpu``
# extra so the workers it spawns inherit an environment with CUDA JAX.
DRIVER_RESOURCES = ResourceConfig(cpu=1, ram="2g")
MAX_CONCURRENT = 8


def _partition(sources: list[str], index: int, count: int) -> list[str]:
    if not 0 <= index < count:
        raise ValueError(f"partition index {index} must be in [0, {count})")
    return sources[index::count]


def build_score_steps(sources: list[str]) -> list[StepSpec]:
    """One fusion score step per source, at the identity the score stage owns."""
    steps = []
    for source in sources:
        step = fusion_score_step(
            name=f"datakit/fusion_scores/{source}",
            normalized=hero_data.normalized(source),
            embedding=hero_data.harrier(source),
            quality_model=hero_data.NEMOTRON_88K,
        )
        steps.append(replace(step, fn=remote(step.fn, resources=DRIVER_RESOURCES, pip_dependency_groups=["gpu"])))
    return steps


def build_bucket_steps(sources: list[str]) -> list[StepSpec]:
    """One bucket step per source, at the identity :func:`hero_data.quality` resolves."""
    steps = []
    for source in sources:
        step = hero_data.quality_step_for(source)
        steps.append(replace(step, fn=remote(step.fn, resources=DRIVER_RESOURCES)))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("score", "bucket"), required=True)
    parser.add_argument(
        "--sources", default=None, help="comma-separated source names (default: every registered source)"
    )
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    sources = hero_data.source_names() if args.sources is None else [s.strip() for s in args.sources.split(",")]
    sources = _partition(sources, args.partition_index, args.partition_count)
    build = build_score_steps if args.stage == "score" else build_bucket_steps
    StepRunner().run(build(sources), max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
