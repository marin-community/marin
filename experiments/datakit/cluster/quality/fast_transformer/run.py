# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce quality data for the registered sources.

Two stages, one chain of steps per source:

* ``pipeline`` runs the quality pipeline from scratch: :func:`score_fusion.fusion_score_step`
  over a source's normalized text and Harrier leaf on GPU workers,
  :func:`content_type.content_type_step` over the same Harrier leaf on CPU workers,
  and :func:`bucket.quality_step` over those two outputs. Every step lands at its
  own identity, so a run never touches what :mod:`experiments.datakit.hero_data`
  registers; registering its output as hero data is a separate, deliberate edit.
* ``bucket`` reruns only the bucket step over the inputs :mod:`hero_data` pins, at
  the identity :func:`hero_data.quality` resolves. A refit calibration moves that
  identity, so this is how a refit reaches the registered data.

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
from experiments.datakit.cluster.quality.fast_transformer.bucket import quality_step
from experiments.datakit.cluster.quality.fast_transformer.content_type import content_type_step
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


def _remote(step: StepSpec, pip_dependency_groups: list[str] | None = None) -> StepSpec:
    return replace(step, fn=remote(step.fn, resources=DRIVER_RESOURCES, pip_dependency_groups=pip_dependency_groups))


def build_pipeline_steps(sources: list[str]) -> list[StepSpec]:
    """Score, type and bucket each source from scratch, each step fed by the one before."""
    steps = []
    for source in sources:
        normalized = hero_data.normalized(source)
        embedding = hero_data.harrier(source)
        # Wrapped before they become deps: the runner schedules a dep by its
        # output path, so the instance it meets first is the one it launches.
        scores = _remote(
            fusion_score_step(
                name=f"datakit/fusion_scores/{source}",
                normalized=normalized,
                embedding=embedding,
                quality_model=hero_data.NEMOTRON_88K,
            ),
            pip_dependency_groups=["gpu"],
        )
        types = _remote(
            content_type_step(
                name=f"datakit/content_type/{source}",
                normalized=normalized,
                embedding=embedding,
                classifier=hero_data.DOMAIN_MLP_V1,
            )
        )
        quality = _remote(
            quality_step(
                name=f"datakit/quality/{source}",
                source=source,
                normalized=normalized,
                scores=scores,
                content_type=types,
                quality_model=hero_data.NEMOTRON_88K,
            )
        )
        steps += [scores, types, quality]
    return steps


def build_bucket_steps(sources: list[str]) -> list[StepSpec]:
    """One bucket step per source over the pinned inputs, at the identity :func:`hero_data.quality` resolves."""
    return [_remote(hero_data.quality_step_for(source)) for source in sources]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pipeline", "bucket"), required=True)
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
    build = build_pipeline_steps if args.stage == "pipeline" else build_bucket_steps
    StepRunner().run(build(sources), max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
