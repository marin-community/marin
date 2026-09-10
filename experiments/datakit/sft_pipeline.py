# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deduplication and eval-contamination attributes for rendered SFT conversations."""

import argparse
import json
import logging
from dataclasses import dataclass, replace

from marin.datakit.sft_sources import DatakitChatSource, all_sft_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PipelineScale,
    decontamination_steps,
    verified_dedup_step,
    zephyr_datakit_steps,
)


@dataclass(frozen=True)
class SftFilterSteps:
    """Attribute outputs keyed to normalized rendered-text IDs."""

    normalized: dict[str, StepSpec]
    exact_dedup: StepSpec
    verified_dedup: StepSpec
    decontam: dict[str, StepSpec]

    @property
    def targets(self) -> list[StepSpec]:
        return [self.exact_dedup, self.verified_dedup, *self.decontam.values()]


def sft_filter_steps(
    sources: dict[str, DatakitChatSource],
    scale: PipelineScale = SMOKE_SCALE,
    zephyr_context: ZephyrContext | None = None,
) -> SftFilterSteps:
    """Build exact/fuzzy dedup and decontamination over rendered conversations.

    Source normalization depends on the canonical render step, so existing
    renders are reused by StepRunner. Only attribute stages are targeted;
    filtering rows into a training dataset is a separate downstream operation.
    """
    if not sources:
        raise ValueError("Select at least one SFT source")
    normalized = {name: source.normalized for name, source in sorted(sources.items())}
    dedup = zephyr_datakit_steps(normalized, scale, zephyr_context)
    _, _, decontam = decontamination_steps(normalized, scale, zephyr_context)
    verified = verified_dedup_step(normalized, dedup.minhash, dedup.fuzzy_dedup, scale)
    return SftFilterSteps(normalized, dedup.exact_dedup, verified, decontam)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", help="SFT registry names; omit for all sources")
    parser.add_argument("--execute", action="store_true", help="Run the DAG; otherwise print target artifact paths")
    parser.add_argument("--pool-workers", type=int, default=SMOKE_SCALE.pool.n_workers)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    if args.pool_workers < 1 or args.max_concurrent < 1:
        parser.error("worker and concurrency limits must be positive")
    configure_logging(logging.INFO)
    registry = all_sft_sources()
    names = args.sources if args.sources is not None else sorted(registry)
    unknown = sorted(set(names) - registry.keys())
    if unknown:
        parser.error(f"Unknown SFT sources: {', '.join(unknown)}")
    sources = {name: registry[name] for name in names}
    scale = replace(SMOKE_SCALE, pool=replace(SMOKE_SCALE.pool, n_workers=args.pool_workers))
    if not args.execute:
        result = sft_filter_steps(sources, scale)
        print(json.dumps({step.name: step.output_path for step in result.targets}, indent=2))
        return
    with ZephyrContext(
        name="datakit-sft-filter",
        resources=scale.pool.worker,
        max_workers=scale.pool.n_workers,
        stage_runner_factory=SubprocessRunner,
    ) as context:
        result = sft_filter_steps(sources, scale, context)
        StepRunner().run(result.targets, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
