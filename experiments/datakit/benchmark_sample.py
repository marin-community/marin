# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent Zephyr inputs bundled with a normalized benchmark sample."""

from dataclasses import dataclass

from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext

from experiments.datakit.reference_pipeline import PipelineScale, zephyr_datakit_steps

BENCHMARK_SAMPLE_INPUTS_DIR = "_benchmark_inputs"


@dataclass(frozen=True)
class BenchmarkSampleFuzzySteps:
    """Sample-owned MinHash steps and the fuzzy step that consumes them."""

    minhash: dict[str, StepSpec]
    fuzzy_dedup: StepSpec


def benchmark_sample_inputs_prefix(sample_prefix: str) -> str:
    return prefix_join(sample_prefix, BENCHMARK_SAMPLE_INPUTS_DIR)


def benchmark_sample_fuzzy_steps(
    sample_prefix: str,
    sources: dict[str, StepSpec],
    scale: PipelineScale,
    zephyr_context: ZephyrContext,
) -> BenchmarkSampleFuzzySteps:
    """Build canonical MinHash inputs and their fuzzy-dedup consumer."""
    steps = zephyr_datakit_steps(
        sources,
        scale,
        zephyr_context,
        output_path_prefix=benchmark_sample_inputs_prefix(sample_prefix),
    )
    return BenchmarkSampleFuzzySteps(minhash=steps.minhash, fuzzy_dedup=steps.fuzzy_dedup)
