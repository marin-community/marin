# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grug-moe baseline on Nemotron CC lowers to a pure ``StepSpec`` graph: the
splits are pinned ``Dataset`` handles (existing tokenized data), the mixture
resolves each component to its pinned path, and the Executor is out of the path.
"""

from marin.execution.executor import executor_context
from marin.execution.lazy import lower, materialized_config

from experiments.grug.moe.launch_lazy import grug_moe_baseline_nemotron
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
)


def test_nemotron_baseline_lowers_to_pinned_pure_graph():
    ckpt = grug_moe_baseline_nemotron()
    with executor_context():
        spec = lower(ckpt)

    # One dependency per Nemotron split, each pinned to its existing tokenized path.
    assert len(spec.deps) == len(NEMOTRON_DATASETS)
    deps_by_path = {dep.override_output_path for dep in spec.deps}
    assert deps_by_path == set(NEMOTRON_LLAMA3_OVERRIDES.values())


def test_mixture_resolves_pinned_paths_and_weights():
    prefix = "gs://marin-golden"
    config = materialized_config(grug_moe_baseline_nemotron(), prefix)

    for split in NEMOTRON_DATASETS:
        component = config.data.components[f"tokenized/nemotron_cc/{split}"]
        # Pinned to the existing tokenized location — never recomputed.
        assert component.cache_dir == f"{prefix}/{NEMOTRON_LLAMA3_OVERRIDES[split]}"
        assert config.data.train_weights[f"tokenized/nemotron_cc/{split}"] == NEMOTRON_WEIGHTS[f"nemotron_cc/{split}"]

    assert config.output_path == f"{prefix}/grug/4_10_baseline_moe_nemotron/v1"
