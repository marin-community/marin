# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full grug-moe baseline training mixture lowers to a pure ``StepSpec`` graph:
the Nemotron CC splits plus starcoder and proofpile are all pinned ``Dataset``
handles (existing tokenized data), the mixture resolves each component to its
pinned path, and the Executor is out of the path.
"""

from marin.execution.executor import executor_context
from marin.execution.lazy import lower, materialized_config

from experiments.grug.moe.launch_lazy import grug_moe_baseline_pure
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
)
from experiments.pretraining_datasets.simple_lazy import (
    _PROOFPILE_LLAMA3_PIN,
    _STARCODER_LLAMA3_PIN,
)

# Component key -> (pinned tokenized path, mixture weight) for the non-Nemotron blend.
_EXTRA_COMPONENTS = {
    "tokenized/starcoderdata": (_STARCODER_LLAMA3_PIN, 0.25),
    "tokenized/proofpile_2": (_PROOFPILE_LLAMA3_PIN, 0.055),
}


def test_baseline_lowers_to_pinned_pure_graph():
    ckpt = grug_moe_baseline_pure()
    with executor_context():
        spec = lower(ckpt)

    # One dependency per training component (Nemotron splits + starcoder + proofpile).
    assert len(spec.deps) == len(NEMOTRON_DATASETS) + len(_EXTRA_COMPONENTS)
    pinned_paths = {dep.override_output_path for dep in spec.deps}
    expected = set(NEMOTRON_LLAMA3_OVERRIDES.values()) | {pin for pin, _ in _EXTRA_COMPONENTS.values()}
    assert pinned_paths == expected


def test_mixture_resolves_pinned_paths_and_weights():
    prefix = "gs://marin-golden"
    config = materialized_config(grug_moe_baseline_pure(), prefix)

    for split in NEMOTRON_DATASETS:
        component = config.data.components[f"tokenized/nemotron_cc/{split}"]
        assert component.cache_dir == f"{prefix}/{NEMOTRON_LLAMA3_OVERRIDES[split]}"
        assert config.data.train_weights[f"tokenized/nemotron_cc/{split}"] == NEMOTRON_WEIGHTS[f"nemotron_cc/{split}"]

    for key, (pin, weight) in _EXTRA_COMPONENTS.items():
        assert config.data.components[key].cache_dir == f"{prefix}/{pin}"
        assert config.data.train_weights[key] == weight

    assert config.output_path == f"{prefix}/grug/4_10_baseline_moe_pure/v1"
