# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from levanter.data.text import HfDatasetSourceConfig
from marin.evaluation.perplexity_gap import _to_dataset_component

from experiments.evals.synthetic_reasoning_ppl import (
    CLRS_STYLE_TASKS,
    DEV_SEED_BASE,
    EXAMPLES_PER_SUBCORPUS,
    NATIVE_ICL_TASKS,
    NATIVE_TASKS,
    STEPMATH_ICL_TASKS,
    STEPMATH_TASKS,
    SYNTHETIC_REASONING_HF_DATASET_ID,
    SYNTHETIC_REASONING_PPL_SLICES,
    SyntheticReasoningFamily,
    synthetic_reasoning_raw_validation_sets,
)


def test_synthetic_reasoning_registry_covers_uploaded_subcorpora() -> None:
    assert len(SYNTHETIC_REASONING_PPL_SLICES) == 44
    assert {slice_.hf_config_name for slice_ in SYNTHETIC_REASONING_PPL_SLICES} == {
        *(f"{SyntheticReasoningFamily.STEPMATH.value}_{task}" for task in STEPMATH_TASKS),
        *(f"{SyntheticReasoningFamily.STEPMATH.value}_{task}" for task, _ in STEPMATH_ICL_TASKS),
        *(f"{SyntheticReasoningFamily.NATIVE.value}_{task}" for task in NATIVE_TASKS),
        *(f"{SyntheticReasoningFamily.NATIVE.value}_{task}" for task, _ in NATIVE_ICL_TASKS),
        *(f"{SyntheticReasoningFamily.CLRS_STYLE.value}_{task}" for task in CLRS_STYLE_TASKS),
    }
    assert len({slice_.registry_key for slice_ in SYNTHETIC_REASONING_PPL_SLICES}) == len(SYNTHETIC_REASONING_PPL_SLICES)


def test_synthetic_reasoning_raw_validation_sets_use_hf_configs() -> None:
    datasets = synthetic_reasoning_raw_validation_sets()

    binary_search_key = "synthetic_reasoning_ppl/native/binary_search"
    algebra_key = "synthetic_reasoning_ppl/stepmath/algebra_linear_equation"
    clrs_key = "synthetic_reasoning_ppl/clrs_style/clrs_binary_search"

    assert datasets[binary_search_key].hf_dataset_id == SYNTHETIC_REASONING_HF_DATASET_ID
    assert datasets[binary_search_key].hf_dataset_name == "native_binary_search"
    assert datasets[binary_search_key].split == "validation"
    assert datasets[binary_search_key].text_key == "text"
    assert datasets[binary_search_key].tags == (
        "synthetic_reasoning_ppl",
        "epic:5005",
        "issue:4148",
        "issue:5052",
        "family:native",
        "task:binary_search",
        f"seed:{DEV_SEED_BASE}",
        f"examples:{EXAMPLES_PER_SUBCORPUS}",
        "source_commit:c4a59c3e1",
    )

    assert datasets[algebra_key].hf_dataset_name == "stepmath_algebra_linear_equation"
    assert datasets[clrs_key].hf_dataset_name == "clrs_style_clrs_binary_search"

    arithmetic_icl = datasets["synthetic_reasoning_ppl/stepmath/arithmetic_5shot_icl"]
    assert arithmetic_icl.hf_dataset_name == "stepmath_arithmetic_5shot_icl"
    assert arithmetic_icl.tags[-1] == "icl_shots:5"


def test_synthetic_reasoning_hf_dataset_component_preserves_validation_split() -> None:
    datasets = synthetic_reasoning_raw_validation_sets()
    component = _to_dataset_component(datasets["synthetic_reasoning_ppl/clrs_style/clrs_bfs"])

    assert component.split == "validation"
    assert component.format.text_key == "text"
    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.id == SYNTHETIC_REASONING_HF_DATASET_ID
    assert component.source.name == "clrs_style_clrs_bfs"
    assert component.source.splits == ["validation"]
