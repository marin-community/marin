# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ResourceConfig

from experiments.evals.long_context_ppl import (
    long_context_raw_validation_sets,
    long_context_supervised_validation_sets,
    long_context_validation_sets,
)
from experiments.evals.perplexity_gap_registry import (
    LONG_CONTEXT_32K_EVAL_LENGTH,
    LONG_CONTEXT_64K_EVAL_LENGTH,
    LONG_CONTEXT_MAX_DOC_BYTES,
    LONG_CONTEXT_MAX_DOCS_PER_DATASET,
    build_registered_perplexity_gap_coverage_plan,
    long_context_64k_bundle,
    long_context_bundle,
    registered_perplexity_gap_bundles,
)

EXPECTED_RAW_KEYS = {
    "long_context/pg19_test",
    "long_context/govreport_validation",
}

EXPECTED_SUPERVISED_KEYS = {
    "long_context/scrolls_qasper",
    "long_context/scrolls_narrative_qa",
    "long_context/scrolls_quality",
}


def test_raw_long_context_slices_are_text_only_hf_sources() -> None:
    datasets = long_context_raw_validation_sets()

    assert set(datasets) == EXPECTED_RAW_KEYS
    for dataset in datasets.values():
        assert dataset.hf_dataset_id is not None
        assert dataset.input_path is None
        assert dataset.input_key is None
        assert dataset.target_key is None


def test_supervised_long_context_slices_set_input_and_target_keys() -> None:
    datasets = long_context_supervised_validation_sets()

    assert set(datasets) == EXPECTED_SUPERVISED_KEYS
    for dataset in datasets.values():
        assert dataset.hf_dataset_id == "tau/scrolls"
        assert dataset.input_key == "input"
        assert dataset.target_key == "output"
        assert dataset.split == "validation"


def test_union_factory_combines_raw_and_supervised_slices() -> None:
    combined = long_context_validation_sets()

    assert set(combined) == EXPECTED_RAW_KEYS | EXPECTED_SUPERVISED_KEYS


def test_default_long_context_bundle_uses_32k_budget() -> None:
    bundle = long_context_bundle()

    assert bundle.key == "long_context_32k"
    assert bundle.max_eval_length == LONG_CONTEXT_32K_EVAL_LENGTH == 32_768
    assert bundle.max_docs_per_dataset == LONG_CONTEXT_MAX_DOCS_PER_DATASET
    assert bundle.max_doc_bytes == LONG_CONTEXT_MAX_DOC_BYTES


def test_64k_bundle_is_opt_in_diagnostic_tier() -> None:
    bundle = long_context_64k_bundle()

    assert bundle.key == "long_context_64k"
    assert bundle.max_eval_length == LONG_CONTEXT_64K_EVAL_LENGTH == 65_536
    # Same slice set as the 32K bundle — only the length tier changes.
    assert set(bundle.datasets()) == EXPECTED_RAW_KEYS | EXPECTED_SUPERVISED_KEYS

    # Opt-in: not part of the default registered bundles tuple.
    registered_keys = {b.key for b in registered_perplexity_gap_bundles()}
    assert "long_context_64k" not in registered_keys
    assert "long_context_32k" in registered_keys


def test_coverage_plan_propagates_long_context_caps_to_score_step() -> None:
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
    )

    score_step = plan.score_steps[("long_context_32k", "marin_8b")]
    assert score_step.config.max_eval_length == LONG_CONTEXT_32K_EVAL_LENGTH
    assert score_step.config.max_doc_bytes == LONG_CONTEXT_MAX_DOC_BYTES
    assert score_step.config.max_docs_per_dataset == LONG_CONTEXT_MAX_DOCS_PER_DATASET
    assert set(score_step.config.datasets) == EXPECTED_RAW_KEYS | EXPECTED_SUPERVISED_KEYS


def test_opt_in_64k_bundle_propagates_caps_when_passed_explicitly() -> None:
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
        bundles=[long_context_64k_bundle()],
    )

    score_step = plan.score_steps[("long_context_64k", "marin_8b")]
    assert score_step.config.max_eval_length == LONG_CONTEXT_64K_EVAL_LENGTH
    assert score_step.config.max_doc_bytes == LONG_CONTEXT_MAX_DOC_BYTES
