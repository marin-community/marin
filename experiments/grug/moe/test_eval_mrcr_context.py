# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.datasets.mrcr import MrcrPromptVariant
from experiments.grug.moe.eval_mrcr_context import (
    PACKAGE_BY_NAME,
    MrcrEvaluationArtifact,
    MrcrEvaluationKey,
    aggregate_262k_evaluation_cells,
    build_default_steps,
    expected_evaluations_for_stage,
    matrix_cell_count,
    primary_evaluation_cells,
    sensitivity_evaluation_cells,
    smoke_evaluation_cells,
    summarize_mrcr_matrix,
)


def _write_losses(
    path: Path,
    gains: tuple[float, ...],
    *,
    tokens: tuple[int, ...] = (1, 3),
    source_ids: tuple[str, ...] = ("a", "b"),
) -> None:
    with path.open("w") as stream:
        for source_id, gain, scored_tokens in zip(source_ids, gains, tokens, strict=True):
            row = {
                "source_id": source_id,
                "prompt_variant": "two_shot",
                "context_cap": 8192,
                "n_needles": 2,
                "distance_band": "distance_0_32768",
                "evidence_distance_tokens": 100,
                "scored_tokens": scored_tokens,
                "full_context_loss_sum": 2.0 * scored_tokens,
                "query_only_loss_sum": (2.0 + gain) * scored_tokens,
            }
            stream.write(json.dumps(row) + "\n")


def _artifact(
    tmp_path: Path,
    package_name: str,
    variant: MrcrPromptVariant,
    gains: tuple[float, ...],
    *,
    suffix: str = "",
) -> MrcrEvaluationArtifact:
    package = PACKAGE_BY_NAME[package_name]
    path = tmp_path / f"{package_name}-{variant}{suffix}.jsonl"
    _write_losses(path, gains)
    return MrcrEvaluationArtifact(
        package_name=package_name,
        prompt_variant=variant,
        context_cap=8192,
        qk_mult=package.qk_mult,
        training_offset=package.training_offset,
        baseline_package_name=package.baseline_package_name,
        example_losses_path=str(path),
    )


def _read_comparisons(output_path: Path) -> list[dict]:
    return [json.loads(line) for line in (output_path / "mrcr_matrix_comparisons.jsonl").read_text().splitlines()]


def _comparison(rows: list[dict], kind: str) -> dict:
    return next(row for row in rows if row["kind"] == kind)


def test_matrix_contract_contains_sixty_primary_and_sixteen_sensitivity_jobs():
    assert matrix_cell_count() == (60, 16)
    assert len(primary_evaluation_cells()) == 60
    assert len(sensitivity_evaluation_cells()) == 16
    assert len(smoke_evaluation_cells()) == 4
    assert {
        cell.package.checkpoint_step for cell in sensitivity_evaluation_cells() if cell.package.training_offset > 0
    } == {157_000}


def test_smoke_builder_exposes_four_independent_cells_with_validation_dependencies():
    steps = build_default_steps("smoke")

    assert len(steps) == 4
    assert {step.config.evaluation.context_cap for step in steps} == {8192, 32768}
    assert {step.config.evaluation.qk_mult for step in steps} == {1.57, 1.75}
    assert all(step.config.evaluation.dataset_stats_path is not None for step in steps)
    assert all(len(step.config.evaluation.dataset_manifest_paths) == 3 for step in steps)
    assert all(step.config.resources.value.preemptible is False for step in steps)


def test_v4_32_smoke_builder_uses_full_data_sharding_and_distinct_outputs():
    steps = build_default_steps("smoke", tpu_variant="v4-32")

    assert len(steps) == 4
    assert all(step.config.resources.value.device.variant == "v4-32" for step in steps)
    assert all(step.config.evaluation.runtime.value.eval_batch_size == 16 for step in steps)
    assert all(step.config.evaluation.runtime.value.data_axis_size == 16 for step in steps)
    assert all(step.config.evaluation.runtime.value.context_axis_size == 1 for step in steps)
    assert all(step.config.resources.value.preemptible is True for step in steps)
    assert all(step.name.endswith("-v432") for step in steps)

    with pytest.raises(ValueError, match="bounded smoke"):
        build_default_steps("primary", tpu_variant="v4-32")


def test_v4_64_smoke_builder_uses_full_data_sharding_without_context_parallelism():
    steps = build_default_steps("smoke", tpu_variant="v4-64")

    assert len(steps) == 4
    assert all(step.config.resources.value.device.variant == "v4-64" for step in steps)
    assert all(step.config.evaluation.runtime.value.eval_batch_size == 32 for step in steps)
    assert all(step.config.evaluation.runtime.value.data_axis_size == 32 for step in steps)
    assert all(step.config.evaluation.runtime.value.context_axis_size == 1 for step in steps)
    assert all(step.config.resources.value.preemptible is True for step in steps)
    assert all(step.name.endswith("-v464") for step in steps)

    with pytest.raises(ValueError, match="bounded smoke"):
        build_default_steps("primary", tpu_variant="v4-64")


def test_aggregate_262k_builder_uses_matched_qk_trajectory_and_context_parallelism():
    steps = build_default_steps("aggregate_262k", tpu_variant="v4-128-cp4")
    probe_steps = build_default_steps("aggregate_262k_probe", tpu_variant="v4-128-cp4")
    fallback_steps = build_default_steps("aggregate_262k_probe", tpu_variant="v4-64-cp2")

    assert len(aggregate_262k_evaluation_cells()) == 3
    assert len(steps) == 3
    assert {step.config.evaluation.context_cap for step in steps} == {262_144}
    assert {step.config.evaluation.qk_mult for step in steps} == {1.57}
    assert {step.config.evaluation.run_id.split("-qk", 1)[0] for step in steps} == {
        "mrcr-67b-step141000",
        "mrcr-67b-step156000",
        "mrcr-67b-step157000",
    }
    assert all(step.config.resources.value.device.variant == "v4-128" for step in steps)
    assert all(step.config.resources.value.preemptible is True for step in steps)
    assert all(step.config.evaluation.runtime.value.eval_batch_size == 16 for step in steps)
    assert all(step.config.evaluation.runtime.value.data_axis_size == 16 for step in steps)
    assert all(step.config.evaluation.runtime.value.context_axis_size == 4 for step in steps)
    assert all(step.name.endswith("-v4128cp4") for step in steps)
    assert len(probe_steps) == 1
    assert probe_steps[0].config.evaluation.run_id.startswith("mrcr-67b-step141000-")
    assert fallback_steps[0].config.resources.value.device.variant == "v4-64"
    assert fallback_steps[0].config.evaluation.runtime.value.eval_batch_size == 16
    assert fallback_steps[0].config.evaluation.runtime.value.data_axis_size == 16
    assert fallback_steps[0].config.evaluation.runtime.value.context_axis_size == 2
    assert fallback_steps[0].name.endswith("-v464cp2")

    with pytest.raises(ValueError, match="aggregate_262k requires a context-parallel TPU shape"):
        build_default_steps("aggregate_262k")


def test_summary_computes_adaptation_arm_qk_and_difference_in_differences(tmp_path: Path):
    artifacts = (
        _artifact(tmp_path, "step-156000-source-qk157", MrcrPromptVariant.TWO_SHOT, (0.0, 0.0)),
        _artifact(tmp_path, "step-156000-source-qk175", MrcrPromptVariant.TWO_SHOT, (1.0, 1.0)),
        _artifact(tmp_path, "qk157-step156250", MrcrPromptVariant.TWO_SHOT, (2.0, 2.0)),
        _artifact(tmp_path, "qk175-step156250", MrcrPromptVariant.TWO_SHOT, (4.0, 4.0)),
    )
    output_path = tmp_path / "summary"
    summarize_mrcr_matrix(
        artifacts,
        summary_stage="offset_250",
        expected_evaluations=expected_evaluations_for_stage("offset_250"),
        output_path=str(output_path),
        claim_gain_floor=0.0,
        claim_min_examples=2,
        bootstrap_samples=100,
    )

    rows = _read_comparisons(output_path)
    adaptations = {row["packages"][0]: row for row in rows if row["kind"] == "adaptation"}
    assert adaptations["qk157-step156250"]["micro_difference"] == pytest.approx(2.0)
    assert adaptations["qk175-step156250"]["macro_difference"] == pytest.approx(3.0)
    assert _comparison(rows, "source_inference_qk")["micro_difference"] == pytest.approx(1.0)
    assert _comparison(rows, "deployable_arm")["micro_difference"] == pytest.approx(2.0)
    assert _comparison(rows, "difference_in_differences")["micro_difference"] == pytest.approx(1.0)


def test_summary_computes_shot_and_prefix_sensitivity_with_paired_sources(tmp_path: Path):
    package_name = "step-156000-source-qk157"
    artifacts = (
        _artifact(tmp_path, package_name, MrcrPromptVariant.TWO_SHOT, (0.0, 2.0)),
        _artifact(tmp_path, package_name, MrcrPromptVariant.ONE_SHOT, (1.0, 3.0)),
        _artifact(tmp_path, package_name, MrcrPromptVariant.TWO_SHOT_NO_PREFIX, (2.0, 4.0)),
    )
    expected = tuple(MrcrEvaluationKey(package_name, variant) for variant in MrcrPromptVariant)
    output_path = tmp_path / "summary"
    summarize_mrcr_matrix(
        artifacts,
        summary_stage="prompt_sensitivity",
        expected_evaluations=expected,
        output_path=str(output_path),
        claim_gain_floor=0.0,
        claim_min_examples=2,
        bootstrap_samples=200,
        bootstrap_seed=17,
    )

    rows = _read_comparisons(output_path)
    assert _comparison(rows, "shot_sensitivity")["micro_difference"] == pytest.approx(1.0)
    assert _comparison(rows, "prefix_sensitivity")["macro_difference"] == pytest.approx(2.0)


def test_summary_bootstrap_is_fixed_seed_reproducible_and_preserves_pairing(tmp_path: Path):
    artifacts = (
        _artifact(tmp_path, "step-156000-source-qk157", MrcrPromptVariant.TWO_SHOT, (0.0, 4.0)),
        _artifact(tmp_path, "step-156000-source-qk175", MrcrPromptVariant.TWO_SHOT, (2.0, 2.0)),
    )
    expected = expected_evaluations_for_stage("source_qk")
    for directory in (tmp_path / "first", tmp_path / "second"):
        summarize_mrcr_matrix(
            artifacts,
            summary_stage="source_qk",
            expected_evaluations=expected,
            output_path=str(directory),
            claim_gain_floor=-10.0,
            claim_min_examples=1,
            bootstrap_samples=500,
            bootstrap_seed=9,
        )

    assert (tmp_path / "first" / "mrcr_matrix_comparisons.jsonl").read_bytes() == (
        tmp_path / "second" / "mrcr_matrix_comparisons.jsonl"
    ).read_bytes()
    row = _comparison(_read_comparisons(tmp_path / "first"), "source_inference_qk")
    assert row["micro_difference"] == pytest.approx(-1.0)
    assert row["macro_difference"] == pytest.approx(0.0)
    assert row["micro_difference_ci95_low"] <= row["micro_difference"] <= row["micro_difference_ci95_high"]


def test_partial_summary_lists_missing_and_blocked_comparisons(tmp_path: Path):
    expected = expected_evaluations_for_stage("offset_250")
    artifacts = (_artifact(tmp_path, "step-156000-source-qk157", MrcrPromptVariant.TWO_SHOT, (0.0, 1.0)),)
    output_path = tmp_path / "summary"
    summarize_mrcr_matrix(
        artifacts,
        summary_stage="offset_250",
        expected_evaluations=expected,
        output_path=str(output_path),
        claim_gain_floor=0.05,
        claim_min_examples=20,
        bootstrap_samples=20,
    )

    summary = json.loads((output_path / "summary.json").read_text())
    assert summary["complete"] is False
    assert len(summary["available_evaluations"]) == 1
    assert len(summary["missing_evaluations"]) == 3
    assert {row["kind"] for row in summary["blocked_comparisons"]} == {
        "adaptation",
        "deployable_arm",
        "difference_in_differences",
        "source_inference_qk",
    }
    assert summary["claim_gain_floor"] == 0.05
    assert summary["claim_min_examples"] == 20


def test_summary_rejects_mismatched_source_ids(tmp_path: Path):
    left = _artifact(tmp_path, "step-156000-source-qk157", MrcrPromptVariant.TWO_SHOT, (0.0, 1.0))
    right = _artifact(tmp_path, "step-156000-source-qk175", MrcrPromptVariant.TWO_SHOT, (0.0, 1.0))
    _write_losses(Path(right.example_losses_path), (0.0, 1.0), source_ids=("a", "c"))

    with pytest.raises(ValueError, match="Mismatched source IDs"):
        summarize_mrcr_matrix(
            (left, right),
            summary_stage="source_qk",
            expected_evaluations=expected_evaluations_for_stage("source_qk"),
            output_path=str(tmp_path / "summary"),
            claim_gain_floor=0.0,
            claim_min_examples=1,
            bootstrap_samples=10,
        )


def test_summary_retry_is_idempotent_but_conflicting_output_fails(tmp_path: Path):
    artifacts = (
        _artifact(tmp_path, "step-156000-source-qk157", MrcrPromptVariant.TWO_SHOT, (0.0, 1.0)),
        _artifact(tmp_path, "step-156000-source-qk175", MrcrPromptVariant.TWO_SHOT, (1.0, 2.0)),
    )
    kwargs = dict(
        summary_stage="source_qk",
        expected_evaluations=expected_evaluations_for_stage("source_qk"),
        output_path=str(tmp_path / "summary"),
        claim_gain_floor=0.0,
        claim_min_examples=1,
        bootstrap_samples=20,
    )
    summarize_mrcr_matrix(artifacts, **kwargs)
    summarize_mrcr_matrix(artifacts, **kwargs)

    with pytest.raises(ValueError, match="Conflicting MRCR summary output"):
        summarize_mrcr_matrix(artifacts, **(kwargs | {"bootstrap_seed": 1}))
