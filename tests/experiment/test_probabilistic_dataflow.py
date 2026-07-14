# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.compiler import (
    AttentionLayout,
    Autoregressive,
    AvailabilityError,
    FactorizationError,
    ParallelQuery,
    Refine,
    TokenCodec,
    compile_query,
    lower_to_transformer,
)
from experiments.probabilistic_dataflow.debug_render import check_example_outputs, render_example_outputs
from experiments.probabilistic_dataflow.dsl import (
    QueryRoleDifference,
    UnorderedPairAxis,
    training_deployment_differences,
)
from experiments.probabilistic_dataflow.synthetic import (
    advection_example,
    advection_problem,
    factorized_structure_problem,
    leaky_normalization_problem,
    scalar_forecast_example,
    scalar_forecast_problem,
)
from experiments.probabilistic_dataflow.training import (
    build_mixed_synthetic_batch,
    build_synthetic_advection_batch,
    build_synthetic_text_batch,
    record_order_equivariance_error,
    train_cross_domain_smoke,
    train_smoke,
)


def test_indirect_environment_leakage_is_rejected_with_provenance() -> None:
    problem = leaky_normalization_problem()

    with pytest.raises(AvailabilityError, match=r"synthetic\.future_statistics"):
        compile_query(problem.query)


def test_scalar_tutorial_lowers_to_one_context_and_one_target_record() -> None:
    problem = scalar_forecast_problem()
    plan = compile_query(problem.query)
    codec = TokenCodec()
    execution = lower_to_transformer(
        problem.program,
        plan,
        scalar_forecast_example(problem),
        codec,
    )

    current = problem.program.value("current")
    assert problem.query.given == (current,)
    assert problem.query.environment == "deployment"
    assert plan.calls[0].context_ids == (current.node_id,)
    assert plan.calls[0].target_ids == (problem.program.value("future").node_id,)
    assert plan.calls[0].approximation_notes == ()

    sequence = execution.calls[0].sequences[0]
    assert sequence.token_ids == (codec.data(3), codec.QUERY_ID)
    assert sequence.target_ids == (-1, codec.data(5))
    assert sequence.loss_weights == (0.0, 1.0)
    assert sequence.rotary_position_ids == (0, 0)


def test_dependent_factors_require_an_ordered_plan() -> None:
    problem, _sequence, contacts, distances = factorized_structure_problem()

    with pytest.raises(FactorizationError, match="depends on the former"):
        compile_query(problem.query, ParallelQuery((contacts, distances)))

    compiled = compile_query(problem.query, Autoregressive((contacts, distances)))
    assert [call.target_ids for call in compiled.calls] == [(contacts.node_id,), (distances.node_id,)]
    assert compiled.calls[1].dependency_call_ids == (compiled.calls[0].id,)
    assert contacts.node_id in compiled.calls[1].context_ids


def test_refinement_feedback_is_an_explicit_call_dependency() -> None:
    problem = advection_problem()
    compiled = compile_query(
        problem.query,
        Refine(ParallelQuery(problem.targets), steps=3, resample_fraction=0.25),
    )

    assert [call.operator for call in compiled.calls] == ["parallel", "refine", "refine"]
    assert [call.dependency_call_ids for call in compiled.calls] == [(), (0,), (1,)]
    assert problem.targets[0].node_id in compiled.calls[1].context_ids


def test_training_deployment_report_identifies_teacher_forced_target() -> None:
    problem = advection_problem()
    initial = problem.program.value("initial")
    forcing = problem.program.value("forcing")
    future = problem.program.value("future")
    differences = training_deployment_differences(
        problem.program,
        training_given=(initial, forcing, future),
        deployment_given=problem.query.given,
    )
    assert differences == (QueryRoleDifference("future", training_role="given", deployment_role="generated"),)


def test_parallel_field_lowering_records_approximation_and_target_alignment() -> None:
    problem = advection_problem()
    compiled = compile_query(problem.query, ParallelQuery(problem.targets))

    assert compiled.calls[0].approximation_notes == (
        "factor synthetic_advection:future:2 is approximated as 12 parallel token marginals",
    )

    batch, _codec = build_mixed_synthetic_batch(examples_per_problem=2, max_seq_len=64)
    supervised = batch.loss_weights > 0
    assert np.all(batch.token_ids[supervised] == TokenCodec.QUERY_ID)
    assert np.all(batch.target_ids[supervised] >= TokenCodec.DATA_OFFSET)
    assert np.all(batch.scientific_position_ids[batch.segment_ids >= 0] >= 0)
    assert {location.example_id.split("-", 1)[0] for location in batch.locations} == {"advection", "contacts"}
    assert batch.token_ids.shape[0] < len(batch.locations)


def test_target_values_are_aligned_labels_and_not_model_inputs() -> None:
    problem = advection_problem()
    plan = compile_query(problem.query, ParallelQuery(problem.targets))
    original = advection_example(problem, seed=0)
    changed = type(original)(
        id=original.id,
        program_name=original.program_name,
        values={
            **original.values,
            "future": tuple((value + 1) % 16 for value in original.values["future"]),
        },
    )
    codec = TokenCodec()

    original_sequence = lower_to_transformer(problem.program, plan, original, codec).calls[0].sequences[0]
    changed_sequence = lower_to_transformer(problem.program, plan, changed, codec).calls[0].sequences[0]

    assert original_sequence.token_ids == changed_sequence.token_ids
    assert original_sequence.scientific_position_ids == changed_sequence.scientific_position_ids
    assert original_sequence.rotary_position_ids == changed_sequence.rotary_position_ids
    assert original_sequence.target_ids != changed_sequence.target_ids


def test_scientific_record_logits_are_equivariant_to_serialization_order() -> None:
    assert record_order_equivariance_error() < 1e-5


def test_text_and_science_calls_share_vocabulary_with_data_dependent_execution() -> None:
    codec = TokenCodec()
    text = build_synthetic_text_batch(codec, repetitions=1)
    science = build_synthetic_advection_batch(codec, examples=1, max_seq_len=32)

    assert text.attention_layout == AttentionLayout.CAUSAL
    assert np.all(text.scientific_position_ids == -1)
    assert np.array_equal(text.rotary_position_ids[0], np.arange(text.token_ids.shape[1]))
    assert np.array_equal(text.target_ids[:, :-1], text.token_ids[:, 1:])

    assert science.attention_layout == AttentionLayout.FULL
    assert np.all(science.rotary_position_ids == 0)
    assert np.all(science.scientific_position_ids[science.segment_ids >= 0] >= 0)
    assert text.token_ids.max() < TokenCodec.DATA_OFFSET
    science_values = science.token_ids[(science.segment_ids >= 0) & (science.token_ids != TokenCodec.QUERY_ID)]
    assert science_values.min() >= TokenCodec.DATA_OFFSET


def test_unordered_pair_program_is_invariant_to_identity_permutation() -> None:
    problem, *_ = factorized_structure_problem()
    pair_axis = problem.program.value("contacts").value_type.axes[0]
    assert isinstance(pair_axis, UnorderedPairAxis)
    pairs = pair_axis.canonical_pairs()
    assert pairs == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    permutation = (2, 0, 3, 1)

    permuted_pairs = {tuple(sorted((permutation[left], permutation[right]))) for left, right in pairs}
    assert permuted_pairs == set(pairs)


def test_debug_rendering_explains_ir_and_document_treatment() -> None:
    outputs = render_example_outputs()

    scalar = outputs["scalar.md"]
    assert "scalar_forecast.current[scalar]" in scalar
    assert "scalar_forecast.future[scalar]" in scalar
    assert "| given | %0 current |" in scalar
    assert "| environment | deployment |" in scalar
    assert "available_at" not in scalar
    assert "| 0 | parallel | 0 | %0 current | %1 future | - | - |" in scalar

    advection = outputs["advection.md"]
    assert "## 1. Probabilistic Dataflow IR" in advection
    assert "## 2. Conditional Query IR" in advection
    assert "## 3. Inference Plan IR" in advection
    assert "## 4. Transformer Execution IR" in advection
    assert "target record | synthetic_advection.future[time=0,cell=(0.0,)]" in advection
    assert "Physical rotary positions: all 0; RoPE is the identity" in advection
    assert "target value is a label, not an input" in advection
    assert "replace those values with the proposal produced by the preceding call" in advection

    structure = outputs["structure.md"]
    assert "c0 --> c1" in structure
    assert "%0 sequence<br>%1 contacts | %2 distances | 0" in structure

    mixed = outputs["mixed-packing.md"]
    assert "advection-0/call0/doc0" in mixed
    assert "contacts-0/call0/doc0" in mixed
    assert "all 0; RoPE is the identity" in mixed
    assert "segment_id; records cannot attend across documents" in mixed

    cross_domain = outputs["cross-domain.md"]
    assert "same parameters for both calls" in cross_domain
    assert "shifted next-token labels" in cross_domain
    assert "aligned scientific-value labels" in cross_domain


def test_checked_in_debug_outputs_are_current() -> None:
    assert check_example_outputs() == ()


@pytest.mark.slow
def test_existing_grug_model_learns_the_mixed_compiler_output() -> None:
    result = train_smoke(steps=10, examples_per_problem=4)

    assert result.final_loss < result.initial_loss
    assert result.final_accuracy > result.initial_accuracy
    assert result.task_families == ("synthetic_advection", "synthetic_contacts")


@pytest.mark.slow
def test_one_grug_model_learns_causal_text_and_full_attention_science() -> None:
    result = train_cross_domain_smoke(steps=10, examples_per_task=4)

    assert result.final_loss < result.initial_loss
    assert result.text.final_accuracy > result.text.initial_accuracy
    assert result.science.final_accuracy > result.science.initial_accuracy
    assert result.task_families == ("synthetic_text", "synthetic_advection")
