# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.probabilistic_dataflow.compiler import (
    AttentionLayout,
    ConcreteExample,
    TokenCodec,
    inference_plan_ir,
    lower_to_transformer,
)
from experiments.probabilistic_dataflow.debug_render import check_example_outputs, render_example_outputs
from experiments.probabilistic_dataflow.dsl import (
    AttentionPattern,
    Budget,
    DocumentSpec,
    FieldType,
    InferenceProgram,
    InferenceRoleDifference,
    PositionMode,
    UnorderedPairAxis,
    training_deployment_differences,
)
from experiments.probabilistic_dataflow.synthetic import (
    advection_example,
    advection_program,
    refined_advection_program,
    scalar_forecast_example,
    scalar_forecast_program,
    structure_program,
)
from experiments.probabilistic_dataflow.training import (
    build_mixed_synthetic_batch,
    build_synthetic_advection_batch,
    build_synthetic_text_batch,
    record_order_equivariance_error,
    train_cross_domain_smoke,
    train_smoke,
)


def test_scalar_tutorial_lowers_to_one_context_and_one_target_record() -> None:
    program = scalar_forecast_program()
    plan = inference_plan_ir(program)
    codec = TokenCodec()
    execution = lower_to_transformer(
        program,
        scalar_forecast_example(program),
        codec,
    )

    current = program.value("current")
    assert program.external_inputs == (current,)
    assert program.outputs == (program.value("future"),)
    assert plan.calls[0].context_ids == (current.node_id,)
    assert plan.calls[0].target_ids == (program.value("future").node_id,)
    assert plan.calls[0].approximation_notes == ()
    assert plan.calls[0].attention_layout == AttentionLayout.FULL
    assert plan.calls[0].position_mode == PositionMode.SCIENTIFIC

    sequence = execution.calls[0].sequences[0]
    assert sequence.token_ids == (codec.data(3), codec.QUERY_ID)
    assert sequence.target_ids == (-1, codec.data(5))
    assert sequence.loss_weights == (0.0, 1.0)
    assert sequence.rotary_position_ids == (0, 0)


def test_document_policy_selects_causal_attention_and_sequence_positions() -> None:
    scalar = FieldType("scalar", bins=4)
    program = InferenceProgram(
        "ordered_scalar",
        budget=Budget(model_calls=1, generated_tokens=1),
    )
    context = program.input_value("context", scalar)
    target = program.generate(
        "target",
        scalar,
        context=(context,),
        document=DocumentSpec(attention=AttentionPattern.CAUSAL, positions=PositionMode.SEQUENCE),
        factor_name="ordered_transition",
    )
    program.finish(target)

    execution = lower_to_transformer(
        program,
        ConcreteExample("ordered-0", program.name, {"context": (1,), "target": (2,)}),
        TokenCodec(),
    )
    call = execution.calls[0]
    sequence = call.sequences[0]

    assert call.attention_layout == AttentionLayout.CAUSAL
    assert call.position_mode == PositionMode.SEQUENCE
    assert sequence.scientific_position_ids == (-1, -1)
    assert sequence.rotary_position_ids == (0, 1)


def test_generated_value_flow_preserves_factor_order() -> None:
    program = structure_program()
    contacts = program.value("contacts")
    distances = program.value("distances")
    compiled = inference_plan_ir(program)
    assert [call.target_ids for call in compiled.calls] == [(contacts.node_id,), (distances.node_id,)]
    assert compiled.calls[1].dependency_call_ids == (compiled.calls[0].id,)
    assert contacts.node_id in compiled.calls[1].context_ids
    assert compiled.output_ids == (contacts.node_id, distances.node_id)


def test_refinement_feedback_is_an_explicit_call_dependency() -> None:
    program = refined_advection_program()
    compiled = inference_plan_ir(program)

    assert [call.operator for call in compiled.calls] == ["generate", "refine", "refine"]
    assert [call.dependency_call_ids for call in compiled.calls] == [(), (0,), (1,)]
    assert program.value("future").node_id in compiled.calls[1].context_ids


def test_training_deployment_report_identifies_teacher_forced_target() -> None:
    program = advection_program()
    initial = program.value("initial")
    forcing = program.value("forcing")
    future = program.value("future")
    differences = training_deployment_differences(
        program,
        training_given=(initial, forcing, future),
    )
    assert differences == (InferenceRoleDifference("future", training_role="given", deployment_role="generated"),)


def test_parallel_field_lowering_records_approximation_and_target_alignment() -> None:
    program = advection_program()
    compiled = inference_plan_ir(program)

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
    program = advection_program()
    original = advection_example(program, seed=0)
    changed = type(original)(
        id=original.id,
        program_name=original.program_name,
        values={
            **original.values,
            "future": tuple((value + 1) % 16 for value in original.values["future"]),
        },
    )
    codec = TokenCodec()

    original_sequence = lower_to_transformer(program, original, codec).calls[0].sequences[0]
    changed_sequence = lower_to_transformer(program, changed, codec).calls[0].sequences[0]

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
    program = structure_program()
    pair_axis = program.value("contacts").value_type.axes[0]
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
    assert "| external inputs | %0 current |" in scalar
    assert "available_at" not in scalar
    assert "| 0 | generate | 0 | %0 current | %1 future | - | full_segment | scientific | - |" in scalar

    advection = outputs["advection.md"]
    assert "## 1. Inference Program Values" in advection
    assert "Conditional Query IR" not in advection
    assert "## 2. Inference Plan IR" in advection
    assert "## 3. Transformer Execution IR" in advection
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
