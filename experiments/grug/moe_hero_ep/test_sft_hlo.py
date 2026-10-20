# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe_hero_ep.sft_hlo import audit_gradient_operations


def test_hlo_weight_dataflow_distinguishes_same_shape_activation_and_weight_gradient():
    # Both contractions return a shape matching a frozen projection. There is no
    # autodiff metadata: provenance must cross the while carry and fusion inputs.
    hlo = """
HloModule train
%projection (w: f32[2,40], x: f32[8,2]) -> f32[8,40] {
  %w = f32[2,40] parameter(0)
  %x = f32[8,2] parameter(1)
  ROOT %forward = f32[8,40] dot(%x, %w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
%condition (state: (f32[2,40], f32[8,2], f32[8,40])) -> pred[] {
  %state = (f32[2,40], f32[8,2], f32[8,40]) parameter(0)
  ROOT %done = pred[] constant(false)
}
%body (state: (f32[2,40], f32[8,2], f32[8,40])) -> (f32[2,40], f32[8,2], f32[8,40]) {
  %state = (f32[2,40], f32[8,2], f32[8,40]) parameter(0)
  %weight = f32[2,40] get-tuple-element(%state), index=0
  %input = f32[8,2] get-tuple-element(%state), index=1
  %activation = f32[8,40] fusion(%weight, %input), calls=%projection
  ROOT %carry = (f32[2,40], f32[8,2], f32[8,40]) tuple(%weight, %input, %activation)
}
ENTRY %main (frozen_w: f32[2,40], x: f32[8,2], left: f32[8,3], right: f32[3,40]) -> (f32[8,40], f32[8,40]) {
  %frozen_w = f32[2,40] parameter(0)
  %x = f32[8,2] parameter(1)
  %left = f32[8,3] parameter(2)
  %right = f32[3,40] parameter(3)
  %zero = f32[8,40] constant(0)
  %initial = (f32[2,40], f32[8,2], f32[8,40]) tuple(%frozen_w, %x, %zero)
  %scan = (f32[2,40], f32[8,2], f32[8,40]) while(%initial), condition=%condition, body=%body
  %activation = f32[8,40] get-tuple-element(%scan), index=2
  %gradient = f32[8,40] dot(%left, %right), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %result = (f32[8,40], f32[8,40]) tuple(%activation, %gradient)
}
"""
    report = audit_gradient_operations(hlo, {(8, 40): {"attention.k"}}, weight_parameter_prefixes=("frozen_",))
    assert report.parsed_weight_operand_dataflow
    assert len(report.certified_activation_dots) == 1
    assert len(report.full_weight_gradient_dot_results) == 1
    assert not report.avoids_full_base_gradients
    forward_only = hlo.replace(
        "%gradient = f32[8,40] dot(%left, %right), lhs_contracting_dims={1}, rhs_contracting_dims={0}",
        "%gradient = f32[8,40] constant(0)",
    )
    assert audit_gradient_operations(
        forward_only, {(8, 40): {"attention.k"}}, weight_parameter_prefixes=("frozen_",)
    ).avoids_full_base_gradients


def test_hlo_generic_grouped_launcher_tuple_weight_output_is_rejected():
    # QuACK activation and Wgrad kernels share a generic launcher. The declared
    # tuple result, not the callee name or autodiff metadata, exposes bank Wgrad.
    hlo = """
HloModule grouped
ENTRY %main (frozen_w: f32[1,8,40], x: f32[3,8], dy: f32[3,40]) -> (f32[3,8], f32[1,8,40]) {
  %frozen_w = f32[1,8,40] parameter(0)
  %x = f32[3,8] parameter(1)
  %dy = f32[3,40] parameter(2)
  ROOT %grouped = (f32[3,8], f32[1,8,40]) custom-call(%frozen_w, %x, %dy), custom_call_target="launcher"
}
"""
    report = audit_gradient_operations(hlo, {(1, 8, 40): {"experts.gate"}}, weight_parameter_prefixes=("frozen_",))
    assert not report.avoids_full_base_gradients
    assert report.full_weight_gradient_dot_results[0]["matching_frozen_paths"] == ["experts.gate"]
    opaque = hlo.replace("(f32[3,8], f32[1,8,40]) custom-call", "token[] custom-call")
    unknown = audit_gradient_operations(opaque, {(1, 8, 40): {"experts.gate"}})
    assert unknown.unknown_custom_call_outputs and not unknown.avoids_full_base_gradients
