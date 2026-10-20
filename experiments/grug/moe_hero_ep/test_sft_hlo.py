# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import zlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.sft import (
    SftConfig,
    compact_optimizer_report,
    compiled_gradient_report,
    initialize_sft,
    make_sft_optimizer_step,
    sft_optimizers,
)
from experiments.grug.moe_hero_ep.sft_hlo import audit_gradient_operations


@pytest.mark.parametrize("opcode", ["fusion", "async"])
def test_hlo_async_computation_cannot_hide_weight_gradient_behind_entry_dot(opcode):
    # The witness used to make the audit pass although the callee's Wgrad was
    # never visited. Static inventory must find it independently of execution
    # semantics, even when the entry contains an unrelated harmless dot.
    fusion_kind = ", kind=kLoop" if opcode == "fusion" else ""
    hlo = f"""
HloModule hidden_gradient
%hidden_gradient (x: f32[4,2], dy: f32[4,3]) -> f32[2,3] {{
  %x = f32[4,2] parameter(0)
  %dy = f32[4,3] parameter(1)
  ROOT %wgrad = f32[2,3] dot(%x, %dy), lhs_contracting_dims={{0}}, rhs_contracting_dims={{0}}
}}
ENTRY %main (x: f32[4,2], dy: f32[4,3], a: f32[1,1]) -> (f32[2,3], f32[1,1]) {{
  %x = f32[4,2] parameter(0)
  %dy = f32[4,3] parameter(1)
  %a = f32[1,1] parameter(2)
  %pending = ((f32[4,2], f32[4,3]), f32[2,3], u32[]) {opcode}-start(%x, %dy){fusion_kind}, calls=%hidden_gradient
  %hidden = f32[2,3] {opcode}-done(%pending)
  %witness = f32[1,1] dot(%a, %a), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
  ROOT %result = (f32[2,3], f32[1,1]) tuple(%hidden, %witness)
}}
"""
    report = audit_gradient_operations(hlo, {(2, 3): {"frozen.router"}})
    assert report.inspected_operations == 2
    assert report.parsed_instruction_inventory
    assert not report.avoids_full_base_gradients
    assert len(report.unverified_weight_shaped_results) == 1
    # The same async control flow qualifies when the callee has a safe shape.
    safe = audit_gradient_operations(hlo, {(8, 40): {"frozen.other_projection"}})
    assert safe.parsed_instruction_inventory and safe.avoids_full_base_gradients


def test_hlo_reducer_body_cannot_hide_full_weight_gradient():
    # The scalar reducer constructs a one-token matrix Wgrad and sums it. Its
    # intermediate matrix allocation still matters even though its result is
    # scalar; inspecting only the entry or reducer output would miss it.
    hlo = """
HloModule reducer_gradient
%sum (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}
%reducer (x: f32[], dy: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %dy = f32[] parameter(1)
  %left = f32[2] broadcast(%x), dimensions={}
  %right = f32[3] broadcast(%dy), dimensions={}
  %gradient = f32[2,3] dot(%left, %right), lhs_contracting_dims={}, rhs_contracting_dims={}
  %zero = f32[] constant(0)
  ROOT %summed = f32[] reduce(%gradient, %zero), dimensions={0,1}, to_apply=%sum
}
ENTRY %main (x: f32[4], q: f32[1,1]) -> (f32[], f32[1,1]) {
  %x = f32[4] parameter(0)
  %q = f32[1,1] parameter(1)
  %zero = f32[] constant(0)
  %hidden = f32[] reduce(%x, %zero), dimensions={0}, to_apply=%reducer
  %witness = f32[1,1] dot(%q, %q), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %result = (f32[], f32[1,1]) tuple(%hidden, %witness)
}
"""
    report = audit_gradient_operations(hlo, {(2, 3): {"frozen.projection"}})
    assert report.parsed_instruction_inventory
    assert report.inspected_operations == 2
    assert len(report.unverified_weight_shaped_results) == 1
    assert not report.avoids_full_base_gradients


@pytest.mark.parametrize(
    "changed",
    [
        "%gradient = f32[2,3] unrecognized-op(%x, %dy)",
        "%gradient = f32[2,3] dot(%missing, %dy), lhs_contracting_dims={0}, rhs_contracting_dims={0}",
        "%gradient = f32[2,3] fusion(%x, %dy), calls=%missing",
        "%gradient = f32[2,3] fusion(%x, %dy)",
        "%gradient = f32[2,3] dot(%x, %dy), lhs_contracting_dims={0}, rhs_contracting_dims={0",
        "%gradient = f32[2,3] dot(%x, %dy) hidden syntax",
        "%gradient = this instruction cannot be parsed",
    ],
)
def test_hlo_unparsed_instruction_cannot_hide_behind_valid_witness(changed):
    hlo = """
HloModule malformed_gradient
ENTRY %main (x: f32[4,2], dy: f32[4,3], q: f32[1,1]) -> (f32[2,3], f32[1,1]) {
  %x = f32[4,2] parameter(0)
  %dy = f32[4,3] parameter(1)
  %q = f32[1,1] parameter(2)
  %gradient = f32[2,3] dot(%x, %dy), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  %witness = f32[1,1] dot(%q, %q), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %result = (f32[2,3], f32[1,1]) tuple(%gradient, %witness)
}
"""
    target = "%gradient = f32[2,3] dot(%x, %dy), lhs_contracting_dims={0}, rhs_contracting_dims={0}"
    report = audit_gradient_operations(hlo.replace(target, changed), {(8, 40): {"frozen.other_projection"}})
    assert not report.parsed_instruction_inventory
    assert report.instruction_inventory_errors
    assert not report.avoids_full_base_gradients


def test_hlo_inventory_keeps_ambiguous_parameter_products_unverified():
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
    report = audit_gradient_operations(hlo, {(8, 40): {"attention.k"}})
    assert report.parsed_instruction_inventory
    assert len(report.unverified_weight_shaped_results) == 2
    assert not report.avoids_full_base_gradients
    forward_only = hlo.replace(
        "%gradient = f32[8,40] dot(%left, %right), lhs_contracting_dims={1}, rhs_contracting_dims={0}",
        "%gradient = f32[8,40] constant(0)",
    )
    # A known weight operand alone cannot prove that this is an activation:
    # reassociated Wgrad can also contain another projection's weight.
    assert not audit_gradient_operations(forward_only, {(8, 40): {"attention.k"}}).avoids_full_base_gradients


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
    report = audit_gradient_operations(hlo, {(1, 8, 40): {"experts.gate"}})
    assert not report.avoids_full_base_gradients
    assert report.unverified_weight_shaped_results[0]["matching_frozen_paths"] == ["experts.gate"]
    opaque = hlo.replace("(f32[3,8], f32[1,8,40]) custom-call", "token[] custom-call")
    unknown = audit_gradient_operations(opaque, {(1, 8, 40): {"experts.gate"}})
    assert unknown.unknown_custom_call_outputs and not unknown.avoids_full_base_gradients


def test_hlo_integer_results_are_parsed_but_integer_wrapped_weight_gradients_fail():
    hlo = """
HloModule integer_results
ENTRY %main (x: f32[8,40]) -> (s32[8,40], (pred[8], u64[])) {
  %x = f32[8,40] parameter(0)
  ROOT %indices = (s32[8,40], (pred[8], u64[])) custom-call(%x), custom_call_target="opaque"
}
"""
    shapes = {(8, 40): {"frozen.router"}}
    report = audit_gradient_operations(hlo, shapes)
    assert report.avoids_full_base_gradients and len(report.known_nonfloating_custom_calls) == 1
    wrapped = """
HloModule integer_wrapped_gradient
%unwrap (bits: s32[8,40]) -> f32[8,40] {
  %bits = s32[8,40] parameter(0)
  ROOT %gradient = f32[8,40] bitcast-convert(%bits)
}
ENTRY %main (x: f32[8,40]) -> f32[8,40] {
  %x = f32[8,40] parameter(0)
  %bits = s32[8,40] custom-call(%x), custom_call_target="opaque"
  ROOT %gradient = f32[8,40] fusion(%bits), calls=%unwrap
}
"""
    wrapped_report = audit_gradient_operations(wrapped, shapes)
    assert not wrapped_report.avoids_full_base_gradients
    assert wrapped_report.unverified_weight_shaped_results


def test_hlo_unsupported_float_result_cannot_hide_full_weight_gradient():
    # Float8 is valid HLO but outside this parser's supported dtype grammar.
    # A missing declared shape used to silently drop the full Wgrad result.
    hlo = """
HloModule float8_weight_gradient
ENTRY %main (x: f8e4m3fn[4,2], dy: f8e4m3fn[4,3]) -> f8e4m3fn[2,3] {
  %x = f8e4m3fn[4,2] parameter(0)
  %dy = f8e4m3fn[4,3] parameter(1)
  ROOT %wgrad = f8e4m3fn[2,3] dot(%x, %dy), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
"""
    report = audit_gradient_operations(hlo, {(2, 3): {"frozen.router"}})
    assert report.inspected_operations == 1
    assert not report.parsed_instruction_inventory
    assert not report.avoids_full_base_gradients


@pytest.mark.parametrize("prefix", ["params.adapters.", "frozen."])
def test_hlo_frozen_shape_contraction_with_parameter_operand_is_not_a_compact_certificate(prefix):
    # This is a real full-bank gradient even though reassociation leaves a
    # parameter operand in the final contraction. Calling it an adapter cannot
    # establish rank-axis identity or result flow into compact gradient leaves.
    hlo = f"""
HloModule reassociated_weight_gradient
ENTRY %main (left: f32[8,3], parameter: f32[3,40]) -> f32[8,40] {{
  %left = f32[8,3] parameter(0)
  %parameter = f32[3,40] parameter(1), metadata={{op_name="{prefix}w"}}
  ROOT %gradient = f32[8,40] dot(%left, %parameter), lhs_contracting_dims={{1}}, rhs_contracting_dims={{0}}
}}
"""
    report = audit_gradient_operations(hlo, {(8, 40): {"frozen.first_layer"}})
    assert not report.avoids_full_base_gradients
    assert len(report.unverified_weight_shaped_results) == 1


def test_hlo_single_token_outer_product_cannot_hide_behind_an_unrelated_dot():
    # For a one-token dense projection, Wgrad is x[:, None] * dy[None, :].
    # XLA need not retain a dot, so inspecting an unrelated dot is insufficient.
    hlo = """
HloModule single_token_wgrad
ENTRY %main (x: f32[2], dy: f32[3], q: f32[1,1]) -> (f32[2,3], f32[1,1]) {
  %x = f32[2] parameter(0)
  %dy = f32[3] parameter(1)
  %q = f32[1,1] parameter(2)
  %left = f32[2,3] broadcast(%x), dimensions={0}
  %right = f32[2,3] broadcast(%dy), dimensions={1}
  %gradient = f32[2,3] multiply(%left, %right)
  %unrelated = f32[1,1] dot(%q, %q), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %result = (f32[2,3], f32[1,1]) tuple(%gradient, %unrelated)
}
"""
    report = audit_gradient_operations(hlo, {(2, 3): {"frozen.projection"}})
    assert not report.avoids_full_base_gradients
    assert len(report.unverified_weight_shaped_results) == 1
    assert "multiply(" in report.unverified_weight_shaped_results[0]["operation"]


def _wire_varint(value):
    data = bytearray()
    while value > 127:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def _wire_message(fields):
    result = bytearray()
    for number, value in fields:
        result.extend(_wire_varint((number << 3) | (2 if isinstance(value, bytes) else 0)))
        if isinstance(value, bytes):
            result.extend(_wire_varint(len(value)))
            result.extend(value)
        else:
            result.extend(_wire_varint(value))
    return bytes(result)


def _transport_hlo(
    *,
    ttir_body="tt.return",
    ptx_body="ret;",
    consumer="destination",
    extra_kernel_fields=(),
    target="triton_kernel_call",
):
    # A serialized JAX TritonAnyKernelCall fixture, with independent PTX and
    # TTIR bodies so a forged debug IR cannot certify different executable code.
    name = "arbitrary_name"
    ttir = (
        f"module {{\n  tt.func public @{name}("
        "%marker: !tt.ptr<i32> {tt.divisibility = 16 : i32}, "
        "%out: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {\n"
        f"    {ttir_body}\n  }}\n}}"
    )
    ptx = f".version 8.8\n.target sm_100a\n.address_size 64\n.visible .entry {name}()\n.reqntid 128\n{{\n{ptx_body}\n}}"
    kernel = _wire_message(
        [(1, name.encode()), (2, 4), (4, ptx.encode()), (5, ttir.encode()), (6, 100), (10, 1), *extra_kernel_fields]
    )
    array = _wire_message([(1, _wire_message([(2, 16)]))])
    call = _wire_message([(1, kernel), (2, 1), (3, 1), (4, 1), (5, array), (5, array)])
    compressed = zlib.compress(_wire_message([(1, call)]))
    encoded = "".join(f"\\{byte:03o}" for byte in compressed)
    use = (
        "ROOT %result = bf16[8,40] ragged-all-to-all(%x, %buffer, %offset, %offset, %offset, %offset)"
        if consumer == "destination"
        else "ROOT %result = bf16[8,40] copy(%buffer)"
    )
    return f"""
HloModule no_write_transport
ENTRY %main (marker: s32[], x: bf16[3,40], offset: s64[4]) -> bf16[8,40] {{
  %marker = s32[] parameter(0)
  %x = bf16[3,40] parameter(1)
  %offset = s64[4] parameter(2)
  %buffer = bf16[8,40] custom-call(%marker), custom_call_target="{target}", backend_config="{encoded}"
  {use}
}}
"""


def test_hlo_proves_no_write_transport_from_both_ir_and_destination_contract():
    report = audit_gradient_operations(_transport_hlo(), {(8, 40): {"frozen.router"}})
    assert report.avoids_full_base_gradients
    assert len(report.certified_transport_calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"ttir_body": "tt.store %out, %marker : !tt.ptr<bf16>\n    tt.return"},
        {"ptx_body": "st.global.b32 [%rd1], 0;\nret;"},
        {"ptx_body": ".loc 1 148 4 st.global.b32 [%rd1], 0;\nret;"},
        {"ttir_body": "tt.call @hidden() : () -> ()\n    tt.return"},
        {"ptx_body": "atom.global.add.f32 %f1, [%rd1], %f2;\nret;"},
        {"consumer": "gradient"},
        {"target": "another_loader_with_uninspected_semantics"},
        {"extra_kernel_fields": ((11, b"uninspected machine image"),)},
    ],
)
def test_hlo_forged_or_mutated_transport_stays_unverified(changes):
    report = audit_gradient_operations(_transport_hlo(**changes), {(8, 40): {"frozen.router"}})
    assert not report.avoids_full_base_gradients
    assert not report.certified_transport_calls
    assert report.unverified_weight_shaped_results


@dataclasses.dataclass(frozen=True)
class CompiledExecutableFixture:
    """External compiled-executable boundary with literal HLO and no execution."""

    text: str
    output_size_in_bytes: int = 0
    temp_size_in_bytes: int = 0

    def as_text(self):
        return self.text

    def memory_analysis(self):
        return self


def test_compiled_report_inventories_unused_short_conv_weight_gradient_partial():
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        attention_implementation="reference",
        sconv=True,
        sconv_kernel=4,
    )
    # Retained GPU .111 ABI, scaled to C=8: the joint bwd call declares dX and
    # FP32 [P,W,C] dW partials. Only dX is consumed, but dW still exists in the
    # custom call's declared buffers. The real model must supply its frozen
    # short-conv shape to the report; no hand-authored forbidden catalog is used.
    text = """
HloModule short_conv_backward
ENTRY %main (x: bf16[1,8,8], ids: s32[1,8], dy: bf16[1,8,8], w: bf16[4,8]) -> bf16[1,8,8] {
  %x = bf16[1,8,8] parameter(0)
  %ids = s32[1,8] parameter(1)
  %dy = bf16[1,8,8] parameter(2)
  %w = bf16[4,8] parameter(3)
  %joint = (bf16[1,8,8], f32[1,4,8]) custom-call(%x, %ids, %dy, %w), custom_call_target="triton_kernel_call"
  ROOT %dx = bf16[1,8,8] get-tuple-element(%joint), index=0
}
"""
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        model = Transformer.init(config, key=jax.random.key(0))
        report = compiled_gradient_report(CompiledExecutableFixture(text), model)
    assert report["parsed_instruction_inventory"]
    assert not report["avoids_full_base_gradients"]
    assert any(
        item["shape"] == [1, 4, 8] and any(".sconv_mlp.weight" in path for path in item["matching_frozen_paths"])
        for item in report["unverified_weight_shaped_results"]
    )


def test_compact_adam_tree_audit_accepts_shape_collision_and_rejects_frozen_parameter():
    config = GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=4,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        attention_implementation="reference",
        sconv=True,
    )
    sft = SftConfig(2, 2.0, (2, 3, 4), 1e-3, 1e-3, 4, 0, 1)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
        frozen, state = initialize_sft(Transformer.init(config, key=jax.random.key(0)), sft)
        frozen_shapes = {x.shape for x in jax.tree.leaves(frozen)}
        assert any(x.shape in frozen_shapes for x in jax.tree.leaves(state.adapter_opt_state))
        assert compact_optimizer_report(state)["compact_optimizer_only"]
        contaminated = eqx.tree_at(
            lambda params: params.adapters.stacked_blocks.stacked.attn.w_q,
            state.params,
            frozen.stacked_blocks.stacked.attn.w_q.astype(jnp.float32),
            is_leaf=lambda value: value is None,
        )
        adam, _ = sft_optimizers(sft)
        wrong_state = dataclasses.replace(state, params=contaminated, adapter_opt_state=adam.init(contaminated.adapters))
        report = compact_optimizer_report(wrong_state)
        assert not report["compact_optimizer_only"]
        assert any("Noncompact parameter" in error for error in report["errors"])
        # A resumed state must carry the same update count in both Adam moments
        # and LR schedules. Shape/byte checks alone miss a corrupted counter.
        optimize = make_sft_optimizer_step(sft)
        gradients = jax.tree.map(jnp.ones_like, state.params)
        for _ in range(2):
            state = optimize(state, gradients, jnp.array(8, jnp.int32), jnp.array(7, jnp.int32))
        assert compact_optimizer_report(state)["compact_optimizer_only"]
        adapter_adam, adapter_schedule = state.adapter_opt_state
        corrupt_adam = dataclasses.replace(
            state, adapter_opt_state=(adapter_adam._replace(count=state.step + 1), adapter_schedule)
        )
        vocab_adam, vocab_schedule = state.vocab_opt_state
        corrupt_schedule = dataclasses.replace(
            state, vocab_opt_state=(vocab_adam, vocab_schedule._replace(count=state.step - 1))
        )
        for corrupted in (corrupt_adam, corrupt_schedule):
            report = compact_optimizer_report(corrupted)
            assert not report["compact_optimizer_only"]
            assert any("counters do not equal SFT step 2" in error for error in report["errors"])
