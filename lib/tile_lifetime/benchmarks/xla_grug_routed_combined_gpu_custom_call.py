#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute generic generated regions in one natural Grug train step."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import jax
import jaxlib
from haliax.partitioning import set_mesh

from lib.tile_lifetime.benchmarks.xla_grug_backward_multi_output_gpu_custom_call_smoke import (
    _compare_under_ordered_fp,
    _tree_hash_evidence,
)
from lib.tile_lifetime.benchmarks.xla_grug_pair_map_custom_call_smoke import (
    _mesh,
    _natural_train_step,
)
from lib.tile_lifetime.benchmarks.xla_pair_map_custom_call_smoke import (
    _compile_cuda_ffi_handler,
    write_gzip_text,
)
from shuttle.experimental.stablehlo_import import import_stablehlo
from tile_lifetime.command_buffer_capture import CaptureSiteManifest, derive_capture_site_manifest
from tile_lifetime.cuda_axis_fold_codegen import generate_cuda_axis_fold_ffi
from tile_lifetime.cuda_contract_map_chain_codegen import (
    GeneratedCudaContractMapChainFfi,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    GeneratedCudaNormalizedExpContractForwardFfi,
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    GeneratedCudaNormalizedExpContractReverseFfi,
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.ffi_command_buffer import (
    DirectLaunchFfiPhysicalCandidate,
    require_custom_call_command_buffers_enabled,
)
from tile_lifetime.jax_contract_map_chain_ffi import register_cuda_contract_map_chain_ffi
from tile_lifetime.jax_hlo_rewrite_runtime import audit_hlo_rewrite_runtime, require_hlo_rewrite_runtime
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_axis_fold_ffi import (
    AxisFoldHloRegionReplacementAudit,
    AxisFoldHloRegionReplacementPlan,
    audit_axis_fold_hlo_region_replacement,
    plan_axis_fold_hlo_region_replacement,
    recover_axis_fold_hlo_region_candidates,
    replace_axis_fold_hlo_region_with_custom_call,
)
from tile_lifetime.xla_contract_relation_fold_ffi import (
    ContractRelationFoldReplacementAudit,
    GeneratedContractRelationFoldFfi,
    audit_contract_relation_fold_replacement,
    generate_cuda_contract_relation_fold_ffi,
    replace_contract_relation_fold_with_custom_call,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    GeneratedLowRankContractMapTrainingAudit,
    GeneratedLowRankContractMapTrainingPlan,
    audit_generated_low_rank_contract_map_training,
    plan_generated_low_rank_contract_map_training,
    replace_generated_low_rank_contract_map_training,
)
from tile_lifetime.xla_normalized_exp_contract_forward import (
    NormalizedExpContractForwardHloReplacementAudit,
    NormalizedExpContractForwardHloReplacementPlan,
    audit_normalized_exp_contract_forward_hlo_replacement,
    plan_normalized_exp_contract_forward_hlo_replacement,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    NormalizedExpContractReverseHloReplacementAudit,
    NormalizedExpContractReverseHloReplacementPlan,
    audit_normalized_exp_contract_reverse_hlo_replacement,
    plan_normalized_exp_contract_reverse_hlo_replacement,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)
from tile_lifetime.xla_rank_two_contract_ffi import (
    GeneratedRankTwoContractFfi,
    generate_cuda_rank_two_contract_ffi,
)
from tile_lifetime.xla_relation_program_recovery import RoutedInputAdjointFfiOperandRole
from tile_lifetime.xla_routed_forward_ffi import generate_cuda_routed_forward_ffi
from tile_lifetime.xla_routed_input_adjoint_ffi import (
    generate_cuda_routed_input_adjoint_ffi,
)
from tile_lifetime.xla_routed_shared_map_training_ffi import (
    RoutedSharedMapTrainingFfiTargets,
    RoutedSharedMapTrainingReplacementAudit,
    RoutedSharedMapTrainingTypedFfiPlan,
    audit_routed_shared_map_training_replacement,
    plan_routed_shared_map_training_typed_ffi,
    replace_routed_shared_map_training_regions_with_custom_calls,
)
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingAndAttentionFfiTargets,
    RoutedTrainingAttentionAndAxisFoldFfiTargets,
    RoutedTrainingFfiTargets,
    audit_routed_training_attention_and_axis_fold_replacement,
    entry_parameter_ancestors,
    plan_routed_training_attention_and_axis_fold_typed_ffi,
    replace_routed_training_attention_and_axis_fold_regions_with_custom_calls,
)
from tile_lifetime.xla_routed_weight_gradient_ffi import (
    generate_cuda_group_batched_contract_ffi,
)
from tile_lifetime.xla_shared_contract_multimap_ffi import (
    GeneratedSharedContractMultiMapFfi,
    generate_cuda_shared_contract_multi_map_ffi,
)
from tile_lifetime.xla_source_indexed_fold_ffi import (
    GeneratedSourceIndexedFoldFfi,
    generate_cuda_source_indexed_fold_ffi,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloRegionReplacementPlan,
    audit_streaming_attention_backward_region_replacement,
    derive_streaming_attention_backward_ffi_output_layouts,
    plan_streaming_attention_backward_hlo_region_replacement,
    replace_streaming_attention_backward_region_with_custom_call,
)
from tile_lifetime.xla_weighted_relation_reverse_ffi import (
    GeneratedRelationEdgeFoldFfi,
    WeightedRelationReverseReplacementAudit,
    WeightedRelationReverseTypedFfiPlan,
    audit_weighted_relation_reverse_replacement,
    generate_cuda_relation_edge_fold_ffi,
    plan_weighted_relation_reverse_typed_ffi,
    replace_weighted_relation_reverse_with_custom_calls,
)

_PASS_NAME = "shuttle_grug_routed_training_gpu_v1"
_ROUTED_ATTENTION_TARGETS = RoutedTrainingAndAttentionFfiTargets(
    routed=RoutedTrainingFfiTargets(
        forward="shuttle.routed_training.forward.v2",
        input_adjoint="shuttle.routed_training.input_adjoint.v2",
        weight_gradients=(
            "shuttle.routed_training.weight_gradient.0.v2",
            "shuttle.routed_training.weight_gradient.1.v2",
        ),
    ),
    attention_backward="shuttle.routed_training.attention_backward.v2",
)
_TARGETS = RoutedTrainingAttentionAndAxisFoldFfiTargets(
    routed_attention=_ROUTED_ATTENTION_TARGETS,
    axis_folds=(
        "shuttle.routed_training.axis_fold.0.v1",
        "shuttle.routed_training.axis_fold.1.v1",
    ),
)
_SHARED_ROUTED_TARGETS = RoutedSharedMapTrainingFfiTargets(
    forward="shuttle.routed_training.shared_map.forward.v1",
    input_contracts=(
        "shuttle.routed_training.shared_map.input_contract.0.v1",
        "shuttle.routed_training.shared_map.input_contract.1.v1",
    ),
    shared_contract_multi_map="shuttle.routed_training.shared_map.multi_map.v1",
    source_fold="shuttle.routed_training.shared_map.source_fold.v1",
    weight_gradients=(
        "shuttle.routed_training.shared_map.weight_gradient.0.v1",
        "shuttle.routed_training.shared_map.weight_gradient.1.v1",
    ),
)
_WEIGHTED_RELATION_CONTRACT_TARGET = "shuttle.routed_training.weighted_relation.contract.v1"
_WEIGHTED_RELATION_FOLD_TARGET = "shuttle.routed_training.weighted_relation.fold.v1"
_WEIGHTED_RELATION_FUSED_TARGET = "shuttle.routed_training.weighted_relation.contract_fold.v1"
_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET = "shuttle.routed_training.normalized_exp_contract_forward.v1"
_NORMALIZED_EXP_CONTRACT_REVERSE_TARGET = "shuttle.routed_training.normalized_exp_contract_reverse.v1"
_LOW_RANK_CONTRACT_MAP_FORWARD_TARGET = "shuttle.routed_training.low_rank_contract_map.forward.v1"
_LOW_RANK_CONTRACT_MAP_REVERSE_TARGET = "shuttle.routed_training.low_rank_contract_map.reverse.v1"


class RoutedTrainingCompositionMode(StrEnum):
    """Generated routed-training ownership boundary used by the benchmark."""

    MONOLITHIC_INPUT_ADJOINT = "monolithic_input_adjoint"
    SHARED_MAP_XLA_REMAINDER = "shared_map_xla_remainder"
    SHARED_MAP_FUSED_WEIGHTED_REVERSE = "shared_map_fused_weighted_reverse"
    SHARED_MAP_FUSED_REVERSES = "shared_map_fused_reverses"
    SHARED_MAP_FUSED_REVERSES_AND_GATED_PRODUCTS = "shared_map_fused_reverses_and_gated_products"

    @property
    def uses_shared_map(self) -> bool:
        """Whether this composition exposes the shared-Map routed subregions."""
        return self is not RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT

    @property
    def fuses_weighted_reverse(self) -> bool:
        """Whether one bounded Contract/Map/Fold call owns weighted reverse."""
        return self in {
            RoutedTrainingCompositionMode.SHARED_MAP_FUSED_WEIGHTED_REVERSE,
            RoutedTrainingCompositionMode.SHARED_MAP_FUSED_REVERSES,
            RoutedTrainingCompositionMode.SHARED_MAP_FUSED_REVERSES_AND_GATED_PRODUCTS,
        }

    @property
    def generates_normalized_exp_pair(self) -> bool:
        """Whether generated targets own normalized-exp forward and reverse."""
        return self in {
            RoutedTrainingCompositionMode.SHARED_MAP_FUSED_REVERSES,
            RoutedTrainingCompositionMode.SHARED_MAP_FUSED_REVERSES_AND_GATED_PRODUCTS,
        }

    @property
    def generates_low_rank_gated_products(self) -> bool:
        """Whether generated Contract/Map families own six primal and four reverse chains."""
        return self is RoutedTrainingCompositionMode.SHARED_MAP_FUSED_REVERSES_AND_GATED_PRODUCTS

    @property
    def independent_custom_call_count(self) -> int:
        """Return the exact generated-call count selected by this composition."""
        if self is RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT:
            return 7
        if self.generates_normalized_exp_pair:
            return 23 if self.generates_low_rank_gated_products else 13
        return 11 if self.fuses_weighted_reverse else 12


class CommandBufferCandidateMode(StrEnum):
    """Bounded generated-handler set offered to XLA command-buffer conversion."""

    DISABLED = "disabled"
    NORMALIZED_EXP_PAIR = "normalized_exp_pair"
    GENERIC_DIRECT_LAUNCH_SET = "generic_direct_launch_set"

    def compatible_targets(self, composition_mode: RoutedTrainingCompositionMode) -> tuple[str, ...]:
        """Return handlers whose host callbacks are capture-only instrumentation."""
        if self is CommandBufferCandidateMode.NORMALIZED_EXP_PAIR:
            if not composition_mode.generates_normalized_exp_pair:
                raise ValueError("the normalized-exp command-buffer candidate requires the generated pair")
            return (
                _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
            )
        if self is CommandBufferCandidateMode.GENERIC_DIRECT_LAUNCH_SET:
            if not composition_mode.generates_low_rank_gated_products:
                raise ValueError("the generic direct-launch set requires the complete 23-call composition")
            return (
                _SHARED_ROUTED_TARGETS.source_fold,
                _WEIGHTED_RELATION_FUSED_TARGET,
                _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
                _LOW_RANK_CONTRACT_MAP_FORWARD_TARGET,
                _LOW_RANK_CONTRACT_MAP_REVERSE_TARGET,
            )
        return ()


@dataclass(frozen=True)
class SharedMapTrainingAttentionAndAxisFoldPlan:
    """Clean shared-Map routed plan composed with reverse, attention, and row Folds."""

    routed: RoutedSharedMapTrainingTypedFfiPlan
    weighted_relation_reverse: WeightedRelationReverseTypedFfiPlan
    normalized_exp_contract_forward: NormalizedExpContractForwardHloReplacementPlan
    normalized_exp_contract_reverse: NormalizedExpContractReverseHloReplacementPlan
    attention_backward: StreamingReverseHloRegionReplacementPlan
    axis_folds: tuple[AxisFoldHloRegionReplacementPlan, ...]


@dataclass(frozen=True)
class SharedMapTrainingAttentionAndAxisFoldAudit:
    """Post-roundtrip evidence for one clean shared-Map composition."""

    routed: RoutedSharedMapTrainingReplacementAudit
    weighted_relation_reverse: WeightedRelationReverseReplacementAudit | ContractRelationFoldReplacementAudit
    normalized_exp_contract_forward: NormalizedExpContractForwardHloReplacementAudit | None
    normalized_exp_contract_reverse: NormalizedExpContractReverseHloReplacementAudit | None
    attention_backward_instruction: str
    axis_folds: tuple[AxisFoldHloRegionReplacementAudit, ...]


_CUSTOM_CALL_TARGET_ATTRIBUTE = re.compile(r'(?:^|,\s*)custom_call_target="(?P<target>[^"]+)"(?=,|$)')


def _custom_call_target_occurrences(hlo_text: str, expected: dict[str, int]) -> dict[str, int]:
    """Count exact target attributes and enforce each selected multiplicity."""
    module = parse_hlo_module_text(hlo_text)
    counts = dict.fromkeys(expected, 0)
    for computation in module.computations:
        for instruction in computation.instructions:
            if instruction.opcode != "custom-call":
                continue
            attributes = _CUSTOM_CALL_TARGET_ATTRIBUTE.findall(instruction.attributes)
            if len(attributes) != 1:
                raise RuntimeError(
                    f"custom-call %{instruction.name} has {len(attributes)} exact custom_call_target attributes"
                )
            target = attributes[0]
            if target in counts:
                counts[target] += 1
    invalid = {
        target: {"expected": expected[target], "actual": count}
        for target, count in counts.items()
        if count != expected[target]
    }
    if invalid:
        raise RuntimeError(f"custom-call target multiplicities changed: {invalid}")
    return counts


def _target_occurrence(hlo_text: str, target: str) -> int:
    """Count an exact custom-call target attribute without requiring its presence."""
    module = parse_hlo_module_text(hlo_text)
    return sum(
        instruction.opcode == "custom-call" and _CUSTOM_CALL_TARGET_ATTRIBUTE.findall(instruction.attributes) == [target]
        for computation in module.computations
        for instruction in computation.instructions
    )


def _final_hlo_capture_site_manifest(
    final_hlo: str,
    *,
    executable: str,
    composition_mode: RoutedTrainingCompositionMode,
    command_buffer_candidate: CommandBufferCandidateMode,
) -> CaptureSiteManifest:
    """Derive and validate the selected capture topology from compiled HLO."""
    targets = command_buffer_candidate.compatible_targets(composition_mode)
    if not targets:
        return CaptureSiteManifest.uninstrumented(executable)
    expected = dict.fromkeys(targets, 1)
    if _LOW_RANK_CONTRACT_MAP_FORWARD_TARGET in expected:
        expected[_LOW_RANK_CONTRACT_MAP_FORWARD_TARGET] = 6
        expected[_LOW_RANK_CONTRACT_MAP_REVERSE_TARGET] = 4
    return derive_capture_site_manifest(
        executable,
        final_hlo,
        {target: target for target in targets},
        expected_target_occurrences=expected,
    )


def _attention_reverse_program() -> tuple[Any, Any, bytes]:
    config = StreamingAttentionBackwardDebugConfig(
        batch=2,
        query_length=4,
        key_length=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=16,
        scale=0.32421875,
    )
    source_stablehlo = export_debug_streaming_attention_backward(config)
    graph = import_stablehlo(source_stablehlo, input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES)
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    return program, schedule, source_stablehlo


def _generate_axis_fold_programs(hlo_text: str) -> tuple[Any, ...]:
    report = recover_axis_fold_hlo_region_candidates(
        hlo_text,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected = tuple(
        plan
        for plan in report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    if len(selected) != len(_TARGETS.axis_folds):
        raise RuntimeError(f"expected two Grug row-axis Folds, found {len(selected)}")
    return tuple(
        generate_cuda_axis_fold_ffi((plan.program,), target_name=target)
        for plan, target in zip(selected, _TARGETS.axis_folds, strict=True)
    )


def _axis_fold_reassociation_report(
    plans: tuple[AxisFoldHloRegionReplacementPlan, ...],
) -> list[str]:
    return [plan.program.reassociation.value for plan in plans]


def _plan_shared_map_composition(
    hlo_text: str,
    attention_program: Any,
    generated_attention: Any,
    generated_axis_folds: tuple[Any, ...],
) -> SharedMapTrainingAttentionAndAxisFoldPlan:
    """Plan disjoint shared-Map calls while retaining collectives and physical views."""
    if len(generated_axis_folds) != len(_TARGETS.axis_folds):
        raise ValueError("shared-Map composition requires one generated body per axis-Fold target")
    axis_folds = tuple(
        plan_axis_fold_hlo_region_replacement(
            hlo_text,
            generated,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )
        for generated in generated_axis_folds
    )
    if len({fold.internal_instructions for fold in axis_folds}) != len(axis_folds):
        raise ValueError("shared-Map composition selected one axis-Fold region more than once")
    normalized_exp_contract_forward = plan_normalized_exp_contract_forward_hlo_replacement(hlo_text)
    normalized_exp_forward_hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
        hlo_text,
        normalized_exp_contract_forward,
        target=_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
    )
    return SharedMapTrainingAttentionAndAxisFoldPlan(
        routed=plan_routed_shared_map_training_typed_ffi(hlo_text),
        weighted_relation_reverse=plan_weighted_relation_reverse_typed_ffi(hlo_text),
        normalized_exp_contract_forward=normalized_exp_contract_forward,
        normalized_exp_contract_reverse=plan_normalized_exp_contract_reverse_hlo_replacement(normalized_exp_forward_hlo),
        attention_backward=plan_streaming_attention_backward_hlo_region_replacement(
            hlo_text,
            attention_program,
            generated_attention,
        ),
        axis_folds=axis_folds,
    )


def _replace_shared_map_composition(
    hlo_text: str,
    plan: SharedMapTrainingAttentionAndAxisFoldPlan,
    *,
    fuse_weighted_reverse: bool,
    generate_normalized_exp_pair: bool = False,
) -> str:
    """Apply one clean shared-Map composition while retaining physical views."""
    rewritten = replace_streaming_attention_backward_region_with_custom_call(
        hlo_text,
        plan.attention_backward,
        target=_ROUTED_ATTENTION_TARGETS.attention_backward,
    )
    rewritten = replace_routed_shared_map_training_regions_with_custom_calls(
        rewritten,
        plan.routed,
        targets=_SHARED_ROUTED_TARGETS,
    )
    if fuse_weighted_reverse:
        rewritten = replace_contract_relation_fold_with_custom_call(
            rewritten,
            plan.weighted_relation_reverse.payload_contract,
            plan.weighted_relation_reverse.edge_fold,
            target=_WEIGHTED_RELATION_FUSED_TARGET,
        )
    else:
        rewritten = replace_weighted_relation_reverse_with_custom_calls(
            rewritten,
            plan.weighted_relation_reverse,
            contract_target=_WEIGHTED_RELATION_CONTRACT_TARGET,
            fold_target=_WEIGHTED_RELATION_FOLD_TARGET,
        )
    if generate_normalized_exp_pair:
        rewritten = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
            rewritten,
            plan.normalized_exp_contract_forward,
            target=_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
        )
        rewritten = replace_normalized_exp_contract_reverse_hlo_region_with_custom_call(
            rewritten,
            plan.normalized_exp_contract_reverse,
            target=_NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
        )
    for axis_fold, target in zip(plan.axis_folds, _TARGETS.axis_folds, strict=True):
        rewritten = replace_axis_fold_hlo_region_with_custom_call(rewritten, axis_fold, target=target)
    return rewritten


def _audit_shared_map_composition(
    original_hlo: str,
    transformed_hlo: str,
    plan: SharedMapTrainingAttentionAndAxisFoldPlan,
    *,
    fuse_weighted_reverse: bool,
    generate_normalized_exp_pair: bool = False,
) -> SharedMapTrainingAttentionAndAxisFoldAudit:
    """Audit every generated call and the XLA-owned physical views."""
    routed = audit_routed_shared_map_training_replacement(
        original_hlo,
        transformed_hlo,
        plan.routed,
        targets=_SHARED_ROUTED_TARGETS,
    )
    if fuse_weighted_reverse:
        weighted_relation_reverse = audit_contract_relation_fold_replacement(
            original_hlo,
            transformed_hlo,
            plan.weighted_relation_reverse.payload_contract,
            plan.weighted_relation_reverse.edge_fold,
            target=_WEIGHTED_RELATION_FUSED_TARGET,
        )
    else:
        weighted_relation_reverse = audit_weighted_relation_reverse_replacement(
            original_hlo,
            transformed_hlo,
            plan.weighted_relation_reverse,
            contract_target=_WEIGHTED_RELATION_CONTRACT_TARGET,
            fold_target=_WEIGHTED_RELATION_FOLD_TARGET,
        )
    attention = audit_streaming_attention_backward_region_replacement(
        original_hlo,
        transformed_hlo,
        plan.attention_backward,
        target=_ROUTED_ATTENTION_TARGETS.attention_backward,
    )
    normalized_exp_contract_forward = None
    normalized_exp_contract_reverse = None
    if generate_normalized_exp_pair:
        forward_only_hlo = replace_normalized_exp_contract_forward_hlo_region_with_custom_call(
            original_hlo,
            plan.normalized_exp_contract_forward,
            target=_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
        )
        normalized_exp_contract_reverse = audit_normalized_exp_contract_reverse_hlo_replacement(
            forward_only_hlo,
            transformed_hlo,
            plan.normalized_exp_contract_reverse,
            target=_NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
        )
        generated_saved_state = "shuttle.generated.normalized_exp_contract_forward.output.1"
        if plan.normalized_exp_contract_reverse.region.saved_state.instruction != generated_saved_state:
            raise ValueError("normalized-exp reverse plan does not consume generated forward state")
        normalized_exp_contract_forward = audit_normalized_exp_contract_forward_hlo_replacement(
            original_hlo,
            transformed_hlo,
            plan.normalized_exp_contract_forward,
            target=_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
            expected_output_users=(
                plan.normalized_exp_contract_forward.external_users[0][1],
                (normalized_exp_contract_reverse.call_instruction,),
            ),
        )
    axis_folds = tuple(
        audit_axis_fold_hlo_region_replacement(
            original_hlo,
            transformed_hlo,
            fold,
            target=target,
        )
        for fold, target in zip(plan.axis_folds, _TARGETS.axis_folds, strict=True)
    )
    return SharedMapTrainingAttentionAndAxisFoldAudit(
        routed=routed,
        weighted_relation_reverse=weighted_relation_reverse,
        normalized_exp_contract_forward=normalized_exp_contract_forward,
        normalized_exp_contract_reverse=normalized_exp_contract_reverse,
        attention_backward_instruction=attention.call_instruction,
        axis_folds=axis_folds,
    )


def _register_cuda_target(library: ctypes.CDLL, target: str) -> None:
    handler = getattr(library, target.replace(".", "_"))
    handler.restype = ctypes.c_void_p
    jax.ffi.register_ffi_target(
        target,
        jax.ffi.pycapsule(handler),
        platform="CUDA",
        api_version=1,
    )


def _runtime_dependency_audit(library_path: Path, source: str) -> tuple[str, ...]:
    if "torch" in source.lower() or "triton" in source.lower():
        raise RuntimeError(f"generated runtime source contains a Torch/Triton reference: {library_path}")
    completed = subprocess.run(("ldd", str(library_path)), check=True, capture_output=True, text=True)
    dependencies = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    forbidden = tuple(line for line in dependencies if "torch" in line.lower() or "triton" in line.lower())
    if forbidden:
        raise RuntimeError(f"generated runtime library links Torch/Triton: {forbidden}")
    return dependencies


def _require_attention_aot_build_dependencies() -> dict[str, str]:
    """Fail before device access when the current attention AOT toolchain is incomplete."""
    try:
        versions = {distribution: importlib.metadata.version(distribution) for distribution in ("torch", "triton")}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"streaming-attention AOT build dependency is missing: {error.name}") from error
    probe = (
        "import importlib; "
        "importlib.import_module('torch'); "
        "importlib.import_module('triton'); "
        "importlib.import_module('triton.language'); "
        "importlib.import_module('triton.tools.compile')"
    )
    completed = subprocess.run(
        (sys.executable, "-c", probe),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "dependency probe failed without output"
        raise RuntimeError(f"streaming-attention AOT build dependency preflight failed: {detail}")
    return versions


def dependency_preflight() -> dict[str, Any]:
    """Audit the complete build/runtime dependency boundary without touching a GPU."""
    hlo = audit_hlo_rewrite_runtime()
    if not hlo.available:
        raise RuntimeError(f"HLO rewrite runtime preflight failed: {hlo.unavailable_reasons}")
    return {
        "jax_version": hlo.jax_version,
        "jaxlib_version": hlo.jaxlib_version,
        "compiler_ir_proto_roundtrip": hlo.compiler_ir_proto_roundtrip,
        "text_parser_backend": hlo.text_parser_backend,
        "transformation_api": hlo.transformation_api,
        "attention_aot_build_dependencies": _require_attention_aot_build_dependencies(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(
    nvcc: Path,
    architecture: str,
    artifact_directory: Path | None,
    *,
    composition_mode: RoutedTrainingCompositionMode,
    repository: Path,
    triton_target: str | None,
    command_buffer_candidate: CommandBufferCandidateMode = CommandBufferCandidateMode.DISABLED,
    warmup: int = 4,
    repeats: int = 30,
) -> dict[str, Any]:
    """Compile, execute, and time the combined routed-plus-attention transform."""
    compatible_targets = command_buffer_candidate.compatible_targets(composition_mode)
    command_buffer_flag_audit = (
        require_custom_call_command_buffers_enabled(os.environ.get("XLA_FLAGS", ""))
        if command_buffer_candidate is not CommandBufferCandidateMode.DISABLED
        else None
    )
    attention_aot_build_dependencies = _require_attention_aot_build_dependencies()
    hlo_runtime = require_hlo_rewrite_runtime()
    if not jax.devices() or jax.devices()[0].platform != "gpu":
        raise RuntimeError("the combined routed training replacement requires a CUDA JAX device")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if repeats <= 0 or repeats % 2:
        raise ValueError("repeats must be a positive even number")
    xla = hlo_runtime.transformation_api
    jax.config.update("jax_enable_compilation_cache", False)
    temporary = None
    if artifact_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="shuttle-grug-routed-training-")
        directory = Path(temporary.name)
    else:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        directory = artifact_directory
    attention_program, attention_schedule, source_attention_stablehlo = _attention_reverse_program()
    default_attention = generate_streaming_attention_backward_ffi(
        attention_program,
        attention_schedule,
        target_name=_TARGETS.routed_attention.attention_backward,
    )
    if artifact_directory is not None:
        (directory / "source-attention-vjp-stablehlo.mlir.bc").write_bytes(source_attention_stablehlo)

    original_modules: list[str] = []
    transformed_modules: list[str] = []
    recovered_plans: list[Any] = []
    generated_programs: list[dict[str, Any]] = []
    replacement_audits: list[Any] = []
    attention_liveness_audits: list[Any] = []
    low_rank_contract_map_plans: list[GeneratedLowRankContractMapTrainingPlan] = []
    low_rank_contract_map_audits: list[GeneratedLowRankContractMapTrainingAudit] = []
    low_rank_contract_map_originals: list[str] = []
    libraries: dict[str, ctypes.CDLL] = {}
    runtime_dependencies: dict[str, tuple[str, ...]] = {}
    low_rank_library_names: list[str] = []
    attention_library_path: Path | None = None
    selected_routed_targets = (
        _SHARED_ROUTED_TARGETS if composition_mode.uses_shared_map else _TARGETS.routed_attention.routed
    )
    if not composition_mode.uses_shared_map:
        selected_targets = (
            selected_routed_targets.forward,
            selected_routed_targets.input_adjoint,
            *selected_routed_targets.weight_gradients,
            _TARGETS.routed_attention.attention_backward,
            *_TARGETS.axis_folds,
        )
    else:
        weighted_targets = (
            (_WEIGHTED_RELATION_FUSED_TARGET,)
            if composition_mode.fuses_weighted_reverse
            else (_WEIGHTED_RELATION_CONTRACT_TARGET, _WEIGHTED_RELATION_FOLD_TARGET)
        )
        selected_targets = (
            selected_routed_targets.forward,
            *selected_routed_targets.input_contracts,
            selected_routed_targets.shared_contract_multi_map,
            selected_routed_targets.source_fold,
            *selected_routed_targets.weight_gradients,
            *weighted_targets,
            *(
                (_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET, _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET)
                if composition_mode.generates_normalized_exp_pair
                else ()
            ),
            *(
                (_LOW_RANK_CONTRACT_MAP_FORWARD_TARGET, _LOW_RANK_CONTRACT_MAP_REVERSE_TARGET)
                if composition_mode.generates_low_rank_gated_products
                else ()
            ),
            _TARGETS.routed_attention.attention_backward,
            *_TARGETS.axis_folds,
        )
    expected_target_occurrences = dict.fromkeys(selected_targets, 1)
    if composition_mode.generates_low_rank_gated_products:
        expected_target_occurrences[_LOW_RANK_CONTRACT_MAP_FORWARD_TARGET] = 6
        expected_target_occurrences[_LOW_RANK_CONTRACT_MAP_REVERSE_TARGET] = 4
    exact_target_occurrences: dict[str, int] | None = None
    final_capture_site_manifest: CaptureSiteManifest | None = None
    try:
        with set_mesh(_mesh()):
            train_step, state, batch = _natural_train_step()
            host_state = jax.device_get(state)
            host_batch = jax.device_get(batch)

            def fresh_inputs() -> tuple[Any, Any]:
                fresh_state = jax.tree.map(jax.numpy.array, host_state)
                fresh_batch = jax.tree.map(jax.numpy.array, host_batch)
                jax.block_until_ready((fresh_state, fresh_batch))
                return fresh_state, fresh_batch

            baseline = train_step.lower(state, batch, compute_watch=False).compile()
            expected = baseline(state, batch)
            jax.block_until_ready(expected)

            def compile_target(source: str, target: str, name: str) -> ctypes.CDLL:
                source_directory = directory / name
                source_directory.mkdir(exist_ok=True)
                (directory / f"generated_{name}.cu").write_text(source)
                library = _compile_cuda_ffi_handler(source, source_directory, nvcc, architecture)
                _register_cuda_target(library, target)
                libraries[name] = library
                runtime_dependencies[name] = _runtime_dependency_audit(Path(library._name), source)
                return library

            def replace(serialized_module: bytes) -> bytes | None:
                nonlocal attention_library_path
                module = hlo_runtime.module_from_serialized_proto(serialized_module)
                if module.name != "jit_train_step":
                    return None
                original = module.to_string()
                original_modules.append(original)
                if artifact_directory is not None:
                    write_gzip_text(directory / "original-gpu-pre-scheduler-hlo.txt.gz", original)
                default_attention_plan = plan_streaming_attention_backward_hlo_region_replacement(
                    original,
                    attention_program,
                    default_attention,
                )
                attention = generate_streaming_attention_backward_ffi(
                    attention_program,
                    attention_schedule,
                    target_name=_TARGETS.routed_attention.attention_backward,
                    output_layouts=derive_streaming_attention_backward_ffi_output_layouts(default_attention_plan),
                )
                generated_axis_folds = _generate_axis_fold_programs(original)
                if composition_mode is RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT:
                    plan = plan_routed_training_attention_and_axis_fold_typed_ffi(
                        original,
                        attention_program,
                        attention,
                        generated_axis_folds,
                        axis_fold_numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
                    )
                    routed_plan = plan.routed_attention.routed
                    attention_plan = plan.routed_attention.attention_backward
                else:
                    plan = _plan_shared_map_composition(
                        original,
                        attention_program,
                        attention,
                        generated_axis_folds,
                    )
                    routed_plan = plan.routed
                    attention_plan = plan.attention_backward
                forward = generate_cuda_routed_forward_ffi(
                    routed_plan.forward,
                    target=(
                        _TARGETS.routed_attention.routed.forward
                        if composition_mode is RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT
                        else _SHARED_ROUTED_TARGETS.forward
                    ),
                )
                input_adjoint = None
                shared_multi_map = None
                input_contracts: tuple[GeneratedRankTwoContractFfi, ...] = ()
                source_fold: GeneratedSourceIndexedFoldFfi | None = None
                weighted_relation_contract: GeneratedRankTwoContractFfi | None = None
                weighted_relation_fold: GeneratedRelationEdgeFoldFfi | None = None
                weighted_relation_fused: GeneratedContractRelationFoldFfi | None = None
                normalized_exp_contract_forward: GeneratedCudaNormalizedExpContractForwardFfi | None = None
                normalized_exp_contract_reverse: GeneratedCudaNormalizedExpContractReverseFfi | None = None
                if not composition_mode.uses_shared_map:
                    input_adjoint = generate_cuda_routed_input_adjoint_ffi(
                        routed_plan.input_adjoint,
                        target=_TARGETS.routed_attention.routed.input_adjoint,
                    )
                    routed_weight_targets = _TARGETS.routed_attention.routed.weight_gradients
                else:
                    input_contracts = tuple(
                        generate_cuda_rank_two_contract_ffi(contract, target=target)
                        for contract, target in zip(
                            routed_plan.input_contracts,
                            _SHARED_ROUTED_TARGETS.input_contracts,
                            strict=True,
                        )
                    )
                    shared_multi_map = generate_cuda_shared_contract_multi_map_ffi(
                        routed_plan.shared_contract_multi_map,
                        target=_SHARED_ROUTED_TARGETS.shared_contract_multi_map,
                    )
                    source_fold = generate_cuda_source_indexed_fold_ffi(
                        routed_plan.source_fold,
                        target=_SHARED_ROUTED_TARGETS.source_fold,
                        physical_candidate=(
                            DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE
                            if _SHARED_ROUTED_TARGETS.source_fold in compatible_targets
                            else DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED
                        ),
                    )
                    if composition_mode.fuses_weighted_reverse:
                        weighted_relation_fused = generate_cuda_contract_relation_fold_ffi(
                            plan.weighted_relation_reverse.payload_contract,
                            plan.weighted_relation_reverse.edge_fold,
                            target=_WEIGHTED_RELATION_FUSED_TARGET,
                            physical_candidate=(
                                DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE
                                if _WEIGHTED_RELATION_FUSED_TARGET in compatible_targets
                                else DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED
                            ),
                        )
                    else:
                        weighted_relation_contract = generate_cuda_rank_two_contract_ffi(
                            plan.weighted_relation_reverse.payload_contract,
                            target=_WEIGHTED_RELATION_CONTRACT_TARGET,
                        )
                        weighted_relation_fold = generate_cuda_relation_edge_fold_ffi(
                            plan.weighted_relation_reverse.edge_fold,
                            target=_WEIGHTED_RELATION_FOLD_TARGET,
                        )
                    if composition_mode.generates_normalized_exp_pair:
                        normalized_exp_contract_forward = generate_cuda_normalized_exp_contract_forward_ffi(
                            plan.normalized_exp_contract_forward,
                            target=_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                            command_buffer_compatible=(_NORMALIZED_EXP_CONTRACT_FORWARD_TARGET in compatible_targets),
                        )
                        normalized_exp_contract_reverse = generate_cuda_normalized_exp_contract_reverse_ffi(
                            plan.normalized_exp_contract_reverse,
                            target=_NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
                            command_buffer_compatible=(_NORMALIZED_EXP_CONTRACT_REVERSE_TARGET in compatible_targets),
                        )
                    routed_weight_targets = _SHARED_ROUTED_TARGETS.weight_gradients
                weights = tuple(
                    generate_cuda_group_batched_contract_ffi(weight, target=target)
                    for weight, target in zip(
                        routed_plan.weight_gradients,
                        routed_weight_targets,
                        strict=True,
                    )
                )
                generated_sources = (
                    forward.source,
                    *((input_adjoint.source,) if input_adjoint is not None else ()),
                    *(contract.source for contract in input_contracts),
                    *((shared_multi_map.source,) if shared_multi_map is not None else ()),
                    *((source_fold.source,) if source_fold is not None else ()),
                    *((weighted_relation_contract.source,) if weighted_relation_contract is not None else ()),
                    *((weighted_relation_fold.source,) if weighted_relation_fold is not None else ()),
                    *((weighted_relation_fused.source,) if weighted_relation_fused is not None else ()),
                    *((normalized_exp_contract_forward.source,) if normalized_exp_contract_forward is not None else ()),
                    *((normalized_exp_contract_reverse.source,) if normalized_exp_contract_reverse is not None else ()),
                    *(weight.source for weight in weights),
                    *(axis_fold.source for axis_fold in generated_axis_folds),
                )
                if any("atomicAdd(" in source for source in generated_sources):
                    raise RuntimeError("generated routed training source contains semantic atomic accumulation")
                forward_target = (
                    _TARGETS.routed_attention.routed.forward
                    if composition_mode is RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT
                    else _SHARED_ROUTED_TARGETS.forward
                )
                compile_target(forward.source, forward_target, "routed_forward")
                if input_adjoint is not None:
                    compile_target(
                        input_adjoint.source,
                        _TARGETS.routed_attention.routed.input_adjoint,
                        "routed_input_adjoint",
                    )
                if shared_multi_map is not None:
                    for index, (contract, target) in enumerate(
                        zip(
                            input_contracts,
                            _SHARED_ROUTED_TARGETS.input_contracts,
                            strict=True,
                        )
                    ):
                        compile_target(contract.source, target, f"rank_two_input_contract_{index}")
                    compile_target(
                        shared_multi_map.source,
                        _SHARED_ROUTED_TARGETS.shared_contract_multi_map,
                        "shared_contract_multi_map",
                    )
                    if source_fold is None:
                        raise RuntimeError("shared-Map composition did not generate its source Fold")
                    compile_target(
                        source_fold.source,
                        _SHARED_ROUTED_TARGETS.source_fold,
                        "source_indexed_fold",
                    )
                    if composition_mode.fuses_weighted_reverse:
                        if weighted_relation_fused is None:
                            raise RuntimeError("shared-Map composition did not generate fused weighted reverse")
                        compile_target(
                            weighted_relation_fused.source,
                            _WEIGHTED_RELATION_FUSED_TARGET,
                            "weighted_relation_contract_fold",
                        )
                    else:
                        if weighted_relation_contract is None or weighted_relation_fold is None:
                            raise RuntimeError("shared-Map composition did not generate weighted reverse bodies")
                        compile_target(
                            weighted_relation_contract.source,
                            _WEIGHTED_RELATION_CONTRACT_TARGET,
                            "weighted_relation_contract",
                        )
                        compile_target(
                            weighted_relation_fold.source,
                            _WEIGHTED_RELATION_FOLD_TARGET,
                            "weighted_relation_fold",
                        )
                    if composition_mode.generates_normalized_exp_pair:
                        if normalized_exp_contract_forward is None or normalized_exp_contract_reverse is None:
                            raise RuntimeError("shared-Map composition did not generate normalized-exp forward/reverse")
                        compile_target(
                            normalized_exp_contract_forward.source,
                            _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                            "normalized_exp_contract_forward",
                        )
                        compile_target(
                            normalized_exp_contract_reverse.source,
                            _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
                            "normalized_exp_contract_reverse",
                        )
                for index, (weight, target) in enumerate(zip(weights, routed_weight_targets, strict=True)):
                    compile_target(weight.source, target, f"group_batched_weight_gradient_{index}")
                for index, (axis_fold, target) in enumerate(zip(generated_axis_folds, _TARGETS.axis_folds, strict=True)):
                    compile_target(axis_fold.source, target, f"axis_fold_{index}")
                compiled_attention = compile_streaming_attention_backward_ffi(
                    attention,
                    repository=repository,
                    directory=directory / "attention_backward",
                    nvcc=nvcc,
                    architecture=architecture,
                    triton_target=triton_target,
                )
                register_streaming_attention_backward_ffi(compiled_attention)
                libraries["attention_backward"] = compiled_attention.library
                runtime_dependencies["attention_backward"] = _runtime_dependency_audit(
                    compiled_attention.library_path,
                    compiled_attention.source_path.read_text(),
                )
                attention_library_path = compiled_attention.library_path
                if not composition_mode.uses_shared_map:
                    rewritten = replace_routed_training_attention_and_axis_fold_regions_with_custom_calls(
                        original,
                        plan,
                        targets=_TARGETS,
                    )
                else:
                    rewritten = _replace_shared_map_composition(
                        original,
                        plan,
                        fuse_weighted_reverse=composition_mode.fuses_weighted_reverse,
                        generate_normalized_exp_pair=composition_mode.generates_normalized_exp_pair,
                    )
                generated_low_rank_contract_maps: tuple[GeneratedCudaContractMapChainFfi, ...] = ()
                if composition_mode.generates_low_rank_gated_products:
                    low_rank_plan = plan_generated_low_rank_contract_map_training(
                        rewritten,
                        forward_target_prefix=_LOW_RANK_CONTRACT_MAP_FORWARD_TARGET,
                        reverse_target_prefix=_LOW_RANK_CONTRACT_MAP_REVERSE_TARGET,
                    )
                    generated_low_rank_contract_maps = tuple(
                        generate_cuda_contract_map_chain_ffi(
                            family.program,
                            forward_target=family.forward_target,
                            reverse_target=family.reverse_target,
                            physical_candidate=(
                                DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE
                                if family.forward_target in compatible_targets
                                and family.reverse_target in compatible_targets
                                else DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED
                            ),
                        )
                        for family in low_rank_plan.families
                    )
                    for index, generated_low_rank in enumerate(generated_low_rank_contract_maps):
                        name = f"low_rank_contract_map_family_{index}"
                        source_directory = directory / name
                        source_directory.mkdir(exist_ok=True)
                        (directory / f"generated_{name}.cu").write_text(generated_low_rank.source)
                        library = _compile_cuda_ffi_handler(
                            generated_low_rank.source,
                            source_directory,
                            nvcc,
                            architecture,
                        )
                        register_cuda_contract_map_chain_ffi(generated_low_rank, library)
                        libraries[name] = library
                        low_rank_library_names.append(name)
                        runtime_dependencies[name] = _runtime_dependency_audit(
                            Path(library._name), generated_low_rank.source
                        )
                    low_rank_contract_map_originals.append(rewritten)
                    rewritten = replace_generated_low_rank_contract_map_training(rewritten, low_rank_plan)
                    low_rank_contract_map_plans.append(low_rank_plan)
                recovered_plans.append(plan)
                generated_programs.append(
                    {
                        "forward": forward,
                        "input_adjoint": input_adjoint,
                        "input_contracts": input_contracts,
                        "shared_multi_map": shared_multi_map,
                        "source_fold": source_fold,
                        "weighted_relation_contract": weighted_relation_contract,
                        "weighted_relation_fold": weighted_relation_fold,
                        "weighted_relation_fused": weighted_relation_fused,
                        "normalized_exp_contract_forward": normalized_exp_contract_forward,
                        "normalized_exp_contract_reverse": normalized_exp_contract_reverse,
                        "low_rank_contract_maps": generated_low_rank_contract_maps,
                        "weights": weights,
                        "attention": compiled_attention,
                        "attention_source_sha256": {
                            "handler": _sha256(compiled_attention.source_path),
                            "aot_sources": [_sha256(path) for path in compiled_attention.aot_sources],
                        },
                        "axis_folds": generated_axis_folds,
                    }
                )
                transformed_module = hlo_runtime.module_from_text(rewritten)
                transformed_text = transformed_module.to_string()
                transformed_modules.append(transformed_text)
                if not composition_mode.uses_shared_map:
                    replacement_audits.append(
                        audit_routed_training_attention_and_axis_fold_replacement(
                            original,
                            transformed_text,
                            plan,
                            targets=_TARGETS,
                        )
                    )
                else:
                    replacement_audits.append(
                        _audit_shared_map_composition(
                            original,
                            transformed_text,
                            plan,
                            fuse_weighted_reverse=composition_mode.fuses_weighted_reverse,
                            generate_normalized_exp_pair=composition_mode.generates_normalized_exp_pair,
                        )
                    )
                if composition_mode.generates_low_rank_gated_products:
                    low_rank_contract_map_audits.append(
                        audit_generated_low_rank_contract_map_training(
                            low_rank_contract_map_originals[-1],
                            transformed_text,
                            low_rank_contract_map_plans[-1],
                        )
                    )
                attention_liveness_audits.append(
                    audit_streaming_attention_backward_region_replacement(
                        original,
                        transformed_text,
                        attention_plan,
                        target=_TARGETS.routed_attention.attention_backward,
                    )
                )
                return transformed_module.as_serialized_hlo_module_proto()

            xla.register_hlo_module_transformation(
                replace,
                name=f"{_PASS_NAME}_{composition_mode.value}",
                stage=xla.PipelineStage.PRE_SCHEDULER,
                platforms="cuda",
            )
            jax.clear_caches()
            transformed_state, transformed_batch = fresh_inputs()
            try:
                transformed = train_step.lower(
                    transformed_state,
                    transformed_batch,
                    compute_watch=False,
                ).compile()
            finally:
                xla.clear_hlo_module_transformation(
                    f"{_PASS_NAME}_{composition_mode.value}",
                    stage=xla.PipelineStage.PRE_SCHEDULER,
                    platforms="cuda",
                )
            if len(transformed_modules) != 1:
                raise RuntimeError("expected one transformed HLO module before execution")
            if artifact_directory is not None:
                write_gzip_text(
                    directory / "transformed-gpu-pre-scheduler-hlo.txt.gz",
                    transformed_modules[0],
                )
            exact_target_occurrences = _custom_call_target_occurrences(
                transformed_modules[0],
                expected_target_occurrences,
            )
            final_hlo = transformed.as_text()
            final_capture_site_manifest = _final_hlo_capture_site_manifest(
                final_hlo,
                executable="transformed",
                composition_mode=composition_mode,
                command_buffer_candidate=command_buffer_candidate,
            )
            if artifact_directory is not None:
                write_gzip_text(directory / "transformed-final-optimized-hlo.txt.gz", final_hlo)
            actual = transformed(transformed_state, transformed_batch)
            jax.block_until_ready(actual)
            comparison = _compare_under_ordered_fp(expected, actual)

            def timed_execution(executable: Any) -> tuple[float, str, list[dict[str, Any]]]:
                timing_state, timing_batch = fresh_inputs()
                start = time.perf_counter()
                output = executable(timing_state, timing_batch)
                jax.block_until_ready(output)
                latency_ms = (time.perf_counter() - start) * 1e3
                output_hash, leaf_hashes = _tree_hash_evidence(output)
                return latency_ms, output_hash, leaf_hashes

            for index in range(warmup):
                order = (baseline, transformed) if index % 2 == 0 else (transformed, baseline)
                for executable in order:
                    timed_execution(executable)

            raw_samples: list[dict[str, Any]] = []
            baseline_samples: list[float] = []
            transformed_samples: list[float] = []
            baseline_hashes: list[str] = []
            transformed_hashes: list[str] = []
            for index in range(repeats):
                order = ("baseline", "transformed") if index % 2 == 0 else ("transformed", "baseline")
                sample: dict[str, Any] = {"order": order}
                for name in order:
                    executable = baseline if name == "baseline" else transformed
                    latency, output_hash, leaf_hashes = timed_execution(executable)
                    sample[name] = {
                        "latency_ms": latency,
                        "output_hash": output_hash,
                        "output_leaf_hashes": leaf_hashes,
                    }
                    if name == "baseline":
                        baseline_samples.append(latency)
                        baseline_hashes.append(output_hash)
                    else:
                        transformed_samples.append(latency)
                        transformed_hashes.append(output_hash)
                raw_samples.append(sample)

            call_counts: dict[str, Any] = {
                "forward": _call_count(libraries["routed_forward"], "shuttle_routed_forward_call_count"),
                "weight_gradients": [
                    _call_count(
                        libraries[f"group_batched_weight_gradient_{index}"],
                        "shuttle_routed_weight_gradient_call_count",
                    )
                    for index in range(2)
                ],
                "attention_backward": _call_count(
                    libraries["attention_backward"],
                    "shuttle_streaming_attention_backward_ffi_call_count",
                ),
                "axis_folds": [
                    _call_count(
                        libraries[f"axis_fold_{index}"],
                        "shuttle_axis_fold_ffi_call_count",
                    )
                    for index in range(2)
                ],
            }
            if not composition_mode.uses_shared_map:
                call_counts["input_adjoint"] = _call_count(
                    libraries["routed_input_adjoint"],
                    "shuttle_routed_input_adjoint_call_count",
                )
            else:
                call_counts["input_contracts"] = [
                    _call_count(
                        libraries[f"rank_two_input_contract_{index}"],
                        "shuttle_rank_two_contract_call_count",
                    )
                    for index in range(2)
                ]
                call_counts["shared_contract_multi_map"] = _call_count(
                    libraries["shared_contract_multi_map"],
                    "shuttle_shared_contract_multi_map_call_count",
                )
                call_counts["source_fold"] = _call_count(
                    libraries["source_indexed_fold"],
                    "shuttle_source_indexed_fold_call_count",
                )
                if composition_mode.fuses_weighted_reverse:
                    call_counts["weighted_relation_contract_fold"] = _call_count(
                        libraries["weighted_relation_contract_fold"],
                        "shuttle_contract_relation_fold_call_count",
                    )
                else:
                    call_counts["weighted_relation_contract"] = _call_count(
                        libraries["weighted_relation_contract"],
                        "shuttle_rank_two_contract_call_count",
                    )
                    call_counts["weighted_relation_fold"] = _call_count(
                        libraries["weighted_relation_fold"],
                        "shuttle_relation_edge_fold_call_count",
                    )
                if composition_mode.generates_normalized_exp_pair:
                    call_counts["normalized_exp_contract_forward"] = _call_count(
                        libraries["normalized_exp_contract_forward"],
                        "shuttle_normalized_exp_contract_forward_call_count",
                    )
                    call_counts["normalized_exp_contract_reverse"] = _call_count(
                        libraries["normalized_exp_contract_reverse"],
                        "shuttle_normalized_exp_contract_reverse_call_count",
                    )
                if composition_mode.generates_low_rank_gated_products:
                    call_counts["low_rank_contract_map_families"] = [
                        {
                            "forward": _call_count(libraries[name], "shuttle_contract_map_chain_forward_call_count"),
                            "reverse": _call_count(libraries[name], "shuttle_contract_map_chain_reverse_call_count"),
                        }
                        for name in low_rank_library_names
                    ]
    finally:
        if artifact_directory is not None:
            for name in (
                "routed_forward",
                "routed_input_adjoint",
                "shared_contract_multi_map",
                "rank_two_input_contract_0",
                "rank_two_input_contract_1",
                "source_indexed_fold",
                "weighted_relation_contract",
                "weighted_relation_fold",
                "weighted_relation_contract_fold",
                "normalized_exp_contract_forward",
                "normalized_exp_contract_reverse",
                "group_batched_weight_gradient_0",
                "group_batched_weight_gradient_1",
                "axis_fold_0",
                "axis_fold_1",
                *low_rank_library_names,
            ):
                compiled_library = directory / name / "generated_pair_map_handler.so"
                if compiled_library.exists():
                    compiled_library.unlink()
            if attention_library_path is not None and attention_library_path.exists():
                attention_library_path.unlink()
        if temporary is not None:
            temporary.cleanup()

    if (
        len(original_modules) != 1
        or len(transformed_modules) != 1
        or len(recovered_plans) != 1
        or len(replacement_audits) != 1
        or len(attention_liveness_audits) != 1
        or (
            composition_mode.generates_low_rank_gated_products
            and (len(low_rank_contract_map_plans) != 1 or len(low_rank_contract_map_audits) != 1)
        )
        or (
            not composition_mode.generates_low_rank_gated_products
            and (low_rank_contract_map_plans or low_rank_contract_map_audits)
        )
    ):
        raise RuntimeError("expected one combined routed-plus-attention-plus-Fold replacement pass")
    original = original_modules[0]
    plan = recovered_plans[0]
    replacement_audit = replacement_audits[0]
    attention_liveness = attention_liveness_audits[0]
    generated = generated_programs[0]
    forward = generated["forward"]
    input_adjoint = generated["input_adjoint"]
    input_contracts: tuple[GeneratedRankTwoContractFfi, ...] = generated["input_contracts"]
    shared_multi_map: GeneratedSharedContractMultiMapFfi | None = generated["shared_multi_map"]
    source_fold: GeneratedSourceIndexedFoldFfi | None = generated["source_fold"]
    weighted_relation_contract: GeneratedRankTwoContractFfi | None = generated["weighted_relation_contract"]
    weighted_relation_fold: GeneratedRelationEdgeFoldFfi | None = generated["weighted_relation_fold"]
    weighted_relation_fused: GeneratedContractRelationFoldFfi | None = generated["weighted_relation_fused"]
    normalized_exp_contract_forward: GeneratedCudaNormalizedExpContractForwardFfi | None = generated[
        "normalized_exp_contract_forward"
    ]
    normalized_exp_contract_reverse: GeneratedCudaNormalizedExpContractReverseFfi | None = generated[
        "normalized_exp_contract_reverse"
    ]
    generated_low_rank_contract_maps: tuple[GeneratedCudaContractMapChainFfi, ...] = generated["low_rank_contract_maps"]
    low_rank_contract_map_plan = low_rank_contract_map_plans[0] if low_rank_contract_map_plans else None
    low_rank_contract_map_audit = low_rank_contract_map_audits[0] if low_rank_contract_map_audits else None
    weights = generated["weights"]
    compiled_attention = generated["attention"]
    attention_source_sha256 = generated["attention_source_sha256"]
    generated_axis_folds = generated["axis_folds"]
    weighted_relation_plan: WeightedRelationReverseTypedFfiPlan | None = None
    weighted_relation_audit: WeightedRelationReverseReplacementAudit | ContractRelationFoldReplacementAudit | None = None
    normalized_exp_contract_forward_audit: NormalizedExpContractForwardHloReplacementAudit | None = None
    normalized_exp_contract_reverse_audit: NormalizedExpContractReverseHloReplacementAudit | None = None
    if not composition_mode.uses_shared_map:
        routed_plan = plan.routed_attention.routed
        attention_plan = plan.routed_attention.attention_backward
        axis_fold_plans = plan.axis_folds
        routed_audit = replacement_audit.routed_attention.routed
        attention_instruction = replacement_audit.routed_attention.attention_backward_instruction
        axis_fold_audits = replacement_audit.axis_folds
        routed_targets = _TARGETS.routed_attention.routed
        mode_target_key = "input_adjoint"
        mode_target = routed_targets.input_adjoint
        mode_generated = input_adjoint
        mode_operands = routed_plan.input_adjoint.operands
        mode_numerical_policy = routed_plan.input_adjoint.region.numerical_policy.value
        input_adjoint_auxiliary = routed_audit.input_adjoint_auxiliary
        retained_input_adjoint_wrappers = ()
        generated_regions = [
            "Contract -> Map -> Contract -> source Fold",
            "Contract -> reverse Map -> Contract -> source Fold",
        ]
    else:
        routed_plan = plan.routed
        attention_plan = plan.attention_backward
        axis_fold_plans = plan.axis_folds
        routed_audit = replacement_audit.routed
        attention_instruction = replacement_audit.attention_backward_instruction
        axis_fold_audits = replacement_audit.axis_folds
        routed_targets = _SHARED_ROUTED_TARGETS
        mode_target_key = "shared_contract_multi_map"
        mode_target = routed_targets.shared_contract_multi_map
        mode_generated = shared_multi_map
        mode_operands = routed_plan.shared_contract_multi_map.operands
        mode_numerical_policy = routed_plan.shared_contract_multi_map.numerical_contract.numerical_policy.value
        input_adjoint_auxiliary = None
        retained_input_adjoint_wrappers = routed_audit.retained_input_adjoint_wrappers
        weighted_relation_plan = plan.weighted_relation_reverse
        weighted_relation_audit = replacement_audit.weighted_relation_reverse
        normalized_exp_contract_forward_audit = replacement_audit.normalized_exp_contract_forward
        normalized_exp_contract_reverse_audit = replacement_audit.normalized_exp_contract_reverse
        generated_regions = [
            "Contract -> Map -> Contract -> source Fold",
            "rank-two input-adjoint Contract 0",
            "shared Contract -> two generated scalar Maps",
            "rank-two input-adjoint Contract 1",
            "deterministic source-indexed Fold",
            (
                "one-CTA weighted RelationProgram reverse Contract -> edge Map -> hidden Fold -> source-slot Fold"
                if composition_mode.fuses_weighted_reverse
                else "weighted RelationProgram reverse Contract -> edge Map -> hidden Fold -> source-slot Fold"
            ),
            *(
                (
                    "compact normalized-exp Contract -> Map/Fold forward with saved state",
                    "normalized-exp saved state -> Map/Fold reverse -> two Contracts",
                )
                if composition_mode.generates_normalized_exp_pair
                else ()
            ),
            *(
                (
                    "six generic two-Contract scalar-Map forward/rematerialization calls",
                    "four JAX-owned reverse Map plus input/weight-adjoint Contract calls",
                )
                if composition_mode.generates_low_rank_gated_products
                else ()
            ),
        ]
    if mode_generated is None:
        raise RuntimeError(f"composition {composition_mode.value} did not generate its routed mode body")
    if composition_mode.uses_shared_map:
        if composition_mode.fuses_weighted_reverse and weighted_relation_fused is None:
            raise RuntimeError("shared-Map composition did not retain fused weighted reverse")
        if not composition_mode.fuses_weighted_reverse and (
            weighted_relation_contract is None or weighted_relation_fold is None
        ):
            raise RuntimeError("shared-Map composition did not retain generated weighted relation reverse bodies")
        if composition_mode.generates_normalized_exp_pair and (
            normalized_exp_contract_forward is None
            or normalized_exp_contract_reverse is None
            or normalized_exp_contract_forward_audit is None
            or normalized_exp_contract_reverse_audit is None
        ):
            raise RuntimeError("shared-Map composition did not retain generated normalized-exp forward/reverse")
    if exact_target_occurrences is None:
        raise RuntimeError("exact custom-call target validation did not run before execution")
    if final_capture_site_manifest is None:
        raise RuntimeError("final-HLO capture-site manifest was not derived before execution")
    target_occurrences: dict[str, Any] = {
        "forward": exact_target_occurrences[routed_targets.forward],
        mode_target_key: exact_target_occurrences[mode_target],
        "weight_gradients": [exact_target_occurrences[target] for target in routed_targets.weight_gradients],
        "attention_backward": exact_target_occurrences[_TARGETS.routed_attention.attention_backward],
        "axis_folds": [exact_target_occurrences[target] for target in _TARGETS.axis_folds],
    }
    if composition_mode.uses_shared_map:
        target_occurrences["input_contracts"] = [
            exact_target_occurrences[target] for target in routed_targets.input_contracts
        ]
        target_occurrences["source_fold"] = exact_target_occurrences[routed_targets.source_fold]
        if composition_mode.fuses_weighted_reverse:
            target_occurrences["weighted_relation_contract_fold"] = exact_target_occurrences[
                _WEIGHTED_RELATION_FUSED_TARGET
            ]
            target_occurrences["eliminated_weighted_relation_targets"] = {
                _WEIGHTED_RELATION_CONTRACT_TARGET: _target_occurrence(
                    transformed_modules[0], _WEIGHTED_RELATION_CONTRACT_TARGET
                ),
                _WEIGHTED_RELATION_FOLD_TARGET: _target_occurrence(
                    transformed_modules[0], _WEIGHTED_RELATION_FOLD_TARGET
                ),
            }
            if any(target_occurrences["eliminated_weighted_relation_targets"].values()):
                raise RuntimeError("separated weighted-reverse custom calls remain in fused transformed HLO")
        else:
            target_occurrences["weighted_relation_contract"] = exact_target_occurrences[
                _WEIGHTED_RELATION_CONTRACT_TARGET
            ]
            target_occurrences["weighted_relation_fold"] = exact_target_occurrences[_WEIGHTED_RELATION_FOLD_TARGET]
        if composition_mode.generates_normalized_exp_pair:
            target_occurrences["normalized_exp_contract_forward"] = exact_target_occurrences[
                _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET
            ]
            target_occurrences["normalized_exp_contract_reverse"] = exact_target_occurrences[
                _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET
            ]
        if composition_mode.generates_low_rank_gated_products:
            if low_rank_contract_map_plan is None or low_rank_contract_map_audit is None:
                raise RuntimeError("generated low-rank Contract/Map evidence is incomplete")
            target_occurrences["low_rank_contract_map_families"] = [
                {
                    "forward": exact_target_occurrences[family.forward_target],
                    "reverse": exact_target_occurrences[family.reverse_target],
                }
                for family in low_rank_contract_map_plan.families
            ]
    minimum_calls = repeats + warmup + 1
    observed_calls = [
        call_counts["forward"],
        call_counts[mode_target_key],
        *call_counts["weight_gradients"],
        call_counts["attention_backward"],
        *call_counts["axis_folds"],
    ]
    capture_only_calls: list[int] = []
    if composition_mode.uses_shared_map:
        observed_calls.extend(call_counts["input_contracts"])
        source_fold_calls = call_counts["source_fold"]
        if _SHARED_ROUTED_TARGETS.source_fold in compatible_targets:
            capture_only_calls.append(source_fold_calls)
        else:
            observed_calls.append(source_fold_calls)
        if composition_mode.fuses_weighted_reverse:
            weighted_calls = call_counts["weighted_relation_contract_fold"]
            if _WEIGHTED_RELATION_FUSED_TARGET in compatible_targets:
                capture_only_calls.append(weighted_calls)
            else:
                observed_calls.append(weighted_calls)
        else:
            observed_calls.extend(
                (
                    call_counts["weighted_relation_contract"],
                    call_counts["weighted_relation_fold"],
                )
            )
        if composition_mode.generates_normalized_exp_pair:
            normalized_calls = (
                call_counts["normalized_exp_contract_forward"],
                call_counts["normalized_exp_contract_reverse"],
            )
            if _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET in compatible_targets:
                capture_only_calls.extend(normalized_calls)
            else:
                observed_calls.extend(normalized_calls)
        if composition_mode.generates_low_rank_gated_products:
            low_rank_calls = [
                count
                for family_counts in call_counts["low_rank_contract_map_families"]
                for count in (family_counts["forward"], family_counts["reverse"])
            ]
            if _LOW_RANK_CONTRACT_MAP_FORWARD_TARGET in compatible_targets:
                capture_only_calls.extend(low_rank_calls)
            else:
                observed_calls.extend(low_rank_calls)

    def write_execution_evidence(status: str, reason: str | None) -> None:
        if artifact_directory is None:
            return
        weighted_relation_evidence = None
        if weighted_relation_fused is not None:
            weighted_relation_evidence = {
                "targets": {"contract_map_fold": _WEIGHTED_RELATION_FUSED_TARGET},
                "semantic_sha256": {"contract_map_fold": weighted_relation_fused.semantic_digest},
                "source_sha256": {"contract_map_fold": weighted_relation_fused.source_digest},
                "physical_cost": {
                    "contract_fma_count": weighted_relation_fused.cost.contract_fma_count,
                    "payload_elements": weighted_relation_fused.cost.payload_elements,
                    "payload_global_bytes": weighted_relation_fused.cost.payload_global_bytes,
                    "kernel_launches": weighted_relation_fused.cost.kernel_launches,
                    "threads_per_block": weighted_relation_fused.cost.threads_per_block,
                    "shared_bytes": weighted_relation_fused.cost.shared_bytes,
                },
                "placement_collective": (
                    weighted_relation_audit.placement_collective if weighted_relation_audit is not None else None
                ),
            }
        elif weighted_relation_contract is not None and weighted_relation_fold is not None:
            weighted_relation_evidence = {
                "targets": {
                    "contract": _WEIGHTED_RELATION_CONTRACT_TARGET,
                    "fold": _WEIGHTED_RELATION_FOLD_TARGET,
                },
                "semantic_sha256": {
                    "contract": weighted_relation_contract.semantic_digest,
                    "fold": weighted_relation_fold.semantic_digest,
                },
                "source_sha256": {
                    "contract": weighted_relation_contract.source_digest,
                    "fold": weighted_relation_fold.source_digest,
                },
                "placement_collective": (
                    weighted_relation_audit.placement_collective if weighted_relation_audit is not None else None
                ),
            }
        normalized_exp_evidence = None
        if normalized_exp_contract_forward is not None and normalized_exp_contract_reverse is not None:
            normalized_exp_evidence = {
                "forward": {
                    "target": _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                    "command_buffer_compatible": normalized_exp_contract_forward.command_buffer_compatible,
                    "semantic_sha256": normalized_exp_contract_forward.semantic_digest,
                    "source_sha256": normalized_exp_contract_forward.source_digest,
                    "inputs": [
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_forward.inputs
                    ],
                    "outputs": [
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_forward.outputs
                    ],
                    "dead_instructions": (
                        normalized_exp_contract_forward_audit.dead_instructions
                        if normalized_exp_contract_forward_audit is not None
                        else None
                    ),
                },
                "reverse": {
                    "target": _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
                    "command_buffer_compatible": normalized_exp_contract_reverse.command_buffer_compatible,
                    "semantic_sha256": normalized_exp_contract_reverse.semantic_digest,
                    "source_sha256": normalized_exp_contract_reverse.source_digest,
                    "inputs": [
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_reverse.inputs
                    ],
                    "outputs": [
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_reverse.outputs
                    ],
                    "dead_instructions": (
                        normalized_exp_contract_reverse_audit.dead_instructions
                        if normalized_exp_contract_reverse_audit is not None
                        else None
                    ),
                    "placement_paths": (
                        normalized_exp_contract_reverse_audit.placement_paths
                        if normalized_exp_contract_reverse_audit is not None
                        else None
                    ),
                },
                "saved_state_link": plan.normalized_exp_contract_reverse.region.saved_state.instruction,
            }
        evidence = {
            "status": status,
            "reason": reason,
            "command_buffer_candidate": command_buffer_candidate.value,
            "command_buffer_compatible_targets": compatible_targets,
            "xla_command_buffer_startup_selection": (
                {
                    "uses_xla_default": command_buffer_flag_audit.uses_xla_default,
                    "selected_entries": command_buffer_flag_audit.selected_entries,
                }
                if command_buffer_flag_audit is not None
                else None
            ),
            "minimum_custom_call_handler_executions": minimum_calls,
            "capture_only_handler_minimum_executions": 1 if capture_only_calls else None,
            "capture_only_handler_targets": compatible_targets,
            "final_hlo_capture_site_manifest": final_capture_site_manifest.to_json(),
            "custom_call_occurrences_in_transformed_hlo": target_occurrences,
            "custom_call_handler_executions": call_counts,
            "generated_runtime_dependencies": runtime_dependencies,
            "weighted_relation_reverse": weighted_relation_evidence,
            "normalized_exp_contract_forward_reverse": normalized_exp_evidence,
            "baseline_samples_ms": baseline_samples,
            "generated_samples_ms": transformed_samples,
            "baseline_output_hashes": baseline_hashes,
            "generated_output_hashes": transformed_hashes,
            "raw_samples": raw_samples,
            "comparison": comparison,
        }
        (directory / "execution-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    def fail_after_execution(message: str) -> None:
        write_execution_evidence("unaccepted", message)
        raise RuntimeError(message)

    if any(count < minimum_calls for count in observed_calls):
        fail_after_execution(f"custom-call handler evidence mismatch: minimum={minimum_calls}, calls={observed_calls}")
    if any(count < 1 for count in capture_only_calls):
        fail_after_execution(f"command-buffer capture handler did not execute: calls={capture_only_calls}")
    weighted_relation_inputs: tuple[str, ...] = ()
    normalized_exp_inputs: tuple[str, ...] = ()
    low_rank_contract_map_inputs: tuple[str, ...] = ()
    if composition_mode.uses_shared_map:
        if weighted_relation_plan is None:
            fail_after_execution("shared-Map composition lost its weighted relation reverse plan")
        weighted_relation_inputs = (
            weighted_relation_plan.payload_contract.lhs.instruction,
            weighted_relation_plan.payload_contract.rhs.instruction,
            weighted_relation_plan.edge_fold.initial.instruction,
            weighted_relation_plan.edge_fold.source_indices.instruction,
            weighted_relation_plan.edge_fold.edge_cotangent.instruction,
        )
        if composition_mode.generates_normalized_exp_pair:
            normalized_exp_inputs = tuple(
                dict.fromkeys(
                    (
                        *(value.instruction for value in plan.normalized_exp_contract_forward.inputs),
                        *(
                            value.instruction
                            for value in plan.normalized_exp_contract_reverse.inputs
                            if not value.instruction.startswith("shuttle.generated.")
                        ),
                    )
                )
            )
        if composition_mode.generates_low_rank_gated_products:
            if low_rank_contract_map_plan is None:
                fail_after_execution("generated low-rank Contract/Map plan is missing")
            low_rank_contract_map_inputs = tuple(
                dict.fromkeys(
                    (
                        *(
                            value.instruction
                            for boundary in low_rank_contract_map_plan.logical.forward
                            for value in (
                                boundary.semantics.down_contract.lhs,
                                boundary.semantics.down_contract.rhs,
                                boundary.semantics.up_contract.rhs,
                            )
                        ),
                        *(
                            value.instruction
                            for boundary in low_rank_contract_map_plan.logical.reverse
                            for value in boundary.cotangent_inputs
                        ),
                    )
                )
            )
    shared_input_operands = (
        (
            *(
                operand.instruction
                for contract in routed_plan.input_contracts
                for operand in (contract.lhs, contract.rhs)
            ),
            routed_plan.source_fold.initial.instruction,
            routed_plan.source_fold.source_indices.instruction,
            routed_plan.source_fold.contributions.instruction,
            *weighted_relation_inputs,
            *normalized_exp_inputs,
            *low_rank_contract_map_inputs,
        )
        if composition_mode.uses_shared_map
        else ()
    )
    all_operands = tuple(
        dict.fromkeys(
            (
                *(operand.value.instruction for operand in routed_plan.forward.operands),
                *(operand.value.instruction for operand in mode_operands),
                *shared_input_operands,
                *(operand.value.instruction for weight in routed_plan.weight_gradients for operand in weight.operands),
                *(value.instruction for value in attention_plan.inputs),
                *(value.instruction for fold in axis_fold_plans for value in fold.inputs),
            )
        )
    )
    parameter_ancestors = entry_parameter_ancestors(original, all_operands)
    static_bindings = [(operand.role.value, operand.value.instruction) for operand in mode_operands]
    if composition_mode.uses_shared_map:
        static_bindings.extend(
            (
                ("source_fold.initial", routed_plan.source_fold.initial.instruction),
                ("source_fold.source_indices", routed_plan.source_fold.source_indices.instruction),
                ("source_fold.contributions", routed_plan.source_fold.contributions.instruction),
                (
                    "weighted_relation_reverse.initial",
                    weighted_relation_plan.edge_fold.initial.instruction,
                ),
            )
        )
    static_roles = tuple(role for role, instruction in static_bindings if not parameter_ancestors[instruction])
    expected_static_roles = (
        (RoutedInputAdjointFfiOperandRole.FOLD_INITIAL.value,)
        if composition_mode is RoutedTrainingCompositionMode.MONOLITHIC_INPUT_ADJOINT
        else ("source_fold.initial", "weighted_relation_reverse.initial")
    )
    if static_roles != expected_static_roles:
        fail_after_execution(f"unexpected routed training static operands: {static_roles}")
    baseline_median = statistics.median(baseline_samples)
    transformed_median = statistics.median(transformed_samples)
    if len(set(transformed_hashes)) != 1:
        fail_after_execution("generated routed composition result is not bitwise deterministic")
    weighted_relation_report: dict[str, Any] = {}
    weighted_relation_targets: dict[str, str] = {}
    weighted_relation_semantic_hashes: dict[str, str] = {}
    weighted_relation_source_hashes: dict[str, str] = {}
    weighted_relation_target_instructions: tuple[str, ...] = ()
    weighted_relation_collectives: tuple[str, ...] = ()
    normalized_exp_report: dict[str, Any] = {}
    normalized_exp_targets: dict[str, str] = {}
    normalized_exp_semantic_hashes: dict[str, str] = {}
    normalized_exp_source_hashes: dict[str, str] = {}
    normalized_exp_target_instructions: tuple[str, ...] = ()
    normalized_exp_collectives: tuple[str, ...] = ()
    low_rank_contract_map_report: dict[str, Any] = {}
    low_rank_contract_map_targets: dict[str, Any] = {}
    low_rank_contract_map_semantic_hashes: dict[str, Any] = {}
    low_rank_contract_map_source_hashes: dict[str, Any] = {}
    low_rank_contract_map_target_instructions: tuple[str, ...] = ()
    if composition_mode.uses_shared_map:
        if weighted_relation_plan is None or weighted_relation_audit is None:
            fail_after_execution("shared-Map weighted relation reverse evidence is incomplete")
        weighted_relation_report = {
            "weighted_relation_reverse": {
                "contract": weighted_relation_plan.payload_contract.numerical_contract.numerical_policy.value,
                "fold": weighted_relation_plan.edge_fold.numerical_contract.numerical_policy.value,
                "payload_policy": weighted_relation_plan.payload_policy.value,
                "contract_output_dtype": weighted_relation_plan.payload_contract.numerical_contract.output_dtype,
                "contract_output_rounding": weighted_relation_plan.payload_contract.numerical_contract.output_rounding,
                "map_dtype": weighted_relation_plan.edge_fold.numerical_contract.map_dtype,
                "inner_state_dtype": weighted_relation_plan.edge_fold.numerical_contract.inner_state_dtype,
                "outer_state_dtype": weighted_relation_plan.edge_fold.numerical_contract.outer_state_dtype,
                "deterministic": weighted_relation_plan.edge_fold.numerical_contract.deterministic,
                "atomic_accumulation": weighted_relation_plan.edge_fold.numerical_contract.atomic_accumulation,
            }
        }
        if composition_mode.fuses_weighted_reverse:
            if weighted_relation_fused is None or not isinstance(
                weighted_relation_audit, ContractRelationFoldReplacementAudit
            ):
                fail_after_execution("fused weighted relation reverse evidence is incomplete")
            weighted_relation_targets = {"weighted_relation_contract_fold": _WEIGHTED_RELATION_FUSED_TARGET}
            weighted_relation_semantic_hashes = {
                "weighted_relation_contract_fold": weighted_relation_fused.semantic_digest
            }
            weighted_relation_source_hashes = {"weighted_relation_contract_fold": weighted_relation_fused.source_digest}
            weighted_relation_target_instructions = (weighted_relation_audit.call_instruction,)
            weighted_relation_report["weighted_relation_reverse"]["physical_cost"] = {
                "contract_fma_count": weighted_relation_fused.cost.contract_fma_count,
                "payload_elements": weighted_relation_fused.cost.payload_elements,
                "payload_global_bytes": weighted_relation_fused.cost.payload_global_bytes,
                "kernel_launches": weighted_relation_fused.cost.kernel_launches,
                "threads_per_block": weighted_relation_fused.cost.threads_per_block,
                "shared_bytes": weighted_relation_fused.cost.shared_bytes,
            }
            weighted_relation_report["weighted_relation_reverse"][
                "dead_instructions"
            ] = weighted_relation_audit.dead_instructions
        else:
            if (
                weighted_relation_contract is None
                or weighted_relation_fold is None
                or not isinstance(weighted_relation_audit, WeightedRelationReverseReplacementAudit)
            ):
                fail_after_execution("separated weighted relation reverse evidence is incomplete")
            weighted_relation_targets = {
                "weighted_relation_contract": _WEIGHTED_RELATION_CONTRACT_TARGET,
                "weighted_relation_fold": _WEIGHTED_RELATION_FOLD_TARGET,
            }
            weighted_relation_semantic_hashes = {
                "weighted_relation_contract": weighted_relation_contract.semantic_digest,
                "weighted_relation_fold": weighted_relation_fold.semantic_digest,
            }
            weighted_relation_source_hashes = {
                "weighted_relation_contract": weighted_relation_contract.source_digest,
                "weighted_relation_fold": weighted_relation_fold.source_digest,
            }
            weighted_relation_target_instructions = (
                weighted_relation_audit.contract_instruction,
                weighted_relation_audit.fold_instruction,
            )
        weighted_relation_collectives = (weighted_relation_audit.placement_collective,)
        if composition_mode.generates_normalized_exp_pair:
            if (
                normalized_exp_contract_forward is None
                or normalized_exp_contract_reverse is None
                or normalized_exp_contract_forward_audit is None
                or normalized_exp_contract_reverse_audit is None
            ):
                fail_after_execution("generated normalized-exp forward/reverse evidence is incomplete")
            normalized_exp_report = {
                "normalized_exp_contract_forward": {
                    "score_contract_boundary": "bf16_rne",
                    "fold_order": "source_ordered_fp32",
                    "dead_instructions": normalized_exp_contract_forward_audit.dead_instructions,
                    "retained_boundary_instructions": (
                        normalized_exp_contract_forward_audit.retained_boundary_instructions
                    ),
                    "outputs_and_users": normalized_exp_contract_forward_audit.output_users,
                    "inputs": tuple(
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_forward.inputs
                    ),
                    "outputs": tuple(
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_forward.outputs
                    ),
                },
                "normalized_exp_contract_reverse": {
                    "score_contract_boundary": "bf16_rne",
                    "score_cotangent_boundary": "bf16_rne",
                    "accumulation": "ordered_fp32",
                    "dead_instructions": normalized_exp_contract_reverse_audit.dead_instructions,
                    "placement_paths": normalized_exp_contract_reverse_audit.placement_paths,
                    "inputs": tuple(
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_reverse.inputs
                    ),
                    "outputs": tuple(
                        {"instruction": value.instruction, "shape": value.shape}
                        for value in plan.normalized_exp_contract_reverse.outputs
                    ),
                    "saved_state_from_forward": plan.normalized_exp_contract_reverse.region.saved_state.instruction,
                },
            }
            normalized_exp_targets = {
                "normalized_exp_contract_forward": _NORMALIZED_EXP_CONTRACT_FORWARD_TARGET,
                "normalized_exp_contract_reverse": _NORMALIZED_EXP_CONTRACT_REVERSE_TARGET,
            }
            normalized_exp_semantic_hashes = {
                "normalized_exp_contract_forward": normalized_exp_contract_forward.semantic_digest,
                "normalized_exp_contract_reverse": normalized_exp_contract_reverse.semantic_digest,
            }
            normalized_exp_source_hashes = {
                "normalized_exp_contract_forward": normalized_exp_contract_forward.source_digest,
                "normalized_exp_contract_reverse": normalized_exp_contract_reverse.source_digest,
            }
            normalized_exp_target_instructions = (
                normalized_exp_contract_forward_audit.call_instruction,
                normalized_exp_contract_reverse_audit.call_instruction,
            )
            normalized_exp_collectives = tuple(path[2] for path in normalized_exp_contract_reverse_audit.placement_paths)
    if composition_mode.generates_low_rank_gated_products:
        if (
            low_rank_contract_map_plan is None
            or low_rank_contract_map_audit is None
            or not generated_low_rank_contract_maps
        ):
            fail_after_execution("generated low-rank Contract/Map evidence is incomplete")
        low_rank_contract_map_report = {
            "low_rank_contract_map_training": {
                "numerical_policy": "source_ordered",
                "generated_calls": low_rank_contract_map_audit.generated_call_count,
                "generated_targets": low_rank_contract_map_audit.generated_target_count,
                "removed_contracts": low_rank_contract_map_audit.removed_dot_count,
                "removed_contract_flops": low_rank_contract_map_audit.removed_dot_flops,
                "weight_adjoint_layouts": [
                    {
                        "first": family.program.first_weight_adjoint_minor_to_major,
                        "second": family.program.second_weight_adjoint_minor_to_major,
                    }
                    for family in low_rank_contract_map_plan.families
                ],
                "target_occurrences": low_rank_contract_map_audit.target_occurrences,
            }
        }
        low_rank_contract_map_targets = {
            "low_rank_contract_map_families": [
                {"forward": family.forward_target, "reverse": family.reverse_target}
                for family in low_rank_contract_map_plan.families
            ]
        }
        low_rank_contract_map_semantic_hashes = {
            "low_rank_contract_map_families": [
                generated.semantic_digest for generated in generated_low_rank_contract_maps
            ]
        }
        low_rank_contract_map_source_hashes = {
            "low_rank_contract_map_families": [generated.source_digest for generated in generated_low_rank_contract_maps]
        }
        low_rank_contract_map_target_instructions = tuple(
            call.call_instruction
            for call in (*low_rank_contract_map_audit.forward, *low_rank_contract_map_audit.reverse)
        )
    write_execution_evidence("execution_checks_passed", None)
    return {
        "kind": "xla_grug_combined_routed_training_attention_and_axis_fold_generated_ffi",
        "composition_mode": composition_mode.value,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": "cuda",
        "device_kind": jax.devices()[0].device_kind,
        "architecture": architecture,
        "natural_frontend": "ordinary one-layer Grug train step with JAX-owned differentiation",
        "generated_regions": [
            *generated_regions,
            "group-batched weight Contract 0",
            "group-batched weight Contract 1",
            "causal GQA attention reverse Contract/Fold/DomainRestriction region",
            "generic 8x32 row-axis Fold 0",
            "generic 8x32 row-axis Fold 1",
        ],
        "numerical_policies": {
            "forward": routed_plan.forward.region.numerical_policy.value,
            mode_target_key: mode_numerical_policy,
            **(
                {
                    "input_contracts": [
                        contract.numerical_contract.numerical_policy.value for contract in routed_plan.input_contracts
                    ],
                    "source_fold": routed_plan.source_fold.numerical_contract.numerical_policy.value,
                }
                if composition_mode.uses_shared_map
                else {}
            ),
            "weight_gradients": [
                weight.numerical_contract.numerical_policy.value for weight in routed_plan.weight_gradients
            ],
            "attention_backward": attention_plan.reassociation,
            "axis_folds": _axis_fold_reassociation_report(axis_fold_plans),
            **weighted_relation_report,
            **normalized_exp_report,
            **low_rank_contract_map_report,
        },
        "external_collectives": (
            (
                routed_audit.source_fold_collective,
                *weighted_relation_collectives,
                *normalized_exp_collectives,
                *routed_audit.weight_gradient_collectives,
            )
            if composition_mode.uses_shared_map
            else routed_audit.weight_gradient_collectives
        ),
        "uses_atomic_accumulation": False,
        "static_operand_roles": static_roles,
        "operand_ancestry": {operand: parameter_ancestors[operand] for operand in all_operands},
        "output_alias_operands": [weight.output_alias_operand for weight in routed_plan.weight_gradients],
        "custom_call_targets": {
            "forward": routed_targets.forward,
            mode_target_key: mode_target,
            **(
                {
                    "input_contracts": routed_targets.input_contracts,
                    "source_fold": routed_targets.source_fold,
                }
                if composition_mode.uses_shared_map
                else {}
            ),
            "weight_gradients": routed_targets.weight_gradients,
            "attention_backward": _TARGETS.routed_attention.attention_backward,
            "axis_folds": _TARGETS.axis_folds,
            **weighted_relation_targets,
            **normalized_exp_targets,
            **low_rank_contract_map_targets,
        },
        "custom_call_occurrences_in_transformed_hlo": target_occurrences,
        "final_hlo_capture_site_manifest": final_capture_site_manifest.to_json(),
        "custom_call_handler_executions": call_counts,
        "custom_call_handler_count_contract": {
            "logical_execution_minimum": minimum_calls,
            "capture_only_targets": compatible_targets,
            "capture_only_minimum": 1 if capture_only_calls else None,
            "note": (
                "Command-buffer-compatible host handlers run while XLA records a graph; graph replay does not "
                "increment their host counters. Target occurrence and profiler evidence replace logical call-count "
                "accounting for those handlers."
                if capture_only_calls
                else "Every selected handler counter covers logical benchmark executions."
            ),
        },
        "input_adjoint_auxiliary": input_adjoint_auxiliary,
        "retained_input_adjoint_wrappers": retained_input_adjoint_wrappers,
        "target_instruction_names": (
            *routed_audit.target_instructions,
            *weighted_relation_target_instructions,
            *normalized_exp_target_instructions,
            *low_rank_contract_map_target_instructions,
            attention_instruction,
            *(axis_fold.call_instruction for axis_fold in axis_fold_audits),
        ),
        "copy_count": dict(zip(("original", "transformed"), routed_audit.copy_count, strict=True)),
        "transpose_count": dict(
            zip(
                ("original", "transformed"),
                routed_audit.transpose_count,
                strict=True,
            )
        ),
        "attention_dead_reverse_closure": attention_liveness.dead_reverse_closure,
        "attention_preserved_shared_users": dict(attention_liveness.preserved_shared_users),
        "generated_semantic_sha256": {
            "forward": forward.semantic_digest,
            mode_target_key: mode_generated.semantic_digest,
            **(
                {
                    "input_contracts": [contract.semantic_digest for contract in input_contracts],
                    "source_fold": source_fold.semantic_digest if source_fold is not None else None,
                }
                if composition_mode.uses_shared_map
                else {}
            ),
            "weight_gradients": [weight.semantic_digest for weight in weights],
            "attention_backward": compiled_attention.generated.semantic_fingerprint,
            "axis_folds": [axis_fold.semantic_fingerprints for axis_fold in generated_axis_folds],
            **weighted_relation_semantic_hashes,
            **normalized_exp_semantic_hashes,
            **low_rank_contract_map_semantic_hashes,
        },
        "generated_source_sha256": {
            "forward": forward.source_digest,
            mode_target_key: mode_generated.source_digest,
            **(
                {
                    "input_contracts": [contract.source_digest for contract in input_contracts],
                    "source_fold": source_fold.source_digest if source_fold is not None else None,
                }
                if composition_mode.uses_shared_map
                else {}
            ),
            "weight_gradients": [weight.source_digest for weight in weights],
            "attention_backward": attention_source_sha256,
            "axis_folds": [axis_fold.source_sha256 for axis_fold in generated_axis_folds],
            **weighted_relation_source_hashes,
            **normalized_exp_source_hashes,
            **low_rank_contract_map_source_hashes,
        },
        "baseline_median_ms": baseline_median,
        "generated_median_ms": transformed_median,
        "generated_over_baseline": transformed_median / baseline_median,
        "generated_minus_baseline_ms": transformed_median - baseline_median,
        "independent_custom_call_count": composition_mode.independent_custom_call_count,
        "command_buffer_candidate": command_buffer_candidate.value,
        "command_buffer_compatible_targets": compatible_targets,
        "xla_command_buffer_startup_selection": (
            {
                "uses_xla_default": command_buffer_flag_audit.uses_xla_default,
                "selected_entries": command_buffer_flag_audit.selected_entries,
            }
            if command_buffer_flag_audit is not None
            else None
        ),
        "latency_delta_per_custom_call_us": (
            (transformed_median - baseline_median) * 1e3 / composition_mode.independent_custom_call_count
        ),
        "latency_delta_attribution": (
            "Whole-step delta includes fixed typed-FFI dispatches and changed kernels; the per-call quotient is "
            "reported only to expose the fixed-call scale, not as causal attribution."
        ),
        "baseline_unique_output_hashes": sorted(set(baseline_hashes)),
        "generated_unique_output_hashes": sorted(set(transformed_hashes)),
        "generated_runtime_dependencies": runtime_dependencies,
        "generated_runtime_torch_triton_free": True,
        "attention_aot_build_dependencies": attention_aot_build_dependencies,
        "raw_samples": raw_samples,
        **comparison,
        "outputs_match": True,
        "remaining_ownership_gap": (
            "The relation index plane, placement collectives, and harmless input-adjoint view wrappers remain "
            "under XLA. Shuttle owns both input-adjoint Contracts, the input source Fold, and the weighted "
            "RelationProgram reverse through its source-slot Fold; the selected composition also owns the generic "
            "normalized-exp Contract/Map/Fold pair when requested, plus six forward/rematerialization and four "
            "JAX-owned reverse low-rank Contract/Map chains when requested. Router normalization remains under XLA."
            if composition_mode.uses_shared_map
            else "The rematerialized routed forward chain, relation index plane, and placement collectives remain "
            "under XLA; the monolithic generated input-adjoint region owns the overlapping reverse path."
        ),
    }


def _call_count(library: ctypes.CDLL, symbol: str) -> int:
    function = getattr(library, symbol)
    function.restype = ctypes.c_int
    return int(function())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--triton-target")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--composition-mode",
        type=RoutedTrainingCompositionMode,
        choices=tuple(RoutedTrainingCompositionMode),
        default=RoutedTrainingCompositionMode.SHARED_MAP_XLA_REMAINDER,
    )
    parser.add_argument(
        "--command-buffer-candidate",
        type=CommandBufferCandidateMode,
        choices=tuple(CommandBufferCandidateMode),
        default=CommandBufferCandidateMode.DISABLED,
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dependency-preflight-only", action="store_true")
    args = parser.parse_args()
    if args.dependency_preflight_only:
        print(json.dumps(dependency_preflight(), indent=2, sort_keys=True))
        return
    result = run_smoke(
        args.nvcc,
        args.architecture,
        args.artifact_directory,
        composition_mode=args.composition_mode,
        command_buffer_candidate=args.command_buffer_candidate,
        repository=args.repository,
        triton_target=args.triton_target,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
