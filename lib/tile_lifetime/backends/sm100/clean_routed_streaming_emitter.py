# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clean source extraction for Shuttle's first SM100 routed streaming emitter.

MiniMax Sparse Attention supplies the initial tcgen05/TMA/layout/pipeline
lineage. The accepted source produced here does not import MSA's softmax, mask,
public sparse-attention interface, or semantic combine. The extraction replaces
the first two with Shuttle-owned generic Fold and DomainRestriction classes and
emits a separate deterministic partial-state merge kernel.

The companion runtime instantiates the emitted physical class directly. It
does not import MSA's public attention interface or semantic combine.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import linecache
import math
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from shuttle.ir import DType
from tile_lifetime.event_dataflow_adapters import sm100_routed_right_resource_descriptor
from tile_lifetime.ffi_command_buffer import finalize_ffi_handler_source
from tile_lifetime.right_resource_event_schedule import (
    RightResourceFoldEventSchedule,
    derive_right_resource_fold_event_schedule,
)
from tile_lifetime.sm100_routed_lowering import (
    SM100RelationOrientation,
    SM100RoutedStreamingLowering,
)
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind, TensorAxis
from tile_lifetime.tiled_fold_finalize import (
    FoldDenominatorPolicy,
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPhysicalAxis,
    FoldReassociationPolicy,
    FoldScalarReduction,
    TiledFoldAxes,
    TiledFoldFinalizeProgram,
    TiledFoldFinalizeSchedule,
    TiledFoldInputLayout,
    normalized_exponential_fold_program,
)

MINIMAX_MSA_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
MINIMAX_MSA_CUTLASS_COMMIT = "eb61c911471867a5fd2466bfd8f29306cea6ebf8"
MINIMAX_MSA_CUTLASS_PATH = "python/fmha_sm100/cutlass"
MINIMAX_MSA_CUTE_ROOT = Path("python/fmha_sm100/cute")

PINNED_SOURCE_SHA256 = {
    "src/sm100/fwd/atten_fwd.py": "69b615bcbeaacd1fd87446870c3dbd5e65300549590f5477701b1cd51dc65510",
    "src/common/softmax.py": "d6756e56a74c638eaeae6ca840d34db2863f7d0836f84afbbce2b4a8015caaeb",
    "src/common/mask.py": "2fbef9d57a60398a65c11e5789225dc15a85d007633f0887c5ee8e3f11e85cb0",
    "src/sm100/prepare_k2q_csr.py": "429ab94135902eec524ec8a0db31857aa51b067c525330b2009412384601199f",
    "src/sm100/prepare_scheduler.py": "55e49e022a9439b21be5b33f9975413332375a6ce37836c6ad13e58d204470ac",
}

FORBIDDEN_IMPORT_PREFIXES = (
    "fmha_sm100",
    "interface",
    "sparse_fmha_adapter",
    "src.common.softmax",
    "src.common.mask",
    "src.sm100.fwd.combine",
)
FORBIDDEN_REFERENCED_NAMES = frozenset(
    {
        "AttentionMask",
        "SoftmaxSm100",
        "SparseAttentionForwardCombine",
        "sparse_atten_func",
    }
)
REQUIRED_PHYSICAL_TOKENS = frozenset(
    {
        "tcgen05",
        "PipelineTmaUmma",
        "PipelineUmmaAsync",
        "tma_gather4_cached",
        "make_tmem_copy",
    }
)
GENERATED_SEMANTICS_MODULE = "shuttle_sm100_generated_semantics"
GENERATED_RELATION_SCHEDULER_MODULE = "shuttle_sm100_generated_relation_scheduler"
GENERATED_RELATION_BUILDER_CLASS = "RightMajorRelationCsrBuilderSm100"
GENERATED_PHYSICAL_CLASS = "RoutedStreamingFoldContractSm100"


class DomainRestrictionKind(StrEnum):
    """Generic score-domain restriction supported by the first template."""

    UNRESTRICTED = "unrestricted"
    CAUSAL = "causal"


class PartialValueDType(StrEnum):
    """Storage policy for the normalized-value component of Fold partials."""

    BF16 = "bf16"
    FP32 = "fp32"


class PartialMergeScheduleKind(StrEnum):
    """Bounded physical schedules for a deterministic partial-state Fold."""

    ROW_BLOCK = "row_block"
    WARP_ROWS = "warp_rows"
    TILED_PIPELINED = "tiled_pipelined"


@dataclass(frozen=True)
class StaticLaunchGridAudit:
    """Source-level proof that a bounded launch grid is host-specialized."""

    capacity_annotation: str
    grid_expression: str
    runtime_work_count_forwarded: bool

    @property
    def clean(self) -> bool:
        """Return whether the physical ABI separates capacity from work count."""
        return (
            self.capacity_annotation == "cutlass.Constexpr[int]"
            and self.grid_expression in {"work_capacity", "num_ctas"}
            and self.runtime_work_count_forwarded
        )


def real_col_to_stg128_half_col(real_col: int) -> int:
    """Map one real BF16 feature column to the physical STG.128 column."""
    tile, col32 = divmod(real_col, 32)
    lane = (col32 % 8) // 2
    group = col32 // 8
    element = col32 % 2
    return tile * 32 + lane * 8 + group * 2 + element


def stg128_half_col_to_real_col(fake_col: int) -> int:
    """Invert the BF16 STG.128 feature-column permutation."""
    tile, fake32 = divmod(fake_col, 32)
    lane, lane_slot = divmod(fake32, 8)
    group, element = divmod(lane_slot, 2)
    return tile * 32 + group * 8 + lane * 2 + element


@dataclass(frozen=True)
class DomainRestrictionProgram:
    """Bounds and optional index predicate applied to one score tile."""

    kind: DomainRestrictionKind
    apply_sequence_bounds: bool

    @property
    def causal(self) -> bool:
        return self.kind is DomainRestrictionKind.CAUSAL

    def accepts(self, query_position: int, key_position: int) -> bool:
        """Evaluate the semantic predicate for a valid token pair."""
        if self.kind is DomainRestrictionKind.UNRESTRICTED:
            return True
        return key_position <= query_position


@dataclass(frozen=True)
class NormalizedExpFoldProgram:
    """Compiler-owned normalized-exponential state and update contract."""

    score_scale: float
    accumulator_dtype: str
    state_fields: tuple[str, str, str]
    exponential_base: int

    def update_numpy(
        self,
        row_max: np.ndarray,
        row_sum_exp: np.ndarray,
        weighted_value: np.ndarray,
        scores: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference one generic online Fold update."""
        scaled_scores = scores.astype(np.float32) * np.float32(self.score_scale)
        tile_max = np.max(scaled_scores, axis=-1)
        new_max = np.maximum(row_max, tile_max)
        old_scale = np.where(row_sum_exp > 0, np.exp(row_max - new_max), 0.0).astype(np.float32)
        probabilities = np.exp(scaled_scores - new_max[..., None]).astype(np.float32)
        new_sum = old_scale * row_sum_exp + probabilities.sum(axis=-1)
        new_weighted = old_scale[..., None] * weighted_value + np.einsum(
            "...k,...kd->...d", probabilities, values.astype(np.float32)
        )
        return new_max.astype(np.float32), new_sum.astype(np.float32), new_weighted.astype(np.float32)


@dataclass(frozen=True)
class PartialStateMergeProgram:
    """Deterministic merge and finalization of disjoint Fold partials."""

    representation: str
    output_scale: float
    value_dtype: PartialValueDType
    threads: int = 256
    schedule_kind: PartialMergeScheduleKind = PartialMergeScheduleKind.ROW_BLOCK
    rows_per_block: int = 1
    accumulator_dtype: str = "fp32"
    source_ordered: bool = True
    partial_extent: int = 16
    feature_tile: int = 64
    pipeline_stages: int = 4
    pipeline_buffers: int = 2
    vector_bytes: int = 16
    generic_program: TiledFoldFinalizeProgram | None = None

    def merge_numpy(self, scalar_state: np.ndarray, value_state: np.ndarray) -> np.ndarray:
        """Merge partial states in ascending partial-slot order."""
        if scalar_state.ndim != 2 or value_state.ndim != 3:
            raise ValueError("partial scalar state must be [P,R] and values must be [P,R,D]")
        if scalar_state.shape != value_state.shape[:2]:
            raise ValueError("partial scalar and value state domains must match")
        if self.representation == "log_normalizer_normalized_value":
            common = np.max(scalar_state.astype(np.float32), axis=0)
            safe_common = np.where(np.isfinite(common), common, 0.0)
            weight = np.where(
                np.isfinite(scalar_state),
                np.exp(scalar_state.astype(np.float32) - safe_common[None, :]),
                0.0,
            )
            denominator = weight.sum(axis=0)
            if np.any(denominator <= 0):
                raise ValueError("partial normalized-exp Fold has an empty row")
            merged = np.sum(weight[..., None] * value_state.astype(np.float32), axis=0)
            return (merged / denominator[:, None] * np.float32(self.output_scale)).astype(np.float32)
        raise ValueError(f"reference merge does not support {self.representation!r}")


def tiled_fold_merge_program(
    generic_program: TiledFoldFinalizeProgram,
    *,
    output_scale: float = 1.0,
) -> PartialStateMergeProgram:
    """Bind generic Fold semantics to the shared-staged SM100 finalizer."""
    if generic_program.semantics.partial_value_dtype is not DType.BF16:
        raise ValueError("the first SM100 tiled Fold binding requires BF16 value partials")
    schedule = generic_program.schedule
    return PartialStateMergeProgram(
        representation="generic_tiled_fold",
        output_scale=output_scale,
        value_dtype=PartialValueDType.BF16,
        threads=schedule.threads,
        schedule_kind=PartialMergeScheduleKind.TILED_PIPELINED,
        rows_per_block=schedule.row_tile,
        source_ordered=(generic_program.semantics.reassociation is FoldReassociationPolicy.SOURCE_ORDERED),
        partial_extent=schedule.axes.partial.extent,
        feature_tile=schedule.feature_tile,
        pipeline_stages=schedule.shared_stages,
        pipeline_buffers=schedule.shared_buffers,
        vector_bytes=schedule.vector_bytes,
        generic_program=generic_program,
    )


@dataclass(frozen=True)
class SM100EmitterPlan:
    """Generic semantics and physical constructor arguments for one lowering."""

    domain_restriction: DomainRestrictionProgram
    normalized_exp_fold: NormalizedExpFoldProgram
    partial_merge: PartialStateMergeProgram
    partial_value_dtype: PartialValueDType
    relation_encoding: SM100RelationEncoding
    event_schedule: RightResourceFoldEventSchedule
    physical_class: str
    physical_constructor: dict[str, Any]
    external_semantic_kernels: tuple[str, ...]


@dataclass(frozen=True)
class SM100RelationEncoding:
    """Map flattened generic relation edges onto MSA-lineage q2k metadata."""

    source_domain: str
    destination_domain: str
    key_value_heads: int
    selected_count: int

    def edge_group(self, route_slot: int) -> int:
        """Decode the GQA group from one flattened relation slot."""
        return route_slot // self.selected_count

    def edge_selected_slot(self, route_slot: int) -> int:
        """Decode the within-group selected-block slot."""
        return route_slot % self.selected_count

    def right_task_key(self, route_slot: int, key_value_block: int) -> tuple[int, int]:
        """Return the KV-major grouping key for one relation edge."""
        return self.edge_group(route_slot), key_value_block


@dataclass(frozen=True)
class SourceAudit:
    """Static imports, references, and local call edges for emitted Python."""

    imported_modules: tuple[str, ...]
    referenced_names: tuple[str, ...]
    local_call_edges: tuple[tuple[str, str], ...]
    forbidden_dependencies: tuple[str, ...]
    required_physical_tokens: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.forbidden_dependencies


@dataclass(frozen=True)
class ExtractedSM100Sources:
    """Pinned physical extraction plus compiler-owned semantic bodies."""

    emitter_plan: SM100EmitterPlan
    physical_source: str
    semantic_source: str
    relation_builder_source: str
    scheduler_source: str
    merge_cuda_source: str
    upstream_source_sha256: dict[str, str]
    generated_source_sha256: dict[str, str]
    physical_audit: SourceAudit
    semantic_audit: SourceAudit
    relation_builder_audit: SourceAudit
    scheduler_audit: SourceAudit
    lineage: dict[str, Any]


@dataclass(frozen=True)
class GeneratedPartialMergeFfi:
    """Torch-free typed-FFI binding for one generated Fold finalizer."""

    target: str
    handler_symbol: str
    source: str
    source_sha256: str
    partial_count: int
    query_count: int
    query_heads: int
    key_value_heads: int
    value_width: int
    value_dtype: PartialValueDType


def emitter_plan_from_lowering(
    lowering: SM100RoutedStreamingLowering,
    *,
    paged_key_value: bool,
    partial_value_dtype: PartialValueDType = PartialValueDType.BF16,
    partial_merge_schedule: PartialMergeScheduleKind = PartialMergeScheduleKind.ROW_BLOCK,
) -> SM100EmitterPlan:
    """Erase routed-attention names into generic Fold/Contract physical inputs."""
    if lowering.schedule.orientation is not SM100RelationOrientation.RIGHT_MAJOR:
        raise ValueError("the first extracted SM100 template implements only right-major relation traversal")
    if lowering.schedule.partial_state_representation != "log_normalizer_normalized_value":
        raise ValueError("the first compiler-owned SM100 merge accepts normalized-value partial state")
    if lowering.score_map.softcap is not None:
        raise ValueError("the first extracted SM100 template does not generate a score softcap")

    restriction_kind = DomainRestrictionKind.CAUSAL if lowering.score_map.causal else DomainRestrictionKind.UNRESTRICTED
    restriction = DomainRestrictionProgram(kind=restriction_kind, apply_sequence_bounds=True)
    fold = NormalizedExpFoldProgram(
        score_scale=lowering.score_map.scale,
        accumulator_dtype="fp32",
        state_fields=("row_max", "row_sum_exp", "weighted_value_accumulator"),
        exponential_base=2,
    )
    generic_program = None
    if partial_merge_schedule is PartialMergeScheduleKind.TILED_PIPELINED:
        axes = TiledFoldAxes(
            partial=TensorAxis(0, lowering.selected_count, "partial"),
            row=TensorAxis(
                1,
                lowering.query_length * lowering.key_value_heads * lowering.head_group_size,
                "row",
            ),
            feature=TensorAxis(2, 128, "feature"),
        )
        generic_program = normalized_exponential_fold_program(
            TiledFoldFinalizeSchedule(
                axes=axes,
                partial_addressing=FoldPartialAddressing.DENSE,
                row_tile=lowering.schedule.partial_merge_tile_rows,
                feature_tile=128,
                vector_bytes=16,
                shared_stages=4,
                threads=lowering.schedule.partial_merge_threads,
                partial_lanes=32,
                shared_buffers=2,
                input_layout=TiledFoldInputLayout(
                    addressing=FoldPartialAddressing.DENSE,
                    value_axis_order=(
                        FoldPhysicalAxis.PARTIAL,
                        FoldPhysicalAxis.ROW,
                        FoldPhysicalAxis.FEATURE,
                    ),
                    scalar_axis_order=(FoldPhysicalAxis.PARTIAL, FoldPhysicalAxis.ROW),
                    feature_layout=FoldFeatureLayout.STG128_LANE_PERMUTED,
                ),
            ),
            partial_value_dtype=DType.BF16,
            output_dtype=DType.BF16,
        )
    if generic_program is not None:
        merge = tiled_fold_merge_program(generic_program, output_scale=lowering.output_scale)
    else:
        merge = PartialStateMergeProgram(
            representation=lowering.schedule.partial_state_representation,
            output_scale=lowering.output_scale,
            value_dtype=partial_value_dtype,
            threads=lowering.schedule.partial_merge_threads,
            schedule_kind=partial_merge_schedule,
            rows_per_block=(
                lowering.schedule.partial_merge_tile_rows
                if partial_merge_schedule is PartialMergeScheduleKind.WARP_ROWS
                else 1
            ),
        )
    relation_encoding = SM100RelationEncoding(
        source_domain="query_token",
        destination_domain="key_value_block",
        key_value_heads=lowering.key_value_heads,
        selected_count=lowering.selected_count,
    )
    event_schedule = derive_right_resource_fold_event_schedule(
        lowering.relation,
        sm100_routed_right_resource_descriptor(lowering),
    )
    return SM100EmitterPlan(
        domain_restriction=restriction,
        normalized_exp_fold=fold,
        partial_merge=merge,
        partial_value_dtype=partial_value_dtype,
        relation_encoding=relation_encoding,
        event_schedule=event_schedule,
        physical_class=GENERATED_PHYSICAL_CLASS,
        physical_constructor={
            "head_dim": 128,
            "qheadperkv": lowering.head_group_size,
            "m_block_size": lowering.schedule.packed_left_rows,
            "n_block_size": lowering.schedule.right_block_size,
            "paged_kv": paged_key_value,
            "page_size": lowering.schedule.right_block_size if paged_key_value else None,
            "has_seqused_k": paged_key_value,
            "causal": restriction.causal,
            "use_prepare_scheduler": True,
        },
        external_semantic_kernels=(),
    )


def q2k_indices_from_lowering(lowering: SM100RoutedStreamingLowering) -> np.ndarray:
    """Encode the generic relation as ``[H_kv, query_token, selected]`` q2k."""
    encoding = SM100RelationEncoding(
        source_domain="query_token",
        destination_domain="key_value_block",
        key_value_heads=lowering.key_value_heads,
        selected_count=lowering.selected_count,
    )
    q2k = np.full(
        (lowering.key_value_heads, lowering.query_length, lowering.selected_count),
        -1,
        dtype=np.int32,
    )
    relation = lowering.relation
    valid_edges = np.flatnonzero(relation.edge_valid.reshape(-1))
    for edge in valid_edges:
        source = int(relation.source_item[edge])
        route_slot = int(relation.route_slot[edge])
        group = encoding.edge_group(route_slot)
        selected_slot = encoding.edge_selected_slot(route_slot)
        destination = int(relation.destination_item[edge])
        if not 0 <= group < lowering.key_value_heads:
            raise ValueError(f"relation route slot {route_slot} decodes to invalid GQA group {group}")
        if q2k[group, source, selected_slot] != -1:
            raise ValueError("multiple relation edges occupy one q2k group/query/selected slot")
        q2k[group, source, selected_slot] = destination
    return q2k


def _sha256(source: str | bytes) -> str:
    payload = source.encode() if isinstance(source, str) else source
    return hashlib.sha256(payload).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _validate_msa_checkout(msa_root: Path) -> dict[str, Any]:
    revision = _git_output(msa_root, "rev-parse", "HEAD")
    if revision != MINIMAX_MSA_COMMIT:
        raise ValueError(f"MSA checkout is {revision}; expected {MINIMAX_MSA_COMMIT}")
    cutlass_revision = _git_output(msa_root, "rev-parse", f"HEAD:{MINIMAX_MSA_CUTLASS_PATH}")
    if cutlass_revision != MINIMAX_MSA_CUTLASS_COMMIT:
        raise ValueError(f"MSA CUTLASS gitlink is {cutlass_revision}; expected {MINIMAX_MSA_CUTLASS_COMMIT}")
    modifications = _git_output(msa_root, "status", "--short").splitlines()
    return {
        "repository": "https://github.com/MiniMax-AI/MSA",
        "commit": revision,
        "cutlass_commit": cutlass_revision,
        "local_modifications": modifications,
    }


def _read_pinned_sources(msa_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_root = msa_root / MINIMAX_MSA_CUTE_ROOT
    sources = {}
    hashes = {}
    for relative, expected_sha256 in PINNED_SOURCE_SHA256.items():
        source = (source_root / relative).read_text()
        observed_sha256 = _sha256(source)
        if observed_sha256 != expected_sha256:
            raise ValueError(f"MSA source {relative} is {observed_sha256}; expected {expected_sha256}")
        sources[relative] = source
        hashes[relative] = observed_sha256
    return sources, hashes


def _replace_semantic_imports(source: str) -> str:
    source = source.replace(
        "from src.common.softmax import SoftmaxSm100",
        f"from {GENERATED_SEMANTICS_MODULE} import NormalizedExpFoldSm100",
    )
    source = source.replace(
        "from src.common.mask import AttentionMask",
        f"from {GENERATED_SEMANTICS_MODULE} import DomainRestrictionSm100",
    )
    source = source.replace("SoftmaxSm100", "NormalizedExpFoldSm100")
    source = source.replace("AttentionMask", "DomainRestrictionSm100")
    source = source.replace("SparseAttentionForwardSm100", GENERATED_PHYSICAL_CLASS)
    return specialize_static_launch_grid(source)


def specialize_static_launch_grid(source: str) -> str:
    """Make the bounded maximum task count a CuTe compile-time parameter."""
    dynamic_signature = "        work_capacity: Int32,\n        stream=None,"
    static_signature = "        work_capacity: cutlass.Constexpr[int],\n        stream=None,"
    if source.count(dynamic_signature) != 1:
        raise ValueError("expected one host launch-capacity parameter in the pinned physical source")
    specialized = source.replace(dynamic_signature, static_signature)
    audit = audit_static_launch_grid(specialized)
    if not audit.clean:
        raise ValueError(f"physical launch-grid specialization is incomplete: {audit}")
    return specialized


def audit_static_launch_grid(source: str) -> StaticLaunchGridAudit:
    """Audit host capacity specialization and runtime work-count forwarding."""
    tree = ast.parse(source)
    physical_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == GENERATED_PHYSICAL_CLASS),
        None,
    )
    if physical_class is None:
        return StaticLaunchGridAudit("", "", False)
    host_call = next(
        (
            node
            for node in physical_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__"
        ),
        None,
    )
    if host_call is None:
        return StaticLaunchGridAudit("", "", False)
    capacity_argument = next(
        (argument for argument in host_call.args.args if argument.arg == "work_capacity"),
        None,
    )
    annotation = (
        ""
        if capacity_argument is None or capacity_argument.annotation is None
        else ast.unparse(capacity_argument.annotation)
    )
    aliases = {"work_capacity"}
    for node in ast.walk(host_call):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in aliases:
            continue
        aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    grid_expression = ""
    runtime_work_count_forwarded = False
    for node in ast.walk(host_call):
        if not isinstance(node, ast.Call):
            continue
        runtime_work_count_forwarded |= any(
            isinstance(argument, ast.Name) and argument.id == "mWorkCount" for argument in node.args
        )
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "launch":
            continue
        grid = next((keyword.value for keyword in node.keywords if keyword.arg == "grid"), None)
        if isinstance(grid, ast.Tuple) and len(grid.elts) == 1 and isinstance(grid.elts[0], ast.Name):
            grid_expression = grid.elts[0].id
    if grid_expression not in aliases:
        grid_expression = ""
    return StaticLaunchGridAudit(annotation, grid_expression, runtime_work_count_forwarded)


def _replace_fold_names(source: str) -> str:
    source = source.replace("from . import utils", "from src.common import utils")
    source = source.replace("SoftmaxSm100", "NormalizedExpFoldSm100")
    source = source.replace("Softmax", "NormalizedExpFold")
    source = source.replace("softmax", "normalized_exp_fold")
    return source


def _replace_domain_names(source: str) -> str:
    source = source.replace("AttentionMask", "DomainRestrictionSm100")
    source = source.replace("MaskGenFn", "RestrictionMaskGenFn")
    return source


def _render_semantic_source(softmax_source: str, mask_source: str) -> str:
    attribution = '''"""Shuttle generic SM100 Fold and DomainRestriction primitives.

Derived from MiniMax MSA 80434d7 under the MIT license. The original copyright
and SPDX notices are preserved immediately below in each extracted section.
"""\n\n'''
    return attribution + _replace_fold_names(softmax_source) + "\n\n" + _replace_domain_names(mask_source)


def render_relation_scheduler_source(source: str) -> str:
    """Erase attention names from the generic right-major worklist helper."""
    replacements = (
        ("SparseAttentionPrepareFwdSplitAtomicSm100", "RightMajorRelationPrepareSplitOwnershipSm100"),
        ("SparseAttentionPrepareFlatScheduleSm100", "RightMajorRelationPrepareFlatScheduleSm100"),
        ("SparseAttentionScheduleModel", "RightMajorRelationScheduleModel"),
        ("SparseAttentionSchedule", "RightMajorRelationSchedule"),
        ("SparseSchedulePlan", "RightMajorRelationSchedulePlan"),
        ("SPARSE_SCHEDULE_MODEL", "RIGHT_MAJOR_RELATION_SCHEDULE_MODEL"),
        ("prepare_sparse_fwd_schedule_and_split", "prepare_right_major_schedule_and_split"),
        ("prepare_sparse_fwd_schedule", "prepare_right_major_schedule"),
        ("prepare_sparse_flat_schedule", "prepare_right_major_flat_schedule"),
        ("_get_sparse_prepare_fwd_split_atomic", "_get_right_major_prepare_split_ownership"),
        ("_get_sparse_prepare_flat_schedule", "_get_right_major_prepare_flat_schedule"),
        ("_decode_sparse_row_linear", "_decode_right_major_row_linear"),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    source = source.replace("SparseAttention_", "ShuttleRelation_")
    source = source.replace("sparse attention", "right-major relation")
    source = source.replace("Sparse attention", "Right-major relation")
    return source


def render_relation_builder_source(source: str) -> str:
    """Expose MSA's generic CSR mechanism through RelationPlan terminology."""
    source = source.replace(
        "from src.sm100.prepare_scheduler import SparseAttentionSchedule, SPARSE_SCHEDULE_MODEL",
        (
            f"from {GENERATED_RELATION_SCHEDULER_MODULE} import (\n"
            "    RIGHT_MAJOR_RELATION_SCHEDULE_MODEL,\n"
            "    RightMajorRelationSchedule,\n"
            ")"
        ),
    )
    physical_builder_class = "_PhysicalRightMajorRelationCsrBuilderSm100"
    source = source.replace("SparseK2qCsrBuilderSm100", physical_builder_class)
    source = source.replace("SparseAttentionSchedule", "RightMajorRelationSchedule")
    source = source.replace("SPARSE_SCHEDULE_MODEL", "RIGHT_MAJOR_RELATION_SCHEDULE_MODEL")
    source = source.replace(
        "run_build_k2q_csr_with_schedule,",
        "run_build_k2q_csr_with_schedule as run_build_right_major_relation_with_schedule,",
    )
    source = source.replace(
        "run_build_k2q_csr,",
        "run_build_k2q_csr as run_build_right_major_relation,",
    )
    source = source.replace("self._run = run_build_k2q_csr", "self._run = run_build_right_major_relation")
    source = source.replace(
        "self._run_with_schedule = run_build_k2q_csr_with_schedule",
        "self._run_with_schedule = run_build_right_major_relation_with_schedule",
    )
    source = source.replace("Sparse k2q CSR", "Right-major relation CSR")
    source = source.replace("sparse attention", "right-major relation")
    source = source.replace("SparseK2qCsr_Pipeline", "ShuttleRightMajorRelationCsr")
    return (
        source
        + f'''\n\n
class {GENERATED_RELATION_BUILDER_CLASS}:
    """Build a grouped right-major physical plan from left-to-right edges."""

    def __init__(self) -> None:
        self._physical = {physical_builder_class}()

    def __call__(
        self,
        left_to_right_indices: torch.Tensor,
        left_offsets: torch.Tensor,
        right_payload_offsets: torch.Tensor,
        *,
        right_payload_extent: int,
        right_item_width: int = 128,
        maximum_right_payload_extent: Optional[int] = None,
        maximum_left_item_count: Optional[int] = None,
        right_item_count: Optional[int] = None,
        left_lanes_per_group: int = 1,
        return_schedule: bool = False,
    ):
        return self._physical(
            left_to_right_indices,
            left_offsets,
            right_payload_offsets,
            total_k=right_payload_extent,
            blk_kv=right_item_width,
            max_seqlen_k=maximum_right_payload_extent,
            max_seqlen_q=maximum_left_item_count,
            total_rows=right_item_count,
            qhead_per_kv=left_lanes_per_group,
            return_schedule=return_schedule,
        )
'''
    )


def _called_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def audit_python_source(source: str) -> SourceAudit:
    """Reject forbidden semantic dependencies and expose a local call graph."""
    tree = ast.parse(source)
    imported_modules = set()
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    definitions: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
    local_calls = set()
    for caller, node in definitions.items():
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            callee = _called_name(call)
            if callee in definitions:
                local_calls.add((caller, callee))

    forbidden = {
        module
        for module in imported_modules
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    }
    forbidden.update(FORBIDDEN_REFERENCED_NAMES & referenced_names)
    required = tuple(sorted(token for token in REQUIRED_PHYSICAL_TOKENS if token in source))
    return SourceAudit(
        imported_modules=tuple(sorted(imported_modules)),
        referenced_names=tuple(sorted(referenced_names)),
        local_call_edges=tuple(sorted(local_calls)),
        forbidden_dependencies=tuple(sorted(forbidden)),
        required_physical_tokens=required,
    )


def _require_clean_physical_source(source: str) -> SourceAudit:
    audit = audit_python_source(source)
    if not audit.clean:
        raise ValueError(f"SM100 physical extraction retains forbidden semantics: {audit.forbidden_dependencies}")
    missing = REQUIRED_PHYSICAL_TOKENS - set(audit.required_physical_tokens)
    if missing:
        raise ValueError(f"SM100 physical extraction lost required mechanisms: {sorted(missing)}")
    return audit


def _render_cuda_scalar_expression(
    expression: ScalarExpression,
    inputs: dict[str, str],
    *,
    ordered_fp: bool = False,
) -> str:
    """Render one backend-neutral scalar AST inside the tiled Fold skeleton."""
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return inputs[expression.input_name]
    if expression.kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        if isinstance(expression.constant, bool):
            return "true" if expression.constant else "false"
        return f"{float(expression.constant):.17e}f"
    operands = tuple(
        _render_cuda_scalar_expression(operand, inputs, ordered_fp=ordered_fp) for operand in expression.operands
    )
    infix = {
        ScalarExpressionKind.ADD: "+",
        ScalarExpressionKind.SUBTRACT: "-",
        ScalarExpressionKind.MULTIPLY: "*",
        ScalarExpressionKind.DIVIDE: "/",
        ScalarExpressionKind.LESS_EQUAL: "<=",
    }
    if ordered_fp and expression.kind is ScalarExpressionKind.ADD:
        return f"__fadd_rn({operands[0]}, {operands[1]})"
    if ordered_fp and expression.kind is ScalarExpressionKind.MULTIPLY:
        return f"__fmul_rn({operands[0]}, {operands[1]})"
    if expression.kind in infix:
        return f"({operands[0]} {infix[expression.kind]} {operands[1]})"
    if expression.kind is ScalarExpressionKind.EXP:
        return f"expf({operands[0]})"
    if expression.kind is ScalarExpressionKind.RSQRT:
        return f"rsqrtf({operands[0]})"
    if expression.kind is ScalarExpressionKind.TANH:
        return f"tanhf({operands[0]})"
    if expression.kind is ScalarExpressionKind.SELECT:
        return f"({operands[0]} ? {operands[1]} : {operands[2]})"
    raise ValueError(f"unsupported tiled Fold scalar expression {expression.kind.value}")


def _render_tiled_pipelined_merge_cuda(program: PartialStateMergeProgram) -> str:
    """Generate the first tiled, shared-staged Fold finalization skeleton."""
    generic = program.generic_program
    if generic is None:
        raise ValueError("the tiled physical schedule requires generic Fold finalization semantics")
    schedule = generic.schedule
    semantics = generic.semantics
    ordered_fp = semantics.reassociation is FoldReassociationPolicy.SOURCE_ORDERED
    if schedule.axes.partial.extent != program.partial_extent:
        raise ValueError("generic Fold partial extent does not match the physical schedule")
    if schedule.row_tile != program.rows_per_block:
        raise ValueError("generic Fold row tile does not match the physical schedule")
    if schedule.feature_tile != program.feature_tile:
        raise ValueError("generic Fold feature tile does not match the physical schedule")
    if schedule.shared_stages != program.pipeline_stages:
        raise ValueError("generic Fold shared stages do not match the physical schedule")
    if schedule.shared_buffers != program.pipeline_buffers:
        raise ValueError("generic Fold shared buffers do not match the physical schedule")
    if schedule.vector_bytes != program.vector_bytes:
        raise ValueError("generic Fold vector width does not match the physical schedule")
    if program.value_dtype is not PartialValueDType.BF16:
        raise ValueError("the first tiled Fold finalizer supports BF16 value partials")
    if program.partial_extent > 32:
        raise ValueError("the first tiled Fold finalizer supports at most one warp of partials")
    if program.rows_per_block != program.threads // 32:
        raise ValueError("the tiled Fold finalizer assigns one logical row to each warp")
    if program.feature_tile not in (64, 128):
        raise ValueError("the tiled Fold finalizer requires a 64- or 128-feature tile")
    if program.pipeline_stages != 4:
        raise ValueError("the first tiled Fold finalizer requires four shared-memory stages")
    if program.pipeline_buffers not in (1, 2):
        raise ValueError("the first tiled Fold finalizer requires one or two shared-memory buffers")
    if program.vector_bytes != 16:
        raise ValueError("the first tiled Fold finalizer requires 128-bit global-to-shared copies")
    features_per_lane = program.feature_tile // 32
    if features_per_lane % 2:
        raise ValueError("the Fold feature tile must assign an even number of BF16 values to each lane")
    input_layout = schedule.input_layout
    assert input_layout is not None
    if input_layout.feature_layout is FoldFeatureLayout.STG128_LANE_PERMUTED:
        shared_features = tuple(
            f"real_col_to_stg128_half_col(real_feature{element}) - feature_base" for element in range(features_per_lane)
        )
    elif input_layout.feature_layout is FoldFeatureLayout.CONTIGUOUS:
        shared_features = tuple(f"real_feature{element} - feature_base" for element in range(features_per_lane))
    else:
        raise ValueError(f"unsupported Fold feature layout {input_layout.feature_layout.value}")
    if input_layout.addressing is FoldPartialAddressing.DENSE:
        row_mapping = """
  const int query = row / query_heads;
  const int query_head = row - query * query_heads;
  const int key_value_head = query_head / query_heads_per_key_value_head;
  const int scheduled_partials =
      partial_metadata[query * (query_heads / query_heads_per_key_value_head) + key_value_head];
  const int valid_partials = min(scheduled_partials, kPartialExtent);
""".rstrip()
        lane_validity = "lane < valid_partials"
        scalar_index = "lane * row_count + row"
        stage_validity = "partial < valid_partials"
        source_row = "static_cast<int64_t>(partial) * row_count + row"
        wrapper_checks = """
  TORCH_CHECK(partial_scalar.dim() == 3, "scalar state must be [P,Q,H]");
  TORCH_CHECK(partial_value.dim() == 4, "value state must be [P,Q,H,D]");
  TORCH_CHECK(partial_metadata.dim() == 2, "valid partial counts must be [Q,G]");
  TORCH_CHECK(partial_scalar.size(0) == kPartialExtent,
              "scalar state partial extent does not match generated schedule");
  TORCH_CHECK(partial_value.size(0) == kPartialExtent,
              "value state partial extent does not match generated schedule");
  TORCH_CHECK(partial_scalar.size(1) == partial_value.size(1), "row counts must match");
  TORCH_CHECK(partial_scalar.size(2) == partial_value.size(2), "row groups must match");
  TORCH_CHECK(partial_metadata.size(0) == partial_value.size(1),
              "valid-count outer domain must match");
  TORCH_CHECK(query_heads_per_key_value_head > 0, "group width must be positive");
  TORCH_CHECK(partial_value.size(2) % query_heads_per_key_value_head == 0,
              "row groups must divide by the count-group width");
  TORCH_CHECK(partial_metadata.size(1) ==
                  partial_value.size(2) / query_heads_per_key_value_head,
              "valid-count group domain must match");
""".rstrip()
        wrapper_shape = """
  const int64_t query_count = partial_value.size(1);
  const int64_t query_heads = partial_value.size(2);
  const int64_t value_width = partial_value.size(3);
  const int64_t row_count = query_count * query_heads;
""".rstrip()
        output_checks = """
  TORCH_CHECK(output.dim() == 3, "dense Fold output must be [Q,H,D]");
  TORCH_CHECK(output.size(0) == query_count && output.size(1) == query_heads &&
                  output.size(2) == value_width,
              "dense Fold output shape must match the partial-value row/feature domain");
""".rstrip()
        output_allocation = """
  auto output = torch::empty(
      {partial_value.size(1), partial_value.size(2), partial_value.size(3)},
      partial_value.options().dtype(torch::kBFloat16));
""".rstrip()
    elif input_layout.addressing is FoldPartialAddressing.INDEXED:
        row_mapping = ""
        lane_validity = "lane < kPartialExtent && partial_metadata[row * kPartialExtent + lane] >= 0"
        scalar_index = "row * kPartialExtent + lane"
        stage_validity = "partial < kPartialExtent && partial_metadata[row * kPartialExtent + partial] >= 0"
        source_row = "partial_metadata[row * kPartialExtent + partial]"
        wrapper_checks = """
  TORCH_CHECK(partial_scalar.dim() == 2, "indexed scalar state must be [R,P]");
  TORCH_CHECK(partial_value.dim() == 2, "indexed source values must be [S,D]");
  TORCH_CHECK(partial_metadata.dim() == 2, "source-row indices must be [R,P]");
  TORCH_CHECK(partial_scalar.size(0) == partial_metadata.size(0), "row counts must match");
  TORCH_CHECK(partial_scalar.size(1) == kPartialExtent,
              "scalar state partial extent does not match generated schedule");
  TORCH_CHECK(partial_metadata.size(1) == kPartialExtent,
              "source-index partial extent does not match generated schedule");
  TORCH_CHECK(query_heads_per_key_value_head == 1,
              "indexed Fold binding does not use grouped count metadata");
""".rstrip()
        wrapper_shape = """
  const int64_t query_count = partial_scalar.size(0);
  const int64_t query_heads = 1;
  const int64_t value_width = partial_value.size(1);
  const int64_t row_count = query_count;
""".rstrip()
        output_checks = """
  TORCH_CHECK(output.dim() == 2, "indexed Fold output must be [R,D]");
  TORCH_CHECK(output.size(0) == query_count && output.size(1) == value_width,
              "indexed Fold output shape must match the row/feature domain");
""".rstrip()
        output_allocation = """
  auto output = torch::empty(
      {partial_scalar.size(0), partial_value.size(1)},
      partial_value.options().dtype(torch::kBFloat16));
""".rstrip()
    else:
        raise ValueError(f"unsupported Fold partial addressing {input_layout.addressing.value}")
    weight_expression = _render_cuda_scalar_expression(
        semantics.weight_expression,
        {
            "partial_scalar": "scalar",
            "reduced_scalar": "common",
            "valid": "semantic_valid",
        },
        ordered_fp=ordered_fp,
    )
    contributions = tuple(
        _render_cuda_scalar_expression(
            semantics.contribution_expression,
            {"partial_value": f"partial_value{element}", "weight": "weight"},
            ordered_fp=ordered_fp,
        )
        for element in range(features_per_lane)
    )
    updates = tuple(
        _render_cuda_scalar_expression(
            semantics.update_expression,
            {"state": f"numerator{element}", "contribution": f"contribution{element}"},
            ordered_fp=ordered_fp,
        )
        for element in range(features_per_lane)
    )
    finalizations = tuple(
        _render_cuda_scalar_expression(
            semantics.finalize_expression,
            {"state": f"numerator{element}", "denominator": "denominator"},
            ordered_fp=ordered_fp,
        )
        for element in range(features_per_lane)
    )
    feature_declarations = "\n".join(
        f"  const int real_feature{element} = feature_base + lane * kFeaturesPerLane + {element};\n"
        f"  const int shared_feature{element} = {shared_features[element]};"
        for element in range(features_per_lane)
    )
    numerator_declarations = "\n".join(f"  float numerator{element} = 0.0f;" for element in range(features_per_lane))
    partial_loads = "\n".join(
        f"        const float partial_value{element} = __bfloat162float(\n"
        f"            staged_value[warp][current_buffer][stage][shared_feature{element}]);"
        for element in range(features_per_lane)
    )
    contribution_updates = "\n".join(
        f"        const float contribution{element} = {contributions[element]};\n"
        f"        numerator{element} = {updates[element]};"
        for element in range(features_per_lane)
    )
    if semantics.scalar_reduction is FoldScalarReduction.MAXIMUM:
        scalar_reduction = f"""
  float scalar = 0.0f;
  bool semantic_valid = {lane_validity};
  if (semantic_valid) {{
    scalar = partial_scalar[{scalar_index}];
    semantic_valid = isfinite(scalar);
  }}
  const float common = fixed_warp_max(semantic_valid ? scalar : -CUDART_INF_F);
""".rstrip()
    elif semantics.scalar_reduction is FoldScalarReduction.NONE:
        scalar_reduction = f"""
  const bool semantic_valid = {lane_validity};
  const float scalar = semantic_valid ? partial_scalar[{scalar_index}] : 0.0f;
  const float common = 0.0f;
""".rstrip()
    else:
        raise ValueError(f"unsupported scalar reduction {semantics.scalar_reduction.value}")
    if semantics.denominator is FoldDenominatorPolicy.SUM_WEIGHTS:
        denominator_reduction = "const float denominator = fixed_warp_sum(local_weight);"
        empty_guard = "denominator > 0.0f"
    elif semantics.denominator is FoldDenominatorPolicy.NONE:
        denominator_reduction = "const float denominator = 1.0f;"
        empty_guard = "true"
    else:
        raise ValueError(f"unsupported denominator policy {semantics.denominator.value}")
    finalized_declarations = "\n".join(
        f"  const float finalized{element} = {empty_guard} ? ({finalizations[element]}) : 0.0f;"
        for element in range(features_per_lane)
    )
    result_stores = "\n".join(
        f"  const __nv_bfloat162 result{pair} = __floats2bfloat162_rn(\n"
        f"      finalized{pair * 2} * {program.output_scale:.17e}f,\n"
        f"      finalized{pair * 2 + 1} * {program.output_scale:.17e}f);\n"
        f"  const int64_t output_pair{pair} =\n"
        f"      (static_cast<int64_t>(row) * value_width + real_feature{pair * 2}) / 2;\n"
        f"  reinterpret_cast<__nv_bfloat162*>(output)[output_pair{pair}] = result{pair};"
        for pair in range(features_per_lane // 2)
    )
    if program.pipeline_buffers == 2:
        issue_next_group = f"""
    if (next_partial_base < kPartialExtent) {{
#pragma unroll
      for (int stage = 0; stage < kPipelineStages; ++stage) {{
        const int partial = next_partial_base + stage;
        if ({stage_validity} && lane < kCopiesPerStage) {{
          const int fake_feature = feature_base + lane * 8;
          const int64_t source_index =
              (static_cast<int64_t>({source_row})) * value_width + fake_feature;
          copy_global_to_shared_16(
              &staged_value[warp][next_buffer][stage][lane * 8],
              &partial_value[source_index]);
        }}
      }}
      commit_async_group();
    }}
""".rstrip()
        finish_next_group = """
    if (next_partial_base < kPartialExtent) {
      wait_for_all_async_groups();
      __syncwarp();
      current_buffer = next_buffer;
    }
""".rstrip()
        pipeline_comment = """
  // Ping-pong schedule: issue the next group into a disjoint shared buffer
  // before evaluating the generated AST over the current group.
""".rstrip()
    else:
        issue_next_group = ""
        finish_next_group = f"""
    if (next_partial_base < kPartialExtent) {{
      // One-buffer ablation: the generated AST must finish consuming the
      // current group before the next asynchronous copies reuse its storage.
#pragma unroll
      for (int stage = 0; stage < kPipelineStages; ++stage) {{
        const int partial = next_partial_base + stage;
        if ({stage_validity} && lane < kCopiesPerStage) {{
          const int fake_feature = feature_base + lane * 8;
          const int64_t source_index =
              (static_cast<int64_t>({source_row})) * value_width + fake_feature;
          copy_global_to_shared_16(
              &staged_value[warp][0][stage][lane * 8],
              &partial_value[source_index]);
        }}
      }}
      commit_async_group();
      wait_for_all_async_groups();
      __syncwarp();
    }}
""".rstrip()
        pipeline_comment = """
  // Serialized one-buffer schedule used as the explicit no-overlap ablation.
""".rstrip()
    return f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic Shuttle partial-state Fold semantics. The physical
// skeleton uses fixed-tree scalar-state reduction and ascending partial-slot
// vector accumulation. No MSA combine source or callable is used.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>

namespace {{

constexpr int kPartialExtent = {program.partial_extent};
constexpr int kRowsPerBlock = {program.rows_per_block};
constexpr int kFeatureTile = {program.feature_tile};
constexpr int kPipelineStages = {program.pipeline_stages};
constexpr int kPipelineBuffers = {program.pipeline_buffers};
constexpr int kFeaturesPerLane = {features_per_lane};
constexpr int kCopiesPerStage = kFeatureTile * sizeof(__nv_bfloat16) / 16;

__device__ __forceinline__ int real_col_to_stg128_half_col(int real_col) {{
  const int tile = real_col / 32;
  const int col32 = real_col - tile * 32;
  const int lane = (col32 % 8) / 2;
  const int group = col32 / 8;
  const int element = col32 % 2;
  return tile * 32 + lane * 8 + group * 2 + element;
}}

__device__ __forceinline__ void copy_global_to_shared_16(void* destination, const void* source) {{
  const unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared_address), "l"(source));
}}

__device__ __forceinline__ void commit_async_group() {{
  asm volatile("cp.async.commit_group;");
}}

__device__ __forceinline__ void wait_for_all_async_groups() {{
  asm volatile("cp.async.wait_group 0;");
}}

__device__ __forceinline__ float fixed_warp_max(float value) {{
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {{
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }}
  return __shfl_sync(0xffffffffu, value, 0);
}}

__device__ __forceinline__ float fixed_warp_sum(float value) {{
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {{
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }}
  return __shfl_sync(0xffffffffu, value, 0);
}}

__global__ __launch_bounds__({program.threads}) void shuttle_tiled_finalize_fold_partials(
    const float* partial_scalar,
    const __nv_bfloat16* partial_value,
    const int* partial_metadata,
    __nv_bfloat16* output,
    int query_count,
    int query_heads,
    int query_heads_per_key_value_head,
    int value_width) {{
  __shared__ __align__(16) __nv_bfloat16
      staged_value[kRowsPerBlock][kPipelineBuffers][kPipelineStages][kFeatureTile];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row_count = query_count * query_heads;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  const int feature_base = blockIdx.y * kFeatureTile;
  if (row >= row_count || feature_base >= value_width) return;

{row_mapping}

{scalar_reduction}
  const float local_weight = {weight_expression};
  {denominator_reduction}

{numerator_declarations}
{feature_declarations}

{pipeline_comment}
  // Prologue: make the first partial group available.
#pragma unroll
  for (int stage = 0; stage < kPipelineStages; ++stage) {{
    const int partial = stage;
    if ({stage_validity} && lane < kCopiesPerStage) {{
      const int fake_feature = feature_base + lane * 8;
      const int64_t source_index =
          (static_cast<int64_t>({source_row})) * value_width + fake_feature;
      copy_global_to_shared_16(
          &staged_value[warp][0][stage][lane * 8],
          &partial_value[source_index]);
    }}
  }}
  commit_async_group();
  wait_for_all_async_groups();
  __syncwarp();

  int current_buffer = 0;
#pragma unroll
  for (int partial_base = 0; partial_base < kPartialExtent;
       partial_base += kPipelineStages) {{
    const int next_partial_base = partial_base + kPipelineStages;
    const int next_buffer = (current_buffer + 1) % kPipelineBuffers;
{issue_next_group}

#pragma unroll
    for (int stage = 0; stage < kPipelineStages; ++stage) {{
      const int partial = partial_base + stage;
      const float weight = partial < kPartialExtent
          ? __shfl_sync(0xffffffffu, local_weight, partial)
          : 0.0f;
      if (partial < kPartialExtent && weight != 0.0f) {{
{partial_loads}
{contribution_updates}
      }}
    }}
{finish_next_group}
  }}

{finalized_declarations}
{result_stores}
}}

}}  // namespace

torch::Tensor shuttle_tiled_finalize_fold_partials_out_cuda(
    torch::Tensor partial_scalar,
    torch::Tensor partial_value,
    torch::Tensor partial_metadata,
    int64_t query_heads_per_key_value_head,
    torch::Tensor output) {{
  TORCH_CHECK(partial_scalar.is_cuda(), "scalar partial state must be CUDA");
  TORCH_CHECK(partial_value.is_cuda(), "value partial state must be CUDA");
  TORCH_CHECK(partial_metadata.is_cuda(), "partial metadata must be CUDA");
  TORCH_CHECK(partial_scalar.scalar_type() == torch::kFloat32,
              "scalar partial state must be FP32");
  TORCH_CHECK(partial_value.scalar_type() == torch::kBFloat16,
              "value partial state must be BF16");
  TORCH_CHECK(partial_metadata.scalar_type() == torch::kInt32,
              "partial metadata must be int32");
  TORCH_CHECK(partial_scalar.is_contiguous(), "scalar state must be contiguous");
  TORCH_CHECK(partial_value.is_contiguous(), "value state must be contiguous");
  TORCH_CHECK(partial_metadata.is_contiguous(), "partial metadata must be contiguous");
  TORCH_CHECK(output.is_cuda(), "Fold output must be CUDA");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "Fold output must be BF16");
  TORCH_CHECK(output.is_contiguous(), "Fold output must be contiguous");
  TORCH_CHECK(output.device() == partial_value.device(),
              "Fold output and partial values must be on the same device");
{wrapper_checks}
  const int64_t feature_extent = partial_value.size(partial_value.dim() - 1);
  TORCH_CHECK(feature_extent % kFeatureTile == 0,
              "feature width must be a multiple of the generated feature tile");

  const c10::cuda::CUDAGuard device_guard(partial_value.device());
{wrapper_shape}
{output_checks}
  const dim3 blocks(
      static_cast<unsigned>((row_count + kRowsPerBlock - 1) / kRowsPerBlock),
      static_cast<unsigned>(value_width / kFeatureTile),
      1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_tiled_finalize_fold_partials<<<blocks, {program.threads}, 0, stream>>>(
      partial_scalar.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(partial_value.data_ptr<at::BFloat16>()),
      partial_metadata.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(query_count),
      static_cast<int>(query_heads),
      static_cast<int>(query_heads_per_key_value_head),
      static_cast<int>(value_width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}}

torch::Tensor shuttle_tiled_finalize_fold_partials_cuda(
    torch::Tensor partial_scalar,
    torch::Tensor partial_value,
    torch::Tensor partial_metadata,
    int64_t query_heads_per_key_value_head) {{
{output_allocation}
  return shuttle_tiled_finalize_fold_partials_out_cuda(
      partial_scalar,
      partial_value,
      partial_metadata,
      query_heads_per_key_value_head,
      output);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("merge", &shuttle_tiled_finalize_fold_partials_cuda);
  module.def("merge_out", &shuttle_tiled_finalize_fold_partials_out_cuda);
}}
""".strip()


def render_partial_merge_cuda(program: PartialStateMergeProgram) -> str:
    """Generate a deterministic standalone merge for normalized Fold partials."""
    if program.schedule_kind is PartialMergeScheduleKind.TILED_PIPELINED:
        return _render_tiled_pipelined_merge_cuda(program)
    if program.representation != "log_normalizer_normalized_value":
        raise ValueError("the initial SM100 source generator supports normalized-value partials")
    if not math.isfinite(program.output_scale):
        raise ValueError("partial merge output scale must be finite")
    if program.accumulator_dtype != "fp32":
        raise ValueError("the initial deterministic merge requires FP32 accumulation")
    if program.threads < 32 or program.threads > 1024 or program.threads % 32:
        raise ValueError("partial merge threads must be a warp-aligned value in [32, 1024]")
    if program.value_dtype is PartialValueDType.BF16:
        value_pointer_type = "const __nv_bfloat16*"
        value_torch_dtype = "torch::kBFloat16"
        value_data_pointer = "reinterpret_cast<const __nv_bfloat16*>(normalized_value.data_ptr<at::BFloat16>())"
        load_value = "__bfloat162float(normalized_value[value_index])"
        feature_mapping = "real_col_to_stg128_half_col(real_feature)"
        value_width_alignment = 32
    elif program.value_dtype is PartialValueDType.FP32:
        value_pointer_type = "const float*"
        value_torch_dtype = "torch::kFloat32"
        value_data_pointer = "normalized_value.data_ptr<float>()"
        load_value = "normalized_value[value_index]"
        feature_mapping = "real_col_to_stg128_float_col(real_feature)"
        value_width_alignment = 16
    else:
        raise ValueError(f"unsupported normalized-value partial dtype {program.value_dtype}")
    if program.schedule_kind is PartialMergeScheduleKind.ROW_BLOCK:
        if program.rows_per_block != 1:
            raise ValueError("the row-block merge schedule requires rows_per_block == 1")
        kernel_body = f"""
  const int row_count = query_count * query_heads;
  const int row = blockIdx.x;
  if (row >= row_count) return;
  const int query = row / query_heads;
  const int query_head = row - query * query_heads;
  const int key_value_head = query_head / query_heads_per_key_value_head;
  __shared__ float partial_weights[32];
  __shared__ float shared_denominator;
  __shared__ int shared_valid_partials;

  if (threadIdx.x == 0) {{
    const int scheduled_partials = split_counts[query * (query_heads / query_heads_per_key_value_head)
                                                + key_value_head];
    const int valid_partials = scheduled_partials < partial_count ? scheduled_partials : partial_count;
    float common = -CUDART_INF_F;
    for (int partial = 0; partial < valid_partials; ++partial) {{
      common = fmaxf(common, log_normalizer[partial * row_count + row]);
    }}
    float denominator = 0.0f;
    for (int partial = 0; partial < valid_partials; ++partial) {{
      const float log_weight = log_normalizer[partial * row_count + row];
      const float weight = isfinite(log_weight) ? expf(log_weight - common) : 0.0f;
      partial_weights[partial] = weight;
      denominator += weight;
    }}
    shared_valid_partials = valid_partials;
    shared_denominator = denominator;
  }}
  __syncthreads();

  for (int real_feature = threadIdx.x; real_feature < value_width; real_feature += blockDim.x) {{
    const int fake_feature = {feature_mapping};
    float numerator = 0.0f;
    for (int partial = 0; partial < shared_valid_partials; ++partial) {{
      const int value_index = (partial * row_count + row) * value_width + fake_feature;
      numerator += partial_weights[partial] * {load_value};
    }}
    const float merged = shared_denominator > 0.0f ? numerator / shared_denominator : 0.0f;
    output[row * value_width + real_feature] = __float2bfloat16_rn(
        merged * {program.output_scale:.17e}f);
  }}
""".rstrip()
    elif program.schedule_kind is PartialMergeScheduleKind.WARP_ROWS:
        warp_count = program.threads // 32
        if program.rows_per_block != warp_count:
            raise ValueError("the warp-rows merge requires one row per physical warp")
        kernel_body = f"""
  const int row_count = query_count * query_heads;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * {program.rows_per_block} + warp;
  if (row >= row_count) return;
  const int query = row / query_heads;
  const int query_head = row - query * query_heads;
  const int key_value_head = query_head / query_heads_per_key_value_head;
  __shared__ float partial_weights[{program.rows_per_block}][32];
  __shared__ float shared_denominator[{program.rows_per_block}];
  __shared__ int shared_valid_partials[{program.rows_per_block}];

  if (lane == 0) {{
    const int scheduled_partials = split_counts[query * (query_heads / query_heads_per_key_value_head)
                                                + key_value_head];
    const int valid_partials = scheduled_partials < partial_count ? scheduled_partials : partial_count;
    float common = -CUDART_INF_F;
    for (int partial = 0; partial < valid_partials; ++partial) {{
      common = fmaxf(common, log_normalizer[partial * row_count + row]);
    }}
    float denominator = 0.0f;
    for (int partial = 0; partial < valid_partials; ++partial) {{
      const float log_weight = log_normalizer[partial * row_count + row];
      const float weight = isfinite(log_weight) ? expf(log_weight - common) : 0.0f;
      partial_weights[warp][partial] = weight;
      denominator += weight;
    }}
    shared_valid_partials[warp] = valid_partials;
    shared_denominator[warp] = denominator;
  }}
  __syncwarp();

  for (int real_feature = lane; real_feature < value_width; real_feature += 32) {{
    const int fake_feature = {feature_mapping};
    float numerator = 0.0f;
    for (int partial = 0; partial < shared_valid_partials[warp]; ++partial) {{
      const int value_index = (partial * row_count + row) * value_width + fake_feature;
      numerator += partial_weights[warp][partial] * {load_value};
    }}
    const float denominator = shared_denominator[warp];
    const float merged = denominator > 0.0f ? numerator / denominator : 0.0f;
    output[row * value_width + real_feature] = __float2bfloat16_rn(
        merged * {program.output_scale:.17e}f);
  }}
""".rstrip()
    else:
        raise ValueError(f"unsupported partial merge schedule {program.schedule_kind}")
    return f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic Shuttle partial-state Fold semantics. No MSA combine
// source or callable is used.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>

namespace {{

__device__ __forceinline__ int real_col_to_stg128_float_col(int real_col) {{
  const int tile = real_col / 16;
  const int col16 = real_col - tile * 16;
  const int pair = col16 / 2;
  const int rank = pair % 4;
  const int value = (pair / 4) * 2 + (col16 % 2);
  return tile * 16 + rank * 4 + value;
}}

__device__ __forceinline__ int real_col_to_stg128_half_col(int real_col) {{
  const int tile = real_col / 32;
  const int col32 = real_col - tile * 32;
  const int lane = (col32 % 8) / 2;
  const int group = col32 / 8;
  const int element = col32 % 2;
  return tile * 32 + lane * 8 + group * 2 + element;
}}

__global__ void shuttle_merge_normalized_exp_partials(
    const float* log_normalizer,
    {value_pointer_type} normalized_value,
    const int* split_counts,
    __nv_bfloat16* output,
    int partial_count,
    int query_count,
    int query_heads,
    int query_heads_per_key_value_head,
    int value_width) {{
{kernel_body}
}}

}}  // namespace

torch::Tensor shuttle_merge_normalized_exp_partials_cuda(
    torch::Tensor log_normalizer,
    torch::Tensor normalized_value,
    torch::Tensor split_counts,
    int64_t query_heads_per_key_value_head) {{
  TORCH_CHECK(log_normalizer.is_cuda(), "log_normalizer must be CUDA");
  TORCH_CHECK(normalized_value.is_cuda(), "normalized_value must be CUDA");
  TORCH_CHECK(split_counts.is_cuda(), "split_counts must be CUDA");
  TORCH_CHECK(log_normalizer.scalar_type() == torch::kFloat32, "log_normalizer must be FP32");
  TORCH_CHECK(normalized_value.scalar_type() == {value_torch_dtype},
              "normalized_value partial dtype does not match the generated numerical policy");
  TORCH_CHECK(split_counts.scalar_type() == torch::kInt32, "split_counts must be int32");
  TORCH_CHECK(log_normalizer.dim() == 3, "log_normalizer must be [P,Q,Hq]");
  TORCH_CHECK(normalized_value.dim() == 4, "normalized_value must be [P,Q,Hq,D]");
  TORCH_CHECK(split_counts.dim() == 2, "split_counts must be [Q,Hkv]");
  TORCH_CHECK(log_normalizer.is_contiguous(), "log_normalizer must be contiguous");
  TORCH_CHECK(normalized_value.is_contiguous(), "normalized_value must be contiguous");
  TORCH_CHECK(split_counts.is_contiguous(), "split_counts must be contiguous");
  TORCH_CHECK(log_normalizer.size(0) == normalized_value.size(0), "partial counts must match");
  TORCH_CHECK(log_normalizer.size(1) == normalized_value.size(1), "query counts must match");
  TORCH_CHECK(log_normalizer.size(2) == normalized_value.size(2), "query-head counts must match");
  TORCH_CHECK(split_counts.size(0) == normalized_value.size(1), "split-count query domain must match");
  TORCH_CHECK(query_heads_per_key_value_head > 0, "GQA ratio must be positive");
  TORCH_CHECK(normalized_value.size(2) % query_heads_per_key_value_head == 0,
              "query heads must divide by the GQA ratio");
  TORCH_CHECK(split_counts.size(1) == normalized_value.size(2) / query_heads_per_key_value_head,
              "split-count KV-head domain must match");
  TORCH_CHECK(normalized_value.size(3) % {value_width_alignment} == 0,
              "value width must be a multiple of {value_width_alignment} for the generated layout");
  TORCH_CHECK(normalized_value.size(0) <= 32, "generated row merge supports at most 32 partials");

  const c10::cuda::CUDAGuard device_guard(normalized_value.device());
  auto output = torch::empty(
      {{normalized_value.size(1), normalized_value.size(2), normalized_value.size(3)}},
      normalized_value.options().dtype(torch::kBFloat16));
  constexpr int threads = {program.threads};
  const int64_t row_count = normalized_value.size(1) * normalized_value.size(2);
  const int blocks = static_cast<int>((row_count + {program.rows_per_block} - 1) / {program.rows_per_block});
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shuttle_merge_normalized_exp_partials<<<blocks, threads, 0, stream>>>(
      log_normalizer.data_ptr<float>(),
      {value_data_pointer},
      split_counts.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(normalized_value.size(0)),
      static_cast<int>(normalized_value.size(1)),
      static_cast<int>(normalized_value.size(2)),
      static_cast<int>(query_heads_per_key_value_head),
      static_cast<int>(normalized_value.size(3)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
  module.def("merge", &shuttle_merge_normalized_exp_partials_cuda);
}}
""".strip()


def render_partial_merge_ffi_cuda(
    program: PartialStateMergeProgram,
    *,
    target: str,
    partial_count: int,
    query_count: int,
    query_heads: int,
    key_value_heads: int,
    value_width: int,
) -> GeneratedPartialMergeFfi:
    """Generate a Torch-free JAX typed-FFI wrapper for one Fold finalizer."""
    if program.schedule_kind is PartialMergeScheduleKind.TILED_PIPELINED:
        raise ValueError("the first typed-FFI Fold finalizer supports row-block and warp-row schedules")
    if min(partial_count, query_count, query_heads, key_value_heads, value_width) <= 0:
        raise ValueError("typed-FFI Fold extents must be positive")
    if query_heads % key_value_heads:
        raise ValueError("Fold query-head count must divide by the partition count")
    symbol = target.replace(".", "_").replace("-", "_")
    if not symbol.isidentifier():
        raise ValueError(f"typed-FFI target {target!r} cannot form a C++ symbol")
    handler_symbol = f"{symbol}_handler"
    torch_source = render_partial_merge_cuda(program)
    namespace_begin = torch_source.index("namespace {")
    namespace_end = torch_source.index("}  // namespace", namespace_begin) + len("}  // namespace")
    kernel_source = torch_source[namespace_begin:namespace_end]
    value_ffi_dtype = {
        PartialValueDType.BF16: "ffi::BF16",
        PartialValueDType.FP32: "ffi::F32",
    }[program.value_dtype]
    q_heads_per_partition = query_heads // key_value_heads
    blocks = math.ceil(query_count * query_heads / program.rows_per_block)
    source_template = f"""
// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
// Generated from generic Shuttle partial-state Fold semantics.
#include <atomic>
#include <cstdint>

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

{kernel_source}

namespace {{
std::atomic<int> call_count{{0}};

ffi::Error ShuttlePartialStateFoldFinalize(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32, 3> scalar_state,
    ffi::Buffer<{value_ffi_dtype}, 4> value_state,
    ffi::Buffer<ffi::S32, 2> partial_counts,
    ffi::Result<ffi::Buffer<ffi::BF16, 3>> output) {{
  constexpr int kBlocks = {blocks};
  constexpr int kThreads = {program.threads};
  shuttle_merge_normalized_exp_partials<<<kBlocks, kThreads, 0, stream>>>(
      scalar_state.typed_data(),
      reinterpret_cast<const {"__nv_bfloat16" if program.value_dtype is PartialValueDType.BF16 else "float"}*>(
          value_state.typed_data()),
      partial_counts.typed_data(),
      reinterpret_cast<__nv_bfloat16*>(output->typed_data()),
      {partial_count},
      {query_count},
      {query_heads},
      {q_heads_per_partition},
      {value_width});
  call_count.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}}

auto ShuttlePartialStateFoldFinalizeBinding() {{
  return ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<ffi::Buffer<ffi::F32, 3>>()
      .Arg<ffi::Buffer<{value_ffi_dtype}, 4>>()
      .Arg<ffi::Buffer<ffi::S32, 2>>()
      .Ret<ffi::Buffer<ffi::BF16, 3>>();
}}
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {handler_symbol},
    ShuttlePartialStateFoldFinalize,
    ShuttlePartialStateFoldFinalizeBinding()__SHUTTLE_FFI_HANDLER_TRAITS__);

extern "C" int shuttle_partial_state_fold_finalize_call_count() {{
  return call_count.load(std::memory_order_relaxed);
}}
""".strip()
    source = finalize_ffi_handler_source(source_template, command_buffer_compatible=True)
    return GeneratedPartialMergeFfi(
        target=target,
        handler_symbol=handler_symbol,
        source=source,
        source_sha256=_sha256(source),
        partial_count=partial_count,
        query_count=query_count,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        value_width=value_width,
        value_dtype=program.value_dtype,
    )


def extract_clean_sm100_sources(
    msa_root: Path,
    lowering: SM100RoutedStreamingLowering,
    *,
    paged_key_value: bool = True,
    partial_value_dtype: PartialValueDType = PartialValueDType.BF16,
    partial_merge_schedule: PartialMergeScheduleKind = PartialMergeScheduleKind.ROW_BLOCK,
) -> ExtractedSM100Sources:
    """Extract and audit the first generic physical source family."""
    root = msa_root.resolve()
    lineage = _validate_msa_checkout(root)
    upstream, upstream_hashes = _read_pinned_sources(root)
    plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=paged_key_value,
        partial_value_dtype=partial_value_dtype,
        partial_merge_schedule=partial_merge_schedule,
    )
    semantic_source = _render_semantic_source(
        upstream["src/common/softmax.py"],
        upstream["src/common/mask.py"],
    )
    physical_source = _replace_semantic_imports(upstream["src/sm100/fwd/atten_fwd.py"])
    relation_builder_source = render_relation_builder_source(upstream["src/sm100/prepare_k2q_csr.py"])
    scheduler_source = render_relation_scheduler_source(upstream["src/sm100/prepare_scheduler.py"])
    merge_source = render_partial_merge_cuda(plan.partial_merge)

    compile(semantic_source, f"<{GENERATED_SEMANTICS_MODULE}>", "exec")
    compile(physical_source, "<shuttle_sm100_physical>", "exec")
    compile(relation_builder_source, "<shuttle_sm100_relation_builder>", "exec")
    compile(scheduler_source, "<shuttle_sm100_scheduler>", "exec")
    semantic_audit = audit_python_source(semantic_source)
    if not semantic_audit.clean:
        raise ValueError(
            f"SM100 semantic extraction retains forbidden dependencies: {semantic_audit.forbidden_dependencies}"
        )
    physical_audit = _require_clean_physical_source(physical_source)
    relation_builder_audit = audit_python_source(relation_builder_source)
    scheduler_audit = audit_python_source(scheduler_source)
    stale_relation_names = tuple(
        sorted(
            name
            for name in (
                "SparseK2qCsrBuilderSm100",
                "SparseAttentionSchedule",
                "SparseAttentionScheduleModel",
                "prepare_sparse_fwd_schedule_and_split",
            )
            if name in relation_builder_source or name in scheduler_source
        )
    )
    if stale_relation_names:
        raise ValueError(f"generic relation extraction retains workload-specific interfaces: {stale_relation_names}")
    forbidden_merge_token = next(
        (token for token in FORBIDDEN_REFERENCED_NAMES if token in merge_source),
        None,
    )
    if forbidden_merge_token is not None:
        raise ValueError(f"generated merge source retains forbidden symbol {forbidden_merge_token}")

    return ExtractedSM100Sources(
        emitter_plan=plan,
        physical_source=physical_source,
        semantic_source=semantic_source,
        relation_builder_source=relation_builder_source,
        scheduler_source=scheduler_source,
        merge_cuda_source=merge_source,
        upstream_source_sha256=upstream_hashes,
        generated_source_sha256={
            "physical": _sha256(physical_source),
            "semantics": _sha256(semantic_source),
            "relation_builder": _sha256(relation_builder_source),
            "scheduler": _sha256(scheduler_source),
            "partial_merge": _sha256(merge_source),
        },
        physical_audit=physical_audit,
        semantic_audit=semantic_audit,
        relation_builder_audit=relation_builder_audit,
        scheduler_audit=scheduler_audit,
        lineage={
            **lineage,
            "retained_physical_mechanisms": (
                "tcgen05 matrix mainloops",
                "TMA movement and gather descriptors",
                "bounded producer-consumer pipelines and barriers",
                "right-major relation CSR/worklist construction",
                "GQA physical packing and STG.128 partial layout",
            ),
            "removed_semantic_interfaces": (
                "MSA public sparse attention callable",
                "MSA semantic combine",
                "MSA Softmax and AttentionMask imports",
                "MSA-named sparse-attention relation builder and scheduler interfaces",
            ),
            "constant_pruned_template_features": (
                "temperature log-sum-exp output",
                "paged key-value addressing",
            ),
            "generic_interfaces": (
                GENERATED_PHYSICAL_CLASS,
                GENERATED_RELATION_BUILDER_CLASS,
                "RightMajorRelationSchedule",
                "NormalizedExpFoldSm100",
                "DomainRestrictionSm100",
            ),
            "event_tensor": {
                "program_fingerprint": plan.event_schedule.program_fingerprint,
                "runtime_fingerprint": plan.event_schedule.runtime_fingerprint,
                "resource_tasks": plan.event_schedule.grouping.task_count,
                "resource_buffer_depth": plan.event_schedule.resource_buffer.capacity,
                "realizations": tuple(
                    (entry.plan_name, entry.kind.value, entry.mechanism)
                    for entry in plan.event_schedule.realization.entries
                ),
                "physical_boundary": (
                    "right-resource staging and slot reuse are primitive-owned; "
                    "the generated Fold finalizer follows on the same device stream"
                ),
            },
            "numerical_policy": {
                "partial_log_normalizer": "fp32",
                "partial_normalized_value": partial_value_dtype.value,
                "partial_merge_accumulator": "fp32",
                "partial_merge_order": "ascending_partial_slot",
                "partial_merge_schedule": partial_merge_schedule.value,
            },
        },
    )


def import_extracted_python_sources(
    sources: ExtractedSM100Sources,
    *,
    msa_root: Path,
    source_directory: Path | None = None,
) -> ModuleType:
    """Import the audited extraction in an environment with CuTe dependencies.

    This loads only the generated semantic module and the extracted physical
    class. It does not import MSA's public sparse-attention interface, semantic
    softmax/mask modules, or combine implementation.
    """
    cute_root = (msa_root / MINIMAX_MSA_CUTE_ROOT).resolve()
    if str(cute_root) not in sys.path:
        sys.path.insert(0, str(cute_root))

    semantics_filename = f"<{GENERATED_SEMANTICS_MODULE}>"
    linecache.cache[semantics_filename] = (
        len(sources.semantic_source),
        None,
        sources.semantic_source.splitlines(keepends=True),
        semantics_filename,
    )
    semantics_spec = importlib.util.spec_from_loader(GENERATED_SEMANTICS_MODULE, loader=None)
    if semantics_spec is None:
        raise RuntimeError("failed to create the generated semantics module specification")
    semantics_module = importlib.util.module_from_spec(semantics_spec)
    prior_semantics = sys.modules.get(GENERATED_SEMANTICS_MODULE)
    sys.modules[GENERATED_SEMANTICS_MODULE] = semantics_module
    try:
        exec(
            compile(sources.semantic_source, semantics_filename, "exec"),
            semantics_module.__dict__,
        )
    except BaseException:
        if prior_semantics is None:
            sys.modules.pop(GENERATED_SEMANTICS_MODULE, None)
        else:
            sys.modules[GENERATED_SEMANTICS_MODULE] = prior_semantics
        raise

    physical_module_name = "shuttle_sm100_extracted_physical"
    physical_path = None
    if source_directory is not None:
        source_directory.mkdir(parents=True, exist_ok=True)
        source_hash = hashlib.sha256(sources.physical_source.encode()).hexdigest()[:16]
        physical_path = source_directory / f"{physical_module_name}_{source_hash}.py"
        if not physical_path.exists() or physical_path.read_text() != sources.physical_source:
            physical_path.write_text(sources.physical_source)
    physical_filename = str(physical_path) if physical_path is not None else f"<{physical_module_name}>"
    linecache.cache[physical_filename] = (
        len(sources.physical_source),
        None,
        sources.physical_source.splitlines(keepends=True),
        physical_filename,
    )
    physical_spec = (
        importlib.util.spec_from_file_location(physical_module_name, physical_path)
        if physical_path is not None
        else importlib.util.spec_from_loader(physical_module_name, loader=None)
    )
    if physical_spec is None:
        raise RuntimeError("failed to create the extracted physical module specification")
    physical_module = importlib.util.module_from_spec(physical_spec)
    prior_physical = sys.modules.get(physical_module_name)
    sys.modules[physical_module_name] = physical_module
    try:
        if physical_path is None:
            exec(
                compile(sources.physical_source, physical_filename, "exec"),
                physical_module.__dict__,
            )
        else:
            if physical_spec.loader is None:
                raise RuntimeError("failed to create the extracted physical source loader")
            physical_spec.loader.exec_module(physical_module)
    except BaseException:
        if prior_physical is None:
            sys.modules.pop(physical_module_name, None)
        else:
            sys.modules[physical_module_name] = prior_physical
        raise
    return physical_module
