# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic helpers for profile op grouping and shape-signature extraction."""

from __future__ import annotations

import math
import re

_SEMANTIC_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("attention_splash", re.compile(r"splash_mha|splash_attention", re.IGNORECASE)),
    ("attention_flash", re.compile(r"flash_attention|fused_attention", re.IGNORECASE)),
    ("loss_xent", re.compile(r"linear_softmax_cross_entropy_loss|softmax_cross_entropy|xent", re.IGNORECASE)),
    ("fusion", re.compile(r"(^|\b)(fusion\.|exponential_reduce_fusion)", re.IGNORECASE)),
    ("copy", re.compile(r"(^|\b)copy(\.|-|$)", re.IGNORECASE)),
    ("convert", re.compile(r"convert_element_type", re.IGNORECASE)),
    (
        "collective",
        re.compile(
            r"all-reduce|all_gather|all-gather|reduce-scatter|all-to-all|alltoall|collective",
            re.IGNORECASE,
        ),
    ),
)
_OUTPUT_TUPLE_RE = re.compile(r"=\s*\((.*?)\)\s*[A-Za-z0-9_.-]+\(", re.DOTALL)
_OUTPUT_SINGLE_RE = re.compile(r"=\s*([A-Za-z0-9_]+\[[^\]]+\])")
_DTYPE_SHAPE_RE = re.compile(r"(?:bf16|f32|f16|s32|u32|s8|u8|pred)\[([^\]]+)\]", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


def canonical_op_name(name: str) -> str:
    """Canonicalize an op name by removing numeric suffixes and `%` prefix."""
    return re.sub(r"\.\d+$", "", name.strip().lstrip("%"))


def classify_semantic_family(name: str) -> str:
    """Classify a raw op name into a semantic family bucket."""
    canonical = canonical_op_name(name)
    for family, pattern in _SEMANTIC_FAMILY_PATTERNS:
        if pattern.search(canonical):
            return family
    return "other"


def extract_shape_signature(long_name: str | None) -> str | None:
    """Extract a compact, deterministic shape signature from HLO `long_name`."""
    if not long_name:
        return None

    tuple_match = _OUTPUT_TUPLE_RE.search(long_name)
    if tuple_match:
        shapes = _DTYPE_SHAPE_RE.findall(tuple_match.group(1))
        cleaned = [_normalize_shape_token(token) for token in shapes if token]
        return "|".join(cleaned) if cleaned else None

    single_match = _OUTPUT_SINGLE_RE.search(long_name)
    if single_match:
        shapes = _DTYPE_SHAPE_RE.findall(single_match.group(1))
        cleaned = [_normalize_shape_token(token) for token in shapes if token]
        return "|".join(cleaned) if cleaned else None

    shapes = _DTYPE_SHAPE_RE.findall(long_name)
    cleaned = [_normalize_shape_token(token) for token in shapes[:8] if token]
    return "|".join(cleaned) if cleaned else None


def parse_shape_signature(shape_signature: str | None) -> list[tuple[int, ...]]:
    """Parse a shape signature into integer-dimension tuples."""
    if not shape_signature:
        return []
    dims: list[tuple[int, ...]] = []
    for token in shape_signature.split("|"):
        values = [int(match.group(0)) for match in _INT_RE.finditer(token)]
        if values:
            dims.append(tuple(values))
    return dims


def estimate_flop_proxy(family: str, shape_signature: str | None) -> float | None:
    """
    Estimate relative per-invocation work from shape signature.

    This is a coarse proxy intended for before/after normalization, not a FLOP counter.
    """
    dims = parse_shape_signature(shape_signature)
    if not dims:
        return None

    if family.startswith("attention"):
        rank4 = [shape for shape in dims if len(shape) >= 4]
        if rank4:
            batch = max(shape[0] for shape in rank4 if shape[0] > 0)
            heads = max(shape[1] for shape in rank4 if shape[1] > 0)
            seq = max(shape[2] for shape in rank4 if shape[2] > 0)
            head_dims = sorted({shape[3] for shape in rank4 if shape[3] > 0})
            if batch > 0 and heads > 0 and seq > 0 and head_dims:
                if len(head_dims) >= 2:
                    d_qk = max(head_dims)
                    d_v = min(head_dims)
                else:
                    d_qk = head_dims[0]
                    d_v = head_dims[0]
                # Approximate attention FLOPs: qk matmul + av matmul.
                return float(2 * batch * heads * seq * seq * (d_qk + d_v))

    if family == "loss_xent":
        rank2_or_more = [shape for shape in dims if len(shape) >= 2]
        if rank2_or_more:
            rows = max(shape[0] for shape in rank2_or_more if shape[0] > 0)
            cols = max(shape[1] for shape in rank2_or_more if shape[1] > 0)
            if rows > 0 and cols > 0:
                return float(2 * rows * cols)

    best = max(_shape_product(shape) for shape in dims)
    if math.isfinite(best) and best > 0:
        return float(best)
    return None


def estimate_work_proxy(family: str, shape_signature: str | None) -> float | None:
    """Backward-compatible alias for estimate_flop_proxy."""
    return estimate_flop_proxy(family, shape_signature)


def _normalize_shape_token(token: str) -> str:
    return ",".join(part.strip() for part in token.split(",") if part.strip())


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        if dim <= 0:
            return 0
        product *= dim
    return product


__all__ = [
    "canonical_op_name",
    "classify_semantic_family",
    "estimate_flop_proxy",
    "estimate_work_proxy",
    "extract_shape_signature",
    "parse_shape_signature",
]
