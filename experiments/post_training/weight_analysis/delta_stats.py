# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight-delta statistics over Grug checkpoints, streamed one block at a time.

A block is one tensor, or one expert of a stacked routed-expert tensor, so the
largest array in memory is one embedding matrix. Every metric is reported over
all weights, per :class:`Component`, and per (layer, expert)
(:class:`~experiments.post_training.weight_analysis.components.Keyed`).

Control C8: deltas are taken against the fp32 view of ``B``. A checkpoint must
be stored in F32. ``B`` may be stored in BF16 (it is upcast on read) or in F32,
but every F32 value must survive a bf16 round trip, since ``B`` is the upcast
bf16 Stage 0 output. A checkpoint whose every tensor is bf16-representable is a
bf16 export mislabelled as fp32, and raises.

Deltas are ``checkpoint - B`` in fp32. Sterbenz's lemma makes the subtraction
exact wherever the two values are within a factor of two, which holds for all
but weights crossing or sitting near zero; there the rounding error is below
2^-24 of the value. Sums run in fp64.

Port of the #9920/#9984 weight-delta analyses (``weight_delta_stats.py``,
``weight_delta_subspace.py``, branch ``task/239``, not on ``origin``), rebuilt
from the method text in those issues.
"""

import itertools
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import ml_dtypes
import numpy as np

from experiments.post_training.weight_analysis.checkpoint_io import SafetensorsCheckpoint
from experiments.post_training.weight_analysis.components import (
    Block,
    Component,
    Keyed,
    component_of,
    layer_of,
)

# Elements per fp64 chunk when summing, so a 2 GB fp32 matrix is never widened whole.
_CHUNK_ELEMENTS = 1 << 22
_BASE_DTYPES = (np.dtype(np.float32), np.dtype(ml_dtypes.bfloat16))


def _chunk_pairs(left: np.ndarray, right: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    for start in range(0, flat_left.size, _CHUNK_ELEMENTS):
        stop = start + _CHUNK_ELEMENTS
        yield flat_left[start:stop].astype(np.float64), flat_right[start:stop].astype(np.float64)


def sum_squares(array: np.ndarray) -> float:
    total = 0.0
    flat = array.reshape(-1)
    for start in range(0, flat.size, _CHUNK_ELEMENTS):
        chunk = flat[start : start + _CHUNK_ELEMENTS].astype(np.float64)
        total += float(chunk @ chunk)
    return total


def _is_bf16_exact(array: np.ndarray) -> bool:
    return bool(np.array_equal(array, array.astype(ml_dtypes.bfloat16).astype(np.float32)))


def checkpoint_blocks(checkpoint: SafetensorsCheckpoint) -> Iterator[Block]:
    """Every block of a checkpoint in its stored dtype, in sorted-name order.

    Routed-expert tensors yield one block per expert, read as a byte-range
    slice; a routed-expert tensor that is not stacked ``[E, out, in]`` raises.
    """
    for name in checkpoint.names():
        component = component_of(name)
        if component is not Component.ROUTED_EXPERTS:
            yield Block(name, component, None, checkpoint.read(name))
            continue
        shape = checkpoint.spec(name).shape
        if len(shape) != 3:
            raise ValueError(f"routed-expert tensor {name!r} has shape {shape}; expected [E, out, in]")
        layer = layer_of(name)
        for expert in range(shape[0]):
            yield Block(name, component, (layer, expert), checkpoint.read_expert(name, expert))


def paired_blocks(checkpoint_path: str, base_path: str) -> Iterator[tuple[Block, Block]]:
    """(B block, checkpoint block) pairs, both fp32, after the C8 checks above."""
    checkpoint = SafetensorsCheckpoint(checkpoint_path)
    base = SafetensorsCheckpoint(base_path)
    names = checkpoint.names()
    if names != base.names():
        raise ValueError(
            f"{checkpoint_path} and {base_path} cover different tensors; "
            f"only in checkpoint: {sorted(set(names) - set(base.names()))[:5]}, "
            f"only in base: {sorted(set(base.names()) - set(names))[:5]}"
        )
    for name in names:
        checkpoint_spec, base_spec = checkpoint.spec(name), base.spec(name)
        if checkpoint_spec.shape != base_spec.shape:
            raise ValueError(f"{name!r}: checkpoint shape {checkpoint_spec.shape} != base shape {base_spec.shape}")
        if checkpoint_spec.dtype != np.float32:
            raise ValueError(f"{name!r} is {checkpoint_spec.dtype} in {checkpoint_path}; C8 needs fp32 masters")
        if base_spec.dtype not in _BASE_DTYPES:
            raise ValueError(f"{name!r} is {base_spec.dtype} in {base_path}; B must be bf16 or its fp32 upcast")

    every_tensor_bf16_exact = True
    for base_block, checkpoint_block in aligned(checkpoint_blocks(base), checkpoint_blocks(checkpoint)):
        base_array = base_block.array.astype(np.float32)
        if base_block.array.dtype == np.float32 and not _is_bf16_exact(base_array):
            raise ValueError(f"{base_block.name!r} in {base_path} is not the fp32 upcast of a bf16 tensor (C8)")
        every_tensor_bf16_exact = every_tensor_bf16_exact and _is_bf16_exact(checkpoint_block.array)
        yield Block(base_block.name, base_block.component, base_block.expert, base_array), checkpoint_block
    if every_tensor_bf16_exact:
        raise ValueError(f"every tensor of {checkpoint_path} is bf16-representable: a bf16 export, not an fp32 master")


def delta_blocks(checkpoint_path: str, base_path: str) -> Iterator[Block]:
    """``checkpoint - B`` per block, in fp32."""
    for base_block, checkpoint_block in paired_blocks(checkpoint_path, base_path):
        delta = checkpoint_block.array - base_block.array
        yield Block(base_block.name, base_block.component, base_block.expert, delta)


def aligned(*streams: Iterable[Block]) -> Iterator[tuple[Block, ...]]:
    """Zip block streams, raising unless every row names the same block and shape."""
    for row in itertools.zip_longest(*streams):
        if any(block is None for block in row):
            raise ValueError("block streams have different lengths")
        first = row[0]
        for block in row[1:]:
            if (block.name, block.expert) != (first.name, first.expert):
                raise ValueError(
                    f"block streams are not aligned: {first.name}/{first.expert} vs {block.name}/{block.expert}"
                )
            if block.array.shape != first.array.shape:
                raise ValueError(
                    f"{first.name}/{first.expert}: shapes {first.array.shape} and {block.array.shape} differ"
                )
        yield row


@dataclass
class DeltaSummary:
    """‖Δ‖, ‖B‖ and the share of weights that changed, for one key."""

    delta_sq: float = 0.0
    base_sq: float = 0.0
    changed: int = 0
    elements: int = 0

    @property
    def delta_norm(self) -> float:
        return math.sqrt(self.delta_sq)

    @property
    def relative_norm(self) -> float:
        """‖Δ‖/‖B‖ in Frobenius norm."""
        if self.base_sq == 0.0:
            raise ValueError("‖B‖ is zero for this key; the relative norm is undefined")
        return math.sqrt(self.delta_sq / self.base_sq)

    @property
    def changed_fraction(self) -> float:
        """The share of weights with a nonzero fp32 delta (#9920's "weights changed")."""
        return self.changed / self.elements


def summarize_deltas(pairs: Iterable[tuple[Block, Block]]) -> Keyed[DeltaSummary]:
    """Per-key delta norms, relative norms and changed fractions from :func:`paired_blocks`."""
    summary = Keyed(overall=DeltaSummary())
    for base_block, checkpoint_block in pairs:
        delta = checkpoint_block.array - base_block.array
        delta_sq = sum_squares(delta)
        base_sq = sum_squares(base_block.array)
        changed = int(np.count_nonzero(delta))
        for target in summary.targets(base_block, DeltaSummary):
            target.delta_sq += delta_sq
            target.base_sq += base_sq
            target.changed += changed
            target.elements += delta.size
    return summary


@dataclass
class PairGeometry:
    """Dot product and squared norms of two arrays accumulated over one key."""

    dot: float = 0.0
    left_sq: float = 0.0
    right_sq: float = 0.0

    def update(self, left: np.ndarray, right: np.ndarray) -> None:
        for chunk_left, chunk_right in _chunk_pairs(left, right):
            self.dot += float(chunk_left @ chunk_right)
            self.left_sq += float(chunk_left @ chunk_left)
            self.right_sq += float(chunk_right @ chunk_right)

    def merge(self, other: "PairGeometry") -> None:
        self.dot += other.dot
        self.left_sq += other.left_sq
        self.right_sq += other.right_sq

    @property
    def defined(self) -> bool:
        return self.left_sq > 0.0 and self.right_sq > 0.0

    @property
    def cosine(self) -> float:
        if not self.defined:
            raise ValueError("a side has zero norm for this key; the cosine is undefined")
        return self.dot / math.sqrt(self.left_sq * self.right_sq)

    @property
    def norm_ratio(self) -> float:
        """‖left‖/‖right‖."""
        if self.right_sq == 0.0:
            raise ValueError("the right side has zero norm for this key; the norm ratio is undefined")
        return math.sqrt(self.left_sq / self.right_sq)


@dataclass
class DeltaComparison:
    """Cosine, sign agreement and changed fractions of two deltas for one key."""

    geometry: PairGeometry
    elements: int = 0
    changed_left: int = 0
    changed_right: int = 0
    changed_both: int = 0
    same_sign: int = 0

    @property
    def cosine(self) -> float:
        return self.geometry.cosine

    @property
    def sign_agreement(self) -> float:
        """Among weights both deltas changed, the share moved in the same direction."""
        if self.changed_both == 0:
            raise ValueError("no weight changed in both deltas for this key; sign agreement is undefined")
        return self.same_sign / self.changed_both


def _new_comparison() -> DeltaComparison:
    return DeltaComparison(PairGeometry())


def compare_deltas(left: Iterable[Block], right: Iterable[Block]) -> Keyed[DeltaComparison]:
    """Task conflict between two deltas (``Δ_A`` vs ``Δ_B``, or a method vs an expert)."""
    comparisons = Keyed(overall=_new_comparison())
    for left_block, right_block in aligned(left, right):
        sign_left = np.sign(left_block.array)
        sign_right = np.sign(right_block.array)
        both = (sign_left != 0) & (sign_right != 0)
        counts = (
            left_block.array.size,
            int(np.count_nonzero(sign_left)),
            int(np.count_nonzero(sign_right)),
            int(np.count_nonzero(both)),
            int(np.count_nonzero(both & (sign_left == sign_right))),
        )
        block_geometry = PairGeometry()
        block_geometry.update(left_block.array, right_block.array)
        for target in comparisons.targets(left_block, _new_comparison):
            target.geometry.merge(block_geometry)
            target.elements += counts[0]
            target.changed_left += counts[1]
            target.changed_right += counts[2]
            target.changed_both += counts[3]
            target.same_sign += counts[4]
    return comparisons


@dataclass
class SubspaceOverlap:
    """Energy of a delta inside a reference delta's half-energy subspace, for one key.

    Aggregates are energy-weighted: Σ projected energy / Σ delta energy over the
    key's matrices, so each ratio is the share of the key's delta energy inside
    the subspaces. This is the reading #9920 and #9984 report ("20% [of OPD's
    update energy] sits inside E's half-energy subspace against 10% at random").
    Whether ``weight_delta_subspace.py`` pooled the same way or averaged per
    matrix is not stated there and needs the ``task/239`` script or its result
    JSONs to confirm.
    """

    delta_energy: float = 0.0
    inside: float = 0.0
    inside_random: float = 0.0
    inside_permuted: float = 0.0
    matrices: int = 0
    zero_reference_matrices: int = 0
    """Matrices where the reference did not move, so it has no subspace. Their
    delta energy is excluded from the denominators."""

    def _share(self, value: float) -> float:
        if self.delta_energy == 0.0:
            raise ValueError("no delta energy in any matrix for this key; overlap is undefined")
        return value / self.delta_energy

    @property
    def overlap(self) -> float:
        return self._share(self.inside)

    @property
    def random_control(self) -> float:
        return self._share(self.inside_random)

    @property
    def permuted_control(self) -> float:
        return self._share(self.inside_permuted)


def _top_right_singular_basis(matrix: np.ndarray, k: int | None) -> tuple[np.ndarray, int]:
    """The top-k right singular directions of ``matrix`` as an [n, k] basis.

    With ``k=None``, k is the smallest rank holding half of the matrix energy.
    The basis comes from an fp64 eigendecomposition of the min-side Gram matrix,
    which costs O(m·n·min(m, n)) and an min(m, n)² buffer; for Grug's tall
    vocab-by-d matrices that is a d-by-d problem, cheaper than a randomized SVD of
    the matrix and exact. Squaring the condition number only blurs directions
    far below the half-energy cut.
    """
    rows, cols = matrix.shape
    if cols <= rows:
        gram = np.zeros((cols, cols))
        for start in range(0, rows, max(1, _CHUNK_ELEMENTS // cols)):
            chunk = matrix[start : start + max(1, _CHUNK_ELEMENTS // cols)].astype(np.float64)
            gram += chunk.T @ chunk
        eigenvalues, vectors = np.linalg.eigh(gram)
        eigenvalues, vectors = eigenvalues[::-1].clip(min=0.0), vectors[:, ::-1]
    else:
        wide = matrix.astype(np.float64)
        eigenvalues, left_vectors = np.linalg.eigh(wide @ wide.T)
        eigenvalues, left_vectors = eigenvalues[::-1].clip(min=0.0), left_vectors[:, ::-1]
    if k is None:
        cumulative = np.cumsum(eigenvalues)
        k = int(np.searchsorted(cumulative, cumulative[-1] / 2.0)) + 1
    if cols <= rows:
        return vectors[:, :k], k
    kept = eigenvalues[:k] > 0.0
    basis = (wide.T @ left_vectors[:, :k][:, kept]) / np.sqrt(eigenvalues[:k][kept])
    return basis, k


def _energy_inside(matrix: np.ndarray, basis: np.ndarray) -> float:
    rows = max(1, _CHUNK_ELEMENTS // matrix.shape[1])
    total = 0.0
    for start in range(0, matrix.shape[0], rows):
        projected = matrix[start : start + rows].astype(np.float64) @ basis
        total += float(np.sum(projected * projected))
    return total


def subspace_overlap(delta: Iterable[Block], reference: Iterable[Block], *, seed: int) -> Keyed[SubspaceOverlap]:
    """Share of ``delta``'s energy inside ``reference``'s top singular directions.

    Per matrix (each 2-D tensor, and each expert of a stacked routed-expert
    tensor), k is the number of right singular directions holding 50% of the
    reference matrix's energy, and the delta's energy is projected onto them.
    Two nulls use the same k: a random k-dimensional subspace, and the top-k
    directions of the reference with its entries permuted within the matrix.
    1-D tensors (norm scales, the selection bias) have no subspace and are
    skipped. See :class:`SubspaceOverlap` for the aggregation.
    """
    generator = np.random.default_rng(seed)
    overlaps = Keyed(overall=SubspaceOverlap())
    for delta_block, reference_block in aligned(delta, reference):
        if delta_block.array.ndim != 2:
            continue
        matrix = delta_block.array
        targets = overlaps.targets(delta_block, SubspaceOverlap)
        if sum_squares(reference_block.array) == 0.0:
            for target in targets:
                target.zero_reference_matrices += 1
            continue
        energy = sum_squares(matrix)
        inside = inside_random = inside_permuted = 0.0
        if energy > 0.0:
            basis, k = _top_right_singular_basis(reference_block.array, None)
            inside = _energy_inside(matrix, basis)
            random_basis, _ = np.linalg.qr(generator.standard_normal((matrix.shape[1], k)))
            inside_random = _energy_inside(matrix, random_basis)
            permuted = generator.permutation(reference_block.array.reshape(-1)).reshape(matrix.shape)
            permuted_basis, _ = _top_right_singular_basis(permuted, k)
            inside_permuted = _energy_inside(matrix, permuted_basis)
        for target in targets:
            target.delta_energy += energy
            target.inside += inside
            target.inside_random += inside_random
            target.inside_permuted += inside_permuted
            target.matrices += 1
    return overlaps
