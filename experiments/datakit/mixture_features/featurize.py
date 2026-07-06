# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-function featurization for the mixing-via-embeddings experiment.

Everything here is I/O-free numpy algebra over frozen artifacts. The persisted objects
(:class:`MixtureBasis`, :class:`DomainHistogram`, :class:`BucketStats`) are produced by
``build_domain_histograms.py`` and consumed by the H1/H2 drivers. Two histograms are
comparable iff their :class:`MixtureBasis` is byte-identical (compared by content hashes);
combining unequal bases raises :class:`BasisMismatchError`.

The random-Fourier-feature (RFF) map is defined here too (``build_rff_map`` / ``rff_features``)
so the histogram builder and the featurizer share one implementation; the map identity
(``rff_dim`` / ``rff_seed`` / ``rff_bandwidth``) is part of basis equality.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BasisMismatchError(ValueError):
    """Raised when histograms/features under unequal :class:`MixtureBasis` are combined."""


class InsufficientSampleError(ValueError):
    """Raised when a domain yielded fewer than the minimum sampled documents."""


# ---------------------------------------------------------------------------
# Basis identity + persisted dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureBasis:
    """Identity of the frozen space a histogram is expressed in.

    Equality is by content, so a re-uploaded centroid file with different bytes is a
    different basis. Every persisted artifact embeds this identity.
    """

    embedder: str
    tokenizer: str
    centroids_path: str
    centroids_sha256: str
    k: int
    view_paths: Mapping[int, str]
    view_sha256: Mapping[int, str]
    quality_scorer: str | None
    quality_scorer_sha256: str | None
    rff_dim: int
    rff_seed: int
    rff_bandwidth: float

    def _identity(self) -> tuple:
        # Path strings are excluded: identity is by content hash, not location.
        return (
            self.embedder,
            self.tokenizer,
            self.centroids_sha256,
            self.k,
            tuple(sorted((int(k), v) for k, v in self.view_sha256.items())),
            self.quality_scorer,
            self.quality_scorer_sha256,
            self.rff_dim,
            self.rff_seed,
            round(float(self.rff_bandwidth), 9),
        )

    def matches(self, other: "MixtureBasis") -> bool:
        return self._identity() == other._identity()


def require_same_basis(histograms: "Sequence[DomainHistogram]") -> "MixtureBasis":
    """Return the shared basis or raise :class:`BasisMismatchError`."""
    if not histograms:
        raise ValueError("no histograms given")
    basis = histograms[0].basis
    for h in histograms[1:]:
        if not basis.matches(h.basis):
            raise BasisMismatchError(f"{h.domain} basis differs from {histograms[0].domain}")
    return basis


@dataclass(frozen=True)
class BucketStats:
    """Per-bucket scalars embeddings cannot see but that stay computable for any new bucket."""

    total_tokens_available: int
    mean_doc_tokens: float
    duplicate_frac: float | None
    loss_masked_frac: float


@dataclass(frozen=True)
class DomainHistogram:
    """Token-weighted content histogram for one domain over a frozen basis."""

    domain: str
    basis: MixtureBasis
    sample_size: int
    token_count: int
    seed: int
    counts: Mapping[tuple[int, int], int]  # (cluster_id, quality_bucket) -> eligible tokens
    rff_mean: tuple[float, ...]
    stats: BucketStats

    def fine_frac(self) -> np.ndarray:
        """Token-mass distribution over the K=5000 fine cells (sums to 1)."""
        v = np.zeros(self.basis.k, dtype=np.float64)
        for (cluster_id, _q), tokens in self.counts.items():
            v[cluster_id] += tokens
        total = v.sum()
        if total <= 0:
            raise ValueError(f"{self.domain}: empty histogram")
        return v / total


# ---------------------------------------------------------------------------
# Random Fourier features (cluster-free arm)
# ---------------------------------------------------------------------------


def build_rff_map(rff_dim: int, input_dim: int, seed: int, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(W, b)`` for ``phi(x) = sqrt(2/D) cos(W x + b)``.

    ``W ~ N(0, 1/bandwidth^2)`` of shape ``(rff_dim, input_dim)`` and ``b ~ U[0, 2 pi)`` of
    shape ``(rff_dim,)``, both drawn from a ``seed``-seeded generator. ``bandwidth`` is the
    median-heuristic sigma frozen at basis creation, so the approximated RBF kernel is
    ``exp(-||x - y||^2 / (2 bandwidth^2))``.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rff_dim, input_dim)).astype(np.float64) / bandwidth
    b = rng.uniform(0.0, 2.0 * np.pi, size=rff_dim).astype(np.float64)
    return w, b


def rff_features(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map ``x`` of shape ``(n, input_dim)`` to ``(n, rff_dim)`` random Fourier features."""
    x = np.asarray(x, dtype=np.float64)
    d = w.shape[0]
    return np.sqrt(2.0 / d) * np.cos(x @ w.T + b)


# ---------------------------------------------------------------------------
# Feature families / phase handling
# ---------------------------------------------------------------------------


class FeatureFamily(StrEnum):
    HIST_K40 = "hist_k40"
    HIST_K1000 = "hist_k1000"
    HIST_K5000 = "hist_k5000"
    KME_MEAN = "kme_mean"
    RFF_MEAN = "rff_mean"
    QUALITY_MASS = "quality_mass"
    BUCKET_STATS = "bucket_stats"
    EXPOSURE_GLOBAL = "exposure_global"
    EXPOSURE_BUCKET = "exposure_bucket"


class PhaseHandling(StrEnum):
    PER_PHASE = "per_phase"
    POOLED = "pooled"


_HIST_FAMILY_K = {
    FeatureFamily.HIST_K40: 40,
    FeatureFamily.HIST_K1000: 1000,
    FeatureFamily.HIST_K5000: 5000,
}


# ---------------------------------------------------------------------------
# Composition matrix + coarsening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionDiagnostics:
    """Rank / conditioning of a composition matrix (H1 consumes these)."""

    shape: tuple[int, int]
    singular_values: np.ndarray
    numerical_rank: int
    condition_number: float


class CompositionMatrix(np.ndarray):
    """``np.ndarray`` subclass carrying ``.diagnostics`` (rank / singular spectrum)."""

    diagnostics: CompositionDiagnostics

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.diagnostics = getattr(obj, "diagnostics", None)


def coarsen_fine(fine: np.ndarray, lookup: np.ndarray, k: int) -> np.ndarray:
    """Scatter-add a fine (K=5000) mass vector/column-stack into ``k`` coarse cells via ``lookup``.

    ``fine`` is ``(5000,)`` or ``(5000, n)``; ``lookup`` maps each fine cell to its coarse cell.
    """
    lookup = np.asarray(lookup)
    if fine.ndim == 1:
        return np.bincount(lookup, weights=fine, minlength=k).astype(fine.dtype)
    out = np.zeros((k, fine.shape[1]), dtype=fine.dtype)
    for j in range(fine.shape[1]):
        out[:, j] = np.bincount(lookup, weights=fine[:, j], minlength=k)
    return out


def composition_matrix(
    histograms: "Sequence[DomainHistogram]",
    k: int,
    views: Mapping[int, np.ndarray] | None = None,
) -> tuple[CompositionMatrix, list[str]]:
    """Stack per-domain fracs into ``V`` of shape ``(k_cells, n_domains)`` at granularity ``k``.

    Coarsens from K=5000 via ``views[k]`` (the fine->coarse lookup array) when ``k < 5000``;
    ``views`` must contain ``k``. Column order is ``sorted(domain)``. Raises
    :class:`BasisMismatchError` on basis disagreement. The returned array carries ``.diagnostics``.
    """
    basis = require_same_basis(histograms)
    if k > basis.k:
        raise ValueError(f"k={k} exceeds basis.k={basis.k}")
    order = sorted(h.domain for h in histograms)
    by_domain = {h.domain: h for h in histograms}
    fine = np.stack([by_domain[d].fine_frac() for d in order], axis=1)  # (5000, n)

    if k == basis.k:
        v = fine
    else:
        if views is None or k not in views:
            raise ValueError(f"coarsening to k={k} requires views[{k}] lookup array")
        v = coarsen_fine(fine, np.asarray(views[k]), k)

    sv = np.linalg.svd(v, compute_uv=False)
    tol = sv.max() * max(v.shape) * np.finfo(sv.dtype).eps if sv.size else 0.0
    numerical_rank = int((sv > tol).sum())
    cond = float(sv.max() / sv[sv > 0].min()) if np.any(sv > 0) else float("inf")
    out = np.ascontiguousarray(v).view(CompositionMatrix)
    out.diagnostics = CompositionDiagnostics(
        shape=(v.shape[0], v.shape[1]),
        singular_values=sv,
        numerical_rank=numerical_rank,
        condition_number=cond,
    )
    return out, order


def mixture_histogram(weights: Mapping[str, float], v: np.ndarray, domain_order: list[str]) -> np.ndarray:
    """``h = V @ w``. Missing domains get weight 0; raises on unknown domain or non-distribution."""
    w = np.zeros(len(domain_order), dtype=np.float64)
    index = {d: i for i, d in enumerate(domain_order)}
    for domain, weight in weights.items():
        if domain not in index:
            raise ValueError(f"unknown domain in weights: {domain!r}")
        w[index[domain]] = weight
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"weights do not sum to 1 (got {w.sum():.6f})")
    return np.asarray(v) @ w


# ---------------------------------------------------------------------------
# Control featurizations
# ---------------------------------------------------------------------------


def shuffled_columns_v(v: np.ndarray, seed: int) -> np.ndarray:
    """Permute the domain -> histogram-column mapping (destroys alignment, keeps geometry)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(v.shape[1])
    return np.asarray(v)[:, perm]


def matched_random_v(v: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute cell indices *within each column*.

    Every column keeps its exact mass profile (non-negative, sums to 1, same entropy/sparsity),
    so the result stays a valid histogram for distance-based predictors, but the shared cell
    coordinate system -- and hence all cross-column content similarity -- is destroyed.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        out[:, j] = v[rng.permutation(v.shape[0]), j]
    return out


# ---------------------------------------------------------------------------
# Run-level features
# ---------------------------------------------------------------------------


class RunLike(Protocol):
    """Structural view of a swarm run (matches ``swarm_runs.SwarmRun``)."""

    model_size: int
    phase_weights: tuple[Mapping[str, float], ...]
    phase_tokens: tuple[int, ...]
    domain_tokens: tuple[Mapping[str, float], ...]


def _pooled_weights(run: RunLike) -> dict[str, float]:
    total = float(sum(run.phase_tokens))
    pooled: dict[str, float] = {}
    for pw, t in zip(run.phase_weights, run.phase_tokens, strict=True):
        for d, w in pw.items():
            pooled[d] = pooled.get(d, 0.0) + w * t / total
    return pooled


def _phase_rff_mean(domain_tokens: Mapping[str, float], histograms: "Sequence[DomainHistogram]") -> np.ndarray:
    by_domain = {h.domain: h for h in histograms}
    acc = np.zeros(len(histograms[0].rff_mean), dtype=np.float64)
    total = 0.0
    for d, tokens in domain_tokens.items():
        if tokens <= 0 or d not in by_domain:
            continue
        acc += tokens * np.asarray(by_domain[d].rff_mean, dtype=np.float64)
        total += tokens
    if total <= 0:
        raise ValueError("run has no positive domain token mass")
    return acc / total


def _family_vector(
    family: FeatureFamily,
    weights: Mapping[str, float],
    domain_tokens: Mapping[str, float],
    histograms: "Sequence[DomainHistogram]",
    v_fine: np.ndarray,
    order: list[str],
    views: Mapping[int, np.ndarray] | None,
    centroids: np.ndarray | None,
    phase_tokens_total: int,
    model_size: int,
) -> np.ndarray:
    if family in _HIST_FAMILY_K:
        k = _HIST_FAMILY_K[family]
        v = v_fine if k == v_fine.shape[0] else coarsen_fine(v_fine, np.asarray(views[k]), k)  # type: ignore[index]
        return mixture_histogram(weights, v, order)
    if family is FeatureFamily.KME_MEAN:
        if centroids is None:
            raise ValueError("KME_MEAN requires centroids")
        h = mixture_histogram(weights, v_fine, order)  # (5000,)
        return h @ np.asarray(centroids, dtype=np.float64)
    if family is FeatureFamily.RFF_MEAN:
        return _phase_rff_mean(domain_tokens, histograms)
    if family is FeatureFamily.EXPOSURE_GLOBAL:
        # Transferable global-exposure scalars: model size + this phase's token budget.
        return np.array([float(model_size), float(phase_tokens_total)], dtype=np.float64)
    if family is FeatureFamily.BUCKET_STATS:
        by_domain = {h.domain: h for h in histograms}
        mean_doc = sum(weights.get(d, 0.0) * by_domain[d].stats.mean_doc_tokens for d in order if d in by_domain)
        loss_masked = sum(weights.get(d, 0.0) * by_domain[d].stats.loss_masked_frac for d in order if d in by_domain)
        return np.array([mean_doc, loss_masked], dtype=np.float64)
    if family in (FeatureFamily.QUALITY_MASS, FeatureFamily.EXPOSURE_BUCKET):
        raise NotImplementedError(f"{family} is locked in v0 (no quality axis / diagnostic-only)")
    raise ValueError(f"unhandled feature family: {family}")


def run_features(
    run: RunLike,
    histograms: "Sequence[DomainHistogram]",
    families: Sequence[FeatureFamily],
    phases: PhaseHandling = PhaseHandling.PER_PHASE,
    *,
    views: Mapping[int, np.ndarray] | None = None,
    centroids: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate the requested families under the given phase handling.

    Deterministic ordering: families as given, cells in index order, phases in order.
    ``views`` supplies coarsening lookups for HIST_K40/K1000; ``centroids`` is required for KME_MEAN.
    """
    v_fine, order = composition_matrix(histograms, k=histograms[0].basis.k)

    if phases is PhaseHandling.POOLED:
        pooled_w = _pooled_weights(run)
        pooled_tokens = {d: pooled_w[d] * sum(run.phase_tokens) for d in pooled_w}
        phase_specs = [(pooled_w, pooled_tokens, int(sum(run.phase_tokens)))]
    else:
        phase_specs = [
            (dict(pw), dict(dt), int(t))
            for pw, dt, t in zip(run.phase_weights, run.domain_tokens, run.phase_tokens, strict=True)
        ]

    blocks: list[np.ndarray] = []
    for weights, domain_tokens, ptokens in phase_specs:
        for family in families:
            blocks.append(
                _family_vector(
                    family,
                    weights,
                    domain_tokens,
                    histograms,
                    np.asarray(v_fine),
                    order,
                    views,
                    centroids,
                    ptokens,
                    run.model_size,
                )
            )
    return np.concatenate(blocks) if blocks else np.zeros(0)


# ---------------------------------------------------------------------------
# Inline self-test of the algebra
# ---------------------------------------------------------------------------


def _selftest() -> None:
    rng = np.random.default_rng(0)
    k_fine, n_dom, k_coarse = 5000, 6, 40
    lookup = rng.integers(0, k_coarse, size=k_fine).astype(np.int32)

    bandwidth = 3.0
    basis = MixtureBasis(
        embedder="test",
        tokenizer="test-tok",
        centroids_path="mem://c",
        centroids_sha256="deadbeef",
        k=k_fine,
        view_paths={40: "mem://40", 1000: "mem://1000"},
        view_sha256={40: "aa", 1000: "bb"},
        quality_scorer=None,
        quality_scorer_sha256=None,
        rff_dim=8,
        rff_seed=0,
        rff_bandwidth=bandwidth,
    )

    def make_hist(domain: str) -> DomainHistogram:
        # sparse random fine mass
        cells = rng.choice(k_fine, size=200, replace=False)
        toks = rng.integers(1, 100, size=200)
        counts = {(int(c), -1): int(t) for c, t in zip(cells, toks, strict=True)}
        return DomainHistogram(
            domain=domain,
            basis=basis,
            sample_size=1000,
            token_count=int(toks.sum()),
            seed=0,
            counts=counts,
            rff_mean=tuple(rng.standard_normal(8)),
            stats=BucketStats(10**9, 42.0, None, 0.0),
        )

    hists = [make_hist(f"dom_{i}") for i in range(n_dom)]
    views = {40: lookup}

    # 1. h = V @ w composition + coarsening consistency.
    v_fine, order = composition_matrix(hists, k=k_fine)
    v40, order40 = composition_matrix(hists, k=k_coarse, views=views)
    assert order == order40
    assert v40.shape == (k_coarse, n_dom)
    # each fine column sums to 1; coarse column sums to 1 too.
    np.testing.assert_allclose(v_fine.sum(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(v40.sum(axis=0), 1.0, atol=1e-9)
    # coarse hist == scatter-add of fine hist (per column).
    expected40 = coarsen_fine(np.asarray(v_fine), lookup, k_coarse)
    np.testing.assert_allclose(np.asarray(v40), expected40, atol=1e-12)

    # mixture: coarsen-then-mix == mix-then-coarsen.
    w = rng.random(n_dom)
    w = w / w.sum()
    weights = dict(zip(order, w, strict=True))
    h_fine = mixture_histogram(weights, np.asarray(v_fine), order)
    h40 = mixture_histogram(weights, np.asarray(v40), order)
    np.testing.assert_allclose(coarsen_fine(h_fine, lookup, k_coarse), h40, atol=1e-12)
    np.testing.assert_allclose(h40.sum(), 1.0, atol=1e-9)

    # 2. Permutation invariance: shuffling columns then mixing with the matching-permuted
    #    weights gives the same histogram.
    vs = shuffled_columns_v(np.asarray(v40), seed=3)
    perm = np.random.default_rng(3).permutation(n_dom)
    shuffled_order = [order[i] for i in perm]
    h_shuf = mixture_histogram(weights, vs, shuffled_order)
    np.testing.assert_allclose(h_shuf, h40, atol=1e-12)

    # 3. matched_random_v preserves each column's mass profile (sorted values identical).
    vm = matched_random_v(np.asarray(v40), seed=5)
    np.testing.assert_allclose(np.sort(vm, axis=0), np.sort(np.asarray(v40), axis=0), atol=1e-12)
    np.testing.assert_allclose(vm.sum(axis=0), 1.0, atol=1e-9)

    # 4. diagnostics present and sane.
    assert v40.diagnostics.numerical_rank <= min(k_coarse, n_dom)
    assert v40.diagnostics.singular_values.shape[0] == min(k_coarse, n_dom)

    # 5. RFF map: features have the right shape and bounded magnitude.
    wmap, bvec = build_rff_map(rff_dim=16, input_dim=4, seed=0, bandwidth=2.0)
    phi = rff_features(rng.standard_normal((5, 4)), wmap, bvec)
    assert phi.shape == (5, 16)
    assert np.all(np.abs(phi) <= np.sqrt(2.0 / 16) + 1e-9)

    # 6. run_features smoke over a tiny two-phase run.
    class _Run:
        model_size = 60_000_000
        phase_weights = (weights, weights)
        phase_tokens = (1000, 3000)
        domain_tokens = (
            {d: weights[d] * 1000 for d in order},
            {d: weights[d] * 3000 for d in order},
        )

    feats = run_features(
        _Run(),
        hists,
        [FeatureFamily.HIST_K40, FeatureFamily.RFF_MEAN, FeatureFamily.EXPOSURE_GLOBAL],
        phases=PhaseHandling.PER_PHASE,
        views=views,
    )
    # per-phase: (40 + 8 + 2) * 2 phases
    assert feats.shape == ((k_coarse + 8 + 2) * 2,), feats.shape
    pooled = run_features(
        _Run(),
        hists,
        [FeatureFamily.HIST_K40],
        phases=PhaseHandling.POOLED,
        views=views,
    )
    assert pooled.shape == (k_coarse,), pooled.shape

    # 7. basis mismatch raises.
    other = dataclasses.replace(hists[0], basis=dataclasses.replace(basis, centroids_sha256="ffff"))
    try:
        composition_matrix([hists[1], other], k=k_fine)
    except BasisMismatchError:
        pass
    else:
        raise AssertionError("expected BasisMismatchError")

    print("featurize self-test: all checks passed")


if __name__ == "__main__":
    _selftest()
