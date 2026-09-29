# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer diagnostics from ``grad_capture.py`` captures: what Online KL-Shampoo assumes, measured per matrix.

Online KL-Shampoo (OKLS) models the gradient covariance of an ``m x n`` matrix as a Kronecker product
``Sigma ~= B (x) A`` (``A``: ``m x m`` row factor, ``B``: ``n x n`` column factor), fits the factors by KL, and steps
along ``A^{-1/2} M B^{-1/2}`` for momentum ``M``, with factors refreshed every step. Muon instead steps along the
polar factor of ``M`` alone. Each claim behind OKLS gets a measurement, per captured matrix and capture window:

- ``spectra``: are the factors anisotropic at all (effective rank, condition number)? Isotropic factors make OKLS
  plain momentum SGD.
- ``whitening``: is the covariance Kronecker? Fit the KL factors on even steps and whiten the odd steps; a Kronecker
  covariance leaves isotropic residual row/column covariances, up to the finite-sample spread of a Gaussian with
  exactly that covariance (``whitening.gaussian``).
- ``drift``: how fast does the factors' eigenbasis move (first vs second half of a window, and across windows)?
  OKLS finds even a one-step-stale preconditioner harmful.
- ``snr``: in the row factor's eigenbasis, is the gradient's persistent (mean) component in the high-variance
  directions, which whitening shrinks, or in the low-variance ones it amplifies?
- ``directions``: momentum is rebuilt from the captured gradients (MuonH's Nesterov 0.95, after a warm-up) and
  turned into the Muon, OKLS, one-sided, and Shampoo directions. Each is scored by its first-order gain on the
  *next* steps' gradients (fresh batches, so only persistent signal counts) at unit Frobenius and unit spectral
  norm, plus its stable rank. The captured applied update validates the momentum rebuild (``cos_applied_muon``).
- ``weights``: the weight's stable and effective rank at each window start.

Usage: ``python -m experiments.grug.fast_track.analyze_grad_capture --root <run>/grad_capture --out metrics.json``.
Run it next to the data: a window of captures is several GB.
"""

import dataclasses
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import click
import fsspec
import numpy as np

logger = logging.getLogger(__name__)

MOMENTUM = 0.95
OKLS_BETA2 = 0.9482
MOMENTUM_WARMUP = 24
FUTURE_STEPS = 4
KL_ITERATIONS = 6
RIDGE = 1e-6
SNR_BINS = 8
_STEP_RE = re.compile(r"grad_capture_step(\d+)\.npz$")


@dataclasses.dataclass(frozen=True)
class Window:
    start: int
    steps: tuple[int, ...]


def windows_of(steps: list[int]) -> list[Window]:
    """Group sorted step numbers into runs of consecutive steps."""
    groups: list[list[int]] = []
    for step in sorted(steps):
        if groups and step == groups[-1][-1] + 1:
            groups[-1].append(step)
        else:
            groups.append([step])
    return [Window(start=g[0], steps=tuple(g)) for g in groups]


def _sym_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, RIDGE * max(values.max(), 1e-30))
    return (vectors * values**power) @ vectors.T


def _ridge(matrix: np.ndarray) -> np.ndarray:
    return matrix + RIDGE * np.trace(matrix) / matrix.shape[0] * np.eye(matrix.shape[0])


def shampoo_factors(grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``E[G G^T] / n`` and ``E[G^T G] / m`` over ``grads`` ``[T, m, n]``."""
    _, m, n = grads.shape
    return np.einsum("tij,tkj->ik", grads, grads) / (len(grads) * n), np.einsum("tji,tjk->ik", grads, grads) / (
        len(grads) * m
    )


def kl_factors(grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KL-optimal Kronecker factors of the samples' covariance (the batch fixed point OKLS tracks online):
    ``A = E[G B^-1 G^T] / n``, ``B = E[G^T A^-1 G] / m``, scaled so ``tr A = m``."""
    _, m, n = grads.shape
    a, b = shampoo_factors(grads)
    for _ in range(KL_ITERATIONS):
        b_inv = np.linalg.inv(_ridge(b))
        a = np.einsum("tij,jk,tlk->il", grads, b_inv, grads) / (len(grads) * n)
        a_inv = np.linalg.inv(_ridge(a))
        b = np.einsum("tji,jk,tkl->il", grads, a_inv, grads) / (len(grads) * m)
    scale = m / np.trace(a)
    return a * scale, b / scale


def spectrum_stats(matrix: np.ndarray) -> dict:
    values = np.clip(np.linalg.eigvalsh(matrix)[::-1], 0, None)
    p = values / values.sum()
    nonzero = p[p > 0]
    return {
        "effective_rank_frac": float(np.exp(-(nonzero * np.log(nonzero)).sum()) / len(values)),
        "log10_cond90": float(np.log10(values[0] / max(values[int(0.9 * (len(values) - 1))], 1e-30))),
        "top1_frac": float(p[0]),
        # 32 log-spaced quantiles of the normalized spectrum, for plotting.
        "profile": [
            float(x) for x in (values / values.mean())[np.unique(np.geomspace(1, len(values), 32).astype(int) - 1)]
        ],
    }


def _residual_anisotropy(whitened: np.ndarray) -> float:
    """log10(largest / median eigenvalue) of the residual row covariance, averaged with the column one."""
    _, m, n = whitened.shape
    left = np.einsum("tij,tkj->ik", whitened, whitened) / (len(whitened) * n)
    right = np.einsum("tji,tjk->ik", whitened, whitened) / (len(whitened) * m)
    out = []
    for c in (left, right):
        values = np.linalg.eigvalsh(c)
        out.append(np.log10(values[-1] / max(np.median(values), 1e-30)))
    return float(np.mean(out))


def whitening_test(grads: np.ndarray, rng: np.random.Generator) -> dict:
    fit, held = grads[0::2], grads[1::2]
    a, b = kl_factors(fit)
    a_is, b_is = _sym_power(a, -0.5), _sym_power(b, -0.5)
    sa, sb = shampoo_factors(fit)
    # A Gaussian with exactly the fitted Kronecker covariance, same sample count: the spread a perfect fit shows.
    a_half, b_half = _sym_power(a, 0.5), _sym_power(b, 0.5)
    gaussian = np.stack([a_half @ rng.standard_normal(g.shape) @ b_half for g in held])
    return {
        "none": _residual_anisotropy(held),
        "shampoo": _residual_anisotropy(np.einsum("ij,tjk,kl->til", _sym_power(sa, -0.5), held, _sym_power(sb, -0.5))),
        "kl": _residual_anisotropy(np.einsum("ij,tjk,kl->til", a_is, held, b_is)),
        "gaussian": _residual_anisotropy(np.einsum("ij,tjk,kl->til", a_is, gaussian, b_is)),
    }


def subspace_overlap(x: np.ndarray, y: np.ndarray, frac: float = 0.1) -> float:
    """Mean squared overlap of the top-``frac`` eigenvector subspaces of two symmetric matrices (1: identical)."""
    k = max(1, int(frac * x.shape[0]))
    ux = np.linalg.eigh(x)[1][:, -k:]
    uy = np.linalg.eigh(y)[1][:, -k:]
    return float(np.linalg.norm(ux.T @ uy) ** 2 / k)


def snr_by_eigenrank(grads: np.ndarray, a: np.ndarray) -> list[float]:
    """log10 SNR of the gradient's rows projected on the row factor's eigenvectors, binned by eigenvalue rank
    (bin 0: the largest eigenvalues). SNR = |mean|^2 / (variance / T), i.e. how far the window mean stands above
    the noise it would show by chance."""
    vectors = np.linalg.eigh(a)[1][:, ::-1]
    projected = np.einsum("im,tin->tmn", vectors, grads)
    mean = projected.mean(0)
    signal = (mean**2).sum(-1)
    noise = ((projected - mean) ** 2).sum((0, 2)) / (len(grads) - 1) / len(grads)
    ratio = np.log10(np.maximum(signal, 1e-30) / np.maximum(noise, 1e-30))
    return [float(chunk.mean()) for chunk in np.array_split(ratio, SNR_BINS)]


def _polar(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def _stable_rank(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    return float((s**2).sum() / s[0] ** 2)


def direction_scores(grads: np.ndarray, applied: np.ndarray) -> dict:
    """Score candidate update directions built from the rebuilt Nesterov momentum (see the module docstring)."""
    _, m, n = grads.shape
    buffer = np.zeros((m, n))
    a = np.eye(m) * 1e-12
    b = np.eye(n) * 1e-12
    sa = np.zeros((m, m))
    sb = np.zeros((n, n))
    names = ("sgd", "muon", "okls", "okls_left", "okls_right", "shampoo")
    scores = {name: {"gain_fro": [], "gain_spec": [], "stable_rank": []} for name in names}
    cos_applied, cos_okls = [], []
    for t in range(len(grads) - FUTURE_STEPS):
        g = grads[t]
        buffer = MOMENTUM * buffer + g
        nesterov = g + MOMENTUM * buffer
        # Online KL factor updates in OKLS's order: the row factor sees the gradient whitened on the right by the
        # current column factor, and vice versa.
        b_is = _sym_power(_ridge(b), -0.5) if t else np.eye(n)
        a = OKLS_BETA2 * a + (1 - OKLS_BETA2) * (g @ b_is) @ (g @ b_is).T / n
        a_is = _sym_power(_ridge(a), -0.5)
        b = OKLS_BETA2 * b + (1 - OKLS_BETA2) * (a_is @ g).T @ (a_is @ g) / m
        b_is = _sym_power(_ridge(b), -0.5)
        sa = OKLS_BETA2 * sa + (1 - OKLS_BETA2) * g @ g.T
        sb = OKLS_BETA2 * sb + (1 - OKLS_BETA2) * g.T @ g
        if t < MOMENTUM_WARMUP:
            continue
        candidates = {
            "sgd": nesterov,
            "muon": _polar(nesterov),
            "okls": a_is @ nesterov @ b_is,
            "okls_left": a_is @ nesterov,
            "okls_right": nesterov @ b_is,
            "shampoo": _sym_power(_ridge(sa), -0.25) @ nesterov @ _sym_power(_ridge(sb), -0.25),
        }
        future = grads[t + 1 : t + 1 + FUTURE_STEPS].sum(0)
        for name, d in candidates.items():
            fro = np.linalg.norm(d)
            spec = np.linalg.norm(d, 2)
            # A descent step moves along -d; gain is the first-order loss decrease on the future batches.
            scores[name]["gain_fro"].append(float((future * d).sum() / fro))
            scores[name]["gain_spec"].append(float((future * d).sum() / spec))
            scores[name]["stable_rank"].append(float(fro**2 / spec**2))
        muon = candidates["muon"]
        okls = candidates["okls"]
        cos_okls.append(float((okls * muon).sum() / (np.linalg.norm(okls) * np.linalg.norm(muon))))
        cos_applied.append(float(-(applied[t] * muon).sum() / (np.linalg.norm(applied[t]) * np.linalg.norm(muon))))
    out = {name: {k: float(np.mean(v)) for k, v in s.items()} for name, s in scores.items()}
    out["cos_applied_muon"] = float(np.mean(cos_applied))
    out["cos_okls_muon"] = float(np.mean(cos_okls))
    return out


def autocorrelation(grads: np.ndarray, lags: int = 8) -> list[float]:
    flat = grads.reshape(len(grads), -1)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-30)
    return [float(np.mean((flat[:-k] * flat[k:]).sum(1))) for k in range(1, lags + 1)]


def _effective_rank_frac(matrix: np.ndarray) -> float:
    s = np.linalg.svd(matrix, compute_uv=False)
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()) / len(s))


def analyze_site(grads: np.ndarray, applied: np.ndarray, param: np.ndarray | None, seed: int) -> dict:
    grads = grads.astype(np.float64)
    applied = applied.astype(np.float64)
    a, b = kl_factors(grads)
    half = len(grads) // 2
    a1, b1 = kl_factors(grads[:half])
    a2, b2 = kl_factors(grads[half:])
    out = {
        "shape": list(grads.shape[1:]),
        "grad_norm": float(np.linalg.norm(grads.reshape(len(grads), -1), axis=1).mean()),
        "spectra": {"row": spectrum_stats(a), "col": spectrum_stats(b)},
        "whitening": whitening_test(grads, np.random.default_rng(seed)),
        "drift": {"row_half_overlap": subspace_overlap(a1, a2), "col_half_overlap": subspace_overlap(b1, b2)},
        "snr_row": snr_by_eigenrank(grads, a),
        "autocorrelation": autocorrelation(grads),
        "directions": direction_scores(grads, applied),
        "grad_stable_rank": float(np.mean([_stable_rank(g) for g in grads[::8]])),
        "factors": (a, b),
    }
    if param is not None:
        out["weights"] = {"stable_rank": _stable_rank(param), "effective_rank_frac": _effective_rank_frac(param)}
    return out


def _load(path: str) -> dict[str, np.ndarray]:
    with fsspec.open(path, "rb") as f:
        with np.load(f) as data:
            return {k: data[k] for k in data.files}


def analyze(root: str, sites: tuple[str, ...] | None = None) -> dict:
    fs, base = fsspec.core.url_to_fs(root)
    paths = {int(m.group(1)): p for p in fs.ls(base) if (m := _STEP_RE.search(p))}
    protocol = root.split("://")[0] + "://" if "://" in root else ""
    results: dict = {"windows": {}}
    previous_factors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for window in windows_of(list(paths)):
        with ThreadPoolExecutor(max_workers=8) as pool:
            steps = list(pool.map(lambda s: _load(protocol + paths[s]), window.steps))
        names = sorted(k[len("grad/") :] for k in steps[0] if k.startswith("grad/"))
        if sites is not None:
            names = [n for n in names if n in sites]
        logger.info("window %d: %d steps, %d sites", window.start, len(steps), len(names))
        per_site = {}
        for i, name in enumerate(names):
            grads = np.stack([s[f"grad/{name}"] for s in steps])
            applied = np.stack([s[f"update/{name}"] for s in steps])
            site = analyze_site(grads, applied, steps[0].get(f"param/{name}"), seed=i)
            a, b = site.pop("factors")
            if name in previous_factors:
                pa, pb = previous_factors[name]
                site["drift"]["row_prev_window_overlap"] = subspace_overlap(pa, a)
                site["drift"]["col_prev_window_overlap"] = subspace_overlap(pb, b)
            previous_factors[name] = (a, b)
            per_site[name] = site
        results["windows"][str(window.start)] = per_site
        del steps
    return results


@click.command()
@click.option("--root", required=True, help="The run's grad_capture directory (fsspec URL).")
@click.option("--out", required=True, help="Where to write the metrics JSON (fsspec URL).")
@click.option("--sites", default="", help="Comma-separated site names to analyze (default: all).")
def main(root: str, out: str, sites: str) -> None:
    logging.basicConfig(level=logging.INFO)
    results = analyze(root, tuple(s for s in sites.split(",") if s) or None)
    with fsspec.open(out, "w") as f:
        json.dump(results, f)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
