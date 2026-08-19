# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""A trajectory head for the 39-bucket surrogate: model the increment, not the endpoint (ATOM-023).

ATOM-022 established, after a correction, that the phase-0 readout is worth using and that the right
coefficient on it DEPENDS ON SCALE: near 1 across aggregate cells, where a plain difference works; about
0.19 within cells, where the ordering question lives; about 0.28 for the run noise, because the readout is
taken far from convergence and is twice as noisy as the endpoint. A single global coefficient serves none
of the three well, and pinning it to 1 everywhere inflates the noise by 73%.

The head here is the shape that nests both ends. The target is the INCREMENT `y1 - y0`, which is the
difference model, and the readout is ALSO carried as a free-sign column placed in the shrunk region of the
design, so its fitted amplitude is `beta - 1`. Shrinking that toward zero shrinks toward pure differencing;
letting it grow recovers the free-coefficient regression. The sign is free because a non-negative solve
cannot otherwise express `beta < 1`, and the pair `(+y0, -y0)` under a non-negativity constraint is exactly
one free coefficient.

Three arms, identical in every other respect:

  endpoint    the committed model, target y1, no readout
  endpoint+y0 target y1, readout carried as a shrunk free-sign column
  trajectory  target y1 - y0, readout carried the same way, predictions converted back to y1 before scoring

Scoring is leave-cell-out over aggregate cells and always on the y1 scale, so the three are comparable.
Theta is selected once per arm on all rows under an identical protocol and only the head is refit per fold;
that is mildly optimistic in the same way for all three, and isolates the target-and-feature question from
noise in the parameter search.

Usage: ``uv run python ... [--scale delphi_3e18] [--folds 5]``
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as gen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_swarm39_phase0_20260817 as phase0,
)

ARMS = ("endpoint", "endpoint+y0", "trajectory", "scale-separated")


def load(scale: str):
    """Policies, endpoint, and phase-0 readout for every row that has all three.

    The three swarms record mixtures differently -- delphi's heldout manifest stores them as JSON keyed by
    bucket name, while the 60M audit and the 300M late-boundary audit use one wide column per bucket -- so
    the shapes are normalised here rather than at every call site.
    """
    fit, _held = swarm39.load_scale("delphi_3e18" if scale == "delphi_3e18" else scale)
    panel, output = phase0.sources(scale)
    readouts = pd.read_csv(output)

    if scale == "delphi_3e18":
        merged = panel.merge(readouts, on="heldout_id", how="inner")
        merged = merged[merged["phase0_uncheatable_bpb"].notna()].reset_index(drop=True)

        def vector(payload: str) -> np.ndarray:
            mapping = json.loads(payload)
            return np.asarray([float(mapping.get(bucket, 0.0)) for bucket in fit.buckets], dtype=float)

        phase_0 = np.stack([vector(x) for x in merged["phase_0_weights_json"]])
        phase_1 = np.stack([vector(x) for x in merged["phase_1_weights_json"]])
        endpoint = merged["uncheatable_bpb"].to_numpy(float)
    elif scale == "60m":
        wide = pd.concat(
            [pd.read_csv(phase0.SIXTY / f"{n}.csv") for n in ("fit_two_phase", "heldout_observations")]
        ).drop_duplicates("run_name")
        wide = wide.merge(readouts.rename(columns={"heldout_id": "run_name"}), on="run_name", how="inner")
        merged = wide[wide["uncheatable_bpb"].notna() & wide["phase0_uncheatable_bpb"].notna()].reset_index(drop=True)
        phase_0 = merged[[f"phase_0_{b}" for b in fit.buckets]].to_numpy(float)
        phase_1 = merged[[f"phase_1_{b}" for b in fit.buckets]].to_numpy(float)
        endpoint = merged["uncheatable_bpb"].to_numpy(float)
    else:
        # The 300M audit supplies run names and mixtures but only per-component evaluations, and a plain
        # mean of those reproduces the canonical macro only to 4.2e-3 -- above this target's run noise of
        # 9.1e-4, so it is NOT the macro's definition and must not be substituted for it. The canonical
        # panels carry the true value, and every one of their 280 two-phase rows matches an audit row on
        # its exact weight vector, so the endpoint is JOINED rather than reconstructed.
        audit = pd.read_csv(phase0.THREE_HUNDRED)
        audit = audit.merge(readouts.rename(columns={"heldout_id": "run_name"}), on="run_name", how="inner")
        columns_0 = [f"phase_0_{b}" for b in fit.buckets]
        columns_1 = [f"phase_1_{b}" for b in fit.buckets]
        by_weights = {
            tuple(np.round(row, 6)): index for index, row in enumerate(audit[columns_0 + columns_1].to_numpy(float))
        }
        canonical = pd.concat(
            [pd.read_csv(phase0.CANONICAL / f"300m_{name}.csv") for name in ("two_phase_fit", "heldouts")]
        )
        canonical_weights = np.column_stack(
            [
                canonical[[f"phase_0_weight::{b}" for b in fit.buckets]].to_numpy(float),
                canonical[[f"phase_1_weight::{b}" for b in fit.buckets]].to_numpy(float),
            ]
        )
        pairs = [
            (row, by_weights[tuple(np.round(weights, 6))])
            for row, weights in enumerate(canonical_weights)
            if tuple(np.round(weights, 6)) in by_weights
        ]
        canonical_rows, audit_rows = (np.array(x) for x in zip(*pairs, strict=True))
        merged = audit.iloc[audit_rows].reset_index(drop=True)
        phase_0 = merged[columns_0].to_numpy(float)
        phase_1 = merged[columns_1].to_numpy(float)
        endpoint = canonical.iloc[canonical_rows]["uncheatable_bpb"].to_numpy(float)

    keep = np.isfinite(endpoint)
    phase_0, phase_1, endpoint = phase_0[keep], phase_1[keep], endpoint[keep]
    weights = np.stack([phase_0, phase_1], axis=1)
    return (
        gen.Panel(weights, fit.c0, fit.c1, fit.family_index),
        endpoint,
        merged["phase0_uncheatable_bpb"].to_numpy(float)[keep],
        fit.alpha * phase_0 + (1.0 - fit.alpha) * phase_1,
    )


def columns(panel: gen.Panel, shape, readout: np.ndarray | None):
    """The committed design, with the readout inserted at the head of the SHRUNK region when used."""
    free, constrained = gen.design(panel, shape, "blended")
    if readout is None:
        return free, constrained, gen.pooled_width(panel)
    pooled = gen.pooled_width(panel)
    extended = np.column_stack([constrained[:, :pooled], readout, -readout, constrained[:, pooled:]])
    return free, extended, pooled


def fit_arm(panel, endpoint, readout, arm, folds, seed, between=None):
    """Leave-cell-out predictions of the ENDPOINT, whatever the arm's internal target is.

    `scale-separated` is the arm the measurements actually point at. Differencing pins the readout's
    coefficient at 1, which is right ACROSS aggregate cells and wrong within them; carrying the readout as
    a shrunk column pulls the coefficient toward 0, which is right within cells and wrong across them. So
    the readout is split into its cell mean, which is differenced out at weight 1, and its within-cell
    deviation, which is carried as a shrunk free-sign column and finds its own smaller weight. Both parts
    are computable from a measured readout, since the cell is determined by the policy's aggregate.
    """
    use = None if arm == "endpoint" else readout
    target = endpoint
    if arm == "trajectory":
        target = endpoint - readout
    elif arm == "scale-separated":
        target = endpoint - between
        use = readout - between

    def build(rows, shape):
        subset = gen.Panel(panel.weights[rows], panel.epochs_early, panel.epochs_late, panel.family_index)
        return columns(subset, shape, None if use is None else use[rows])

    def objective(vector):
        shape, ridge = gen.unpack(vector, panel.n_families)
        total = 0.0
        for train, test in folds:
            free, constrained, pooled = build(train, shape)
            offsets, amplitudes = gen.fit_head(free, constrained, target[train], ridge, pooled)
            free_t, constrained_t, _ = build(test, shape)
            residual = free_t @ offsets + constrained_t @ amplitudes - target[test]
            if not np.isfinite(residual).all():
                return 1e6
            total += float(residual @ residual)
        return total

    vector = differential_evolution(
        objective,
        list(gen.bounds(panel.n_families)),
        rng=np.random.default_rng(20260817 + seed),
        popsize=8,
        maxiter=12,
        tol=1e-12,
        polish=True,
        init="sobol",
    ).x
    shape, ridge = gen.unpack(vector, panel.n_families)
    predicted = np.empty(len(endpoint))
    for train, test in folds:
        free, constrained, pooled = build(train, shape)
        offsets, amplitudes = gen.fit_head(free, constrained, target[train], ridge, pooled)
        free_t, constrained_t, _ = build(test, shape)
        predicted[test] = free_t @ offsets + constrained_t @ amplitudes
    if arm == "trajectory":
        return predicted + readout
    if arm == "scale-separated":
        return predicted + between
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="delphi_3e18")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    panel, endpoint, readout, aggregate = load(args.scale)
    cells = collections.defaultdict(list)
    for index, row in enumerate(np.round(aggregate, 6)):
        cells[tuple(row)].append(index)
    blocks = [np.array(v) for v in cells.values()]
    order = np.random.default_rng(args.seed).permutation(len(blocks))
    folds = []
    for k in range(args.folds):
        test = np.concatenate([blocks[j] for j in order[k :: args.folds]])
        folds.append((np.setdiff1d(np.arange(len(endpoint)), test), test))

    untied = 0.5 * np.abs(panel.weights[:, 1] - panel.weights[:, 0]).sum(axis=1) > 1e-9
    print(f"ATOM-023 trajectory head: {args.scale}, {len(endpoint)} rows, {len(blocks)} aggregate cells")
    print(f"{int(untied.sum())} untied; endpoint sd {endpoint.std():.5f}, readout sd {readout.std():.5f}\n")
    print(f"{'arm':14s} {'leave-cell-out RMSE on y1':>26s} {'vs endpoint':>12s} {'within-cell rho':>16s}")

    between = np.empty(len(readout))
    for rows in blocks:
        between[rows] = readout[rows].mean()

    baseline = None
    for arm in ARMS:
        predicted = fit_arm(panel, endpoint, readout, arm, folds, args.seed, between)
        rmse = float(np.sqrt(np.mean((predicted - endpoint) ** 2)))
        baseline = rmse if baseline is None else baseline
        rhos = []
        for rows in blocks:
            base, alternatives = rows[~untied[rows]], rows[untied[rows]]
            if len(base) == 0 or len(alternatives) < 4:
                continue
            observed = endpoint[alternatives] - endpoint[base].mean()
            expected = predicted[alternatives] - predicted[base].mean()
            rhos.append(spearmanr(expected, observed).statistic)
        rhos = np.array([r for r in rhos if not math.isnan(r)])
        print(
            f"{arm:14s} {rmse:26.6f} {rmse / baseline:11.3f}x "
            f"{np.median(rhos):9.3f} ({int((rhos > 0).sum())}/{len(rhos)})"
        )


if __name__ == "__main__":
    main()
