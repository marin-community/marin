# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Reproduce the Wave-1 branch panel and audit whether the draft local Wave-2b design identifies its model.

Two jobs, in order. First reproduce every frozen fact in the handoff from the canonical materialization and
say plainly which ones do not reproduce. Second, ask whether the proposed 80-row local acquisition can
actually estimate the model it is designed for, which is the question that decides whether Wave 2 is worth
running at all.

The anchored critic is

    dL(z) = beta' z + alpha ||z||^2 + gamma R_rep(z),   dL(0) = 0

on a simplex-tangent coordinate z around the tied continuation. Antithetic pairs separate the channels:
the odd half `(L(rd) - L(-rd))/2 ~= r beta'd` carries direction value, and the even half
`(L(rd) + L(-rd))/2 - L(0) ~= alpha r^2 + gamma R_rep` carries curvature and repetition. That separation is
the design's whole justification, so the audit checks the two things that can break it: whether the fit
directions span enough of the tangent space to see beta, and whether alpha and gamma are separable given
the radii actually proposed rather than in principle.

Nothing here launches a job or writes to the panel. Paths default to the canonical worktrees named in the
handoff and can be overridden.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

WAVE2_WORKTREE = Path("/Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-wave2-acquisition-20260825")
LOCAL_WORKTREE = Path("/Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-local-wave2b-20260825")
REFERENCE = "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
WAVE1 = "delphi_phase1_kl0p05_wave1_results_20260825"
NOISE = "delphi_phase1_kl0p05_noise_results_20260825"
LOCAL = "delphi_phase1_kl0p05_local_wave2b_20260825"

EXPECTED_HASHES = {
    f"{WAVE1}/materialization_manifest.json": "0d3f239002637768f696decb095877591bd7406193ba7573ffa6fc0e87ed5ebc",
    f"{WAVE1}/branch_fit_matrix.csv": "399ec79150a4f88de6d31917ac7fc1807410804f69c84400611a9eeaa6636e3c",
    f"{WAVE1}/branch_results.csv": "8841baadddc90efa8da8fa95bd76bb31748a2973388380ea38ea3dfcbe94e54d",
    f"{WAVE1}/uncheatable_metrics_long.csv": "c9849f1283f3e943337fb69de5b249265bfe3b1b717c8a9fbc57e9e18b2a99c4",
    f"{NOISE}/materialization_manifest.json": "aefdf412fe73fb5cade0193c1061b2f942950d8e59da30428aa19dc647051e67",
}
EXPECTED = {
    "tied": 0.98989356,
    "best_fit": 1.00035429,
    "best_minus_tied": 0.01046073,
    "run_noise_sd": 0.00084656,
    "tv_min": 0.43359,
    "tv_median": 0.58667,
    "tv_max": 0.81006,
}
TOLERANCE = 5e-8
SIMPLEX_TANGENT_DIM = 38


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def total_variation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(left - right).sum(axis=-1)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(0.5 * ((np.sqrt(left) - np.sqrt(right)) ** 2).sum(axis=-1))


def reproduce(root: Path) -> dict[str, object]:
    """Every frozen fact in the handoff, recomputed. Returns findings rather than asserting."""
    findings: dict[str, object] = {}
    reference = root / REFERENCE
    findings["hashes"] = {
        name: "OK" if (reference / name).is_file() and sha256_of(reference / name) == want else "FAIL"
        for name, want in EXPECTED_HASHES.items()
    }

    results = pd.read_csv(reference / WAVE1 / "branch_results.csv")
    fit = pd.read_csv(reference / WAVE1 / "branch_fit_matrix.csv")
    findings["rows"] = {"results": len(results), "fit": len(fit)}

    phase0 = [c for c in results.columns if c.startswith("phase_0_")]
    phase1 = [c for c in results.columns if c.startswith("phase_1_")]
    prefix = results[phase0].to_numpy(float)
    findings["prefix_constant"] = bool(np.abs(prefix - prefix[0]).max() < 1e-12)

    tied_row = results[results.continuation_role == "tied_control"].iloc[0]
    tied_weights = tied_row[phase1].to_numpy(float)
    findings["tied_equals_prefix"] = float(total_variation(tied_weights, prefix[0]))
    findings["tied"] = float(tied_row.uncheatable_bpb)

    best = fit.loc[fit.uncheatable_bpb.idxmin()]
    findings["best_fit"] = float(best.uncheatable_bpb)
    findings["best_fit_id"] = str(best.continuation_id)
    findings["best_minus_tied"] = findings["best_fit"] - findings["tied"]
    findings["n_better_than_tied"] = int((fit.uncheatable_bpb < findings["tied"]).sum())
    findings["best_fit_cpp"] = float(best.github_cpp_bpb)
    findings["tied_cpp"] = float(tied_row.github_cpp_bpb)

    distance = total_variation(fit[phase1].to_numpy(float), tied_weights)
    findings["tv"] = {
        "min": float(distance.min()),
        "median": float(np.median(distance)),
        "max": float(distance.max()),
    }

    noise = pd.read_csv(reference / NOISE / "noise_results_n5.csv")
    noise = noise.assign(base=noise.continuation_id.str.replace(r"_noise\d+$", "", regex=True))
    grouped = noise.groupby("base").uncheatable_bpb.agg(["count", "mean", "std"])
    residual = noise.uncheatable_bpb - noise.groupby("base").uncheatable_bpb.transform("mean")
    findings["noise_groups"] = {
        str(name): {"n": int(row["count"]), "sd": float(row["std"])} for name, row in grouped.iterrows()
    }
    findings["run_noise_sd_conservative"] = float(grouped["std"].max())
    findings["run_noise_sd_pooled"] = float(np.sqrt((residual**2).sum() / (len(noise) - len(grouped))))

    controls = results[results.continuation_role != "tied_control"]
    controls = controls[~controls.continuation_role.str.contains("maximin")]
    findings["controls"] = [
        {
            "id": str(row.continuation_id),
            "role": str(row.continuation_role),
            "bpb": float(row.uncheatable_bpb),
            "tv_from_tied": float(total_variation(row[phase1].to_numpy(float), tied_weights)),
        }
        for _, row in controls.iterrows()
    ]
    return findings


def audit_design(root: Path) -> dict[str, object]:
    """Can the proposed 80-row panel estimate beta, alpha and gamma?"""
    output = root / REFERENCE / LOCAL
    directions = pd.read_csv(output / "directions.csv")
    summary = pd.read_csv(output / "continuation_summary.csv")

    clr = [c for c in directions.columns if c.startswith("clr_direction_")]
    fit_directions = directions[directions.direction_id.isin(set(summary[summary.fit_budget == 1].direction_id))]
    matrix = fit_directions[clr].to_numpy(float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int((singular > singular[0] * 1e-10).sum())

    findings: dict[str, object] = {
        "n_directions_total": len(directions),
        "n_fit_directions": len(fit_directions),
        "directional_rank": rank,
        "tangent_dim": SIMPLEX_TANGENT_DIM,
        "spanned_fraction": rank / SIMPLEX_TANGENT_DIM,
        "unspanned_dims": SIMPLEX_TANGENT_DIM - rank,
        "family_counts": fit_directions.direction_family.value_counts().to_dict(),
        "outcome_status": fit_directions.direction_outcome_status.value_counts().to_dict(),
        "clr_matrix": matrix,
    }

    fit_rows = summary[summary.fit_budget == 1]
    findings["n_fit_rows"] = len(fit_rows)
    findings["radii_per_direction"] = (
        fit_rows.groupby("direction_id").target_hellinger.nunique().value_counts().sort_index().to_dict()
    )

    # Even-channel identifiability: alpha rides r^2, gamma rides a label-blind repetition channel. If the
    # two columns are collinear over the rows actually proposed, no amount of data separates them.
    positive = fit_rows[fit_rows.sign == "plus"] if "sign" in fit_rows.columns else fit_rows
    radius = positive.achieved_hellinger_to_tied.to_numpy(float)
    repetition = np.maximum(positive.max_total_materialized_epoch.to_numpy(float) - 1.0, 0.0)
    even = np.column_stack([radius**2, repetition])
    even = even / np.maximum(np.linalg.norm(even, axis=0), 1e-300)
    findings["even_channel"] = {
        "n_rows": len(positive),
        "correlation_r2_vs_repetition": float(np.corrcoef(even[:, 0], even[:, 1])[0, 1]),
        "condition_number": float(np.linalg.cond(even)),
        "repetition_range": [float(repetition.min()), float(repetition.max())],
        "distinct_radii": sorted(np.unique(np.round(radius, 3)).tolist()),
    }
    return findings


def recovery_trial(directions: np.ndarray, sparsity: int, amplitude: float, noise: float, rng) -> dict[str, float]:
    """One synthetic recovery of a sparse beta from odd-channel contrasts on a given direction set.

    The odd half of an antithetic pair estimates `r beta'd` with noise `sigma / sqrt(2)`, so a design is
    only as good as its ability to invert that from however many directions it buys. Support here is drawn
    in the tangent space, not in bucket space, because a sparse mixture perturbation is not a sparse CLR
    vector; this is the friendlier of the two assumptions for the draft design.
    """
    tangent = directions.shape[1]
    beta = np.zeros(tangent)
    support = rng.choice(tangent, size=sparsity, replace=False)
    beta[support] = amplitude * rng.choice([-1.0, 1.0], size=sparsity)
    observed = directions @ beta + rng.normal(0.0, noise, size=len(directions))
    model = LassoCV(cv=min(5, len(directions) // 3), n_alphas=40, max_iter=20000, random_state=0)
    model.fit(directions, observed)
    estimated = model.coef_
    recovered = set(np.flatnonzero(np.abs(estimated) > 0.2 * amplitude).tolist())
    truth = set(support.tolist())
    aligned = float(np.dot(beta, estimated) / max(np.linalg.norm(beta) * np.linalg.norm(estimated), 1e-300))
    return {
        "support_recall": len(recovered & truth) / max(len(truth), 1),
        "cosine": aligned,
        "direction_error": float(
            np.linalg.norm(estimated / max(np.linalg.norm(estimated), 1e-300) - beta / np.linalg.norm(beta))
        ),
    }


def simulate_allocations(directions: np.ndarray, noise: float, trials: int = 40) -> pd.DataFrame:
    """Compare the draft's 28 directions against wider allocations under one 80-row budget.

    Every allocation spends the same forty antithetic pairs. Buying more distinct directions widens the
    span that beta can be seen in and costs the radius replication that the even channel uses to separate
    curvature from repetition, so the comparison is a real trade rather than a free improvement.

    `signal_to_noise` is defined on the OBSERVED contrast, not on beta itself: the simulated coefficient
    vector stands for `r * beta`, so an SNR of five means the whole odd-channel signal across directions is
    five odd-channel noise units. That keeps the statement radius-free. For scale, the far panel's response
    SD is 0.0174 at median Hellinger 0.59; if response grew linearly in radius, a ray at 0.15 would carry
    about 0.004, or roughly seven noise units, so five is near the optimistic end of what to expect.

    Support is drawn uniformly in the tangent space rather than in bucket space, because a sparse mixture
    perturbation is not a sparse CLR vector. That is the friendlier assumption for the draft design.
    """
    rng = np.random.default_rng(20260825)
    tangent = directions.shape[1]
    rows = []
    for label, count in (
        ("draft (actual 28)", None),
        ("32 directions", 32),
        ("36 directions", 36),
        ("40 directions", 40),
    ):
        for sparsity in (3, 6, 10):
            for signal_to_noise in (2.0, 5.0):
                scores = []
                for _trial in range(trials):
                    if count is None:
                        design = directions
                    else:
                        raw = rng.normal(size=(count, tangent))
                        design = raw / np.linalg.norm(raw, axis=1, keepdims=True)
                    amplitude = signal_to_noise * noise / np.sqrt(max(sparsity, 1))
                    scores.append(recovery_trial(design, sparsity, amplitude, noise, rng))
                rows.append(
                    {
                        "allocation": label,
                        "n_directions": len(directions) if count is None else count,
                        "radius_replicated": 10 if count is None else max(0, 40 - (count or 0)),
                        "sparsity": sparsity,
                        "snr": signal_to_noise,
                        "support_recall": float(np.mean([s["support_recall"] for s in scores])),
                        "cosine": float(np.mean([s["cosine"] for s in scores])),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave2-root", type=Path, default=WAVE2_WORKTREE)
    parser.add_argument("--local-root", type=Path, default=LOCAL_WORKTREE)
    args = parser.parse_args()

    findings = reproduce(args.wave2_root)
    print("=== reproduction ===")
    for name, status in findings["hashes"].items():
        print(f"  {status:4s} {name}")
    print(f"  rows: {findings['rows']}")
    print(f"  prefix identical across all rows: {findings['prefix_constant']}")
    print(f"  tied continuation vs prefix, TV: {findings['tied_equals_prefix']:.8f}")
    for key in ("tied", "best_fit", "best_minus_tied"):
        delta = abs(findings[key] - EXPECTED[key])
        verdict = "OK" if delta < TOLERANCE else "DIFFERS"
        print(f"  {key:16s} {findings[key]:.8f}  expected {EXPECTED[key]:.8f}  {verdict}")
    print(f"  best fitted branch: {findings['best_fit_id']}")
    print(f"  fit actions better than tied: {findings['n_better_than_tied']} of {findings['rows']['fit']}")
    print(f"  github c++: best fit {findings['best_fit_cpp']:.8f} vs tied {findings['tied_cpp']:.8f}")
    for key in ("min", "median", "max"):
        expected = EXPECTED[f"tv_{key}"]
        got = findings["tv"][key]
        verdict = "OK" if abs(got - expected) < 1e-5 else "DIFFERS"
        print(f"  TV-from-tied {key:6s} {got:.5f}  expected {expected:.5f}  {verdict}")
    print("  run noise:")
    for name, stats in findings["noise_groups"].items():
        print(f"      {name:24s} n={stats['n']} sd={stats['sd']:.8f}")
    conservative = findings["run_noise_sd_conservative"]
    print(f"      conservative (max group) {conservative:.8f}  expected {EXPECTED['run_noise_sd']:.8f}")
    print(f"      pooled within-group      {findings['run_noise_sd_pooled']:.8f}")
    print("  controls, by distance from tied:")
    for control in sorted(findings["controls"], key=lambda c: c["tv_from_tied"]):
        print(f"      {control['id']:38s} TV {control['tv_from_tied']:.5f}  bpb {control['bpb']:.8f}")

    design = audit_design(args.local_root)
    print("\n=== draft local Wave-2b identifiability ===")
    print(f"  fit rows {design['n_fit_rows']}, fit directions {design['n_fit_directions']}")
    print(f"  directions total (fit + sealed referee) {design['n_directions_total']}")
    print(f"  families {design['family_counts']}")
    print(f"  outcome status {design['outcome_status']}")
    print(
        f"  directional rank {design['directional_rank']} of {design['tangent_dim']} "
        f"({design['spanned_fraction']:.1%}); {design['unspanned_dims']} tangent dimensions unmeasured"
    )
    print(f"  directions by distinct radius count: {design['radii_per_direction']}")
    even = design["even_channel"]
    print(f"  even channel over {even['n_rows']} plus-sign rows: radii {even['distinct_radii']}")
    corr = even["correlation_r2_vs_repetition"]
    print(f"      corr(r^2, repetition) {corr:+.4f}  condition {even['condition_number']:.2f}")
    print(f"      repetition excess-epoch range {even['repetition_range'][0]:.4f} .. {even['repetition_range'][1]:.4f}")
    noise = findings["run_noise_sd_conservative"] / np.sqrt(2.0)
    matrix = design.pop("clr_matrix")
    print(f"\n=== can 40 antithetic pairs recover a sparse beta? (odd-channel noise {noise:.6f}) ===")
    table = simulate_allocations(matrix, noise)
    pivot = table.pivot_table(index=["allocation", "n_directions"], columns=["sparsity", "snr"], values="support_recall")
    print("  mean support recall, by true sparsity and signal-to-noise:")
    print(pivot.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n" + json.dumps({"reproduction_ok": all(v == "OK" for v in findings["hashes"].values())}))


if __name__ == "__main__":
    main()
