# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Why the 80/20 WSD global optimum does not lie on the fiber through the best aggregate.

Two fixed-aggregate fibers were measured in the two-bucket StarCoder/Nemotron setting. One passes
through the best tied policy, aggregate 0.30. The other passes through the observed global optimum,
aggregate 0.18. Along the first, ordering buys nothing; along the second it buys a great deal. The
global optimum sits on the second, so the aggregate that is best for the one-phase policy class is not
the aggregate that is best for the two-phase class.

The decomposition that explains this is an identity, not a fit. Write a policy as an aggregate
``a = b0*p0 + b1*p1`` and a phase contrast ``d = p1 - p0``, so ``p0 = a - b1*d`` and ``p1 = a + b0*d``.
Split the loss along a fiber into its odd and even parts about the tied policy:

    o(a, d) = [L(a, d) - L(a, -d)] / 2          the ordering effect
    c(a, d) = [L(a, d) + L(a, -d)] / 2 - L(a,0) the asymmetry cost

Then ``min(L(a,d), L(a,-d)) = L(a,0) + c - |o|``, so the fiber beats its own tied policy exactly when
``|o| > c``. Define the phase gain ``g(a) = max_d [|o(a,d)| - c(a,d)]``. The one-phase class minimizes
``L(a,0)``; the two-phase class minimizes ``L(a,0) - g(a)``. These have the same argmin only when
``g`` is flat in ``a``, which is what the panel tests.

Two structural facts about the fiber geometry matter for reading the result. The feasible contrast
range is asymmetric: ``d`` runs from ``-a/b0`` to ``min(a/b1, (1-a)/b0)``, so antithetic pairs exist
only for ``|d| <= a/b0``. And that bound shrinks with the aggregate, so a low-aggregate fiber has a
shorter paired region and a longer unpaired tail than a high-aggregate one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "wsd80_two_fiber_decomposition_20260728"

PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
REFERENCE_SEED = 20260711
# Matching a contrast to its mirror image needs a tolerance because the two fibers use different
# uniform grids in p1; pairs are exact by construction and this only guards float representation.
CONTRAST_TOLERANCE = 1e-9
# Under WSD the decay window is the second phase. Averaging a linear decay to zero over that window
# gives it half the learning rate of the stable phase, so learning-rate-weighted domain exposure is
# not the token aggregate. The fibers hold the token aggregate fixed, so this is the obvious rival
# explanation for any measured ordering effect and is computed alongside it.
DECAY_MEAN_LEARNING_RATE_FRACTION = 0.5


def contrast(frame: pd.DataFrame) -> pd.Series:
    return frame["phase_1_starcoder"] - frame["phase_0_starcoder"]


def feasible_contrast_range(aggregate: float) -> tuple[float, float]:
    """Range of phase contrasts reachable at a fixed aggregate, from the simplex constraints."""
    low = max(-aggregate / PHASE_0_FRACTION, (aggregate - 1.0) / PHASE_1_FRACTION)
    high = min(aggregate / PHASE_1_FRACTION, (1.0 - aggregate) / PHASE_0_FRACTION)
    return low, high


def odd_even(frame: pd.DataFrame, tied: float) -> pd.DataFrame:
    """Ordering effect and asymmetry cost at every contrast whose mirror image was also trained."""
    by_contrast = frame.set_index("contrast")["wsd80_bpb"]
    rows = []
    for value in sorted(c for c in by_contrast.index if c > CONTRAST_TOLERANCE):
        mirror = by_contrast.index[np.isclose(by_contrast.index, -value, atol=CONTRAST_TOLERANCE)]
        if not len(mirror):
            continue
        plus, minus = float(by_contrast.loc[value]), float(by_contrast.loc[mirror[0]])
        rows.append(
            {
                "abs_contrast": value,
                "late_heavy_bpb": plus,
                "early_heavy_bpb": minus,
                "ordering_effect": 0.5 * (plus - minus),
                "asymmetry_cost": 0.5 * (plus + minus) - tied,
                "fiber_gain": abs(0.5 * (plus - minus)) - (0.5 * (plus + minus) - tied),
            }
        )
    return pd.DataFrame(rows)


def local_expansion(paired: pd.DataFrame) -> dict[str, float]:
    """Slope of the odd part and curvature of the even part, fit through the origin.

    ``L(a, d) = L(a,0) + kappa*d + rho/2*d^2 + ...`` puts every odd power in ``o`` and every even power
    in ``c``, so the leading coefficients come out of a one-parameter fit on each part separately.

    The largest contrast on each fiber is the feasibility endpoint, where the early-heavy arm sets the
    domain share in one phase to zero. That is a boundary regime, not a perturbation of the tied
    policy, and it dominates a least-squares fit. The fit therefore drops it, and the per-magnitude
    ratios below let a reader see whether the leading term actually describes the response.
    """
    interior = paired.iloc[:-1] if len(paired) > 2 else paired
    d = interior["abs_contrast"].to_numpy()
    kappa = float(np.linalg.lstsq(d[:, None], interior["ordering_effect"].to_numpy(), rcond=None)[0][0])
    rho = 2.0 * float(np.linalg.lstsq((0.5 * d**2)[:, None], interior["asymmetry_cost"].to_numpy(), rcond=None)[0][0])
    return {
        "kappa": kappa,
        "rho": rho,
        "dropped_endpoint_contrast": float(paired["abs_contrast"].iloc[-1]),
        "quadratic_optimum_contrast": abs(kappa) / rho if rho > 0 else float("nan"),
        "quadratic_best_gain": kappa**2 / (2.0 * rho) if rho > 0 else float("nan"),
    }


def learning_rate_weighted_exposure(phase_0: float, phase_1: float) -> float:
    """Domain exposure weighted by learning rate rather than by token count."""
    weight_0 = PHASE_0_FRACTION
    weight_1 = PHASE_1_FRACTION * DECAY_MEAN_LEARNING_RATE_FRACTION
    return (weight_0 * phase_0 + weight_1 * phase_1) / (weight_0 + weight_1)


def paired_seed_delta(frame: pd.DataFrame, contrast_value: float) -> dict[str, float] | None:
    """Within-seed difference between a contrast and its own fiber's tied policy."""
    tied = frame[np.isclose(frame["contrast"], 0.0, atol=CONTRAST_TOLERANCE)].set_index("data_seed")["wsd80_bpb"]
    treated = frame[np.isclose(frame["contrast"], contrast_value, atol=CONTRAST_TOLERANCE)].set_index("data_seed")[
        "wsd80_bpb"
    ]
    shared = sorted(set(tied.index) & set(treated.index))
    if len(shared) < 2:
        return None
    deltas = np.array([treated.loc[s] - tied.loc[s] for s in shared], dtype=float)
    result = stats.ttest_1samp(deltas, 0.0)
    half_width = stats.t.ppf(0.975, len(deltas) - 1) * deltas.std(ddof=1) / np.sqrt(len(deltas))
    return {
        "seeds": len(deltas),
        "mean_delta": float(deltas.mean()),
        "sd_delta": float(deltas.std(ddof=1)),
        "ci_low": float(deltas.mean() - half_width),
        "ci_high": float(deltas.mean() + half_width),
        "p_value": float(result.pvalue),
        "wins": int((deltas < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(args.source_dir / "wsd80_measured_fiber_observations.csv")
    observations["contrast"] = contrast(observations)
    surface = pd.read_csv(args.source_dir / "wsd80_observed_metrics.csv")
    surface["contrast"] = contrast(surface)

    diagonal = surface[np.isclose(surface["contrast"], 0.0)].sort_values("phase_0_starcoder")
    best_tied = diagonal.loc[diagonal["wsd80_bpb"].idxmin()]
    best_overall = surface.loc[surface["wsd80_bpb"].idxmin()]

    print("=" * 100)
    print("SURFACE  (166 coordinates, reference seed)")
    print("=" * 100)
    best_overall_aggregate = (
        PHASE_0_FRACTION * best_overall["phase_0_starcoder"] + PHASE_1_FRACTION * best_overall["phase_1_starcoder"]
    )
    print(
        f"  best one-phase policy   p = {best_tied['phase_0_starcoder']:.4f}"
        f"                 {best_tied['wsd80_bpb']:.6f}"
    )
    print(
        f"  best two-phase policy   p0 = {best_overall['phase_0_starcoder']:.4f}, "
        f"p1 = {best_overall['phase_1_starcoder']:.4f}   {best_overall['wsd80_bpb']:.6f}"
        f"   aggregate {best_overall_aggregate:.4f}"
    )
    print("  the two-phase optimum sits at a different aggregate than the one-phase optimum\n")

    summary: dict[str, object] = {}
    for _fiber_id, block in observations.groupby("fiber_id"):
        aggregate = float(block["aggregate_starcoder_share_80_20"].iloc[0])
        label = block["fiber_label"].iloc[0]
        low, high = feasible_contrast_range(aggregate)
        reference = block[block["data_seed"] == REFERENCE_SEED]
        tied_reference = float(reference.loc[np.isclose(reference["contrast"], 0.0), "wsd80_bpb"].iloc[0])

        print("=" * 100)
        print(f"FIBER  aggregate {aggregate:.2f}   ({label})")
        print("=" * 100)
        print(f"  feasible contrast range   [{low:+.4f}, {high:+.4f}]")
        print(f"  antithetic pairs exist only for |d| <= {aggregate / PHASE_0_FRACTION:.4f}")
        print(f"  tied policy (reference seed)                {tied_reference:.6f}")

        paired = odd_even(reference, tied_reference)
        print("\n  reference-seed odd/even decomposition on the paired region:")
        print("    |d|      late-heavy   early-heavy   ordering o   cost c     |o|-c      o/|d|    2c/d^2   verdict")
        for position, (_, row) in enumerate(paired.iterrows()):
            verdict = "two-phase wins" if row["fiber_gain"] > 0 else "tied wins"
            if position == len(paired) - 1:
                verdict += "   <- feasibility endpoint"
            odd_slope = row["ordering_effect"] / row["abs_contrast"]
            even_curvature = 2 * row["asymmetry_cost"] / row["abs_contrast"] ** 2
            print(
                f"    {row['abs_contrast']:.4f}   {row['late_heavy_bpb']:.6f}    {row['early_heavy_bpb']:.6f}"
                f"   {row['ordering_effect']:+.6f}   {row['asymmetry_cost']:+.6f}  {row['fiber_gain']:+.6f}"
                f"   {odd_slope:+.4f}   {even_curvature:.4f}   {verdict}"
            )
        expansion = local_expansion(paired)
        print(
            f"\n    local expansion (endpoint |d|={expansion['dropped_endpoint_contrast']:.4f} excluded):"
            f" kappa {expansion['kappa']:+.4f}, rho {expansion['rho']:+.4f}"
            f"  ->  quadratic optimum at |d| = {expansion['quadratic_optimum_contrast']:.4f},"
            f" gain {expansion['quadratic_best_gain']:.6f}"
        )

        best_row = reference.loc[reference["wsd80_bpb"].idxmin()]
        best_contrast = float(best_row["contrast"])
        in_paired_region = abs(best_contrast) <= aggregate / PHASE_0_FRACTION + CONTRAST_TOLERANCE
        print(
            f"\n  best point on this fiber (reference seed): d = {best_contrast:+.4f}"
            f"  (p0 {best_row['phase_0_starcoder']:.4f}, p1 {best_row['phase_1_starcoder']:.4f})"
            f"  {best_row['wsd80_bpb']:.6f}"
        )
        print(f"    gain over own tied policy: {tied_reference - best_row['wsd80_bpb']:+.6f}")
        print(
            f"    lies in the paired region: {in_paired_region}"
            + ("" if in_paired_region else "   <- its mirror image is infeasible, so o and c are undefined there")
        )

        print("\n  multi-seed paired tests against this fiber's own tied policy:")
        seed_rows = []
        for value in sorted(block["contrast"].unique()):
            if abs(value) < CONTRAST_TOLERANCE:
                continue
            test = paired_seed_delta(block, value)
            if test is None:
                continue
            test |= {"contrast": value}
            seed_rows.append(test)
            print(
                f"    d = {value:+.4f}   {test['mean_delta']:+.6f}  "
                f"95% CI [{test['ci_low']:+.6f}, {test['ci_high']:+.6f}]  p={test['p_value']:.2e}  "
                f"{test['wins']}/{test['seeds']} seeds better"
            )

        summary[f"aggregate_{aggregate:.2f}"] = {
            "label": label,
            "aggregate": aggregate,
            "feasible_contrast_range": [low, high],
            "paired_region_bound": aggregate / PHASE_0_FRACTION,
            "tied_reference_bpb": tied_reference,
            "paired_decomposition": paired.to_dict("records"),
            "local_expansion": expansion,
            "best_contrast": best_contrast,
            "best_bpb": float(best_row["wsd80_bpb"]),
            "best_in_paired_region": bool(in_paired_region),
            "paired_seed_tests": seed_rows,
        }
        paired.to_csv(args.output_dir / f"odd_even_aggregate_{aggregate:.2f}.csv", index=False)
        print()

    print("=" * 100)
    print("WHY THE OPTIMUM MOVED")
    print("=" * 100)
    low_key, high_key = "aggregate_0.18", "aggregate_0.30"
    low, high = summary[low_key], summary[high_key]

    # Multi-seed tied policies at each aggregate, on the seeds both fibers share.
    tied_by_fiber = {}
    for _fiber_id, block in observations.groupby("fiber_id"):
        aggregate = float(block["aggregate_starcoder_share_80_20"].iloc[0])
        tied_by_fiber[aggregate] = block[np.isclose(block["contrast"], 0.0)].set_index("data_seed")["wsd80_bpb"]
    shared_seeds = sorted(set(tied_by_fiber[0.18].index) & set(tied_by_fiber[0.30].index))
    aggregate_penalty = np.array(
        [tied_by_fiber[0.18].loc[s] - tied_by_fiber[0.30].loc[s] for s in shared_seeds], dtype=float
    )
    phase_gain = [t for t in low["paired_seed_tests"] if abs(t["contrast"] - 0.40) < 1e-6]
    assert phase_gain, "expected the global optimum contrast d=+0.40 among the repeated coordinates"
    gain = phase_gain[0]

    print("  one-phase optimum is at aggregate 0.30; the two-phase optimum is at aggregate 0.18.")
    print("  moving the aggregate from 0.30 to 0.18 costs the one-phase class")
    print(
        f"      {aggregate_penalty.mean():+.6f} BPB   (paired over {len(shared_seeds)} shared seeds,"
        f" SD {aggregate_penalty.std(ddof=1):.6f})"
    )
    print("  ordering at aggregate 0.18 then buys back")
    print(
        f"      {gain['mean_delta']:+.6f} BPB   (paired over {gain['seeds']} seeds,"
        f" 95% CI [{gain['ci_low']:+.6f}, {gain['ci_high']:+.6f}], {gain['wins']}/{gain['seeds']} wins)"
    )
    net = gain["mean_delta"] + aggregate_penalty.mean()
    print("  net advantage of the two-phase optimum over the best tied policy")
    print(f"      {net:+.6f} BPB")
    print()
    print(
        f"  phase gain at aggregate 0.30 (best over the paired region): "
        f"{max(r['fiber_gain'] for r in high['paired_decomposition']):+.6f}"
    )
    print(
        f"  phase gain at aggregate 0.18 (best over the paired region): "
        f"{max(r['fiber_gain'] for r in low['paired_decomposition']):+.6f}"
    )
    print(
        f"  ordering sensitivity kappa:  {high['local_expansion']['kappa']:+.4f} at 0.30"
        f"   vs {low['local_expansion']['kappa']:+.4f} at 0.18"
        f"   (ratio {low['local_expansion']['kappa'] / high['local_expansion']['kappa']:.2f}x)"
    )
    print(
        f"  asymmetry curvature rho:     {high['local_expansion']['rho']:+.4f} at 0.30"
        f"   vs {low['local_expansion']['rho']:+.4f} at 0.18"
        f"   (ratio {low['local_expansion']['rho'] / high['local_expansion']['rho']:.2f}x)"
    )

    print()
    print("=" * 100)
    print("RIVAL EXPLANATION: IS THIS JUST LEARNING-RATE-WEIGHTED EXPOSURE?")
    print("=" * 100)
    print("  The fibers hold the token aggregate fixed but cannot hold learning-rate-weighted exposure")
    print("  fixed, because phase 1 is the decay window. If the winner simply had a better effective")
    print("  dose, the ordering interpretation would be wrong. Weighting a linear decay to zero at half")
    print("  the stable learning rate:")
    rivals = [
        ("best one-phase        p=0.30", 0.30, 0.30),
        ("tied at the low aggregate p=0.18", 0.18, 0.18),
        ("two-phase optimum  (0.10, 0.50)", 0.10, 0.50),
    ]
    for name, phase_0, phase_1 in rivals:
        token = PHASE_0_FRACTION * phase_0 + PHASE_1_FRACTION * phase_1
        print(
            f"    {name:<34} token aggregate {token:.4f}   "
            f"learning-rate-weighted {learning_rate_weighted_exposure(phase_0, phase_1):.4f}"
        )
    optimum_weighted = learning_rate_weighted_exposure(0.10, 0.50)
    nearest_tied = diagonal.iloc[(diagonal["phase_0_starcoder"] - optimum_weighted).abs().argsort().iloc[0]]
    print()
    print(f"  The optimum's learning-rate-weighted exposure ({optimum_weighted:.4f}) is LOWER than its own")
    print("  tied policy's (0.1800) and far below the best tied policy's (0.3000). The one-phase policy")
    print(f"  matched to it, p={nearest_tied['phase_0_starcoder']:.4f}, scores {nearest_tied['wsd80_bpb']:.6f} --")
    print(f"  {nearest_tied['wsd80_bpb'] - best_overall['wsd80_bpb']:+.6f} worse than the optimum.")
    print("  Reweighting exposure by learning rate therefore makes the winner look worse supplied, not")
    print("  better. It cannot explain the gain.")
    print()

    summary["learning_rate_weighted_check"] = {
        "optimum_token_aggregate": 0.18,
        "optimum_learning_rate_weighted": optimum_weighted,
        "matched_tied_share": float(nearest_tied["phase_0_starcoder"]),
        "matched_tied_bpb": float(nearest_tied["wsd80_bpb"]),
    }
    summary["comparison"] = {
        "shared_seeds": shared_seeds,
        "aggregate_penalty_mean": float(aggregate_penalty.mean()),
        "aggregate_penalty_sd": float(aggregate_penalty.std(ddof=1)),
        "phase_gain_at_0.18": gain,
        "net_two_phase_advantage": float(net),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
