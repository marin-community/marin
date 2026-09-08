# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///
"""Test whether the odd/even radius-exponent ordering depends on the phase split.

Background
----------
A two-phase run spends a fraction ``alpha`` of its tokens in phase 0 and the rest
in phase 1. With per-phase mixtures ``p0`` and ``p1``:

    a = alpha * p0 + (1 - alpha) * p1     aggregate (time-averaged) mixture
    d = p1 - p0                           phase contrast; d = 0 is single phase

At fixed ``a``, reversing the sign of ``d`` swaps which mixture runs late. Any
response splits uniquely into an odd part that depends on the ordering and an
even part that depends only on how much the phases differ:

    O(a, d) = [L(a, d) - L(a, -d)] / 2
    C(a, d) = [L(a, d) + L(a, -d)] / 2 - L(a, 0)

Choosing the better of the two orders yields ``C - |O|``, so a two-phase policy
beats its single-phase counterpart only where ``|O| > C``. Writing
``rho = phase TV`` and

    |O| ~ kappa * rho^p,      C ~ c * rho^q,

the sign of ``q - p`` decides the shape of the whole problem. If ``q > p`` the
symmetric cost eventually dominates and the attainable gain is capped at
``max_rho [kappa rho^p - c rho^q]``. If ``q < p`` the ordering benefit keeps
winning as the contrast grows.

The 39-bucket Delphi 3e18 panel gives ``p = 1.66-1.85`` and ``q = 2.10-2.68``, so
``q > p`` and the cap is near 0.003 BPB. Every 39-bucket panel available locally
(Delphi, 300M, 60M) sits at ``alpha = 0.8``, so that family cannot say whether the
ordering is a fact about two-phase training or an artifact of a short late phase.

The two dense StarCoder surfaces are the only local data at a different split:
cosine 50/50 has ``alpha = 0.50`` and WSD 80/20 has ``alpha = 0.80``, over the same
two buckets. This script estimates ``p`` and ``q`` on each.

Scope limits, stated up front
-----------------------------
``alpha`` is confounded with the learning-rate schedule (cosine versus WSD), the
geometry is two buckets rather than 39, and the target is StarCoder BPB. This is a
mechanism probe, not a transfer claim.

A separate, purely geometric consequence of ``alpha`` is also reported. Requiring
both ``+d`` and ``-d`` to keep both phase mixtures inside the simplex bounds the
reversible contrast radius by

    |d| <= min(a, 1 - a) / max(alpha, 1 - alpha),

which is largest at ``alpha = 0.5``. A lopsided split therefore shrinks the part
of contrast space that is reversible at all, independently of any response law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from phase_order_spine_20260725 import REFERENCE_OUTPUTS, build_spine, provenance
from scipy.optimize import least_squares
from scipy.stats import pearsonr

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_phase_split_alpha_20260725"

# Aggregate bins over the code-bucket share. Non-overlapping so that no
# observation contributes to two local fits.
AGGREGATE_BINS = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.01)
MINIMUM_PER_SIGN = 3
EXPONENT_BOUNDS = (0.2, 6.0)
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_SEED = 20260725
CODE_BUCKET = "starcoder"


def reversible_radius(aggregate: np.ndarray, alpha: float) -> np.ndarray:
    """Largest contrast radius at which both sign choices stay inside the simplex."""
    return np.minimum(aggregate, 1.0 - aggregate) / max(alpha, 1.0 - alpha)


def surface_frame(surface) -> pd.DataFrame:
    """Aggregate/contrast coordinates for a two-bucket dense surface."""
    index = surface.buckets.index(CODE_BUCKET)
    aggregate = surface.aggregate[:, index]
    contrast = surface.contrast[:, index]
    frame = pd.DataFrame(
        {
            "aggregate": aggregate,
            "contrast": contrast,
            "radius": np.abs(contrast),
            "bpb": surface.bpb,
            "reversible_radius": reversible_radius(aggregate, surface.alpha),
        }
    )
    frame["reversible"] = frame["radius"] <= frame["reversible_radius"] + 1e-9
    frame["bin"] = np.digitize(frame["aggregate"], np.asarray(AGGREGATE_BINS)) - 1
    return frame


def usable_bins(frame: pd.DataFrame) -> list[int]:
    """Bins holding enough observations of each contrast sign to separate O from C."""
    keep = []
    for bin_id, group in frame.groupby("bin"):
        positive = int((group["contrast"] > 1e-9).sum())
        negative = int((group["contrast"] < -1e-9).sum())
        if positive >= MINIMUM_PER_SIGN and negative >= MINIMUM_PER_SIGN:
            keep.append(int(bin_id))
    return sorted(keep)


def fit_exponents(frame: pd.DataFrame, bins: list[int]) -> dict[str, object]:
    """Joint fit of global exponents with per-bin aggregate and phase amplitudes.

    Within each aggregate bin the model is

        L = F_b + g_b (a - a_bar_b) + s_b sign(d) |d|^p + c_b |d|^q,

    so the steep aggregate response is absorbed locally and the two global
    exponents are identified from the contrast structure. ``s_b`` is the odd
    amplitude and ``c_b`` the even amplitude of bin ``b``.
    """
    blocks = [frame[frame["bin"] == b] for b in bins]
    counts = [len(block) for block in blocks]
    n_bins = len(bins)

    def unpack(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        intercept = z[0:n_bins]
        slope = z[n_bins : 2 * n_bins]
        odd = z[2 * n_bins : 3 * n_bins]
        even = z[3 * n_bins : 4 * n_bins]
        return intercept, slope, odd, even, float(z[-2]), float(z[-1])

    def residual(z: np.ndarray) -> np.ndarray:
        intercept, slope, odd, even, power_odd, power_even = unpack(z)
        out = []
        for position, block in enumerate(blocks):
            centered = block["aggregate"].to_numpy() - block["aggregate"].mean()
            radius = block["radius"].to_numpy()
            sign = np.sign(block["contrast"].to_numpy())
            predicted = (
                intercept[position]
                + slope[position] * centered
                + odd[position] * sign * radius**power_odd
                + even[position] * radius**power_even
            )
            out.append(predicted - block["bpb"].to_numpy())
        return np.concatenate(out)

    start = np.concatenate(
        [
            [block["bpb"].mean() for block in blocks],
            np.zeros(n_bins),
            np.full(n_bins, -0.05),
            np.full(n_bins, 0.05),
            [1.5, 2.0],
        ]
    )
    lower = np.concatenate(
        [
            np.full(n_bins, -np.inf),
            np.full(n_bins, -np.inf),
            np.full(n_bins, -np.inf),
            np.full(n_bins, -np.inf),
            EXPONENT_BOUNDS[:1] * 2,
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_bins, np.inf),
            np.full(n_bins, np.inf),
            np.full(n_bins, np.inf),
            np.full(n_bins, np.inf),
            EXPONENT_BOUNDS[1:] * 2,
        ]
    )
    solution = least_squares(residual, start, bounds=(lower, upper), method="trf")
    intercept, slope, odd, even, power_odd, power_even = unpack(solution.x)
    residuals = residual(solution.x)
    degrees = max(len(residuals) - len(solution.x), 1)
    return {
        "bins": bins,
        "counts": counts,
        "intercept": intercept,
        "slope": slope,
        "odd_amplitude": odd,
        "even_amplitude": even,
        "power_odd": power_odd,
        "power_even": power_even,
        "residual_sd": float(np.sqrt((residuals**2).sum() / degrees)),
        "n_observations": len(residuals),
        "n_parameters": len(solution.x),
    }


def attainable_gain(
    odd_amplitude: float, even_amplitude: float, power_odd: float, power_even: float, ceiling: float
) -> dict[str, float]:
    """Best ``|O| - C`` over the reversible radius range, and where it occurs."""
    magnitude = abs(odd_amplitude)
    if even_amplitude <= 0:
        # No symmetric penalty: the ordering benefit is limited only by geometry.
        return {
            "optimal_radius": ceiling,
            "attainable_gain_bpb": magnitude * ceiling**power_odd,
            "interior_optimum": False,
        }
    if power_even > power_odd:
        interior = (magnitude * power_odd / (even_amplitude * power_even)) ** (1.0 / (power_even - power_odd))
    else:
        interior = np.inf
    radius = float(min(interior, ceiling))
    return {
        "optimal_radius": radius,
        "attainable_gain_bpb": float(magnitude * radius**power_odd - even_amplitude * radius**power_even),
        "interior_optimum": bool(np.isfinite(interior) and interior < ceiling),
    }


def analyze(surface, restrict_reversible: bool, rng: np.random.Generator) -> dict[str, object]:
    frame = surface_frame(surface)
    working = frame[frame["reversible"]] if restrict_reversible else frame
    bins = usable_bins(working)
    if len(bins) < 2:
        return {
            "surface": surface.name,
            "alpha": surface.alpha,
            "sample": "reversible" if restrict_reversible else "all",
            "usable_bins": len(bins),
            "note": "fewer than two aggregate bins carry both contrast signs",
        }
    fit = fit_exponents(working, bins)
    gap = fit["power_even"] - fit["power_odd"]

    gaps = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(bins), len(bins))
        chosen = [bins[j] for j in np.unique(picked)]
        if len(chosen) < 2:
            continue
        try:
            boot = fit_exponents(working, chosen)
        except (ValueError, np.linalg.LinAlgError):
            continue
        gaps.append(boot["power_even"] - boot["power_odd"])
    gaps_array = np.asarray(gaps) if gaps else np.zeros(1)

    return {
        "surface": surface.name,
        "alpha": surface.alpha,
        "sample": "reversible" if restrict_reversible else "all",
        "usable_bins": len(bins),
        "n_observations": fit["n_observations"],
        "n_parameters": fit["n_parameters"],
        "residual_sd_bpb": fit["residual_sd"],
        "power_odd": fit["power_odd"],
        "power_even": fit["power_even"],
        "exponent_gap": gap,
        "gap_ci95_low": float(np.quantile(gaps_array, 0.025)),
        "gap_ci95_high": float(np.quantile(gaps_array, 0.975)),
        "probability_gap_positive": float(np.mean(gaps_array > 0)),
        "bootstrap_draws": len(gaps_array),
        "_fit": fit,
        "_frame": working,
    }


def bin_table(result: dict[str, object]) -> pd.DataFrame:
    fit = result["_fit"]
    frame = result["_frame"]
    rows = []
    for position, bin_id in enumerate(fit["bins"]):
        block = frame[frame["bin"] == bin_id]
        ceiling = float(block["reversible_radius"].median())
        gain = attainable_gain(
            float(fit["odd_amplitude"][position]),
            float(fit["even_amplitude"][position]),
            fit["power_odd"],
            fit["power_even"],
            ceiling,
        )
        rows.append(
            {
                "surface": result["surface"],
                "alpha": result["alpha"],
                "sample": result["sample"],
                "aggregate_bin_low": AGGREGATE_BINS[bin_id],
                "aggregate_bin_high": AGGREGATE_BINS[bin_id + 1],
                "n": len(block),
                "n_positive_contrast": int((block["contrast"] > 1e-9).sum()),
                "n_negative_contrast": int((block["contrast"] < -1e-9).sum()),
                "median_reversible_radius": ceiling,
                "odd_amplitude": float(fit["odd_amplitude"][position]),
                "even_amplitude": float(fit["even_amplitude"][position]),
                "code_late_helps": bool(fit["odd_amplitude"][position] < 0),
                **gain,
            }
        )
    return pd.DataFrame(rows)


def observed_reflection_pairs(surface, tolerance: float = 0.02) -> pd.DataFrame:
    """Model-free odd/even estimates from near-reflected observation pairs.

    Pairs are matched on aggregate and on opposite contrast of similar magnitude,
    then corrected to a common radius. This is a check that the parametric fit is
    not manufacturing the odd term.
    """
    frame = surface_frame(surface)
    rows = []
    values = frame.to_numpy()
    aggregate = frame["aggregate"].to_numpy()
    contrast = frame["contrast"].to_numpy()
    bpb = frame["bpb"].to_numpy()
    tied = frame[frame["radius"] < 1e-9]
    for i in range(len(values)):
        if contrast[i] <= 1e-9:
            continue
        for j in range(len(values)):
            if contrast[j] >= -1e-9:
                continue
            if abs(aggregate[i] - aggregate[j]) > tolerance:
                continue
            if abs(contrast[i] + contrast[j]) > tolerance:
                continue
            mean_aggregate = 0.5 * (aggregate[i] + aggregate[j])
            radius = 0.5 * (abs(contrast[i]) + abs(contrast[j]))
            reference = np.nan
            if len(tied):
                nearest = (tied["aggregate"] - mean_aggregate).abs().idxmin()
                if abs(tied.loc[nearest, "aggregate"] - mean_aggregate) <= tolerance:
                    reference = float(tied.loc[nearest, "bpb"])
            rows.append(
                {
                    "surface": surface.name,
                    "alpha": surface.alpha,
                    "aggregate": mean_aggregate,
                    "radius": radius,
                    "odd": 0.5 * (bpb[i] - bpb[j]),
                    "even_minus_tied": 0.5 * (bpb[i] + bpb[j]) - reference,
                }
            )
    return pd.DataFrame(rows)


def plot_channels(pairs: pd.DataFrame, fits: pd.DataFrame, path: Path) -> None:
    figure = go.Figure()
    palette = {0.5: "#2166ac", 0.8: "#b2182b"}
    for alpha, group in pairs.groupby("alpha"):
        colour = palette.get(float(alpha), "#444444")
        figure.add_trace(
            go.Scatter(
                x=group["radius"],
                y=group["odd"].abs(),
                mode="markers",
                name=f"|odd| alpha={alpha}",
                marker={"color": colour, "size": 8},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=group["radius"],
                y=group["even_minus_tied"],
                mode="markers",
                name=f"even alpha={alpha}",
                marker={"color": colour, "size": 8, "symbol": "diamond-open"},
            )
        )
    for row in fits.itertuples():
        if not np.isfinite(row.power_odd):
            continue
        grid = np.linspace(0.02, 1.0, 80)
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=grid**row.power_odd,
                mode="lines",
                name=f"rho^{row.power_odd:.2f} (odd, alpha={row.alpha})",
                line={"color": palette.get(float(row.alpha), "#444444")},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=grid**row.power_even,
                mode="lines",
                name=f"rho^{row.power_even:.2f} (even, alpha={row.alpha})",
                line={"color": palette.get(float(row.alpha), "#444444"), "dash": "dash"},
            )
        )
    figure.update_layout(
        title="StarCoder odd and even phase channels versus contrast radius, by phase split",
        xaxis={"title": "contrast radius (phase TV)", "type": "log"},
        yaxis={"title": "BPB", "type": "log"},
        template="plotly_white",
        height=560,
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    spine = build_spine()
    surfaces = (spine.starcoder_cosine, spine.starcoder_wsd)

    geometry_rows = []
    for surface in surfaces:
        frame = surface_frame(surface)
        geometry_rows.append(
            {
                "surface": surface.name,
                "alpha": surface.alpha,
                "rows": len(frame),
                "tied_rows": int((frame["radius"] < 1e-9).sum()),
                "positive_contrast": int((frame["contrast"] > 1e-9).sum()),
                "negative_contrast": int((frame["contrast"] < -1e-9).sum()),
                "reversible_rows": int(frame["reversible"].sum()),
                "reversible_fraction": float(frame["reversible"].mean()),
                "max_reversible_radius_at_balanced_aggregate": 0.5 / max(surface.alpha, 1 - surface.alpha),
                "median_reversible_radius": float(frame["reversible_radius"].median()),
            }
        )
    geometry = pd.DataFrame(geometry_rows)

    summaries, bins_frames = [], []
    for surface in surfaces:
        for restrict in (True, False):
            result = analyze(surface, restrict, rng)
            if "_fit" in result:
                bins_frames.append(bin_table(result))
                summaries.append({k: v for k, v in result.items() if not k.startswith("_")})
            else:
                summaries.append(result)
    summary = pd.DataFrame(summaries)
    bins_table = pd.concat(bins_frames, ignore_index=True) if bins_frames else pd.DataFrame()

    pairs = pd.concat([observed_reflection_pairs(s) for s in surfaces], ignore_index=True)

    primary = summary[(summary["sample"] == "reversible") & summary["power_odd"].notna()]
    plot_channels(pairs, primary, output / "starcoder_odd_even_by_alpha.html")

    geometry.to_csv(output / "reversibility_geometry.csv", index=False)
    summary.to_csv(output / "exponent_summary.csv", index=False)
    bins_table.to_csv(output / "per_bin_amplitudes.csv", index=False)
    pairs.to_csv(output / "observed_reflection_pairs.csv", index=False)

    protocol = {
        "aggregate_bins": list(AGGREGATE_BINS),
        "minimum_per_sign": MINIMUM_PER_SIGN,
        "exponent_bounds": list(EXPONENT_BOUNDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "delphi_reference": {"power_odd": [1.665, 1.850], "power_even": [2.097, 2.677], "alpha": 0.8},
        "confounds": [
            "alpha varies with the learning-rate schedule (cosine 50/50 versus WSD 80/20)",
            "two buckets rather than the 39-bucket swarm geometry",
            "target is StarCoder BPB, not Uncheatable or Table-9",
        ],
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    print("=== reversibility geometry ===")
    print(geometry.to_string(index=False))
    print("\n=== exponent fits ===")
    columns = [
        "surface",
        "alpha",
        "sample",
        "usable_bins",
        "n_observations",
        "power_odd",
        "power_even",
        "exponent_gap",
        "gap_ci95_low",
        "gap_ci95_high",
        "probability_gap_positive",
        "residual_sd_bpb",
    ]
    print(summary.reindex(columns=columns).to_string(index=False))
    print("\n=== per-bin amplitudes and attainable gain (reversible sample) ===")
    if not bins_table.empty:
        show = bins_table[bins_table["sample"] == "reversible"]
        print(
            show[
                [
                    "surface",
                    "aggregate_bin_low",
                    "aggregate_bin_high",
                    "n",
                    "n_positive_contrast",
                    "n_negative_contrast",
                    "odd_amplitude",
                    "even_amplitude",
                    "code_late_helps",
                    "optimal_radius",
                    "attainable_gain_bpb",
                ]
            ].to_string(index=False)
        )
    print(f"\n=== model-free reflection pairs: {len(pairs)} ===")
    if len(pairs):
        for alpha, group in pairs.groupby("alpha"):
            usable = group.dropna(subset=["even_minus_tied"])
            print(f"alpha={alpha}: {len(group)} pairs, {len(usable)} with a tied reference")
            print(f"  mean |odd| = {group['odd'].abs().mean():.5f}  mean even = {usable['even_minus_tied'].mean():.5f}")
            if len(group) >= 4:
                r, p = pearsonr(np.log(group["radius"]), np.log(group["odd"].abs().clip(lower=1e-6)))
                print(f"  log-log |odd| versus radius: pearson {r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
