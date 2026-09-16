# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Split the 3e18 frontier into an aggregate channel and an order channel.

The same decomposition was measured at 60M on a balanced designed swarm: moving from the proportional
mixture to the best one-phase policy bought 43.4 run sigma, and moving from there to the best two-phase
policy bought 1.32 more. That comparison is clean because both classes had exactly 242 policies.

At 3e18 the question is more interesting and the arithmetic is more dangerous. It is more interesting
because both frontiers were pushed hard over many months and many methods rather than sampled from one
design, so the one-phase number is a real optimization result rather than the best of a sweep. It is
more dangerous because the two classes are badly unbalanced -- 1533 two-phase policies against 430
one-phase -- and the minimum of a larger sample is lower even when the two classes are identical. Taking
the raw difference of minima would credit the order channel with a selection artifact.

So the order channel is reported twice: once raw, and once after subsampling the two-phase set down to
the one-phase count and bootstrapping the minimum. The gap between those two numbers is the part of the
apparent order gain that comes from having simply tried more two-phase policies.

The panel also answers whether the phase degrees of freedom found a *different* place to stand, by
comparing the best two-phase policy's aggregate mixture against the best one-phase mixture. At 60M they
were the same point, and the phase freedom only re-timed a mixture the aggregate search had already
located.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_REGISTRY = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "two_channels_3e18_20260728"

OBJECTIVES = (("uncheatable", "uncheatable_bpb"), ("table9", "table9_macro_bpb"))
RUN_SIGMA = {"uncheatable": 0.000913, "table9": 0.003772}
# The proportional mixture and its deliberate replicates. Their spread is an independent read on run
# noise at 3e18, computed at a single policy, which is what run sigma is supposed to mean.
PROPORTIONAL_PATTERN = "proportional"
SUBSAMPLE_DRAWS = 4000
SUBSAMPLE_SEED = 20260728


def load(registry: Path) -> pd.DataFrame:
    frame = pd.read_csv(registry)
    assert set(frame["policy_class"]) == {"single_phase_tied", "two_phase"}, "unexpected policy classes"
    return frame


def proportional_baseline(frame: pd.DataFrame, column: str) -> tuple[float, float, int]:
    """Mean and spread of the proportional mixture across its replicates."""
    mask = frame["wandb_run_name"].astype(str).str.contains(PROPORTIONAL_PATTERN, case=False, na=False)
    values = frame.loc[mask, column].dropna()
    assert len(values) >= 2, f"need replicates of the proportional mixture, found {len(values)}"
    return float(values.mean()), float(values.std(ddof=1)), len(values)


def matched_minimum(values: np.ndarray, size: int, draws: int, seed: int) -> tuple[float, float, float]:
    """Distribution of the minimum when only ``size`` of these policies are drawn.

    The two-phase class was searched far harder than the one-phase class, so its observed minimum is
    partly a reward for sample size. Subsampling to the smaller class's count removes that and leaves
    only the advantage the policy class itself confers.
    """
    generator = np.random.default_rng(seed)
    minima = np.array([values[generator.choice(values.size, size=size, replace=False)].min() for _ in range(draws)])
    return float(np.median(minima)), float(np.quantile(minima, 0.05)), float(np.quantile(minima, 0.95))


def aggregate_mixture(row: pd.Series) -> np.ndarray | None:
    """Token-weighted aggregate of a policy's two phase mixtures, in a fixed domain order."""
    early = row.get("phase_0_weights_json")
    late = row.get("phase_1_weights_json")
    if not isinstance(early, str) or not isinstance(late, str):
        return None
    early_weights, late_weights = json.loads(early), json.loads(late)
    fraction = float(row["phase_0_fraction"])
    domains = sorted(early_weights)
    return np.array(
        [fraction * early_weights[d] + (1.0 - fraction) * late_weights.get(d, early_weights[d]) for d in domains],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=SUBSAMPLE_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load(args.registry)
    counts = frame["policy_class"].value_counts()
    print(
        f"3e18 registry: {len(frame)} policies -- {counts['single_phase_tied']} one-phase, "
        f"{counts['two_phase']} two-phase"
    )
    print(f"training series contributing: {frame['training_series'].nunique()}\n")

    rows = []
    for objective, column in OBJECTIVES:
        sigma = RUN_SIGMA[objective]
        block = frame.dropna(subset=[column])
        one = block[block["policy_class"] == "single_phase_tied"]
        two = block[block["policy_class"] == "two_phase"]
        baseline, baseline_sd, replicates = proportional_baseline(block, column)
        best_one = float(one[column].min())
        best_two = float(two[column].min())

        median_two, low_two, high_two = matched_minimum(two[column].to_numpy(), len(one), args.draws, SUBSAMPLE_SEED)
        aggregate_channel = baseline - best_one
        order_raw = best_one - best_two
        order_matched = best_one - median_two

        print("=" * 96)
        print(
            f"{objective.upper()}   (run sigma {sigma:.6f}; proportional replicates give "
            f"{baseline_sd:.6f} over {replicates} runs)"
        )
        print("=" * 96)
        print(f"  proportional mixture      {baseline:.6f}")
        print(f"  best one-phase            {best_one:.6f}   ({len(one)} policies)")
        print(f"  best two-phase            {best_two:.6f}   ({len(two)} policies)")
        print()
        print(
            f"  AGGREGATE channel  proportional -> best one-phase   "
            f"{aggregate_channel:.6f} BPB = {aggregate_channel / sigma:6.1f} sigma"
        )
        print(
            f"  ORDER channel      best one-phase -> best two-phase "
            f"{order_raw:.6f} BPB = {order_raw / sigma:6.2f} sigma   (raw, unbalanced)"
        )
        print(
            f"  ORDER channel at matched sample size ({len(one)} draws)     "
            f"{order_matched:.6f} BPB = {order_matched / sigma:6.2f} sigma"
        )
        print(
            f"    subsampled two-phase minimum: median {median_two:.6f}, "
            f"90% interval [{low_two:.6f}, {high_two:.6f}]"
        )
        print(f"    selection inflation removed: {(order_raw - order_matched) / sigma:.2f} sigma")
        print(f"  ratio aggregate : order = {aggregate_channel / max(order_matched, 1e-12):.0f} : 1 (matched)")
        rows.append(
            {
                "objective": objective,
                "proportional": baseline,
                "proportional_sd": baseline_sd,
                "best_one_phase": best_one,
                "best_two_phase": best_two,
                "n_one_phase": len(one),
                "n_two_phase": len(two),
                "aggregate_channel_bpb": aggregate_channel,
                "aggregate_channel_sigma": aggregate_channel / sigma,
                "order_channel_raw_bpb": order_raw,
                "order_channel_raw_sigma": order_raw / sigma,
                "order_channel_matched_bpb": order_matched,
                "order_channel_matched_sigma": order_matched / sigma,
                "matched_two_phase_p05": low_two,
                "matched_two_phase_p95": high_two,
            }
        )

        # Does the two-phase optimum stand somewhere the one-phase search did not reach?
        best_two_row = two.loc[two[column].idxmin()]
        best_one_row = one.loc[one[column].idxmin()]
        two_aggregate = aggregate_mixture(best_two_row)
        one_aggregate = aggregate_mixture(best_one_row)
        print()
        if two_aggregate is None or one_aggregate is None:
            print("  THE TELL: mixture weights not recorded for one of the optima; cannot compare aggregates")
        else:
            distance = 0.5 * float(np.abs(two_aggregate - one_aggregate).sum())
            print(f"  THE TELL: best two-phase run  {best_two_row['wandb_run_name']}")
            print(f"            best one-phase run  {best_one_row['wandb_run_name']}")
            print(f"            aggregate total variation between them: {distance:.4f}")
            print(
                "            -> same aggregate; phase freedom only re-timed it"
                if distance < 0.01
                else "            -> different aggregate; the two-phase optimum stands elsewhere"
            )
        print()

    pd.DataFrame(rows).to_csv(args.output_dir / "two_channels_3e18.csv", index=False)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
