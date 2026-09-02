# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Test phase-order capacity on antithetic pairs, where the aggregate is held fixed by construction.

An earlier version of this test regressed on every untied row of the fit panel and claimed to measure
phase-order capacity. It did not: those rows all have different aggregate mixtures, so aggregate
composition and phase order were confounded and any apparent capacity could have been the model
reading the aggregate. The fix is to use a design where the aggregate cannot vary.

The 60M fixed-aggregate archive is that design. Each pair holds one aggregate and one contrast
direction, trains ``+d`` and ``-d`` under a shared seed, and trains a tied control at the same
aggregate. The antithetic difference ``y(+d) - y(-d)`` therefore isolates the odd part of the response
exactly, with the aggregate, the direction and the seed all differenced away. Nothing else in the
archive can leak into it.

Two questions are asked of it. Does the *sign* of the antithetic difference agree across seed
replicates and across the two objectives, since a phase-order mechanism that reverses between
objectives at fixed scale is not a mechanism. And is the odd part large enough to overcome the
asymmetry cost measured on the same pair, since that comparison and not the raw difference is what
decides whether a two-phase policy can beat its tied control at all.

Effects are reported per direction family rather than pooled, because pooling directions with
opposite true signs would cancel a real effect into a null.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_ARCHIVE = REFERENCE_OUTPUTS / "60m_fixed_aggregate_phase_order_results_20260726"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "antithetic_phase_order_capacity_20260727"
# 60M control standard deviations measured from same-mixture repeats.
RUN_SIGMA = {"uncheatable": 0.000993, "table9": 0.0031}
# Families that are replicate controls rather than treatments carry no direction to test.
EXCLUDED_FAMILIES = ("sentinel_repeat",)
MIN_PAIRS_FOR_TEST = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_pairs(archive: Path) -> pd.DataFrame:
    """Antithetic pairs with their odd effect, asymmetry cost and same-seed tied control."""
    pairs = pd.read_csv(archive / "pair_decomposition.csv")
    # The archive already stores the antithetic half-difference and the symmetric cost, both computed
    # against a control trained on the same seed. Recomputing them here would only risk disagreeing
    # with the panel that produced them.
    required = {
        "target",
        "direction_family",
        "direction_id",
        "phase_tv",
        "plus_bpb",
        "minus_bpb",
        "same_seed_control_bpb",
        "order_half_effect_plus_minus",
        "symmetric_asymmetry_cost",
    }
    missing = required - set(pairs.columns)
    assert not missing, f"archive is missing columns {sorted(missing)}"
    return pairs[~pairs["direction_family"].isin(EXCLUDED_FAMILIES)].reset_index(drop=True)


def sign_agreement(frame: pd.DataFrame, column: str) -> float:
    """Share of rows agreeing with the majority sign, which is 1.0 for a perfectly consistent effect."""
    signs = np.sign(frame[column].to_numpy())
    signs = signs[signs != 0]
    if signs.size == 0:
        return float("nan")
    return float(max((signs > 0).mean(), (signs < 0).mean()))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.archive)
    print(f"{len(pairs)} antithetic pairs over {pairs['direction_id'].nunique()} directions, ")
    print(f"targets {sorted(pairs['target'].unique())}, TV levels {sorted(pairs['phase_tv'].unique())}\n")

    rows = []
    for (target, family), group in pairs.groupby(["target", "direction_family"]):
        sigma = RUN_SIGMA[target]
        odd = group["order_half_effect_plus_minus"].to_numpy()
        cost = group["symmetric_asymmetry_cost"].to_numpy()
        # A two-phase policy beats its tied control only when the odd part is larger in magnitude than
        # the cost of being asymmetric at all, so that comparison is the operative one.
        can_win = np.abs(odd) > cost
        test = stats.ttest_1samp(np.abs(odd) - cost, 0.0) if len(group) >= MIN_PAIRS_FOR_TEST else None
        record = {
            "target": target,
            "direction_family": family,
            "pairs": len(group),
            "mean_abs_odd": float(np.mean(np.abs(odd))),
            "mean_cost": float(np.mean(cost)),
            "mean_margin": float(np.mean(np.abs(odd) - cost)),
            "margin_p_value": float(test.pvalue) if test is not None else float("nan"),
            "pairs_where_two_phase_wins": int(can_win.sum()),
            "odd_sign_agreement": sign_agreement(group, "order_half_effect_plus_minus"),
        }
        rows.append(record)
        print(f"  {target:<12} {family:<24} n={len(group):>3}")
        print(
            f"    mean |odd| {record['mean_abs_odd'] / sigma:+5.2f}s   mean cost {record['mean_cost'] / sigma:+5.2f}s   "
            f"margin {record['mean_margin'] / sigma:+5.2f}s"
            + (f"  p={record['margin_p_value']:.3f}" if test is not None else "")
        )
        print(
            f"    two-phase beats tied in {record['pairs_where_two_phase_wins']}/{len(group)} pairs   "
            f"odd-sign agreement {record['odd_sign_agreement'] * 100:.0f}%"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "family_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("DOES THE ORDER EFFECT AGREE ACROSS OBJECTIVES AT FIXED AGGREGATE AND FIXED SCALE?")
    print("=" * 100)
    per_direction = []
    for direction, group in pairs.groupby("direction_id"):
        by_target = group.groupby("target")["order_half_effect_plus_minus"].mean()
        if len(by_target) < 2:
            continue
        signs = np.sign(by_target.to_numpy())
        agree = bool(np.all(signs == signs[0]))
        per_direction.append({"direction_id": direction, "agrees_across_targets": agree, **by_target.to_dict()})
        rendered = "  ".join(f"{name} {value:+.6f}" for name, value in by_target.items())
        print(f"  {direction:<32}{rendered}   {'AGREE' if agree else 'REVERSES'}")
    directions = pd.DataFrame(per_direction)
    directions.to_csv(args.output_dir / "direction_cross_target.csv", index=False)

    print("\n" + "=" * 100)
    print("OUT-OF-SAMPLE ORIENTATION TRANSFER: the same test without the winner's curse")
    print("=" * 100)
    print("  Choosing the better orientation after seeing its outcome inflates every margin above, so")
    print("  the orientation is taken from the OTHER objective and applied blind to this one.")
    transfer = orientation_transfer(pairs)
    transfer.to_csv(args.output_dir / "orientation_transfer.csv", index=False)
    for _, row in transfer.iterrows():
        sigma = RUN_SIGMA[row["evaluated_on"]]
        print(f"\n  orientation chosen on {row['chosen_on']}, evaluated on {row['evaluated_on']}")
        print(
            f"    directions {int(row['directions'])}   orientation correct {int(row['orientation_correct'])}"
            f"/{int(row['directions'])}   blind gain vs tied {row['mean_gain_vs_tied']:+.6f} "
            f"({row['mean_gain_vs_tied'] / sigma:+.2f}s)  p={row['gain_p_value']:.3f}"
        )
        print(
            f"    post-hoc best-orientation gain for comparison {row['mean_posthoc_gain']:+.6f} "
            f"({row['mean_posthoc_gain'] / sigma:+.2f}s)  -> selection inflates by "
            f"{(row['mean_posthoc_gain'] - row['mean_gain_vs_tied']) / sigma:+.2f}s"
        )

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    total_pairs = int(summary["pairs"].sum())
    wins = int(summary["pairs_where_two_phase_wins"].sum())
    agreeing = int(directions["agrees_across_targets"].sum()) if len(directions) else 0
    print(f"  antithetic pairs analyzed                            : {total_pairs}")
    print(f"  directions whose order effect agrees on sign          : {agreeing}/{len(directions)}")
    print(f"  pairs where post-hoc best |odd| exceeds the cost      : {wins}/{total_pairs}  (BIASED, see above)")
    blind = transfer[(transfer["gain_p_value"] < 0.05) & (transfer["mean_gain_vs_tied"] < 0)]
    print(f"  transfer directions with a significant blind gain     : {len(blind)}/{len(transfer)}")
    if len(blind):
        for _, row in blind.iterrows():
            print(
                f"    {row['chosen_on']} -> {row['evaluated_on']}: {row['mean_gain_vs_tied']:+.6f} BPB "
                f"(p={row['gain_p_value']:.3f})"
            )
    else:
        print("    none -- with the orientation chosen blind, no transfer direction beats its tied control")
    print(f"\nwrote {args.output_dir}")


def orientation_transfer(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pick each direction's orientation on one objective, then score it blind on the other.

    Comparing the better orientation against a tied control after seeing both outcomes is a selection
    statistic, not an effect: with two arms it wins roughly half the time on noise alone. Choosing the
    orientation on an independent objective removes that, and the gap between the two numbers is the
    size of the curse.
    """
    targets = sorted(pairs["target"].unique())
    assert len(targets) == 2, f"orientation transfer needs exactly two objectives, found {targets}"
    rows = []
    for chosen_on, evaluated_on in ((targets[0], targets[1]), (targets[1], targets[0])):
        chooser = pairs[pairs["target"] == chosen_on].groupby("direction_id")
        evaluator = pairs[pairs["target"] == evaluated_on].groupby("direction_id")
        blind_gains, posthoc_gains, correct = [], [], 0
        for direction, evaluated in evaluator:
            if direction not in chooser.groups:
                continue
            # Sign of the odd effect on the other objective says which orientation to deploy here.
            prefer_plus = float(chooser.get_group(direction)["order_half_effect_plus_minus"].mean()) < 0.0
            plus = float(evaluated["plus_bpb"].mean())
            minus = float(evaluated["minus_bpb"].mean())
            tied = float(evaluated["same_seed_control_bpb"].mean())
            blind_gains.append((plus if prefer_plus else minus) - tied)
            posthoc_gains.append(min(plus, minus) - tied)
            correct += int(prefer_plus == (plus < minus))
        blind = np.asarray(blind_gains)
        result = stats.ttest_1samp(blind, 0.0) if blind.size >= MIN_PAIRS_FOR_TEST else None
        rows.append(
            {
                "chosen_on": chosen_on,
                "evaluated_on": evaluated_on,
                "directions": blind.size,
                "orientation_correct": correct,
                "mean_gain_vs_tied": float(np.mean(blind)),
                "gain_p_value": float(result.pvalue) if result is not None else float("nan"),
                "mean_posthoc_gain": float(np.mean(posthoc_gains)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
