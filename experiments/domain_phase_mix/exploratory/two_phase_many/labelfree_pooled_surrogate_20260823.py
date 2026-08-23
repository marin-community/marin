# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""A label-blind pooling for the two-phase surrogate, chosen by nested anchor folds (GEN-040).

GEN-001 pools every structure over the SEMANTIC bucket families -- broad_text, tech_code, reasoning. That
is a human labelling of what each bucket contains, it is badly unbalanced at 31/6/2, and a surrogate that
needs to be told which bucket is code cannot be deployed on a pool whose contents nobody has classified.

This replaces it with poolings derived from a measured numeric property, the bucket's epochs per unit
weight, which is available before training and says nothing about content:

  strata     equal-count bins of log10(c0 + c1); balanced at 13/13/13 for three groups
  smooth     Gaussian bumps over the same axis, so pooling weight varies continuously rather than by bin

Which pooling to use is not fixed. It is selected per target by LEAVE-ONE-ANCHOR-OUT on the fixed-aggregate
fiber, and evaluated on the anchor that selection never saw. That protocol matters more than the pooling:
every in-panel selector tried on this problem either ignores or inverts the fiber ordering -- fit-panel
cross-validation picks a pooling that scores 0.021 below the best available, and a near-aggregate-matched
pair proxy built from 300 fit-panel pairs ANTI-correlates with fiber performance at Spearman -0.50. There
is no known way to choose this hyperparameter without an aggregate-matched fiber.

Measured on the 3e18 fiber, selecting on one anchor and scoring on the other:

  HellaSwag     nested +0.541 against semantic +0.485, winning both folds (+0.073, +0.038)
  GitHub C++    nested +0.516 against semantic +0.516; selection returns the semantic pooling both times

So the labelling can be dropped at no cost, and on one target dropping it helps. Two anchors give two
folds, which is weak: the sign is consistent but 2/2 is p = 0.25. The fiber extension in
`design_phase_fiber_extension_20260823.py` raises that to six.

Usage: ``uv run python ... [--targets ...] [--out <dir>]``
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fit_swarm39_split_damage_20260817 as split_damage  # noqa: E402
import general_mixture_surrogate_20260809 as gen  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy import stats  # noqa: E402

PANEL = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_observed_components_20260724" / "observed_component_panel.csv"
DEFAULT_TARGETS = (
    "olmo_base_eval/easy_bpb/mt_mbpp_cpp/bpb",
    "olmo_base_eval/easy_bpb/hellaswag/bpb",
)
_ORIGINAL_FAMILY_SUMS = gen.family_sums


def log_rate(fit) -> np.ndarray:
    """Standardised log epochs-per-unit-weight: the pooling axis, measured and content-free."""
    rate = np.log10(np.asarray(fit.c0, dtype=float) + np.asarray(fit.c1, dtype=float))
    return (rate - rate.mean()) / rate.std()


def smooth_weights(axis: np.ndarray, groups: int, width: float) -> np.ndarray:
    """Row-normalised Gaussian bumps: a continuous pooling with no bin edges."""
    centres = np.linspace(axis.min(), axis.max(), groups)
    weights = np.exp(-0.5 * ((axis[None, :] - centres[:, None]) / width) ** 2)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def poolings(fit, budget: int | None) -> list[tuple[str, np.ndarray | None, np.ndarray]]:
    """Candidate poolings, coarsest first, truncated to what the selection folds can support.

    The semantic partition appears only as the baseline to beat; it is not a deployable fallback. The
    ordering is by increasing group count, because coarser pooling carries fewer free parameters and is
    the more stable choice at a 280-row fit budget. ``budget`` is the number of fiber anchors available
    to select on; ``None`` keeps the whole set, which lets the selection cost itself be measured.
    """
    axis = log_rate(fit)
    probe = gen.Panel(np.stack([fit.phase0, fit.phase1], axis=1), fit.c0, fit.c1, fit.family_index)
    candidates: list[tuple[str, np.ndarray | None, np.ndarray]] = [("semantic (baseline)", None, fit.family_index)]
    for groups, width in ((2, 1.1), (3, 0.8), (4, 0.6), (5, 0.5), (6, 0.45)):
        candidates.append((f"strata n={groups}", None, probe.exposure_stratum(groups)))
        candidates.append((f"smooth n={groups}", smooth_weights(axis, groups, width), np.arange(len(axis)) % groups))
    return candidates if budget is None else candidates[: max(2, budget + 1)]


def _install(weights: np.ndarray | None):
    """Swap the pooling used by the shared surrogate, restoring it afterwards."""
    if weights is None:
        gen.family_sums = _ORIGINAL_FAMILY_SUMS
    else:

        def pooled(values, family_index):
            return values @ weights.T

        gen.family_sums = pooled
    split_damage.gen.family_sums = gen.family_sums


def antithetic_pairs(frame: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Plus/minus rows of one direction at one anchor, with their measured contrast."""
    pairs = []
    for _direction, group in frame.groupby("direction_id"):
        if group["sign"].nunique() < 2 or len(group) < 2:
            continue
        group = group.sort_values("sign")
        low, high = group.iloc[0], group.iloc[-1]
        if high["y"] != low["y"]:
            pairs.append((int(low.name), int(high.name), float(high["y"] - low["y"])))
    return pairs


def contrast_correlation(prediction: np.ndarray, pairs: list[tuple[int, int, float]]) -> float:
    """How well predicted contrasts track measured ones -- the statistic sign accuracy throws away.

    Sign agreement reads 54.2% on this fiber, indistinguishable from chance, while the same predictions
    correlate at +0.52. The model is right on the large contrasts and noisy on the small ones, which is the
    useful way round for optimisation, so magnitude has to stay in the statistic.
    """
    predicted = np.array([prediction[j] - prediction[i] for i, j, _ in pairs])
    measured = np.array([value for _, _, value in pairs])
    if np.std(predicted) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(predicted, measured).statistic)


def selection_capture(predicted: np.ndarray, pairs: list[tuple[int, int, float]], measured: np.ndarray) -> float:
    """Realized share of the oracle's best-of-two gain, on the sign choices the model makes.

    Zero is a coin flip and one is the oracle, so a negative value means the model is actively worse
    than not modelling at all. This is the promotion criterion; contrast correlation is not, because a
    model can rank the bulk of a fiber well and still put its optimum on the wrong side.
    """
    realized = 0.0
    ideal = 0.0
    for low, high, _contrast in pairs:
        centre = 0.5 * (measured[low] + measured[high])
        pick = low if predicted[low] <= predicted[high] else high
        realized += centre - measured[pick]
        ideal += centre - min(measured[low], measured[high])
    return float(realized / ideal) if ideal > 0 else float("nan")


def evaluate(target: str, budget: int | None, seeds: int = 1) -> pd.DataFrame:
    """Per-pooling contrast correlation at every anchor, from one fit on the canonical rows."""
    fit, _held = swarm39.load_scale("delphi_3e18")
    buckets = list(fit.buckets)
    frame = pd.read_csv(PANEL)
    early = frame[["phase_0_" + b for b in buckets]].to_numpy(float)
    late = frame[["phase_1_" + b for b in buckets]].to_numpy(float)
    outcome = frame[target].to_numpy(float)
    panel = frame["panel"].to_numpy()
    train = np.flatnonzero((panel == "two_phase_fit") & np.isfinite(outcome))

    rows = []
    for label, weights, index in poolings(fit, budget):
        _install(weights)
        try:
            fitted = [
                split_damage.fit_variant(
                    gen.Panel(np.stack([early[train], late[train]], axis=1), fit.c0, fit.c1, index),
                    outcome[train],
                    "split",
                    seed,
                )
                for seed in range(seeds)
            ]
            for anchor in pd.unique(frame.loc[panel == "frontier_phase_fiber", "anchor_id"]):
                selected = np.flatnonzero(
                    (panel == "frontier_phase_fiber") & (frame["anchor_id"].to_numpy() == anchor) & np.isfinite(outcome)
                )
                query = gen.Panel(np.stack([early[selected], late[selected]], axis=1), fit.c0, fit.c1, index)
                prediction = np.mean([split_damage.predict(query, f, "split") for f in fitted], axis=0)
                local = frame.iloc[selected].reset_index(drop=True)
                local["y"] = outcome[selected]
                pairs = antithetic_pairs(local)
                if len(pairs) >= 10:
                    rows.append(
                        {
                            "pooling": label,
                            "anchor": anchor,
                            "r": contrast_correlation(prediction, pairs),
                            "capture": selection_capture(prediction, pairs, local["y"].to_numpy(float)),
                        }
                    )
        finally:
            _install(None)
    return pd.DataFrame(rows)


def nested(scores: pd.DataFrame) -> pd.DataFrame:
    """Select the pooling on every OTHER anchor, then score the held-out one."""
    table = scores.pivot(index="pooling", columns="anchor", values="r")
    rows = []
    for held in table.columns:
        others = table.drop(columns=[held]).mean(axis=1)
        chosen = str(others.idxmax())
        rows.append(
            {
                "held_out_anchor": held,
                "selected_pooling": chosen,
                "r": float(table.loc[chosen, held]),
                "semantic_r": float(table.loc["semantic (baseline)", held]),
            }
        )
    return pd.DataFrame(rows)


def family_of(pooling: str) -> str:
    """Which construction a pooling came from, so results can be read at family level."""
    return "semantic" if "semantic" in pooling else pooling.split()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = parser.parse_args()
    targets = args.targets.split(",")

    collected = []
    for target in targets:
        scores = evaluate(target, None)
        if scores.empty:
            print(f"{target}: no usable anchors")
            continue
        scores["cell"] = target.split("/")[-2] + "/" + scores["anchor"].astype(str)
        collected.append(scores)
    if not collected:
        return
    scores = pd.concat(collected)

    capture = scores.pivot(index="pooling", columns="cell", values="capture")
    correlation = scores.pivot(index="pooling", columns="cell", values="r")
    capture["worst"] = capture.min(axis=1)
    capture["mean"] = capture.drop(columns=["worst"]).mean(axis=1)

    print("=== share of oracle best-of-two gain captured, per anchor x target cell ===")
    print(capture.sort_values("worst", ascending=False).to_string(float_format=lambda v: f"{v:+.3f}"))

    capture["family"] = [family_of(name) for name in capture.index]
    print("\nworst-cell capture by construction:")
    print(capture.groupby("family")["worst"].agg(["min", "max", "count"]).to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\nrank agreement between contrast correlation and capture:")
    for cell in correlation.columns:
        print(f"  {cell:42s} spearman {stats.spearmanr(correlation[cell], capture[cell]).statistic:+.3f}")

    print("\nnested leave-one-anchor-out selection, per target (2 anchors is not enough to select on):")
    for target in targets:
        subset = scores[scores["cell"].str.startswith(target.split("/")[-2])]
        if subset.empty:
            continue
        folds = nested(subset.drop(columns=["cell"]))
        print(f"  {target.split('/')[-2]:14s} r {folds['r'].mean():+.3f} vs semantic {folds['semantic_r'].mean():+.3f}")


if __name__ == "__main__":
    main()
