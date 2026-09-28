# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Score 39-bucket surrogates by the policy they SELECT, against baselines sharing the panel (ATOM-033).

Regret against the best observed row is not comparable across targets or panels. That reference is a
minimum over noisy outcomes, so it sits below the true best by the expected minimum of n draws: at 3e18
that is -0.01123 BPB on Table 9 against -0.00223 on Uncheatable, purely because Table 9's run noise is
five times larger. Raw regret therefore charges a noisier target about five times more for identical
skill, and 43% of Table 9's apparent Regret@1 at 3e18 is that floor rather than model error. The 300M
heldout has no replicated coordinates at all, so its noise cannot be estimated this way in the first place.

The primary metric is the SELECTED-POLICY VALUE: the actual held-out outcome of the policy a rule picks,
compared against what other rules pick on the same panel. Both are single noisy draws from one panel, so
the selection floor cancels in the difference and the comparison is paired.

Baselines see exactly what the surrogate sees -- the fit panel and the held-out COORDINATES, never the
held-out outcomes:

  proportional  the token-proportional policy, or the observed row nearest it
  best_fit_tied the best single-phase policy in the FIT panel, transferred to its nearest held-out row.
                This is the one-phase incumbent and the bar a two-phase claim has to clear.
  nearest       the held-out row nearest the fit panel's best row: a memorisation control
  random        panel mean, the no-skill floor

Beating `random` earns nothing. A candidate has to beat `best_fit_tied` to matter and `nearest` to show it
models rather than memorises. Noise-corrected regret is reported as a secondary diagnostic with its floor
stated, never as the gate.

Usage: ``uv run python ... [--scales 300m,delphi_3e18] [--seeds 3]``
"""

import argparse
import collections
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
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.stats import norm  # noqa: E402

TARGETS = (swarm39.UNCHEATABLE, swarm39.TABLE9)
SELECTION_K = (1, 3, 20)


def run_noise(held, target: str) -> float:
    """Median within-coordinate standard deviation; nan when nothing is replicated."""
    groups = collections.defaultdict(list)
    for index, (early, late) in enumerate(zip(np.round(held.phase0, 8), np.round(held.phase1, 8), strict=True)):
        groups[(tuple(early), tuple(late))].append(index)
    values = held.targets[target]
    spreads = [
        float(np.std(values[np.asarray(members)], ddof=1))
        for members in groups.values()
        if len(members) > 1 and np.isfinite(values[np.asarray(members)]).all()
    ]
    return float(np.median(spreads)) if spreads else float("nan")


def selection_floor(noise: float, count: int) -> float:
    """How far below the true best the best OBSERVED row sits, from selection on noise alone."""
    return float(noise * norm.ppf(1.0 / (count + 1))) if np.isfinite(noise) else float("nan")


def _total_variation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(left - right).sum(axis=-1)


def baselines(fit, held, observed: np.ndarray, target: str) -> dict[str, float]:
    """Selected-policy value for each rule that never sees a held-out outcome."""
    early, late = held.phase0[observed], held.phase1[observed]
    values = held.targets[target][observed]
    result: dict[str, float] = {"random": float(values.mean())}

    proportional = fit.proportional
    nearness = _total_variation(early, proportional) + _total_variation(late, proportional)
    result["proportional"] = float(values[int(np.argmin(nearness))])

    fit_values = fit.targets[target]
    usable = np.isfinite(fit_values)
    fit_separation = _total_variation(fit.phase1, fit.phase0)
    tied = usable & (fit_separation <= 1e-9)
    if tied.any():
        best_tied = int(np.flatnonzero(tied)[np.argmin(fit_values[tied])])
        distance = _total_variation(early, fit.phase0[best_tied]) + _total_variation(late, fit.phase1[best_tied])
        result["best_fit_tied"] = float(values[int(np.argmin(distance))])

    # The empirical one-phase bar: the best single-phase policy actually observed in this panel. It uses
    # held-out outcomes, which is correct for a BAR -- it is the thing a two-phase claim must beat, not a
    # competitor that has to stay blind. The transferred fit-panel version above is the deployable proxy,
    # and at 3e18 it lands at 1.13723, worse than the panel mean, so it is a poor stand-in for the bar.
    held_separation = _total_variation(late, early)
    held_tied = held_separation <= 1e-9
    if held_tied.any():
        result["best_heldout_tied"] = float(values[held_tied].min())

    best_fit = int(np.flatnonzero(usable)[np.argmin(fit_values[usable])])
    distance = _total_variation(early, fit.phase0[best_fit]) + _total_variation(late, fit.phase1[best_fit])
    result["nearest"] = float(values[int(np.argmin(distance))])
    return result


def selected_values(prediction: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """The best actual outcome among the k policies a prediction ranks highest."""
    order = np.argsort(prediction)
    return {f"selected@{k}": float(values[order[:k]].min()) for k in SELECTION_K}


POOLINGS = ("semantic", "strata")


def bucket_pooling(fit, pooling: str) -> np.ndarray:
    """Which buckets share readout parameters.

    ``semantic`` is the hand-assigned family partition. ``strata`` cuts equal-count strata on log epochs
    per unit weight, at the same ``n_strata`` the model already commits to for ``Shape.boundary_scale``,
    so it needs no knowledge of what any bucket contains.
    """
    if pooling == "semantic":
        return fit.family_index
    probe = gen.Panel(np.stack([fit.phase0, fit.phase1], axis=1), fit.c0, fit.c1, fit.family_index)
    return probe.exposure_stratum()


def panels(scale: str, target: str, pooling: str = "semantic"):
    """Fit and held-out panels restricted to rows whose outcome exists."""
    fit, held = swarm39.load_scale(scale)
    fit_ok = np.isfinite(fit.targets[target])
    held_ok = np.isfinite(held.targets[target])
    index = bucket_pooling(fit, pooling)
    fit_panel = gen.Panel(np.stack([fit.phase0[fit_ok], fit.phase1[fit_ok]], axis=1), fit.c0, fit.c1, index)
    held_panel = gen.Panel(np.stack([held.phase0[held_ok], held.phase1[held_ok]], axis=1), fit.c0, fit.c1, index)
    return fit, held, fit_ok, held_ok, fit_panel, held_panel


def report(scale: str, target: str, candidates: dict[str, list[np.ndarray]]) -> None:
    fit, held, _fit_ok, held_ok, _fp, _hp = panels(scale, target)
    values = held.targets[target][held_ok]
    noise = run_noise(held, target)
    floor = selection_floor(noise, len(values))
    base = baselines(fit, held, held_ok, target)

    single_phase = _total_variation(held.phase1[held_ok], held.phase0[held_ok]) <= 1e-9
    headroom = float(values[single_phase].min() - values.min()) if single_phase.any() else float("nan")

    print(f"\n=== {scale} / {target} ===")
    print(f"  heldout n={len(values)}  of which single-phase {int(single_phase.sum())}")
    print(f"  best observed {values.min():.5f}  best single-phase {values[single_phase].min():.5f}  ")
    print(f"  run noise {noise:.5f}  selection floor on the reference {floor:+.5f}")
    ratio = headroom / noise if np.isfinite(noise) and noise > 0 else float("nan")
    print(f"  TWO-PHASE HEADROOM in this panel {headroom:+.5f} = {ratio:.2f} run noises")
    if not single_phase.all() and headroom <= 0:
        print("      no two-phase row beats the best single-phase row: nothing here for a surrogate to find")
    if single_phase.all():
        print("      every held-out row is single-phase: this panel cannot test a two-phase claim at all")

    order = sorted(base.items(), key=lambda item: item[1])
    print("  baselines (selected-policy value, lower is better):")
    for name, value in order:
        print(f"      {name:16s} {value:.5f}")

    # The matched comparison. `best_heldout_tied` is an ORACLE over the single-phase rows -- it reads
    # held-out outcomes -- so scoring a blind surrogate against it charges the surrogate for the oracle's
    # selection advantage as well as for any modelling error. Letting the same surrogate, on the same
    # information, pick first from every row and then from the single-phase rows alone isolates what the
    # two-phase freedom is worth, with the selection floor common to both sides.
    print("  same surrogate, candidate set restricted vs unrestricted; positive = two-phase freedom paid:")
    for name, predictions in candidates.items():
        unrestricted = np.array([selected_values(p, values)["selected@1"] for p in predictions])
        restricted = np.array(
            [selected_values(p[single_phase], values[single_phase])["selected@1"] for p in predictions]
        )
        gain = float(np.median(restricted) - np.median(unrestricted))
        print(
            f"      {name:22s} all {np.median(unrestricted):.5f} [{unrestricted.min():.5f},"
            f"{unrestricted.max():.5f}]  single-phase-only {np.median(restricted):.5f}   realised {gain:+.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="300m,delphi_3e18")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    for scale in args.scales.split(","):
        for target in TARGETS:
            candidates = {}
            for pooling in POOLINGS:
                fit, _held, fit_ok, _held_ok, fit_panel, held_panel = panels(scale, target, pooling)
                response = fit.targets[target][fit_ok]
                candidates[f"split / {pooling}"] = [
                    split_damage.predict(
                        held_panel, split_damage.fit_variant(fit_panel, response, "split", seed), "split"
                    )
                    for seed in range(args.seeds)
                ]
            report(scale, target, candidates)


if __name__ == "__main__":
    main()
