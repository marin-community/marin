# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Extend the fixed-aggregate phase fiber until the selection claim is testable (ATOM-034).

The 2026-08-23 round established that the existing surrogate predicts fixed-aggregate phase contrasts on
the 3e18 fiber at Pearson r = +0.519, 95% CI [+0.316, +0.658], p < 0.0001 over 96 antithetic directions,
across three atomic targets. What it could NOT establish is that the prediction converts into a policy
gain: selecting the top five directions by prediction beats a random five in 6 of 6 target-anchor cells at
a median 1.90 run noises, yet no cell survives Holm correction, and the three targets share the same runs
so the independent unit is the ANCHOR -- of which the fiber has two.

Two anchors cannot certify a selection effect. This design adds anchors, which is the only thing that can.

WHAT IS BEING TESTED

Per anchor, all policies share one aggregate mixture, so nothing here can be won by choosing a better
aggregate -- the confound that made pooled top-5 selection look like a 7.7 run-noise gain when most of it
was picking the better of the two existing anchors. Within an anchor the surrogate ranks the antithetic
directions, and the gate asks whether its top five beat five drawn at random. The tied control at the
anchor is replicated so the one-phase comparison has a real standard error rather than a single draw.

SIZING

The observed per-anchor advantage on the primary target is 0.80 and 2.75 run noises at the two existing
anchors: mean 1.77, between-anchor sd 1.38. A one-sided 95% test at 80% power therefore needs 4 anchors;
6 are budgeted so the test survives one anchor failing to materialise, and so a leave-one-anchor-out check
is possible.

Twenty directions per anchor rather than the existing forty-nine: the statistic is "best of five selected
from the pool", and a pool of twenty keeps that a genuine 25% selection while cutting the per-anchor cost
by more than half. Four replicated controls per anchor, matching the existing design.

  6 anchors x (20 directions x 2 signs + 4 controls) = 264 runs, about 132 TPU-hours at 3e18.

DESIGN RULES CARRIED FORWARD FROM ATOM-030

Nothing here refers to what a bucket contains. Directions are drawn zero-sum and chosen by maximin on the
unit direction, anchors come from the observed panel rather than from a fitted argmin, no bucket exceeds
ten materialized epochs in a phase, and weights below one sample per block are snapped to zero. Each of
those was violated at some point in ATOM-030 while the manifest still looked reasonable, so each is
asserted here rather than assumed.

Usage: ``uv run python ... [--anchors 6] [--directions 20] [--out <dir>]``
"""

import argparse
import hashlib
import json
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

PANEL = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_observed_components_20260724" / "observed_component_panel.csv"
DEFAULT_OUT = SCRIPT_DIR / "reference_outputs" / "phase_fiber_extension_20260823"
PRIMARY_TARGET = "olmo_base_eval/easy_bpb/mt_mbpp_cpp/bpb"
SECONDARY_TARGETS = (
    "olmo_base_eval/easy_bpb/hellaswag/bpb",
    "olmo_base_eval/easy_bpb/mt_mbpp_python/bpb",
)
DESIGN_SEED = 20260823
MAX_EPOCHS_PER_PHASE = 10.0
BLOCK_SIZE = 4096
CONTROL_SEEDS = 4
SEPARATIONS = (0.02, 0.05, 0.09, 0.14)  # spanning the existing fiber's range, median 0.013 to max 0.156
MINIMUM_DIRECTIONAL_DIMENSIONS = 8.0


def stable_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return DESIGN_SEED + int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big") % 100_000


def snap(weights: np.ndarray) -> np.ndarray:
    """Zero any weight that cannot earn one sample per block, then renormalise."""
    snapped = np.where(weights * BLOCK_SIZE < 1.0, 0.0, weights)
    total = snapped.sum()
    if total <= 0:
        raise ValueError("snapping removed all weight")
    return snapped / total


def _below_block(weights: np.ndarray) -> bool:
    """Any weight that is positive but too small to earn a sample per block."""
    positive = weights[weights > 0.0]
    return bool(positive.size and positive.min() < 1.0 / BLOCK_SIZE)


def separation_of(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(right - left).sum())


def max_epochs(weights: np.ndarray, rate: np.ndarray) -> float:
    return float(np.max(weights * rate))


def zero_sum_directions(count: int, base: np.ndarray, seed: int) -> list[np.ndarray]:
    """Unit-total-variation zero-sum directions PROPORTIONAL to the anchor, drawn label-blind.

    An additive direction moves every bucket by the same amount, so on a snapped anchor -- where many
    weights sit at zero or just above one sample per block -- it pushes small buckets negative and the
    design rejects them. The first revision lost 81% of its directions that way. Scaling each component by
    its own anchor weight makes the perturbation multiplicative, so a bucket can never be driven below zero
    and small buckets move proportionally little. This is a statement about the numeric weight, not about
    what the bucket contains.
    """
    generator = np.random.default_rng(seed)
    found: list[np.ndarray] = []
    while len(found) < count:
        draw = generator.normal(size=len(base)) * base
        draw -= base * (draw.sum() / max(base.sum(), 1e-12))
        scale = 0.5 * np.abs(draw).sum()
        if scale > 1e-9:
            found.append(draw / scale)
    return found


def directional_dimensions(rows: list[np.ndarray]) -> float:
    centred = np.stack(rows) - np.mean(rows, axis=0)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    share = spectrum / spectrum.sum()
    return float(np.exp(-np.sum(share * np.log(share + 1e-300))))


def anchors(frame: pd.DataFrame, buckets: list[str], fit, count: int) -> list[dict]:
    """Aggregate mixtures to build fibers around, taken from observed rows across the outcome range.

    Drawn from what was actually run rather than from a fitted argmin: the surrogate's own optimum sits
    0.649 total variation from the best observed one-phase policy, so anchoring on it would put every new
    fiber in a region the panel says is bad.
    """
    early = frame[["phase_0_" + b for b in buckets]].to_numpy(float)
    late = frame[["phase_1_" + b for b in buckets]].to_numpy(float)
    aggregate = fit.alpha * early + (1.0 - fit.alpha) * late
    outcome = frame[PRIMARY_TARGET].to_numpy(float)
    rate = np.asarray(fit.c0, dtype=float)
    usable = np.flatnonzero(np.isfinite(outcome))
    usable = np.array([i for i in usable if max_epochs(snap(aggregate[i]), rate) <= MAX_EPOCHS_PER_PHASE])
    # Span the observed outcome range so the claim is not confined to one quality band.
    quantiles = np.linspace(0.02, 0.60, count)
    chosen, taken = [], set()
    for q in quantiles:
        target = np.quantile(outcome[usable], q)
        order = usable[np.argsort(np.abs(outcome[usable] - target))]
        for index in order:
            key = tuple(np.round(aggregate[index], 8))
            if key not in taken:
                taken.add(key)
                chosen.append(
                    {
                        "source_row": int(index),
                        "aggregate": snap(aggregate[index]),
                        "observed": float(outcome[index]),
                        "quantile": float(q),
                    }
                )
                break
    if len(chosen) != count:
        raise ValueError(f"selected {len(chosen)} anchors, expected {count}")
    return chosen


def build(anchor_count: int, direction_count: int) -> pd.DataFrame:
    fit, _held = swarm39.load_scale("delphi_3e18")
    buckets = list(fit.buckets)
    frame = pd.read_csv(PANEL)
    rate_early = np.asarray(fit.c0, dtype=float)
    rate_late = np.asarray(fit.c1, dtype=float)
    rows: list[dict] = []

    for anchor_index, anchor in enumerate(anchors(frame, buckets, fit, anchor_count)):
        base = anchor["aggregate"]
        pool = zero_sum_directions(direction_count * 12, base, stable_seed("dir", anchor_index))
        # Maximin on the unit direction: spread the fiber over the simplex rather than along one axis.
        picked = [pool[0]]
        while len(picked) < direction_count:
            spread = [min(separation_of(c, t) for t in picked) for c in pool]
            picked.append(pool[int(np.argmax(spread))])
            pool.pop(int(np.argmax(spread)))
        if directional_dimensions(picked) < MINIMUM_DIRECTIONAL_DIMENSIONS:
            raise ValueError(f"anchor {anchor_index}: directions span too few dimensions")

        for control in range(CONTROL_SEEDS):
            rows.append(
                _row(
                    fit,
                    buckets,
                    anchor_index,
                    anchor,
                    base,
                    base,
                    "center_control",
                    direction_id=-1,
                    sign="center",
                    seed_offset=control,
                )
            )
        for direction_index, direction in enumerate(picked):
            separation = SEPARATIONS[direction_index % len(SEPARATIONS)]
            for sign_name, sign in (("plus", +1.0), ("minus", -1.0)):
                # Parameterise by the CONTRAST, not by moving each phase independently. With
                # w0 = a - (1-alpha)d and w1 = a + alpha*d the aggregate alpha*w0 + (1-alpha)*w1 is
                # exactly `a` by construction, and the total variation between the phases is |d|.
                # Moving the phases separately and rescaling one by alpha/(1-alpha) = 3.95 is what
                # produced 0.53 separations and an 8.4e-02 aggregate drift in the first revision.
                delta = sign * separation * direction
                early = base - (1.0 - fit.alpha) * delta
                late = base + fit.alpha * delta
                # Reject rather than clip or re-snap: both destroy the exact aggregate cancellation.
                if early.min() < 0.0 or late.min() < 0.0:
                    continue
                if _below_block(early) or _below_block(late):
                    continue
                if max_epochs(early, rate_early) > MAX_EPOCHS_PER_PHASE:
                    continue
                if max_epochs(late, rate_late) > MAX_EPOCHS_PER_PHASE:
                    continue
                rows.append(
                    _row(
                        fit,
                        buckets,
                        anchor_index,
                        anchor,
                        early,
                        late,
                        "domain_vs_rest",
                        direction_id=direction_index,
                        sign=sign_name,
                        seed_offset=0,
                    )
                )
    return pd.DataFrame(rows)


def _row(fit, buckets, anchor_index, anchor, early, late, family, direction_id, sign, seed_offset) -> dict:
    return {
        "anchor_index": anchor_index,
        "anchor_source_row": anchor["source_row"],
        "anchor_observed_primary": anchor["observed"],
        "anchor_quantile": round(anchor["quantile"], 4),
        "contrast_family": family,
        "direction_id": direction_id,
        "sign": sign,
        "separation": round(separation_of(early, late), 6),
        "realised_aggregate_drift": round(
            separation_of(fit.alpha * early + (1 - fit.alpha) * late, anchor["aggregate"]), 8
        ),
        "phase_0_weights_json": json.dumps(dict(zip(buckets, np.round(early, 8), strict=True)), sort_keys=True),
        "phase_1_weights_json": json.dumps(dict(zip(buckets, np.round(late, 8), strict=True)), sort_keys=True),
        "data_seed": DESIGN_SEED + seed_offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--directions", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = build(args.anchors, args.directions)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "phase_fiber_extension_manifest.csv"
    manifest.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    drift = manifest["realised_aggregate_drift"]
    # Asserted: a fiber that has drifted is no longer aggregate-matched, and every gain it shows could be
    # aggregate selection rather than phase order -- the confound that inflated a 1.9 run-noise effect to
    # 7.7 when the two existing anchors were pooled.
    if drift.max() > 1e-9:
        raise ValueError(f"fibers are not fixed-aggregate: max drift {drift.max():.2e}")
    if manifest["separation"].max() > max(SEPARATIONS) + 1e-9:
        raise ValueError(f"separation exceeds the design ladder: {manifest['separation'].max():.4f}")
    print(f"wrote {path}  sha256 {digest[:16]}")
    print(f"  rows {len(manifest)}  anchors {manifest['anchor_index'].nunique()}")
    print(manifest.groupby("contrast_family").size().to_string())
    print(f"  separation: min {manifest['separation'].min():.4f} max {manifest['separation'].max():.4f}")
    print(f"  aggregate drift within a fiber: max {drift.max():.2e} (should be ~0; fibers are fixed-aggregate)")
    print(f"  budget: {len(manifest)} runs, about {len(manifest) * 0.5:.0f} TPU-h at 3e18")
    (args.out / "design.json").write_text(
        json.dumps(
            {
                "scale": "delphi_3e18",
                "primary_target": PRIMARY_TARGET,
                "secondary_targets": list(SECONDARY_TARGETS),
                "manifest_sha256": digest,
                "rows": len(manifest),
                "anchors": int(manifest["anchor_index"].nunique()),
                "gate": (
                    "per anchor, the surrogate's top-5 directions by predicted value against five drawn at "
                    "random; combined across anchors one-sided at 95%. Powered for the observed 1.77 "
                    "run-noise advantage with between-anchor sd 1.38."
                ),
                "requirements": [
                    "replicate the tied control at each anchor across CONTROL_SEEDS data seeds",
                    "fit the surrogate on the canonical 280 two-phase rows only",
                    "select directions by prediction before any new outcome is read",
                    "parent placement --region us-east5 --zone us-east5-a; MARIN_PREFIX gs://marin-us-east5",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
