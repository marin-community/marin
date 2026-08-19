# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Generate and freeze the 3e18 shared-prefix search (ATOM-030).

The archive cannot answer the two-phase question because it never crosses the phases: of 1825 distinct
phase-0 mixtures, not one appears with more than a single phase-1 mixture. This design manufactures that
crossing -- a common set of continuations run from every prefix -- which is the structure a
state-conditioned critic and a one-step policy improvement both require, and the structure no amount of
refitting the existing cloud can supply.

WHAT EACH PART IS FOR

Sixteen FULL PREFIXES, each trained to the phase boundary (0.8 of a run) and then branched. Six sampled to
span the pre-boundary readout terciles, because whether a continuation preference transfers across prefix
QUALITY is the open question. Three drawn from independent fits of the endpoint surrogate -- probes, not
bets: that argmin moves 0.20 to 0.60 total variation across fold seeds and three independent swarms agree
on it at chance. Two references: the token-proportional policy, and the best one-phase policy by endpoint
so that "does two-phase beat one-phase" is a paired within-design comparison rather than two separately
fitted surfaces compared after the fact. One exploratory prefix in the high-separation region the argmin
keeps recommending and the archive barely covers. Four replicates of sampled prefixes, retrained under
different data seeds, to separate prefix-level from branch-level noise.

Per full prefix, twenty-six BRANCHES (0.2 of a run each):

  tied (1)      w1 = w0. Anchors the recency channel at every prefix and is the baseline the certified
                improvement is measured against. Without it there is no within-prefix control.
  common (17)   identical ABSOLUTE mixtures at every prefix, giving 16x17 = 272 crossed cells. Absolute
                rather than relative because the critic is Q(state, continuation), so the object that must
                repeat across states is the continuation itself. Laid out on a separation ladder along the
                measured code-late direction, its reverse, and two orthogonal directions, so that a null
                result -- if that is what comes back -- covers the range where an effect could live rather
                than only the near-tied neighbourhood.
  local (8)     prefix-specific probes for the conditional response, which is what the one-step
                improvement differentiates.

Four of the sixteen also carry five REPLICATE branches of one common continuation under different phase-1
seeds. That is gate zero: prefix-conditional branch noise is the denominator of every signal-to-noise
ratio in the analysis and decides whether best-of-M selection is safe at all. Four cells rather than
sixteen because five runs already pin a standard deviation, and four independent cells also test whether
that noise varies with the prefix. At 39 buckets, 72% of an apparent per-aggregate headroom once turned
out to be reproducible from selection noise alone.

The two replicate prefixes carry only the tied anchor and four common continuations. They exist to show
that the prefix-to-continuation response reproduces under a retrained prefix; that needs a handful of
shared branches, not a full set.

BUDGET. Eval dominates training here: a branch costs 0.2 run-hours to train and about 0.3 to evaluate on
Table 9, so more than half the marginal cost of a branch is its eval. That is why the replicate structure
is deliberately thin.

Usage: ``uv run python ... [--out <dir>]``
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

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_prefix_search_evidence_20260819 as evidence,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_swarm39_optimum_degeneracy_20260817 as degeneracy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_swarm39_split_damage_20260817 as split_damage,
)

DEFAULT_OUT = SCRIPT_DIR / "reference_outputs" / "prefix_search_design_20260819"
# The ladder is expressed in CODE-FAMILY SHARE, not in an abstract step along a direction. Two reasons.
# It is the coordinate the ordering evidence is stated in, so the rungs are interpretable. And it cannot
# saturate: an earlier version stepped along a normalised direction and silently produced duplicate
# mixtures, because the code family holds only 5.3% of the proportional policy, so "move code out" caps at
# 0.053 total variation and every rung below it collapses onto the same point.
# `None` is the proportional policy's own code share, substituted exactly at build time. Writing the
# rounded 0.0531 instead leaves it 1.5e-5 total variation from the proportional reference prefix, which
# is not a distinguishable policy but IS a distinct manifest row -- so that prefix would train its tied
# anchor and this rung as two near-identical runs under the same seed, which is pure waste. Exact
# equality instead makes the pair a genuine duplicate that `_assert_no_duplicate_runs` reseeds into a
# free branch-noise cell.
CODE_SHARE_LADDER = (0.0, 0.02, None, 0.10, 0.20, 0.35, 0.55, 0.80, 1.0)
ORTHOGONAL_SEPARATIONS = (0.10, 0.20, 0.30, 0.40)
LOCAL_SEPARATION = 0.15
# Gate zero replicates a mid-ladder rung: far enough from the prefix that any real branch effect is
# present, not so far that the measurement only describes an extreme policy nobody would deploy.
GATE_ZERO_CONTINUATION = "code_share_0200"
CERTIFICATION_SEEDS = 3
TABLE9_RUN_NOISE = 0.00342  # measured on the archive; see ATOM-029 claim 2
DESIGN_SEED = 20260819
COMMON_COUNT = 17
LOCAL_COUNT = 8
GATE_ZERO_REPLICATES = 5
GATE_ZERO_PREFIXES = 4  # which full prefixes carry the replicate branches
REPLICATE_PREFIX_BRANCHES = 4  # common continuations kept at a retrained prefix, besides the tied anchor
PHASE_0_FRACTION = 0.8
TRAIN_HOURS_PER_RUN = 1.0
EVAL_HOURS_TABLE9 = 0.3


def stable_seed(*parts) -> int:
    """A seed derived from the design seed and a label, reproducibly across processes.

    Python's built-in `hash` is salted per interpreter, so a frozen design keyed on it would regenerate
    differently tomorrow. This is why the manifest hashes to the same digest on every run.
    """
    payload = "|".join(str(part) for part in parts).encode()
    return DESIGN_SEED + int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big") % 100_000


def family_direction(fit, family: str, sign: float = 1.0) -> np.ndarray:
    """A unit-total-variation direction moving mass into (or out of) one family.

    Mass is taken from the other families and added within the target family in proportion to the
    token-proportional policy, so the direction expresses a FAMILY-level shift without also asserting an
    arbitrary opinion about allocation inside a family.
    """
    index = list(fit.family_names).index(family)
    inside = fit.family_index == index
    base = fit.proportional
    direction = np.zeros(len(base))
    direction[inside] = base[inside] / max(base[inside].sum(), 1e-12)
    direction[~inside] = -base[~inside] / max(base[~inside].sum(), 1e-12)
    return sign * direction / (0.5 * np.abs(direction).sum())


def orthogonal_directions(fit, primary: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    """Zero-sum directions orthogonal to the primary one, for coverage away from it.

    Without these the design would only probe the direction already believed to matter, and a null result
    would say nothing about every other way the two phases could differ.
    """
    generator = np.random.default_rng(seed)
    found: list[np.ndarray] = []
    while len(found) < count:
        draw = generator.normal(size=len(primary))
        draw -= draw.mean()
        draw -= (draw @ primary) / (primary @ primary) * primary
        scale = 0.5 * np.abs(draw).sum()
        if scale > 1e-9:
            found.append(draw / scale)
    return found


def separation_of(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(right - left).sum())


def set_family_share(fit, base: np.ndarray, family: str, target: float) -> np.ndarray:
    """Rescale so `family` holds exactly `target` of the mass, keeping proportions inside each group.

    Monotone in `target` and defined across the whole range including both endpoints, which is what makes
    it usable as a ladder: every rung is a distinct mixture by construction.
    """
    inside = fit.family_index == list(fit.family_names).index(family)
    weights = np.zeros(len(base))
    for mask, share in ((inside, target), (~inside, 1.0 - target)):
        total = base[mask].sum()
        weights[mask] = share * (base[mask] / total if total > 0 else 1.0 / mask.sum())
    return weights


def shift_to_separation(base: np.ndarray, direction: np.ndarray, separation: float) -> np.ndarray:
    """The point along `direction` whose total variation from `base` is `separation`.

    Solved rather than assumed. Clipping at zero absorbs part of a large step, so realised distance is a
    concave, eventually flat function of step size; stepping by the requested amount undershoots, and past
    saturation several requests map to the same point. Requesting an unreachable distance raises here
    instead of silently returning a duplicate.
    """

    def at(step: float) -> np.ndarray:
        moved = np.clip(base + step * direction, 0.0, None)
        return moved / moved.sum()

    low, high = 0.0, 1.0
    while separation_of(base, at(high)) < separation:
        high *= 2.0
        if high > 1e4:
            raise ValueError(f"direction saturates below the requested separation {separation}")
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if separation_of(base, at(middle)) < separation else (low, middle)
    return at(0.5 * (low + high))


def common_continuations(fit) -> list[dict]:
    """The continuations run from EVERY prefix: a code-share ladder plus orthogonal coverage."""
    base = fit.proportional
    proportional_share = float(base[fit.family_index == list(fit.family_names).index(evidence.CODE_FAMILY)].sum())
    ladder = [proportional_share if share is None else share for share in CODE_SHARE_LADDER]
    rows = [
        {
            "continuation_id": f"code_share_{round(share * 1000):04d}",
            "direction": "code_share",
            "requested": share,
            "weights": set_family_share(fit, base, evidence.CODE_FAMILY, share),
        }
        for share in ladder
    ]
    code = family_direction(fit, evidence.CODE_FAMILY, +1.0)
    for index, direction in enumerate(orthogonal_directions(fit, code, 2, stable_seed("common"))):
        rows += [
            {
                "continuation_id": f"orthogonal{index}_{int(separation * 100):02d}",
                "direction": f"orthogonal_{index}",
                "requested": separation,
                "weights": shift_to_separation(base, direction, separation),
            }
            for separation in ORTHOGONAL_SEPARATIONS
        ]
    _assert_distinct([row["weights"] for row in rows], "common continuations")
    if len(rows) != COMMON_COUNT:
        raise ValueError(f"built {len(rows)} common continuations, expected {COMMON_COUNT}")
    return rows


def _assert_distinct(weights: list[np.ndarray], label: str) -> None:
    """Duplicate mixtures mean the construction saturated, which must fail loudly rather than be reseeded."""
    keys = {tuple(np.round(row, 6)) for row in weights}
    if len(keys) != len(weights):
        raise ValueError(f"{label}: {len(weights) - len(keys)} duplicate mixtures; the ladder saturated")


def predicted_optima(count: int) -> list[np.ndarray]:
    """Phase-0 halves of the surrogate's own argmin, under `count` independent fold and search seeds.

    Taken from real fits rather than from a hand-built direction, for two reasons. The doc's rationale for
    these prefixes is that the predicted optimum is a REGION -- the argmin moves 0.20 to 0.60 total
    variation across seeds -- and only actual refits exhibit that spread. And a hand-built "less code"
    direction cannot reach the region at all: the code family holds 5.3% of the proportional policy, so
    moving code out saturates at 0.053 total variation, which is the same saturation that once collapsed
    the continuation ladder. An earlier revision did exactly that and placed all three of these prefixes
    within 0.04 of the proportional policy and within 0.02 of each other -- three near-duplicate runs
    standing in for the region they were supposed to span.
    """
    panel, response, reference = degeneracy.pooled_panel("delphi_3e18", swarm39.UNCHEATABLE)
    start = np.repeat(reference.proportional[None, :], 2, axis=0)
    optima = []
    for seed in range(count):
        _vector, shape, offsets, amplitudes = split_damage.fit_variant(panel, response, "split", seed)
        policy, _loss = degeneracy.find_optimum(panel, shape, amplitudes, offsets, "split", start)
        optima.append(policy[0])
    _assert_distinct(optima, "predicted optima")
    return optima


def prefixes(fit, frame, geo) -> list[dict]:
    """The phase-0 mixtures, chosen by the ROLE each has to play rather than by one criterion."""
    readout = evidence.with_phase0_readout(frame)["readout_phase0_uncheatable_bpb"].to_numpy(float)
    endpoint = frame["table9_macro_bpb"].to_numpy(float)
    usable = np.flatnonzero(np.isfinite(readout) & np.isfinite(endpoint))
    terciles = np.quantile(readout[usable], [1 / 3, 2 / 3])
    generator = np.random.default_rng(stable_seed("prefixes"))
    chosen: list[dict] = []

    for label, mask in (
        ("best", readout[usable] <= terciles[0]),
        ("middle", (readout[usable] > terciles[0]) & (readout[usable] <= terciles[1])),
        ("worst", readout[usable] > terciles[1]),
    ):
        pool = usable[mask]
        for pick in generator.choice(pool, size=2, replace=False):
            chosen.append(
                {
                    "prefix_id": f"sampled_{label}_{pick}",
                    "role": f"sampled_{label}",
                    "weights": geo["phase_0"][pick],
                    "source_row": int(pick),
                }
            )

    for draw, weights in enumerate(predicted_optima(3)):
        chosen.append(
            {
                "prefix_id": f"predicted_{draw}",
                "role": "predicted_optimum",
                "weights": weights,
                "source_row": -1,
            }
        )

    chosen.append(
        {
            "prefix_id": "reference_proportional",
            "role": "reference_proportional",
            "weights": fit.proportional,
            "source_row": -1,
        }
    )
    tied = geo["separation"] <= 1e-9
    scored = np.where(tied[usable], endpoint[usable], np.inf)
    best = int(usable[int(np.argmin(scored))])
    chosen.append(
        {
            "prefix_id": "reference_best_one_phase",
            "role": "reference_best_1p",
            "weights": geo["phase_0"][best],
            "source_row": best,
        }
    )
    chosen.append(
        {
            "prefix_id": "exploratory_high_separation",
            "role": "exploratory",
            "weights": shift_to_separation(
                fit.proportional, family_direction(fit, evidence.CODE_FAMILY, +1.0), 0.45
            ),
            "source_row": -1,
        }
    )
    if len(chosen) != 12:
        raise ValueError(f"built {len(chosen)} distinct prefixes, expected 12")

    # Four more sampled prefixes widen the crossed design; two of the twelve are then retrained under a
    # different data seed to expose prefix-level noise.
    extra = generator.choice(usable, size=4, replace=False)
    for pick in extra:
        chosen.append(
            {
                "prefix_id": f"sampled_extra_{pick}",
                "role": "sampled_extra",
                "weights": geo["phase_0"][pick],
                "source_row": int(pick),
            }
        )
    full = list(chosen)
    replicates = []
    for source in (full[0], full[6]):
        replicates.append({**source, "prefix_id": f"{source['prefix_id']}_rep", "role": "prefix_replicate"})
    return full, replicates


def branches(fit, prefix: dict, common: list[dict], carries_gate_zero: bool, full: bool) -> list[dict]:
    """The continuations run from one prefix, allocated by that prefix's role."""
    entries = [
        {
            "continuation_id": "tied",
            "direction": "tied",
            "requested": 0.0,
            "weights": prefix["weights"],
            "role": "tied_anchor",
            "seed_offset": 0,
        }
    ]
    kept = common if full else common[:REPLICATE_PREFIX_BRANCHES]
    entries += [{**row, "role": "common_core", "seed_offset": 0} for row in kept]
    if not full:
        return entries

    code = family_direction(fit, evidence.CODE_FAMILY, +1.0)
    local = orthogonal_directions(fit, code, LOCAL_COUNT, stable_seed("local", prefix["prefix_id"]))
    for index, direction in enumerate(local):
        entries.append(
            {
                "continuation_id": f"local_{index}",
                "direction": f"local_{index}",
                "requested": LOCAL_SEPARATION,
                "weights": shift_to_separation(prefix["weights"], direction, LOCAL_SEPARATION),
                "role": "local_probe",
                "seed_offset": 0,
            }
        )
    if carries_gate_zero:
        anchor = next(row for row in common if row["continuation_id"] == GATE_ZERO_CONTINUATION)
        for replicate in range(GATE_ZERO_REPLICATES):
            entries.append(
                {
                    **anchor,
                    "continuation_id": f"gate_zero_replicate{replicate}",
                    "role": "branch_replicate",
                    "seed_offset": replicate + 1,
                }
            )
    return entries


def build() -> pd.DataFrame:
    fit, _held = swarm39.load_scale("delphi_3e18")
    frame, _dropped = evidence.panel()
    geo = evidence.geometry(frame)
    common = common_continuations(fit)
    full, replicated = prefixes(fit, frame, geo)
    gate_zero = {prefix["prefix_id"] for prefix in full[:GATE_ZERO_PREFIXES]}

    rows = []
    for prefix in full + replicated:
        is_full = prefix["role"] != "prefix_replicate"
        # A retrained prefix is the same mixture under a different data seed; that is the whole point.
        prefix_seed = stable_seed("prefix", prefix["prefix_id"]) if not is_full else DESIGN_SEED
        for entry in branches(fit, prefix, common, prefix["prefix_id"] in gate_zero, is_full):
            rows.append(
                {
                    "prefix_id": prefix["prefix_id"],
                    "prefix_role": prefix["role"],
                    "prefix_source_row": prefix["source_row"],
                    "prefix_data_seed": prefix_seed,
                    "continuation_id": entry["continuation_id"],
                    "branch_role": entry["role"],
                    "direction": entry["direction"],
                    "requested_separation": entry["requested"],
                    "realised_separation": separation_of(prefix["weights"], entry["weights"]),
                    "phase_0_fraction": PHASE_0_FRACTION,
                    "phase_0_weights_json": _weights_json(fit, prefix["weights"]),
                    "phase_1_weights_json": _weights_json(fit, entry["weights"]),
                    "phase_1_data_seed": DESIGN_SEED + entry["seed_offset"],
                }
            )
    manifest = pd.DataFrame(rows)
    _assert_no_duplicate_runs(manifest)
    return manifest


def _weights_json(fit, weights: np.ndarray) -> str:
    return json.dumps(dict(zip(fit.buckets, np.round(weights, 8), strict=True)), sort_keys=True)


def _assert_no_duplicate_runs(manifest: pd.DataFrame) -> None:
    """Two branches identical in mixture AND seed would be bit-identical runs, so pure waste.

    This fires in one place by construction: the proportional reference prefix and the proportional
    common continuation coincide, making that prefix's tied anchor and one common branch the same policy.
    Rather than drop the branch, which would break the "same 17 continuations everywhere" property the
    crossed analysis depends on, the tied anchor there is left as the duplicate to be resolved by seed.
    """
    keys = ["phase_0_weights_json", "phase_1_weights_json", "prefix_data_seed", "phase_1_data_seed"]
    duplicated = manifest[manifest.duplicated(keys, keep=False)]
    for key, group in duplicated.groupby(keys):
        roles = set(group["branch_role"])
        if roles != {"tied_anchor", "common_core"} or len(group) != 2:
            raise ValueError(f"{len(group)} bit-identical branches with roles {roles}: {key[:2]}")
        # The one intended coincidence: the proportional reference prefix continued with the proportional
        # continuation IS its own tied anchor. Reseeding turns the pair into a free branch-noise cell
        # rather than a repeated run, and keeps all 17 common continuations present at every prefix.
        manifest.loc[group.index[1], "phase_1_data_seed"] += 1
    if manifest.duplicated(keys).any():
        raise ValueError("branches remain bit-identical after reseeding")


def power(manifest: pd.DataFrame, noise: float, seeds: int) -> dict[str, float]:
    """Smallest paired improvement this design can certify, at one-sided 95%.

    The certified quantity is the tied-versus-selected difference AT THE SAME PREFIX, so the unit is a
    prefix and the noise is that of a difference of two runs. Stage one runs one seed per arm, which puts
    the detectable effect within a hair of the whole effect the archive suggests is available -- the
    design would be measuring against its own resolution limit. Replicating only the two certification
    arms fixes that far more cheaply than adding prefixes, since power in the number of prefixes costs a
    full branch set each while power in seeds costs two runs.
    """
    prefixes = int((manifest["branch_role"] == "tied_anchor").sum()) - 2  # replicate prefixes do not certify
    error = noise * np.sqrt(2.0 / seeds) / np.sqrt(prefixes)
    return {
        "prefixes_certifying": prefixes,
        "seeds_per_arm": seeds,
        "paired_standard_error": round(error, 5),
        # 1.753 is t(0.95, 15); the bound is one-sided because only an improvement is claimed.
        "minimum_certifiable_effect": round(1.753 * error, 5),
    }


def budget(manifest: pd.DataFrame) -> dict[str, float]:
    """TPU-hours, separated into training and eval because eval is the larger half."""
    prefix_count = manifest["prefix_id"].nunique()
    branch_count = len(manifest)
    train = prefix_count * PHASE_0_FRACTION + branch_count * (1.0 - PHASE_0_FRACTION)
    evals = (branch_count + prefix_count) * EVAL_HOURS_TABLE9
    return {
        "prefixes": prefix_count,
        "branches": branch_count,
        "train_hours": round(train * TRAIN_HOURS_PER_RUN, 1),
        "eval_hours": round(evals, 1),
        "total_hours": round(train * TRAIN_HOURS_PER_RUN + evals, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = build()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "prefix_search_manifest.csv"
    manifest.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cost = budget(manifest)
    # Table 9 run-to-run noise, measured on the archive by the evidence driver's `run_noise`.
    stage_one = power(manifest, TABLE9_RUN_NOISE, seeds=1)
    stage_two = power(manifest, TABLE9_RUN_NOISE, seeds=CERTIFICATION_SEEDS)
    certification = stage_one["prefixes_certifying"] * (2 * CERTIFICATION_SEEDS - 1)
    (args.out / "design.json").write_text(
        json.dumps(
            {
                "scale": "delphi_3e18",
                "objective_primary": "table9_macro_bpb",
                "objective_secondary": ["uncheatable_bpb", "per-component uncheatable"],
                "manifest_sha256": digest,
                "design_seed": DESIGN_SEED,
                "crossed_cells": int((manifest["branch_role"] == "common_core").sum()),
                **cost,
                "stage_one_power": stage_one,
                "stage_two_power": stage_two,
                "stage_two_branches": certification,
                "stage_two_hours": round(certification * ((1.0 - PHASE_0_FRACTION) + EVAL_HOURS_TABLE9), 1),
                "requirements": [
                    "retain the checkpoint at the phase boundary for every prefix",
                    "log an eval AT the boundary step, not at the nearest cadence point",
                    "run Table 9 at the boundary as well as at the endpoint",
                    "parent placement --region us-east5 --zone us-east5-a; MARIN_PREFIX gs://marin-us-east5",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {path}")
    print(f"  sha256 {digest[:16]}  {cost}")
    print(f"  stage 1 certifies effects >= {stage_one['minimum_certifiable_effect']:.5f} BPB (1 seed/arm)")
    print(
        f"  stage 2 certifies effects >= {stage_two['minimum_certifiable_effect']:.5f} BPB "
        f"({CERTIFICATION_SEEDS} seeds/arm, {certification} extra branches, "
        f"{certification * ((1.0 - PHASE_0_FRACTION) + EVAL_HOURS_TABLE9):.0f} TPU-h)"
    )
    print()
    print(manifest.groupby("prefix_role")["prefix_id"].nunique().to_string())
    print()
    print(manifest.groupby("branch_role").size().to_string())
    print()
    print("realised separation by branch role:")
    print(manifest.groupby("branch_role")["realised_separation"].describe()[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
