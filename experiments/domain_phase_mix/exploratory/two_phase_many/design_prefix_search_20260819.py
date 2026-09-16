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
QUALITY is the open question. Three taken from the surrogate's own argmin under independent fold and search
seeds -- probes, not bets: that argmin is a region rather than a point, and three independent swarms agree
on it at chance. Two references: the token-proportional policy, and the best one-phase policy by endpoint
so that "does two-phase beat one-phase" is a paired within-design comparison rather than two separately
fitted surfaces compared after the fact. One exploratory prefix in the high-separation region the argmin
keeps recommending and the archive barely covers. Two replicates -- one sampled prefix and one predicted
one -- retrained under different data seeds, to separate prefix-level from branch-level noise.

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
# Continuations are drawn from Dirichlet(concentration * proportional) and chosen by maximin, with no
# reference to what any bucket CONTAINS. The first revision laid them on a code-family-share ladder, which
# was circular -- a design built around the code axis that then finds a code-axis effect has confirmed its
# own premise -- and it collapsed the core to 1.6 effective dimensions out of 38 available, with 94.3% of
# the variance in two directions. Measured over 17 continuations: hand-built 1.6, Dirichlet at a single
# concentration 9.4-11.3, mixed concentrations i.i.d. 6.1 (worse, because mixing radii makes the set
# anisotropic), mixed concentrations with maximin selection 12.9, and 11.5 once the epoch cap applies.
#
# Concentration is the label-agnostic replacement for a separation ladder: it sets distance from
# proportional without naming a direction. 500 gives total variation 0.07-0.12, 5 gives 0.41-0.74.
CONCENTRATIONS = (500.0,) * 4 + (100.0,) * 4 + (20.0,) * 4 + (5.0,) * 4
CANDIDATES_PER_CONCENTRATION = 25
LOCAL_SEPARATION = 0.15
# Gate zero replicates a mid-spread continuation: far enough from the prefix that any real branch effect
# is present, not so far that the measurement only describes an extreme policy nobody would deploy.
GATE_ZERO_CONTINUATION = "common_08"
# No bucket may exceed this many materialized epochs within a phase. Repetition is a different mechanism
# from ordering, and an uncapped Dirichlet tail reaches 178 epochs. The cap costs 0.0-3.9% rejection
# depending on concentration and only 12.9 -> 11.5 effective dimensions.
#
# Its interpretive cost is recorded rather than assumed: the archive sits deep in the repeated regime
# (phase-0 max epoch median 9.65, p90 60.1, max 204, with 89% of rows above 4 epochs), so a phase-0 cap of
# 10 retains 54.7% of archive rows and sampled prefixes therefore span the readout terciles of the
# low-repetition HALF of the archive rather than of the archive itself.
MAX_EPOCHS_PER_PHASE = 10.0
# Below one sample per block a weight is quantisation noise: nominally positive, realised as zero or one
# sequence at random. Levanter drops exact zeros but keeps these, so the realised mixture would differ
# from the frozen one. Snapping moves a mixture by at most 6.2e-4 total variation. The archive's smallest
# positive weights are 4.97e-122 and 8.58e-261, which is the signature of unsnapped Dirichlet generation.
BLOCK_SIZE = 4096
# A floor on DIRECTIONAL diversity, not a target: the first revision spanned 1.6 raw dimensions and read
# as a perfectly reasonable design.
MINIMUM_EFFECTIVE_DIMENSIONS = 9.0
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


def isotropic_directions(count: int, dimension: int, seed: int) -> list[np.ndarray]:
    """Zero-sum unit-total-variation directions drawn without reference to any coordinate's meaning.

    Used for the per-prefix local probes, which measure the conditional response the one-step improvement
    differentiates. They are isotropic rather than aimed, for the same reason the common core is drawn
    rather than laid out: aiming them requires naming a direction, and naming a direction is what made the
    first revision circular.
    """
    generator = np.random.default_rng(seed)
    found: list[np.ndarray] = []
    while len(found) < count:
        draw = generator.normal(size=dimension)
        draw -= draw.mean()
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


def prepare_prefix(weights: np.ndarray, base: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Snap a prefix and bring it inside the epoch cap by shrinking toward proportional.

    Every prefix goes through this, not only the generated ones. Archive-sampled prefixes carry the
    archive's own unsnapped weights -- its smallest positive weights are 4.97e-122, the signature of
    unsnapped Dirichlet generation -- and a tied anchor copies its prefix verbatim, so without this the
    quantisation noise the snap exists to remove would re-enter through the prefix side. The surrogate's
    predicted optima are unconstrained argmins and can land far outside the cap.

    Shrinking toward proportional rather than rejecting keeps the prefix's identity: a predicted optimum
    pulled 20% toward proportional is still that optimum's direction, whereas a redraw is a different
    policy answering a different question.
    """
    snapped = snap(weights)
    if max_epochs(snapped, rate) <= MAX_EPOCHS_PER_PHASE:
        return snapped
    for step in np.linspace(0.05, 1.0, 20):
        candidate = snap((1.0 - step) * snapped + step * base)
        if max_epochs(candidate, rate) <= MAX_EPOCHS_PER_PHASE:
            return candidate
    raise ValueError("no shrink toward proportional satisfies the epoch cap")


def exploratory_prefix(fit) -> np.ndarray:
    """One prefix deliberately far from proportional, drawn rather than aimed."""
    generator = np.random.default_rng(stable_seed("exploratory"))
    rate = np.asarray(fit.c0, dtype=float)
    while True:
        candidate = snap(generator.dirichlet(5.0 * fit.proportional))
        if max_epochs(candidate, rate) <= MAX_EPOCHS_PER_PHASE and separation_of(fit.proportional, candidate) > 0.4:
            return candidate


def local_probe(base: np.ndarray, direction: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """A probe at `LOCAL_SEPARATION` from its prefix, shortened only if the epoch cap requires it.

    Shortening rather than redrawing keeps the probe's DIRECTION, which is what the conditional response
    is estimated from; only its length gives way to the cap.
    """
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1):
        candidate = snap(shift_to_separation(base, direction, LOCAL_SEPARATION * scale))
        if max_epochs(candidate, rate) <= MAX_EPOCHS_PER_PHASE:
            return candidate
    raise ValueError("no probe length along this direction satisfies the epoch cap")


def snap(weights: np.ndarray) -> np.ndarray:
    """Zero any weight that cannot earn one sample per block, then renormalise.

    A weight below 1/BLOCK_SIZE is nominally positive but realised as zero or one sequence at random, so
    the mixture that trains is not the mixture that was frozen. Levanter drops exact zeros cleanly but
    keeps these, which is why the snap has to happen here rather than being left to the trainer.
    """
    snapped = np.where(weights * BLOCK_SIZE < 1.0, 0.0, weights)
    total = snapped.sum()
    if total <= 0:
        raise ValueError("snapping removed all weight")
    return snapped / total


def max_epochs(weights: np.ndarray, rate: np.ndarray) -> float:
    """Materialized epochs of the most-repeated bucket, given that phase's epochs-per-unit-weight."""
    return float(np.max(weights * rate))


def common_continuations(fit) -> list[dict]:
    """The continuations run from EVERY prefix, drawn without reference to what any bucket contains.

    Dirichlet(concentration * proportional) has mean at the token-proportional policy, and concentration
    controls how far a draw lands from it. Drawing a pool across a concentration ladder and then choosing
    by maximin spreads the set over the simplex instead of along one axis, which is what makes the crossed
    cells informative about more than a single direction.
    """
    base = fit.proportional
    rate = np.asarray(fit.c1, dtype=float)
    generator = np.random.default_rng(stable_seed("common"))
    bands = sorted(set(CONCENTRATIONS), reverse=True)
    per_band = (COMMON_COUNT - 1) // len(bands)
    rejected = 0
    chosen: list[np.ndarray] = [base]

    # Selection separates DIRECTION from RADIUS, because the two goals conflict. Maximin over the pooled
    # candidates maximises dimensional spread (12.9 effective dimensions) but every pick comes from the
    # widest band, leaving a median distance of 0.75 from proportional and nothing in the near range a
    # deployed policy would occupy. Maximin within a band fixes the radius coverage but collapses the
    # spread to 6.8, because a band's candidates all sit at a similar distance.
    #
    # Taking the radius from the band and choosing on the unit direction gets both: each band contributes
    # its own distance scale, and within the band the picks point as far apart as the pool allows.
    directions: list[np.ndarray] = []
    for concentration in bands:
        pool: list[np.ndarray] = []
        while len(pool) < CANDIDATES_PER_CONCENTRATION * per_band:
            candidate = snap(generator.dirichlet(concentration * base))
            if max_epochs(candidate, rate) > MAX_EPOCHS_PER_PHASE:
                rejected += 1
                continue
            pool.append(candidate)
        candidates = np.stack(pool)
        unit = np.array([(row - base) / max(separation_of(base, row), 1e-12) for row in candidates])
        for _ in range(per_band):
            if directions:
                spread = np.array([min(separation_of(row, taken) for taken in directions) for row in unit])
            else:
                spread = np.linalg.norm(unit, axis=1)
            pick = int(np.argmax(spread))
            chosen.append(candidates[pick])
            directions.append(unit[pick])
            candidates = np.delete(candidates, pick, axis=0)
            unit = np.delete(unit, pick, axis=0)

    rows = [
        {
            "continuation_id": "common_proportional" if index == 0 else f"common_{index:02d}",
            "direction": "dirichlet_maximin",
            "requested": float(separation_of(base, weights)),
            "weights": weights,
        }
        for index, weights in enumerate(chosen)
    ]
    _assert_distinct([row["weights"] for row in rows], "common continuations")
    _assert_spread([row["weights"] for row in rows], base, "common continuations")
    if len(rows) != COMMON_COUNT:
        raise ValueError(f"built {len(rows)} common continuations, expected {COMMON_COUNT}")
    print(f"  common core: {rejected} candidates rejected by the {MAX_EPOCHS_PER_PHASE:.0f}-epoch cap")
    return rows


def effective_dimensions(rows: list[np.ndarray]) -> float:
    """Participation ratio of the centred set's singular spectrum."""
    centred = np.stack(rows) - np.mean(rows, axis=0)
    spectrum = np.linalg.svd(centred, compute_uv=False) ** 2
    share = spectrum / spectrum.sum()
    return float(np.exp(-np.sum(share * np.log(share + 1e-300))))


def directional_dimensions(weights: list[np.ndarray], base: np.ndarray) -> float:
    """The same ratio over UNIT directions from `base`, which is the property the design actually needs.

    Applied to raw weights the ratio is variance-weighted, so a continuation at total variation 0.07 from
    proportional contributes about a thirty-sixth of the variance of one at 0.45 and is discounted almost
    to nothing. Since covering near AND far distances is a deliberate goal here, that metric penalises the
    design for doing what it is supposed to do: the same set scores 7.0 on raw weights and 11.3 on
    directions. What makes the crossed cells informative is how many distinct DIRECTIONS the continuations
    explore, and distance is covered separately and on purpose, so the assertion is on this.
    """
    units = [(row - base) / separation_of(base, row) for row in weights if separation_of(base, row) > 1e-12]
    return effective_dimensions(units)


def _assert_spread(weights: list[np.ndarray], base: np.ndarray, label: str) -> None:
    dimensions = directional_dimensions(weights, base)
    if dimensions < MINIMUM_EFFECTIVE_DIMENSIONS:
        raise ValueError(
            f"{label}: {dimensions:.1f} directional dimensions, below the {MINIMUM_EFFECTIVE_DIMENSIONS} floor; "
            "the set has collapsed onto too few directions to be worth crossing"
        )


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
    phase_0_epochs = np.max(geo["phase_0"] * np.asarray(fit.c0, dtype=float), axis=1)
    # The cap applies to prefixes too, which costs coverage and is recorded as such: it retains 54.7% of
    # archive rows, so these span the readout terciles of the low-repetition half rather than of the whole.
    usable = np.flatnonzero(np.isfinite(readout) & np.isfinite(endpoint) & (phase_0_epochs <= MAX_EPOCHS_PER_PHASE))
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
            "weights": exploratory_prefix(fit),
            "source_row": -1,
        }
    )
    if len(chosen) != 12:
        raise ValueError(f"built {len(chosen)} distinct prefixes, expected 12")

    # Four more sampled prefixes widen the crossed design; two of the twelve are then retrained under a
    # different data seed to expose prefix-level noise.
    # Excluded by MIXTURE rather than by row index. The archive re-runs the same policy under different
    # seeds, so two distinct rows can carry identical weights -- `reference_best_one_phase` and one extra
    # draw collided exactly that way -- and two prefixes with the same weights and the same seed are
    # bit-identical runs rather than extra coverage.
    taken = {tuple(np.round(row["weights"], 8)) for row in chosen}
    remaining = np.array([index for index in usable if tuple(np.round(geo["phase_0"][index], 8)) not in taken])
    extra = generator.choice(remaining, size=4, replace=False)
    for pick in extra:
        chosen.append(
            {
                "prefix_id": f"sampled_extra_{pick}",
                "role": "sampled_extra",
                "weights": geo["phase_0"][pick],
                "source_row": int(pick),
            }
        )
    rate = np.asarray(fit.c0, dtype=float)
    for row in chosen:
        row["weights"] = prepare_prefix(row["weights"], fit.proportional, rate)
    _assert_distinct([row["weights"] for row in chosen], "prefixes after preparation")
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

    rate = np.asarray(fit.c1, dtype=float)
    local = isotropic_directions(LOCAL_COUNT, len(prefix["weights"]), stable_seed("local", prefix["prefix_id"]))
    for index, direction in enumerate(local):
        entries.append(
            {
                "continuation_id": f"local_{index}",
                "direction": f"local_{index}",
                "requested": LOCAL_SEPARATION,
                "weights": local_probe(prefix["weights"], direction, rate),
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


def _report_design_properties(manifest: pd.DataFrame) -> None:
    """The properties that make the design worth running, printed so a regression is visible."""
    fit, _held = swarm39.load_scale("delphi_3e18")
    base = fit.proportional
    buckets = list(fit.buckets)

    def vector(payload: str) -> np.ndarray:
        mapping = json.loads(payload)
        return np.asarray([mapping[bucket] for bucket in buckets], dtype=float)

    common = manifest[manifest["branch_role"].eq("common_core")].drop_duplicates("continuation_id")
    weights = [vector(payload) for payload in common["phase_1_weights_json"]]
    separations = np.sort([separation_of(base, row) for row in weights])
    phase_1 = np.stack([vector(payload) for payload in manifest["phase_1_weights_json"]])
    phase_0 = np.stack([vector(payload) for payload in manifest["phase_0_weights_json"]])
    epochs_1 = np.max(phase_1 * np.asarray(fit.c1, dtype=float), axis=1)
    epochs_0 = np.max(phase_0 * np.asarray(fit.c0, dtype=float), axis=1)
    positive = phase_1[phase_1 > 0]
    # Asserted, not merely printed: each of these is a property the design's validity rests on, and each
    # was violated at some point during construction while the manifest still looked entirely reasonable.
    if epochs_0.max() > MAX_EPOCHS_PER_PHASE or epochs_1.max() > MAX_EPOCHS_PER_PHASE:
        raise ValueError(f"epoch cap violated: phase-0 {epochs_0.max():.2f}, phase-1 {epochs_1.max():.2f}")
    if positive.min() < 1.0 / BLOCK_SIZE:
        raise ValueError(f"unsnapped weight {positive.min():.2e} below 1/{BLOCK_SIZE}")
    print(
        f"  directional dimensions {directional_dimensions(weights, base):.1f} of {len(weights) - 1}"
        f" (floor {MINIMUM_EFFECTIVE_DIMENSIONS}), raw {effective_dimensions(weights):.1f}"
    )
    print(f"  common-core separations {np.round(separations, 3).tolist()}")
    print(
        f"  max epochs: phase-0 {epochs_0.max():.2f}, phase-1 {epochs_1.max():.2f}"
        f" (cap {MAX_EPOCHS_PER_PHASE:.0f}); smallest positive weight {positive.min():.2e}"
        f" (>= 1/{BLOCK_SIZE} = {1 / BLOCK_SIZE:.2e})"
    )


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
    _report_design_properties(manifest)
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
