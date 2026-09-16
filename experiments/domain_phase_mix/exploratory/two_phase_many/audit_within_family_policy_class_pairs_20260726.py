# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed-matched paired test of two-phase versus tied policy classes at 3e18.

Each surrogate family proposed an optimum twice: once optimized over the tied
(single-phase) policy class and once over the two-phase class, holding the
surrogate, the proposal target, and the aggregate-KL coefficient fixed. The run
names encode the policy class in a single token, so those proposals pair
exactly. The paired difference is the within-family estimate of
V^{part,two} - V^{part,single} at an anchor that was not already the tied
optimum, which is the regime the frontier-anchored panels cannot probe.

Reported alongside the pairs is a run-to-run noise floor measured from replicate
runs of identical mixtures, because a paired lead is only meaningful relative to
the reproducibility of a single training run.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKET = SCRIPT_DIR / "reference_outputs" / "two_phase_surrogate_collaborator_packet_20260721"
CANONICAL = PACKET / "data" / "canonical" / "delphi_3e18_heldouts.csv"
REGISTRY = PACKET / "data" / "raw" / "delphi_3e18_heldouts" / "heldout_registry.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "within_family_policy_class_pairs_20260726"

SEALED_SERIES_FRAGMENT = "targeted_pairwise"

UNCHEATABLE = "uncheatable_bpb"
TABLE9 = "table9_macro_bpb"

# Policy-class tokens that differ between otherwise identical run names.
TIED_TOKENS = ("_1p_", "_tied_", "_onephase_")
TWO_PHASE_TOKENS = ("_2p_", "_twophase_")
CLASS_PLACEHOLDER = "_@CLASS@_"

# Run-name fragments identifying the objective a proposal was optimized for.
UNCHEATABLE_FRAGMENTS = ("uncheatable", "unch", "uncheat")
TABLE9_FRAGMENTS = ("table9", "t9")

BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260726

# A pair is exposure-matched when the two-phase proposal's exposure-average
# aggregate is close to its tied sibling's mixture, so the comparison isolates
# phase structure rather than a different aggregate found by the optimizer.
EXPOSURE_MATCH_TV = 0.02
# Below this phase contrast a "two-phase" proposal is tied for practical purposes.
MIN_PHASE_CONTRAST_TV = 0.05

# Best observed tied policy per target, used to express how suboptimal each
# family's anchor was before the two-phase comparison.
FRONTIER_BPB = {UNCHEATABLE: 0.982455, TABLE9: 1.056690}


@dataclass(frozen=True)
class MatchedPair:
    family: str
    pair_key: str
    target: str
    tied_run: str
    two_phase_run: str
    tied_bpb: float
    two_phase_bpb: float
    tied_seed: int
    two_phase_seed: int
    phase_contrast_tv: float
    aggregate_shift_tv: float

    @property
    def delta(self) -> float:
        """Two-phase minus tied; negative means the two-phase class wins."""
        return self.two_phase_bpb - self.tied_bpb


def assert_sealed_absent(frame: pd.DataFrame) -> None:
    hits = frame["training_series"].astype(str).str.contains(SEALED_SERIES_FRAGMENT, na=False)
    assert not hits.any(), f"sealed series present in panel: {frame.loc[hits, 'training_series'].unique()}"


def load_panel() -> tuple[pd.DataFrame, list[str], list[str]]:
    canonical = pd.read_csv(CANONICAL, low_memory=False)
    registry = pd.read_csv(REGISTRY, low_memory=False)
    assert_sealed_absent(canonical)

    phase_0_columns = [c for c in canonical.columns if c.startswith("phase_0_weight::")]
    phase_1_columns = [c for c in canonical.columns if c.startswith("phase_1_weight::")]
    buckets_0 = [c.split("::", 1)[1] for c in phase_0_columns]
    buckets_1 = [c.split("::", 1)[1] for c in phase_1_columns]
    assert buckets_0 == buckets_1, "phase weight columns misaligned"

    columns = ["heldout_id", "wandb_run_base", "data_seed", "policy_class", "mixture_sha256", "phase_0_fraction"]
    keep = ["row_id", UNCHEATABLE, TABLE9, "training_series", *phase_0_columns, *phase_1_columns]
    panel = canonical[keep].merge(
        registry[columns], left_on="row_id", right_on="heldout_id", how="inner", validate="one_to_one"
    )
    assert len(panel) == len(canonical), f"join dropped rows: {len(panel)} vs {len(canonical)}"
    return panel, phase_0_columns, phase_1_columns


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.abs(left - right).sum())


def normalize_run_base(run_base: str) -> str | None:
    """Replace the policy-class token with a placeholder so siblings share a key."""
    for token in TIED_TOKENS + TWO_PHASE_TOKENS:
        if token in run_base:
            return run_base.replace(token, CLASS_PLACEHOLDER, 1)
    return None


def target_of(run_base: str) -> str | None:
    """Infer the proposal target from the run name, requiring an unambiguous match."""
    lowered = run_base.lower()
    is_uncheatable = any(re.search(rf"_{frag}[_0-9]", lowered) for frag in UNCHEATABLE_FRAGMENTS)
    is_table9 = any(re.search(rf"_{frag}[_0-9]", lowered) for frag in TABLE9_FRAGMENTS)
    if is_uncheatable == is_table9:
        return None
    return UNCHEATABLE if is_uncheatable else TABLE9


def family_of(run_base: str) -> str:
    return run_base.split("_", 1)[0]


def build_pairs(
    panel: pd.DataFrame, phase_0_columns: list[str], phase_1_columns: list[str]
) -> tuple[list[MatchedPair], list[str]]:
    """Pair tied and two-phase proposals that differ only in the policy-class token."""
    tied = panel[panel["policy_class"] == "single_phase_tied"]
    two_phase = panel[panel["policy_class"] == "two_phase"]

    two_phase_by_key: dict[str, pd.Series] = {}
    for _, row in two_phase.iterrows():
        key = normalize_run_base(row["wandb_run_base"])
        if key is not None:
            two_phase_by_key.setdefault(key, row)
    # Token-free two-phase names pair with `_onephase_`-prefixed tied names.
    plain_two_phase = {row["wandb_run_base"]: row for _, row in two_phase.iterrows()}

    pairs: list[MatchedPair] = []
    unmatched: list[str] = []
    for _, tied_row in tied.iterrows():
        tied_run = tied_row["wandb_run_base"]
        key = normalize_run_base(tied_run)
        two_phase_row = two_phase_by_key.get(key) if key is not None else None
        if two_phase_row is None and "_onephase_" in tied_run:
            two_phase_row = plain_two_phase.get(tied_run.replace("_onephase_", "_", 1))
        if two_phase_row is None:
            unmatched.append(tied_run)
            continue

        target = target_of(tied_run)
        if target is None:
            unmatched.append(tied_run)
            continue
        if pd.isna(tied_row[target]) or pd.isna(two_phase_row[target]):
            unmatched.append(tied_run)
            continue

        alpha = float(two_phase_row["phase_0_fraction"])
        two_phase_0 = two_phase_row[phase_0_columns].to_numpy(float)
        two_phase_1 = two_phase_row[phase_1_columns].to_numpy(float)
        tied_mixture = tied_row[phase_0_columns].to_numpy(float)
        assert (
            total_variation(tied_mixture, tied_row[phase_1_columns].to_numpy(float)) < 1e-9
        ), f"tied control {tied_run} is not phase-tied"

        pairs.append(
            MatchedPair(
                family=family_of(tied_run),
                pair_key=key or tied_run,
                target=target,
                tied_run=tied_run,
                two_phase_run=two_phase_row["wandb_run_base"],
                tied_bpb=float(tied_row[target]),
                two_phase_bpb=float(two_phase_row[target]),
                tied_seed=int(tied_row["data_seed"]),
                two_phase_seed=int(two_phase_row["data_seed"]),
                phase_contrast_tv=total_variation(two_phase_1, two_phase_0),
                aggregate_shift_tv=total_variation(alpha * two_phase_0 + (1.0 - alpha) * two_phase_1, tied_mixture),
            )
        )
    return pairs, unmatched


def replicate_noise(panel: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run standard deviation across replicate runs of an identical mixture."""
    rows = []
    for sha, group in panel.groupby("mixture_sha256"):
        if len(group) < 3:
            continue
        for target in (UNCHEATABLE, TABLE9):
            values = group[target].dropna().to_numpy(float)
            if len(values) < 3:
                continue
            rows.append(
                {
                    "mixture_sha256": str(sha)[:12],
                    "example_run": group["wandb_run_base"].iloc[0],
                    "policy_class": group["policy_class"].iloc[0],
                    "target": target,
                    "n_replicates": len(values),
                    "sd_bpb": float(values.std(ddof=1)),
                    "range_bpb": float(values.max() - values.min()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "n_replicates"], ascending=[True, False])


def cluster_bootstrap(deltas: np.ndarray, families: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean paired delta, resampling whole families to respect clustering."""
    unique = np.unique(families)
    index_by_family = {fam: np.flatnonzero(families == fam) for fam in unique}
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_family[fam] for fam in picked])
        draws[draw] = deltas[idx].mean()
    return draws


def pair_frame(pairs: list[MatchedPair]) -> pd.DataFrame:
    """One row per matched pair, deduplicated to one row per two-phase proposal.

    Some families ran both a `_1p_` and a `_tied_` control against a single
    two-phase proposal. Those are two measurements of the same comparison, so
    their deltas are averaged rather than counted as independent pairs.
    """
    frame = pd.DataFrame(
        {
            "family": [p.family for p in pairs],
            "target": [p.target for p in pairs],
            "pair_key": [p.pair_key for p in pairs],
            "tied_run": [p.tied_run for p in pairs],
            "two_phase_run": [p.two_phase_run for p in pairs],
            "tied_bpb": [p.tied_bpb for p in pairs],
            "two_phase_bpb": [p.two_phase_bpb for p in pairs],
            "delta_bpb": [p.delta for p in pairs],
            "same_data_seed": [p.tied_seed == p.two_phase_seed for p in pairs],
            "phase_contrast_tv": [p.phase_contrast_tv for p in pairs],
            "aggregate_shift_tv": [p.aggregate_shift_tv for p in pairs],
        }
    )
    collapsed = (
        frame.groupby(["target", "family", "two_phase_run"], as_index=False)
        .agg(
            n_controls=("tied_bpb", "size"),
            tied_bpb=("tied_bpb", "mean"),
            two_phase_bpb=("two_phase_bpb", "first"),
            delta_bpb=("delta_bpb", "mean"),
            same_data_seed=("same_data_seed", "all"),
            phase_contrast_tv=("phase_contrast_tv", "first"),
            aggregate_shift_tv=("aggregate_shift_tv", "min"),
        )
        .sort_values(["target", "family", "two_phase_run"])
    )
    return collapsed


def summarize_arm(frame: pd.DataFrame, arm: str, rng: np.random.Generator) -> list[dict]:
    summaries = []
    for target, group in frame.groupby("target"):
        deltas = group["delta_bpb"].to_numpy(float)
        families = group["family"].to_numpy()
        if len(deltas) == 0:
            continue
        draws = cluster_bootstrap(deltas, families, rng)
        summaries.append(
            {
                "arm": arm,
                "scope": "all_families",
                "target": target,
                "n_pairs": len(deltas),
                "n_families": len(np.unique(families)),
                "mean_delta_bpb": float(deltas.mean()),
                "median_delta_bpb": float(np.median(deltas)),
                "ci95_low_bpb": float(np.quantile(draws, 0.025)),
                "ci95_high_bpb": float(np.quantile(draws, 0.975)),
                "prob_two_phase_better": float((draws < 0).mean()),
                "fraction_pairs_better": float((deltas < 0).mean()),
                "best_delta_bpb": float(deltas.min()),
                "worst_delta_bpb": float(deltas.max()),
            }
        )
        for family, sub in group.groupby("family"):
            values = sub["delta_bpb"].to_numpy(float)
            summaries.append(
                {
                    "arm": arm,
                    "scope": family,
                    "target": target,
                    "n_pairs": len(values),
                    "n_families": 1,
                    "mean_delta_bpb": float(values.mean()),
                    "median_delta_bpb": float(np.median(values)),
                    "ci95_low_bpb": np.nan,
                    "ci95_high_bpb": np.nan,
                    "prob_two_phase_better": np.nan,
                    "fraction_pairs_better": float((values < 0).mean()),
                    "best_delta_bpb": float(values.min()),
                    "worst_delta_bpb": float(values.max()),
                }
            )
    return summaries


def anchor_suboptimality_law(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Relate the paired two-phase gain to how far the tied anchor sits from the frontier.

    If two-phase only recovers a badly chosen aggregate, the gain shrinks toward
    zero as the anchor approaches the frontier, and the fitted intercept is the
    extrapolated gain available at the frontier itself.
    """
    rows = []
    for arm, subset in (
        ("all_pairs", frame),
        (
            "exposure_matched",
            frame[
                (frame["aggregate_shift_tv"] < EXPOSURE_MATCH_TV) & (frame["phase_contrast_tv"] >= MIN_PHASE_CONTRAST_TV)
            ],
        ),
    ):
        for target, group in subset.groupby("target"):
            if len(group) < 3:
                continue
            suboptimality = group["tied_bpb"].to_numpy(float) - FRONTIER_BPB[target]
            deltas = group["delta_bpb"].to_numpy(float)
            families = group["family"].to_numpy()
            slope, intercept = np.polyfit(suboptimality, deltas, 1)

            unique = np.unique(families)
            index_by_family = {fam: np.flatnonzero(families == fam) for fam in unique}
            intercepts = []
            for _ in range(BOOTSTRAP_DRAWS):
                picked = rng.choice(unique, size=len(unique), replace=True)
                idx = np.concatenate([index_by_family[fam] for fam in picked])
                if np.ptp(suboptimality[idx]) < 1e-9:
                    continue
                intercepts.append(np.polyfit(suboptimality[idx], deltas[idx], 1)[1])
            intercepts = np.asarray(intercepts, dtype=float)

            rows.append(
                {
                    "arm": arm,
                    "target": target,
                    "n_pairs": len(deltas),
                    "n_families": len(unique),
                    "corr_suboptimality_delta": float(np.corrcoef(suboptimality, deltas)[0, 1]),
                    "slope": float(slope),
                    "fraction_of_gap_recovered": float(-slope),
                    "intercept_at_frontier_bpb": float(intercept),
                    "intercept_ci95_low_bpb": float(np.quantile(intercepts, 0.025)) if len(intercepts) else np.nan,
                    "intercept_ci95_high_bpb": float(np.quantile(intercepts, 0.975)) if len(intercepts) else np.nan,
                    "n_beating_frontier": int((group["two_phase_bpb"] < FRONTIER_BPB[target]).sum()),
                    "best_two_phase_bpb": float(group["two_phase_bpb"].min()),
                    "best_gap_to_frontier_bpb": float(group["two_phase_bpb"].min() - FRONTIER_BPB[target]),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Stratify into the confounded policy-class arm and the exposure-matched arm."""
    exposure_matched = frame[
        (frame["aggregate_shift_tv"] < EXPOSURE_MATCH_TV) & (frame["phase_contrast_tv"] >= MIN_PHASE_CONTRAST_TV)
    ]
    summaries = summarize_arm(frame, "all_pairs", rng)
    summaries += summarize_arm(exposure_matched, "exposure_matched", rng)
    return pd.DataFrame(summaries)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    panel, phase_0_columns, phase_1_columns = load_panel()
    pairs, unmatched = build_pairs(panel, phase_0_columns, phase_1_columns)
    assert pairs, "no matched policy-class pairs found"

    frame = pair_frame(pairs)
    summary = summarize(frame, rng)
    law = anchor_suboptimality_law(frame, rng)
    noise = replicate_noise(panel)

    frame.to_csv(OUTPUT_DIR / "matched_pairs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    law.to_csv(OUTPUT_DIR / "anchor_suboptimality_law.csv", index=False)
    noise.to_csv(OUTPUT_DIR / "replicate_noise.csv", index=False)
    (OUTPUT_DIR / "unmatched_tied_runs.json").write_text(json.dumps(sorted(unmatched), indent=2))

    print(f"raw matches: {len(pairs)}  deduplicated pairs: {len(frame)}  unmatched tied runs: {len(unmatched)}")
    print(f"pairs sharing a data seed: {int(frame.same_data_seed.sum())}/{len(frame)}")
    matched = frame[
        (frame["aggregate_shift_tv"] < EXPOSURE_MATCH_TV) & (frame["phase_contrast_tv"] >= MIN_PHASE_CONTRAST_TV)
    ]
    print(f"exposure-matched pairs with real phase contrast: {len(matched)}/{len(frame)}")

    noise_floor = {}
    for target in (UNCHEATABLE, TABLE9):
        sub = noise[(noise["target"] == target) & (noise["n_replicates"] >= 5)]
        typical = float(sub["sd_bpb"].median())
        noise_floor[target] = typical
        print(f"  run noise sd({target}) = {typical:.6f} -> paired-difference sd = {typical * np.sqrt(2):.6f}")

    print("\n=== paired summary (negative delta = two-phase class wins) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== anchor-suboptimality law ===")
    print(law.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== run-to-run replicate noise (>=5 replicates) ===")
    print(noise[noise["n_replicates"] >= 5].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== exposure-matched pair detail ===")
    print(
        matched[
            [
                "target",
                "family",
                "two_phase_run",
                "tied_bpb",
                "two_phase_bpb",
                "delta_bpb",
                "phase_contrast_tv",
                "aggregate_shift_tv",
                "same_data_seed",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )


if __name__ == "__main__":
    main()
