# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Materialize an a priori 280-row Delphi 3e18 swarm design for the 39-bucket single-phase mixing problem.

Every row is fixed before any surrogate is fitted: no coordinate comes from a model or from a measured optimum.
The design uses only bucket metadata (unique-token counts, the bucket type read from its name) and prior knowledge
about repetition. Blocks:

- reserved rows: the proportional, UniMax and uniform baselines, three proportional repeats, and the 39
  leave-one-bucket-out deletions from the proportional control (all a priori, reused from the current panel);
- fixed-share subsampled-pool rows: an anchor mixture with one bucket (or the CC-high group) restricted to a half
  or a quarter of its unique tokens, so that bucket's epochs double or quadruple while every share stays fixed
  (implemented downstream with per-dataset ``max_train_batches`` and ``max_train_batches_subset_seed``);
- a Latin hypercube over per-type relative epoch levels with per-bucket jitter (shares follow from inventories);
- single-bucket share ladders at two type-level anchors;
- Common-Crawl-removal corners;
- reused random panel rows chosen by farthest-point sampling and the highest-epoch conditional dose ladders.

Outputs the launcher-format table (``phase_0_<domain>``/``phase_1_<domain>`` columns, equal across phases, plus
``pool_fraction_<domain>`` columns), a per-row summary, and a manifest with the seed and input hashes. Nothing is
launched. The optional coverage check reads the heldout registry only to report distances after the fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round4_cap_policies_20260903 as policies,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (  # noqa: E402
    DOMAIN_NAMES,
    PHASE_NAMES,
)

PANEL = "delphi_3e18_39bucket"
BUDGET_ROWS = 280
SEED = 20_260_904
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_apriori_swarm_280_20260904"
TYPES = ("cc_high", "cc_low", "code", "curated", "math", "synthetic")
LHS_ROWS = 77
LHS_LOG_BOX = {
    "cc_high": (np.log(0.25), np.log(2.0)),
    "cc_low": (np.log(0.25), np.log(2.0)),
    "code": (np.log(0.25), np.log(16.0)),
    "curated": (np.log(0.25), np.log(16.0)),
    "math": (np.log(0.25), np.log(16.0)),
    "synthetic": (np.log(0.25), np.log(16.0)),
}
LHS_JITTER = 0.35
LHS_MAX_EPOCHS = 128.0
ANCHORS = {
    "B_small_pools_forward": {"cc_high": 1.0, "cc_low": 0.5, "code": 2.0, "curated": 4.0, "math": 4.0, "synthetic": 4.0},
    "C_code_math_forward": {"cc_high": 0.5, "cc_low": 0.5, "code": 6.0, "curated": 2.0, "math": 6.0, "synthetic": 2.0},
}
LADDER_BUCKETS = (
    "dolmino_synth_qa",
    "dolmino_olmocr_pdfs_hq",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolma3_wikipedia",
    "dolmino_stem_heavy_crawl",
    "dolmino_synth_math",
    "dolmino_synth_code",
)
LADDER_MULTIPLIERS = (0.25, 3.0, 8.0)
SUBSAMPLE_TARGETS = (
    "dolmino_synth_qa",
    "dolmino_olmocr_pdfs_hq",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolma3_wikipedia",
    "dolmino_stem_heavy_crawl",
    "dolmino_synth_math",
    "cc_high_group",
)
SUBSAMPLE_CORNER_TARGETS = ("dolmino_synth_qa", "dolmino_olmocr_pdfs_hq", "dolma3_stack_edu", "cc_high_group")
POOL_FRACTIONS = (0.5, 0.25)
CORNER_CC = ((1.0, 0.0), (0.5, 0.0), (0.25, 0.25))
CORNER_OTHERS = (2.0, 4.0, 8.0)
PROPORTIONAL_REPEATS = 3
REUSED_RANDOM_PANEL_ROWS = 18
REUSED_DOSE_ROWS = 40
DOSE_SOURCE = "conditional_epoch_dose_response"


def shares_from_levels(levels: np.ndarray, inventory: np.ndarray) -> np.ndarray:
    """Relative epoch levels per bucket to normalized shares (share = level / inventory, renormalized)."""
    weights = levels / inventory
    return weights / weights.sum()


def level_vector(spec: dict[str, float], types: np.ndarray) -> np.ndarray:
    return np.array([spec[kind] for kind in types])


def farthest_point(points: np.ndarray, count: int, start: np.ndarray) -> np.ndarray:
    chosen: list[int] = []
    reference = [start]
    for _ in range(count):
        distance = 0.5 * np.abs(points[:, None, :] - np.array(reference)[None, :, :]).sum(-1).min(1)
        distance[chosen] = -1.0
        index = int(np.argmax(distance))
        chosen.append(index)
        reference.append(points[index])
    return np.array(chosen)


def build_design(panel: harness.BenchPanel, dose_weights: np.ndarray) -> pd.DataFrame:
    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    types = np.array([policies.bucket_type(bucket) for bucket in buckets])
    weights_panel = panel.features.weights
    names = list(panel.runs)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    def add(
        kind: str,
        block: str,
        weights: np.ndarray,
        *,
        anchor: str = "",
        target: str = "",
        multiplier: float = 1.0,
        pool_fractions: dict[str, float] | None = None,
        source: str = "new",
    ):
        weights = np.asarray(weights, dtype=float)
        if weights.min() < 0 or not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError(f"{kind}: weights must be a nonnegative simplex point")
        rows.append(
            {
                "kind": kind,
                "block": block,
                "anchor": anchor,
                "target_bucket": target,
                "multiplier": multiplier,
                "pool_fractions": dict(pool_fractions or {}),
                "source": source,
                "weights": weights / weights.sum(),
            }
        )

    proportional = weights_panel[names.index("singleavg_fit_000_baseline_proportional")]
    add("baseline_proportional", "reserved", proportional, source="reused_panel")
    add(
        "baseline_unimax",
        "reserved",
        weights_panel[names.index("singleavg_fit_001_baseline_unimax")],
        source="reused_panel",
    )
    add(
        "baseline_uniform",
        "reserved",
        weights_panel[next(i for i, n in enumerate(names) if "stratified" in n)],
        source="reused_panel",
    )
    for repeat in range(PROPORTIONAL_REPEATS):
        add(f"proportional_repeat_{repeat + 1}", "reserved", proportional, source="new")
    for index, name in enumerate(names):
        if "_pctrl_del_" in name:
            add("pctrl_del_" + name.split("_pctrl_del_")[1], "reserved", weights_panel[index], source="reused_panel")

    anchor_weights = {"A_proportional": proportional}
    for name, spec in ANCHORS.items():
        anchor_weights[name] = shares_from_levels(level_vector(spec, types), inventory)
    corner_anchor = shares_from_levels(
        level_vector({"cc_high": 0.5, "cc_low": 0.0, "code": 4.0, "curated": 4.0, "math": 4.0, "synthetic": 4.0}, types),
        inventory,
    )
    anchor_weights["D_cc_half_no_low"] = corner_anchor
    for name in ("B_small_pools_forward", "C_code_math_forward", "D_cc_half_no_low"):
        add(f"anchor_{name}", "anchor", anchor_weights[name], anchor=name)
    for anchor in ("A_proportional", "B_small_pools_forward"):
        for target in SUBSAMPLE_TARGETS:
            for fraction in POOL_FRACTIONS:
                members = (
                    [b for b, t in zip(buckets, types, strict=True) if t == "cc_high"]
                    if target == "cc_high_group"
                    else [target]
                )
                add(
                    f"subsample_{anchor}_{target}_pool{fraction:g}",
                    "subsampled_pool",
                    anchor_weights[anchor],
                    anchor=anchor,
                    target=target,
                    pool_fractions={m: fraction for m in members},
                )
    for target in SUBSAMPLE_CORNER_TARGETS:
        for fraction in POOL_FRACTIONS:
            members = (
                [b for b, t in zip(buckets, types, strict=True) if t == "cc_high"]
                if target == "cc_high_group"
                else [target]
            )
            add(
                f"subsample_D_cc_half_no_low_{target}_pool{fraction:g}",
                "subsampled_pool",
                corner_anchor,
                anchor="D_cc_half_no_low",
                target=target,
                pool_fractions={m: fraction for m in members},
            )

    sampler = qmc.LatinHypercube(d=len(TYPES), seed=SEED)
    produced = 0
    while produced < LHS_ROWS:
        for unit in sampler.random(LHS_ROWS):
            if produced >= LHS_ROWS:
                break
            levels = {
                kind: float(np.exp(LHS_LOG_BOX[kind][0] + unit[j] * (LHS_LOG_BOX[kind][1] - LHS_LOG_BOX[kind][0])))
                for j, kind in enumerate(TYPES)
            }
            raw = level_vector(levels, types) * np.exp(rng.normal(0.0, LHS_JITTER, len(buckets)))
            weights = shares_from_levels(raw, inventory)
            if (weights * inventory).max() > LHS_MAX_EPOCHS:
                continue
            add(f"lhs_type_epoch_ratios_{produced:03d}", "lhs", weights)
            produced += 1
    for anchor in ANCHORS:
        base = level_vector(ANCHORS[anchor], types)
        for target in LADDER_BUCKETS:
            for multiplier in LADDER_MULTIPLIERS:
                levels = base.copy()
                levels[buckets.index(target)] *= multiplier
                add(
                    f"ladder_{anchor}_{target}_x{multiplier:g}",
                    "ladder",
                    shares_from_levels(levels, inventory),
                    anchor=anchor,
                    target=target,
                    multiplier=multiplier,
                )
    for cc_high, cc_low in CORNER_CC:
        for other in CORNER_OTHERS:
            spec = {
                "cc_high": cc_high,
                "cc_low": cc_low,
                "code": other,
                "curated": other,
                "math": other,
                "synthetic": other,
            }
            add(
                f"corner_cchigh{cc_high:g}_cclow{cc_low:g}_others{other:g}",
                "corner",
                shares_from_levels(level_vector(spec, types), inventory),
            )

    random_rows = np.array([i for i, n in enumerate(names) if "_run_" in n])
    for index in random_rows[
        farthest_point(weights_panel[random_rows], REUSED_RANDOM_PANEL_ROWS, weights_panel.mean(0))
    ]:
        add(
            f"random_panel_{names[index].split('_run_')[1]}",
            "reused_random_panel",
            weights_panel[index],
            source="reused_panel",
        )
    dose_order = np.argsort(-(dose_weights * inventory[None, :]).max(1))[:REUSED_DOSE_ROWS]
    for rank, index in enumerate(dose_order):
        add(f"dose_ladder_reused_{rank:02d}", "reused_dose_ladder", dose_weights[index], source="reused_dose")

    frame = pd.DataFrame(rows)
    if len(frame) != BUDGET_ROWS:
        raise ValueError(f"design has {len(frame)} rows, budget {BUDGET_ROWS}")
    if frame["kind"].duplicated().any():
        raise ValueError("duplicate row kinds")
    frame.insert(0, "run_name", [f"apriori_swarm_{i:03d}_{k}"[:96] for i, k in enumerate(frame["kind"])])
    return frame


def launcher_table(design: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    """Launcher-format columns: equal weights in both phases, pool fractions per domain (1.0 = full pool)."""
    records = []
    for _, row in design.iterrows():
        weights = dict(zip(buckets, row["weights"], strict=True))
        record: dict[str, object] = {
            "run_name": row["run_name"],
            "source_run_name": row["run_name"],
            "kind": row["kind"],
            "block": row["block"],
            "anchor": row["anchor"],
            "target_bucket": row["target_bucket"],
            "multiplier": row["multiplier"],
            "source": row["source"],
        }
        for phase in PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                record[f"{phase}_{domain}"] = weights[domain]
        for domain in DOMAIN_NAMES:
            record[f"pool_fraction_{domain}"] = float(row["pool_fractions"].get(domain, 1.0))
        records.append(record)
    return pd.DataFrame(records)


def row_summary(design: pd.DataFrame, panel: harness.BenchPanel) -> pd.DataFrame:
    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    types = np.array([policies.bucket_type(bucket) for bucket in buckets])
    proportional = panel.features.weights[list(panel.runs).index("singleavg_fit_000_baseline_proportional")]
    records = []
    for _, row in design.iterrows():
        weights = np.asarray(row["weights"])
        fractions = np.array([row["pool_fractions"].get(bucket, 1.0) for bucket in buckets])
        epochs = weights * inventory / fractions
        positive = weights[weights > 0]
        records.append(
            {
                "run_name": row["run_name"],
                "kind": row["kind"],
                "block": row["block"],
                "source": row["source"],
                "effective_buckets": float(np.exp(-(positive * np.log(positive)).sum())),
                "nonzero_buckets": int((weights > 0).sum()),
                "max_epochs": float(epochs.max()),
                "max_epochs_bucket": buckets[int(np.argmax(epochs))],
                "cc_share": float(weights[np.char.startswith(types.astype(str), "cc")].sum()),
                "synth_qa_share": float(weights[buckets.index("dolmino_synth_qa")]),
                "stack_share": float(
                    weights[[buckets.index("dolma3_stack_edu"), buckets.index("dolmino_stack_edu_fim")]].sum()
                ),
                "tv_to_proportional": float(0.5 * np.abs(weights - proportional).sum()),
            }
        )
    return pd.DataFrame(records)


def identifiability_report(design: pd.DataFrame, panel: harness.BenchPanel) -> dict[str, object]:
    """Per bucket: can epochs be told apart from share? Residual spread of log epochs after conditioning on share."""
    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    weights = np.vstack(design["weights"].to_numpy())
    fractions = np.array([[row.get(bucket, 1.0) for bucket in buckets] for row in design["pool_fractions"]])
    epochs = weights * inventory[None, :] / fractions
    report: dict[str, object] = {}
    for column, bucket in enumerate(buckets):
        positive = weights[:, column] > 0
        share, log_epochs = weights[positive, column], np.log(epochs[positive, column])
        residual = log_epochs - np.log(share * inventory[column])  # zero wherever the pool is full
        report[bucket] = {
            "rows_with_reduced_pool": int((fractions[:, column] < 1.0).sum()),
            "log_epoch_spread_given_share": float(residual.std()),
            "max_epochs_full_pool": float((weights[:, column] * inventory[column]).max()),
            "max_epochs_with_subsampling": float(epochs[:, column].max()),
        }
    panel_weights = panel.features.weights
    report["panel_note"] = (
        "in the current panel every bucket's epochs equal share times inventory exactly, "
        "so the spread given share is 0 for all 39 buckets"
    )
    )
    report["panel_max_epochs"] = {
        bucket: float((panel_weights[:, column] * inventory[column]).max()) for column, bucket in enumerate(buckets)
    }
    return report


def coverage_report(design: pd.DataFrame, panel: harness.BenchPanel, registry_dir: Path) -> dict[str, object]:
    """Post-hoc diagnostic only: distances from measured coordinates to the design; never used to place rows."""
    harness.HELDOUT_DIR = registry_dir.resolve()
    bank = selection.load_bank(panel, "table9")
    _frame, features = harness.heldout_features(panel, "table9")
    top = np.argsort(bank.measured)[:10]
    points = np.vstack(design["weights"].to_numpy())
    new_points = np.vstack(design.loc[design["source"].eq("new"), "weights"].to_numpy())

    def nearest(pts: np.ndarray) -> dict[str, float]:
        distance = (0.5 * np.abs(features.weights[top][:, None, :] - pts[None, :, :]).sum(-1)).min(1)
        return {
            "median_tv_to_top10": float(np.median(distance)),
            "min_tv_to_top10": float(distance.min()),
            "max_tv_to_top10": float(distance.max()),
        }

    return {
        "note": "diagnostic against measured Table-9 coordinates; the design was placed without them",
        "current_panel": nearest(panel.features.weights),
        "design_all_rows": nearest(points),
        "design_new_rows": nearest(new_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=harness.HELDOUT_DIR,
        help="heldout registry (dose rows to reuse; coverage diagnostic)",
    )
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    coordinates, _components, registry_hashes = harness.heldout_registry()
    dose = coordinates[coordinates["panel"].eq(PANEL) & coordinates["sources"].astype(str).str.contains(DOSE_SOURCE)]
    dose_weights = dose[[f"weight::{bucket}" for bucket in panel.buckets]].to_numpy(float)
    design = build_design(panel, dose_weights)
    buckets = list(panel.buckets)
    table = launcher_table(design, buckets)
    table.to_csv(args.output_dir / "swarm_mixtures.csv", index=False)
    summary = row_summary(design, panel)
    summary.to_csv(args.output_dir / "design_rows.csv", index=False)
    budget = float(TARGET_BUDGET_DOLMA3_COMMON_CRAWL)
    manifest = {
        "seed": SEED,
        "rows": len(design),
        "new_runs": int(design["source"].eq("new").sum()),
        "blocks": design.groupby("block")["source"].value_counts().unstack(fill_value=0).to_dict(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "panel_inputs": panel.input_hashes,
        "registry_hashes": registry_hashes,
        "target_budget_tokens": budget,
        "unique_tokens": {bucket: int(TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket]) for bucket in buckets},
        "mixtures_sha256": hashlib.sha256((args.output_dir / "swarm_mixtures.csv").read_bytes()).hexdigest(),
    }
    manifest["identifiability"] = identifiability_report(design, panel)
    if not args.skip_coverage:
        manifest["coverage_diagnostic"] = coverage_report(design, panel, args.registry_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    pd.set_option("display.width", 250)
    print(
        summary.groupby(["block", "source"])
        .agg(
            rows=("run_name", "size"),
            eff_min=("effective_buckets", "min"),
            eff_median=("effective_buckets", "median"),
            max_epochs_median=("max_epochs", "median"),
            max_epochs_max=("max_epochs", "max"),
            cc_share_min=("cc_share", "min"),
        )
        .round(2)
        .to_string()
    )
    print(json.dumps({k: manifest[k] for k in ("rows", "new_runs", "coverage_diagnostic") if k in manifest}, indent=2))


if __name__ == "__main__":
    main()
