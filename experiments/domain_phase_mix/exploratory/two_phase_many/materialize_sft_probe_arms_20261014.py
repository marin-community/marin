# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the pretraining arms of the SFT midtraining probe (Delphi 3e18, 2026-10-14).

Arms, all at MARINER's Table-9 data seed 662009 with trainer seeds 0-2 so they pair with the existing
``lwspu_t9_snc_cap08`` runs (arm A):

- B1/B2/B3: the three best measured two-phase OlmoBaseEval Easy schedules whose late phase moves at least
  5% of the tokens (phase TV >= 0.05) and whose phase-weighted average is the Table-9 frontier anchor
  (``dphase_t9b075_tied``): the decoupled-panel canonical schedule ``dphase_t9b075_can_e0p005`` and the July
  panel's ``recipe_synthetic_all_plus_tv010`` and ``partition_11_plus_tv010``. B is chosen among them by the mean
  measured Table-9 BPB over the three trainer seeds.
- C: the frontier anchor trained single-phase, the shared phase-weighted average of B1-B3.
- BP (B'): a Dolmino-late schedule at arm A's aggregate with late Dolmino share 0.75 (the July construction),
  so A is its own single-phase control.

Writes ``reference_outputs/sft_midtraining_probe_20261014/arms_source_panel.csv`` (one row per run with
``phase_{0,1}_<domain>`` columns in the launcher's layout), a long ``arms_phase_weights.csv`` and a summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as augmented

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUT_DIR / "sft_midtraining_probe_20261014"
AGGRESSIVE_PANEL = REFERENCE_OUTPUT_DIR / "delphi_3e18_aggressive_phase_asymmetry_20260722" / "launcher_source_panel.csv"
DECOUPLED_MIXTURES = REFERENCE_OUTPUT_DIR / "decoupled_phase_information_validation_panel_20260712" / "mixtures"
MARINER_CANDIDATES = REFERENCE_OUTPUT_DIR / "delphi_fairness_repeats_3e18_20260908" / "candidate_weights.csv"

ARM_A_CANDIDATE = "lwspu_t9_snc_cap08"
ANCHOR_CANDIDATE = "agphase_a1_center_center_tv000_seed00"
DATA_SEED = 662_009
TRAINER_SEEDS = (0, 1, 2)
RUN_ID_BASE = 7_510_000
LATE_DOLMINO_SHARE = 0.75
# Realized phase fractions of the 3,007-step Delphi 3e18 schedule (phase 1 starts at the block-aligned step 2,400).
ALPHA_0 = 2400 / 3007
ALPHA_1 = 1.0 - ALPHA_0
SIMPLEX_TOLERANCE = 1e-9
# dphase_t9b075_can_e0p005 was built with exact 0.8/0.2 fractions, so its realized aggregate misses the anchor by
# at most 7e-5 in one coordinate (below the 1/2048 block quantization).
AGGREGATE_TOLERANCE = {"B1": 1e-4, "B2": 1e-10, "B3": 1e-10, "C": 1e-10, "BP": 1e-10, "CL": 1e-10, "ML": 1e-10}
MIN_PHASE_TV = 0.05

ARMS = {
    "B1": ("two_phase", "dphase_t9b075_can_e0p005", "canonical phase-information schedule, decoupled panel 2026-07-12"),
    "B2": ("two_phase", "agphase_a1_recipe_synthetic_all_plus_tv010", "synthetic-all late recipe, TV 0.10, July panel"),
    "B3": ("two_phase", "agphase_a1_partition_11_plus_tv010", "balanced partition 11 plus, TV 0.10, July panel"),
    "C": ("one_phase", "dphase_t9b075_tied", "Table-9 frontier anchor trained single-phase (B's 0.8/0.2 average)"),
    "BP": ("two_phase", f"{ARM_A_CANDIDATE}_dolmino_late_075", "Dolmino-late 0.75 at arm A's aggregate"),
    # Added 2026-10-14 10:40 PT (Calvin): dose response of July's recipe_code (-0.015 on Pycode at TV 0.10) and a
    # stronger late math shift for GSM8K, both at the anchor (C's aggregate) with the July recipe construction.
    "CL": ("two_phase", "recipe_code_plus_tv020", "code-late recipe at TV 0.20 at the Table-9 anchor"),
    "ML": (
        "two_phase",
        "recipe_math_reasoning_plus_tv020",
        "math-reasoning-late recipe at TV 0.20 at the Table-9 anchor",
    ),
}
RECIPE_TV = 0.20
RECIPES = {
    "CL": (
        "dolma3_cc/electronics_and_hardware_high",
        "dolma3_cc/electronics_and_hardware_low",
        "dolma3_stack_edu",
        "dolmino_stack_edu_fim",
        "dolmino_synth_code",
    ),
    "ML": (
        "dolma3_cc/science_math_and_technology_high",
        "dolma3_cc/science_math_and_technology_low",
        "dolma3_arxiv",
        "dolma3_finemath_3plus",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    ),
}


def _phase_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{domain}" for domain in augmented.DOMAIN_NAMES]


def _anchor_weights(panel: pd.DataFrame) -> np.ndarray:
    row = panel.loc[panel["candidate_id"] == ANCHOR_CANDIDATE].iloc[0]
    return np.asarray([float(row[c]) for c in _phase_cols("phase_0")])


def _panel_phase_weights(panel: pd.DataFrame, candidate_id: str) -> tuple[np.ndarray, np.ndarray]:
    row = panel.loc[panel["candidate_id"] == candidate_id].iloc[0]
    return (
        np.asarray([float(row[c]) for c in _phase_cols("phase_0")]),
        np.asarray([float(row[c]) for c in _phase_cols("phase_1")]),
    )


def _decoupled_phase_weights(candidate_id: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(DECOUPLED_MIXTURES / f"{candidate_id}.csv").set_index("domain")
    frame = frame.loc[list(augmented.DOMAIN_NAMES)]
    return frame["phase_0_weight"].to_numpy(dtype=float), frame["phase_1_weight"].to_numpy(dtype=float)


def _arm_a_weights() -> np.ndarray:
    frame = pd.read_csv(MARINER_CANDIDATES)
    frame = frame.loc[frame["candidate_id"] == ARM_A_CANDIDATE].set_index("domain").loc[list(augmented.DOMAIN_NAMES)]
    return frame["weight"].to_numpy(dtype=float)


def _dolmino_late(anchor: np.ndarray, late_share: float) -> tuple[np.ndarray, np.ndarray]:
    """The July panel's construction: phase 1 holds ``late_share`` Dolmino mass, phase 0 restores the aggregate."""
    dolmino = np.asarray([domain.startswith("dolmino_") for domain in augmented.DOMAIN_NAMES])
    aggregate_share = float(anchor[dolmino].sum())
    phase_1 = np.empty_like(anchor)
    phase_1[dolmino] = anchor[dolmino] * late_share / aggregate_share
    phase_1[~dolmino] = anchor[~dolmino] * (1.0 - late_share) / (1.0 - aggregate_share)
    phase_0 = (anchor - ALPHA_1 * phase_1) / ALPHA_0
    return phase_0, phase_1


def _recipe_late(anchor: np.ndarray, recipients: tuple[str, ...], phase_tv: float) -> tuple[np.ndarray, np.ndarray]:
    """July's handcrafted-recipe construction: a unit-TV group transfer toward ``recipients``, scaled to ``phase_tv``,
    applied with the realized phase fractions so the aggregate is the anchor."""
    mask = np.asarray([domain in recipients for domain in augmented.DOMAIN_NAMES])
    direction = np.empty_like(anchor)
    direction[mask] = anchor[mask] / anchor[mask].sum()
    direction[~mask] = -anchor[~mask] / anchor[~mask].sum()
    contrast = phase_tv * direction
    phase_0 = anchor - ALPHA_1 * contrast
    phase_1 = anchor + ALPHA_0 * contrast
    return phase_0, phase_1


def build_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    panel = pd.read_csv(AGGRESSIVE_PANEL)
    anchor = _anchor_weights(panel)
    arm_a = _arm_a_weights()
    phase_weights: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "B1": _decoupled_phase_weights("dphase_t9b075_can_e0p005"),
        "B2": _panel_phase_weights(panel, "agphase_a1_recipe_synthetic_all_plus_tv010"),
        "B3": _panel_phase_weights(panel, "agphase_a1_partition_11_plus_tv010"),
        "C": (anchor.copy(), anchor.copy()),
        "BP": _dolmino_late(arm_a, LATE_DOLMINO_SHARE),
        "CL": _recipe_late(anchor, RECIPES["CL"], RECIPE_TV),
        "ML": _recipe_late(anchor, RECIPES["ML"], RECIPE_TV),
    }
    targets = {"B1": anchor, "B2": anchor, "B3": anchor, "C": anchor, "BP": arm_a, "CL": anchor, "ML": anchor}
    rows = []
    arm_summaries = {}
    order = 0
    for arm, (policy_class, source, label) in ARMS.items():
        phase_0, phase_1 = phase_weights[arm]
        for weights in (phase_0, phase_1):
            if float(weights.min()) < -SIMPLEX_TOLERANCE or abs(float(weights.sum()) - 1.0) > SIMPLEX_TOLERANCE:
                raise ValueError(f"{arm} phase weights are off the simplex")
        aggregate = ALPHA_0 * phase_0 + ALPHA_1 * phase_1
        aggregate_error = float(np.max(np.abs(aggregate - targets[arm])))
        if aggregate_error > AGGREGATE_TOLERANCE[arm]:
            raise ValueError(f"{arm} aggregate misses its target by {aggregate_error}")
        phase_tv = float(0.5 * np.abs(phase_1 - phase_0).sum())
        if policy_class == "two_phase" and phase_tv < MIN_PHASE_TV:
            raise ValueError(f"{arm} phase TV {phase_tv} is below the {MIN_PHASE_TV} floor")
        if policy_class == "one_phase" and phase_tv > SIMPLEX_TOLERANCE:
            raise ValueError(f"{arm} is not single-phase")
        dolmino = np.asarray([domain.startswith("dolmino_") for domain in augmented.DOMAIN_NAMES])
        arm_summaries[arm] = {
            "policy_class": policy_class,
            "source_candidate": source,
            "label": label,
            "phase_tv": phase_tv,
            "aggregate_target": "arm_a" if arm == "BP" else "table9_frontier_anchor",
            "aggregate_max_abs_error": aggregate_error,
            "phase_0_dolmino_share": float(phase_0[dolmino].sum()),
            "phase_1_dolmino_share": float(phase_1[dolmino].sum()),
        }
        for trainer_seed in TRAINER_SEEDS:
            row = {
                "run_order": order,
                "run_id": RUN_ID_BASE + order,
                "candidate_id": f"sftp_{arm.lower()}_t{trainer_seed}",
                "arm": arm,
                "policy_class": policy_class,
                "source_candidate": source,
                "data_seed": DATA_SEED,
                "trainer_seed": trainer_seed,
                "phase_tv": phase_tv,
                "aggregate_max_abs_error": aggregate_error,
            }
            row.update({c: float(v) for c, v in zip(_phase_cols("phase_0"), phase_0, strict=True)})
            row.update({c: float(v) for c, v in zip(_phase_cols("phase_1"), phase_1, strict=True)})
            rows.append(row)
            order += 1
    frame = pd.DataFrame(rows)
    summary = {
        "data_seed": DATA_SEED,
        "trainer_seeds": list(TRAINER_SEEDS),
        "alpha_0": ALPHA_0,
        "alpha_1": ALPHA_1,
        "late_dolmino_share": LATE_DOLMINO_SHARE,
        "arm_a_candidate": ARM_A_CANDIDATE,
        "arm_a_source": str(MARINER_CANDIDATES.relative_to(REFERENCE_OUTPUT_DIR)),
        "anchor_candidate": ANCHOR_CANDIDATE,
        "n_runs": len(frame),
        "arms": arm_summaries,
    }
    return frame, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, summary = build_rows()
    panel_path = OUTPUT_DIR / "arms_source_panel.csv"
    frame.to_csv(panel_path, index=False)
    summary["arms_source_panel_sha256"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    long_rows = [
        {
            "candidate_id": row["candidate_id"],
            "arm": row["arm"],
            "phase": phase,
            "domain": domain,
            "weight": row[f"{phase}_{domain}"],
        }
        for _, row in frame.iterrows()
        for phase in augmented.PHASE_NAMES
        for domain in augmented.DOMAIN_NAMES
    ]
    pd.DataFrame(long_rows).to_csv(OUTPUT_DIR / "arms_phase_weights.csv", index=False)
    (OUTPUT_DIR / "arms_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
