# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize a phase-total-variation ladder at the 3e18 Uncheatable frontier anchor.

The composite-proposal panel that just finished measured one point on this ladder and settled the
qualitative question: at phase total variation 0.24 the ordering effect is real and negative
(-0.002081 BPB, -2.28 run sigma, with the predicted orientation winning in both seed blocks) but the
cost of the asymmetry itself is larger (+0.003027, +3.32 sigma), so the two-phase policy loses to its
tied control in both blocks. The model's predicted gain was wrong by 9.58 sigma.

What that single point cannot say is whether a smaller tilt wins. Writing the response as
``gain(t) = -kappa t + (rho/2) t^2`` and solving it from that panel alone gives ``kappa = 0.00867``
and ``rho = 0.105``, hence an interior optimum at ``t* = 0.083`` worth ``0.00036`` BPB. That is a
sharp, falsifiable prediction, it sits far below the earlier low-epsilon path estimate of 0.0023, and
the existing ladder design was aimed at 0.19 -- past the optimum this panel implies.

So the ladder brackets 0.083 from both sides and keeps 0.24 as a replication check against the panel
that produced the estimate. Both orientations are run at every level because the odd/even
decomposition needs them: without the antithetic partner, a level's gain cannot be split into the
ordering effect and the asymmetry cost, and it is their difference that decides whether two-phase can
win at all.

The anchor and the contrast direction are recovered from the finished panel's own rows rather than
re-derived, so the direction is identical at every level and the 0.24 arm is a true replicate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
from pathlib import Path

import fsspec
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.launch_delphi_augmented_swarm_3e18 import (  # noqa: E402
    DOMAIN_NAMES,
    PHASE_NAMES,
)

logger = logging.getLogger(__name__)

PANEL_ID = "delphi_3e18_uncheatable_phase_tv_ladder_20260727"
DEFAULT_OUTPUT_PREFIX = f"gs://marin-us-east5/pinlin_calvin_xu/data_mixture/{PANEL_ID}/source"
REFERENCE_OUTPUTS = SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs"
LOCAL_ARTIFACT_DIR = REFERENCE_OUTPUTS / PANEL_ID
SOURCE_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_composite_proposal_validation_20260726"
SOURCE_PANEL_GLOB = "validation_panel-*.csv"

RUN_ID_BASE = 7_270_000
TRAINER_SEED = 0
SEED_BLOCKS = (0, 1, 2)
GEOMETRY_TOLERANCE = 2e-12
# Phase-0 share of training at 3e18. Must match the source panel, and is asserted against it.
ALPHA_0 = 0.7981376787495837
ALPHA_1 = 1.0 - ALPHA_0

# Brackets the implied optimum at 0.083 from both sides; 0.24 replicates the finished panel.
PHASE_TV_LEVELS = (0.06, 0.10, 0.16, 0.24)
SIGNS = ("plus", "minus")
DIRECTION_ID = "technical_specialization"
ANCHOR_ID = "uncheatable_frontier"
REPLICATION_LEVEL = 0.24

# Quadratic solved from the finished panel, recorded before any run of this ladder exists.
IMPLIED_RESPONSE = {
    "kappa_per_tv": 0.008671,
    "rho_per_tv_squared": 0.105104,
    "implied_optimum_tv": 0.0825,
    "implied_gain_at_optimum_bpb": -0.000358,
    "source_odd_effect_at_tv0p24": -0.002081,
    "source_asymmetry_cost_at_tv0p24": 0.003027,
    "run_sigma": 0.000913,
    "note": (
        "Predicted gain at the optimum is 0.39 run sigma, so this ladder is powered to locate the "
        "optimum and bound the effect, not to demonstrate a deployable win."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--local-only", action="store_true", help="Write the local copy without uploading.")
    return parser.parse_args()


def load_source_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Recover the anchor and the unit contrast direction from the finished panel.

    Reading them back rather than re-deriving them guarantees the ladder sweeps the same direction the
    measured point sits on, which is what makes the 0.24 arm a replicate instead of a near-miss.
    """
    panel_files = sorted(SOURCE_PANEL_DIR.glob(SOURCE_PANEL_GLOB))
    assert len(panel_files) == 1, f"expected one source panel, found {panel_files}"
    rows = list(csv.DictReader(panel_files[0].open()))
    tied = next(row for row in rows if row["sign"] == "center")
    plus = next(row for row in rows if row["sign"] == "plus")

    anchor = np.array([float(tied[f"phase_0_{name}"]) for name in DOMAIN_NAMES], dtype=float)
    tied_phase_1 = np.array([float(tied[f"phase_1_{name}"]) for name in DOMAIN_NAMES], dtype=float)
    assert np.abs(anchor - tied_phase_1).max() == 0.0, "tied control is not tied"

    plus_phase_0 = np.array([float(plus[f"phase_0_{name}"]) for name in DOMAIN_NAMES], dtype=float)
    plus_phase_1 = np.array([float(plus[f"phase_1_{name}"]) for name in DOMAIN_NAMES], dtype=float)
    source_tv = 0.5 * float(np.abs(plus_phase_0 - plus_phase_1).sum())
    assert abs(source_tv - REPLICATION_LEVEL) < 1e-9, f"source panel TV is {source_tv}, expected {REPLICATION_LEVEL}"

    aggregate = ALPHA_0 * plus_phase_0 + ALPHA_1 * plus_phase_1
    assert np.abs(aggregate - anchor).max() < GEOMETRY_TOLERANCE, "source phase split does not preserve the anchor"

    unit = (plus_phase_0 - anchor) / (ALPHA_1 * source_tv)
    assert abs(unit.sum()) <= GEOMETRY_TOLERANCE, "direction left the simplex tangent space"
    assert abs(0.5 * float(np.abs(unit).sum()) - 1.0) < 1e-9, "direction is not unit phase total variation"
    return anchor, unit


def orientation(anchor: np.ndarray, unit: np.ndarray, phase_tv: float, sign: str) -> tuple[np.ndarray, np.ndarray]:
    """Phase mixtures at a signed distance along the contrast, holding the aggregate fixed."""
    contrast = (1.0 if sign == "plus" else -1.0) * phase_tv * unit
    phase_0 = anchor + ALPHA_1 * contrast
    phase_1 = anchor - ALPHA_0 * contrast
    assert phase_0.min() > -GEOMETRY_TOLERANCE, f"{sign} at tv {phase_tv} left the simplex in phase 0"
    assert phase_1.min() > -GEOMETRY_TOLERANCE, f"{sign} at tv {phase_tv} left the simplex in phase 1"
    return np.maximum(phase_0, 0.0), np.maximum(phase_1, 0.0)


def build_rows(anchor: np.ndarray, unit: np.ndarray) -> list[dict[str, object]]:
    """Every ladder level in both orientations, plus one tied control per seed block."""
    rows: list[dict[str, object]] = []
    run_id = RUN_ID_BASE
    for seed_block in SEED_BLOCKS:
        data_seed = RUN_ID_BASE + seed_block
        for phase_tv in PHASE_TV_LEVELS:
            for sign in SIGNS:
                phase_0, phase_1 = orientation(anchor, unit, phase_tv, sign)
                rows.append(
                    _row(
                        candidate_id=f"tvladder_{DIRECTION_ID}_tv{phase_tv:g}_{sign}_s{seed_block}",
                        run_id=run_id,
                        sign=sign,
                        phase_tv=phase_tv,
                        seed_block=seed_block,
                        data_seed=data_seed,
                        phase_0=phase_0,
                        phase_1=phase_1,
                        anchor=anchor,
                    )
                )
                run_id += 1
        rows.append(
            _row(
                candidate_id=f"tvladder_tied_control_s{seed_block}",
                run_id=run_id,
                sign="center",
                phase_tv=0.0,
                seed_block=seed_block,
                data_seed=data_seed,
                phase_0=anchor.copy(),
                phase_1=anchor.copy(),
                anchor=anchor,
            )
        )
        run_id += 1
    return rows


def _row(
    candidate_id: str,
    run_id: int,
    sign: str,
    phase_tv: float,
    seed_block: int,
    data_seed: int,
    phase_0: np.ndarray,
    phase_1: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, object]:
    aggregate = ALPHA_0 * phase_0 + ALPHA_1 * phase_1
    aggregate_error = float(np.abs(aggregate - anchor).max())
    assert aggregate_error < GEOMETRY_TOLERANCE, f"{candidate_id} moved the aggregate by {aggregate_error}"
    realized_tv = 0.5 * float(np.abs(phase_0 - phase_1).sum())
    assert abs(realized_tv - phase_tv) < 1e-9, f"{candidate_id} realized tv {realized_tv}, wanted {phase_tv}"
    policy = np.concatenate([phase_0, phase_1])
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "anchor_id": ANCHOR_ID,
        "direction_id": DIRECTION_ID if sign != "center" else "tied_control",
        "contrast_family": "phase_tv_ladder" if sign != "center" else "center_control",
        "sign": sign,
        "phase_tv": phase_tv,
        "seed_block": seed_block,
        "replicate_index": 0,
        "data_seed": data_seed,
        "trainer_seed": TRAINER_SEED,
        "aggregate_max_abs_error": aggregate_error,
        "is_replication_of_finished_panel": bool(sign != "center" and phase_tv == REPLICATION_LEVEL),
        "policy_sha256": hashlib.sha256(policy.tobytes()).hexdigest(),
    }
    for index, name in enumerate(DOMAIN_NAMES):
        row[f"{PHASE_NAMES[0]}_{name}"] = float(phase_0[index])
        row[f"{PHASE_NAMES[1]}_{name}"] = float(phase_1[index])
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    anchor, unit = load_source_geometry()
    rows = build_rows(anchor, unit)

    expected = len(SEED_BLOCKS) * (len(PHASE_TV_LEVELS) * len(SIGNS) + 1)
    assert len(rows) == expected, f"built {len(rows)} rows, expected {expected}"
    assert len({row["candidate_id"] for row in rows}) == len(rows), "duplicate candidate ids"
    assert len({row["run_id"] for row in rows}) == len(rows), "duplicate run ids"
    treatments = sum(1 for row in rows if row["sign"] != "center")
    logger.info("built %d rows: %d treatments, %d tied controls", len(rows), treatments, len(rows) - treatments)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode()
    digest = hashlib.sha256(payload).hexdigest()

    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    panel_name = f"ladder_panel-{digest[:16]}.csv"
    (LOCAL_ARTIFACT_DIR / panel_name).write_bytes(payload)
    manifest = {
        "panel_id": PANEL_ID,
        "rows": len(rows),
        "phase_tv_levels": list(PHASE_TV_LEVELS),
        "seed_blocks": list(SEED_BLOCKS),
        "anchor_id": ANCHOR_ID,
        "direction_id": DIRECTION_ID,
        "alpha_0": ALPHA_0,
        "source_panel_sha256": "342a448c4278739432fa73e3bb37e7a4864ad398f14d3c8b7a2748909e3cf66d",
        "source_panel_id": "delphi_3e18_composite_proposal_validation_20260726",
        "implied_response": IMPLIED_RESPONSE,
        "panel_sha256": digest,
        "panel_file": panel_name,
    }
    (LOCAL_ARTIFACT_DIR / "ladder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.local_only:
        remote_path = f"{args.output_prefix.rstrip('/')}/{panel_name}"
        with fsspec.open(remote_path, "wb") as handle:
            handle.write(payload)
        logger.info("uploaded %s", remote_path)

    print(f"panel rows       : {len(rows)}")
    print(f"tv levels        : {PHASE_TV_LEVELS} x {SIGNS} x {len(SEED_BLOCKS)} seed blocks")
    print(f"panel sha256     : {digest}")
    print(f"local artifact   : {LOCAL_ARTIFACT_DIR / panel_name}")


if __name__ == "__main__":
    main()
