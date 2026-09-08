# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the 3e18 validation panel for the composite surrogate's own top proposal.

The composite surrogate -- geometry-augmented hierarchical replay, band-ensembled, fitted on
140 exposure-matched pairs for a 280-run budget -- was asked to rank fixed-aggregate two-phase
policies at the 3e18 Uncheatable frontier aggregate. Its top pick was the
``technical_specialization`` direction in the ``plus`` orientation at phase total variation
0.24, predicted to beat the tied control by 0.0078 BPB.

The same model made the same kind of call at the 60M frontier and was right: it proposed
``curated_noncc_vs_cc`` plus at TV 0.24 and the observed plus orientation beat its same-seed
tied control by 0.00548, which is 7.2 sigma of that anchor's control standard deviation, with
the losing orientation coming in at +0.01308. That 60M check used an existing panel the model
was never fitted on. This panel is the 3e18 half of the same test.

Four rows, two seed blocks:

* ``plus`` -- the proposal, named group late. The model says this wins.
* ``minus`` -- the antithetic partner, named group early. The model says this loses.
* two tied controls at the identical aggregate, one per seed block.

The antithetic partner matters more than it looks. Comparing only the proposal against a tied
control cannot distinguish "the model chose the right orientation" from "any asymmetry in this
direction helps", and the odd/even decomposition needs both orientations to separate the order
effect from the cost of the asymmetry itself. The aggregate is held identical across all four
rows to machine precision, so nothing here varies except phase ordering and seed.

Predictions are recorded in the manifest before any run starts, so the comparison is
confirmatory rather than fitted after the fact.
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

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_3e18 import (  # noqa: E402
    DOMAIN_NAMES,
    PHASE_NAMES,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PREFIX = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_composite_proposal_validation_20260726/source"
)
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / ("delphi_3e18_composite_proposal_validation_20260726")
)

RUN_ID_BASE = 7_260_000
TRAINER_SEED = 0
SEED_BLOCKS = (0, 1)
GEOMETRY_TOLERANCE = 2e-12

# Phase-0 share of training at 3e18, used to keep the aggregate fixed while ordering varies.
ALPHA_0 = 0.7981376787495837
ALPHA_1 = 1.0 - ALPHA_0

# The composite's prediction for this panel, recorded before the runs exist.
PROPOSAL = {
    "direction_id": "technical_specialization",
    "phase_tv": 0.24,
    "winning_sign": "plus",
    "predicted_gain_vs_tied_bpb": -0.00780,
    "predicted_tied_bpb": 0.91163,
    "model": "geometry-augmented hierarchical replay, band ensemble, 140 exposure-matched pairs",
    "sixty_m_precedent": {
        "direction_id": "curated_noncc_vs_cc",
        "predicted_gain": -0.00712,
        "observed_gain": -0.00548,
        "observed_in_control_sigma": 7.2,
    },
}

# Bucket groups for technical_specialization, taken verbatim from the 60M panel's registry so
# the direction is identical at both scales.
TECHNICAL_TOPICS = ("science_math", "education_and_jobs", "electronics_and_hardware")
TECHNICAL_EXPLICIT = (
    "dolma3_stack_edu",
    "dolma3_arxiv",
    "dolma3_finemath_3plus",
    "dolmino_stack_edu_fim",
    "dolmino_stem_heavy_crawl",
    "dolmino_synth_code",
    "dolmino_synth_math",
    "dolmino_synth_thinking",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-npy", type=Path, required=True, help="3e18 frontier aggregate, canonical order.")
    parser.add_argument("--buckets-json", type=Path, required=True, help="Bucket order matching the anchor vector.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--local-only", action="store_true", help="Write the local copy without uploading.")
    return parser.parse_args()


def technical_group(domains: tuple[str, ...]) -> set[str]:
    return {domain for domain in domains if any(topic in domain for topic in TECHNICAL_TOPICS)} | set(TECHNICAL_EXPLICIT)


def unit_direction(anchor: np.ndarray, domains: tuple[str, ...], left: set[str]) -> np.ndarray:
    """Left group negative, right group positive, normalized to unit phase total variation.

    Matches ``_group_direction`` in the 60M materializer exactly, including the sign convention
    that makes ``plus`` place the named left group later.
    """
    left_index = np.array([index for index, domain in enumerate(domains) if domain in left], dtype=int)
    right_index = np.array([index for index, domain in enumerate(domains) if domain not in left], dtype=int)
    left_mass, right_mass = float(anchor[left_index].sum()), float(anchor[right_index].sum())
    assert left_mass > 0 and right_mass > 0, "technical split has a zero-mass side"
    direction = np.zeros_like(anchor)
    direction[left_index] = -anchor[left_index] / left_mass
    direction[right_index] = anchor[right_index] / right_mass
    direction -= direction.sum() * anchor
    direction -= direction.sum() / len(direction)
    assert abs(direction.sum()) <= GEOMETRY_TOLERANCE, "direction left the simplex tangent space"
    return direction / (0.5 * np.abs(direction).sum())


def orientation(anchor: np.ndarray, unit: np.ndarray, phase_tv: float, sign: str) -> tuple[np.ndarray, np.ndarray]:
    contrast = (1.0 if sign == "plus" else -1.0) * phase_tv * unit
    phase_0 = anchor + ALPHA_1 * contrast
    phase_1 = anchor - ALPHA_0 * contrast
    assert phase_0.min() > -GEOMETRY_TOLERANCE and phase_1.min() > -GEOMETRY_TOLERANCE, f"{sign} left the simplex"
    return np.maximum(phase_0, 0.0), np.maximum(phase_1, 0.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    anchor_raw = np.load(args.anchor_npy)
    buckets = tuple(json.loads(args.buckets_json.read_text()))
    assert len(anchor_raw) == len(buckets), "anchor and bucket order disagree"
    assert set(buckets) == set(DOMAIN_NAMES), "bucket set does not match the launcher's domain set"
    assert set(buckets) == set(TOP_LEVEL_DOMAIN_TOKEN_COUNTS), "bucket set does not match the token-count table"

    # Reindex onto the launcher's canonical domain order.
    order = [buckets.index(domain) for domain in DOMAIN_NAMES]
    anchor = anchor_raw[order]
    assert abs(anchor.sum() - 1.0) < 1e-9, f"anchor does not sum to one: {anchor.sum()}"

    unit = unit_direction(anchor, tuple(DOMAIN_NAMES), technical_group(tuple(DOMAIN_NAMES)))
    rows: list[dict[str, object]] = []
    run_id = RUN_ID_BASE
    for seed_block in SEED_BLOCKS:
        for sign in ("plus", "minus", "center"):
            if sign == "center" and seed_block not in SEED_BLOCKS:
                continue
            if sign == "center":
                phase_0 = phase_1 = anchor
            else:
                phase_0, phase_1 = orientation(anchor, unit, PROPOSAL["phase_tv"], sign)
            aggregate = ALPHA_0 * phase_0 + ALPHA_1 * phase_1
            aggregate_error = float(np.max(np.abs(aggregate - anchor)))
            assert aggregate_error < GEOMETRY_TOLERANCE, f"aggregate drifted by {aggregate_error:.3e}"
            candidate_id = (
                f"composite_val_{PROPOSAL['direction_id']}_tv{PROPOSAL['phase_tv']:.2f}_{sign}_s{seed_block}"
                if sign != "center"
                else f"composite_val_tied_control_s{seed_block}"
            )
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "anchor_id": "uncheatable_frontier",
                "direction_id": PROPOSAL["direction_id"] if sign != "center" else "tied_control",
                "contrast_family": "composite_proposal" if sign != "center" else "center_control",
                "sign": sign,
                "phase_tv": 0.0 if sign == "center" else PROPOSAL["phase_tv"],
                "seed_block": seed_block,
                "replicate_index": 0,
                "data_seed": 7_260_000 + seed_block,
                "trainer_seed": TRAINER_SEED,
                "aggregate_max_abs_error": aggregate_error,
                "model_predicted_gain_vs_tied_bpb": (
                    PROPOSAL["predicted_gain_vs_tied_bpb"] if sign == PROPOSAL["winning_sign"] else ""
                ),
                "policy_sha256": hashlib.sha256(np.concatenate([phase_0, phase_1]).astype("<f8").tobytes()).hexdigest(),
            }
            for phase_name, weights in zip(PHASE_NAMES, (phase_0, phase_1), strict=True):
                for domain, weight in zip(DOMAIN_NAMES, weights, strict=True):
                    row[f"{phase_name}_{domain}"] = f"{weight:.17g}"
            rows.append(row)
            run_id += 1

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode()
    digest = hashlib.sha256(payload).hexdigest()

    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_ARTIFACT_DIR / f"validation_panel-{digest[:16]}.csv"
    local_path.write_bytes(payload)
    (LOCAL_ARTIFACT_DIR / "proposal.json").write_text(
        json.dumps({**PROPOSAL, "rows": len(rows), "source_panel_sha256": digest}, indent=2)
    )

    remote_path = f"{args.output_prefix}/validation_panel-{digest[:16]}.csv"
    if not args.local_only:
        with fsspec.open(remote_path, "wb") as handle:
            handle.write(payload)
        logger.info("Uploaded %s", remote_path)

    print(f"rows: {len(rows)}")
    print(f"local: {local_path}")
    print(f"remote: {remote_path}")
    print(f"sha256: {digest}")
    print(f"max aggregate error across rows: {max(float(row['aggregate_max_abs_error']) for row in rows):.3e}")


if __name__ == "__main__":
    main()
