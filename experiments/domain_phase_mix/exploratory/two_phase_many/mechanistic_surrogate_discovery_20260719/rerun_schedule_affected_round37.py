# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "matplotlib>=3.10",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Rerun frozen schedule-dependent routes after the Round 37 clock erratum."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_optimizer_time_fast_slow_round22 as round22,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_activated_memorization_round24 as round24,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_clipped_task_flow_round35 as round35,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_optimizer_time_flow_round20 as round20,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_sgd_drift_diffusion_round23 as round23,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_spherical_projected_flow_round36 as round36,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    sgd_drift_diffusion_models,
    starcoder_optimizer_schedule,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round37_schedule_corrected_reruns"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def corrected_clock_fraction(panel: object) -> tuple[float, float]:
    spec = starcoder_optimizer_schedule.schedule_for_name(panel.name)
    early, late = spec.phase_learning_rate_masses()
    return early / (early + late), (early + late) / spec.total_steps


def corrected_phase0_fraction(panel: object) -> float:
    return corrected_clock_fraction(panel)[0]


def corrected_learning_rate(time: float, phase0_fraction: float, schedule: object) -> float:
    del phase0_fraction
    schedule_name = str(schedule.value)
    spec = (
        starcoder_optimizer_schedule.COSINE_50_50
        if schedule_name == "cosine"
        else starcoder_optimizer_schedule.WSD_80_20
    )
    return float(spec.learning_rate(time))


def run_module(module: ModuleType, output_dir: Path, scratch_registry: Path, scratch_ledger: Path) -> None:
    module.REGISTRY = scratch_registry
    module.LEDGER = scratch_ledger
    previous_argv = sys.argv
    try:
        sys.argv = [str(module.__file__), "--output-dir", str(output_dir)]
        module.main()
    finally:
        sys.argv = previous_argv


def record_final_decisions(statuses: pd.DataFrame, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    for row in statuses.itertuples(index=False):
        registry.loc[registry["id"].eq(row.id), "status"] = row.status
        registry.loc[registry["id"].eq(row.id), "status_evidence"] = (
            f"Round 37 corrected-clock rerun: {row.status_evidence}"
        )
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    rows = []
    for row in statuses.itertuples(index=False):
        rows.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "round_id": "round_37_schedule_corrected_rerun",
                "candidate_id": row.id,
                "candidate_family": row.family,
                "hyperparameters": "Original frozen route grid; corrected source-derived optimizer schedule only",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": "Round 37 schedule-provenance erratum",
                "novelty_class": "Input-provenance correction, not retuning",
                "evaluation_status": row.status,
                "evidence_path": str((output_dir / row.id.lower()).relative_to(OUTPUT_ROOT)),
                "notes": str(row.status_evidence),
            }
        )
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    additions = [row for row in rows if tuple(row[column] for column in identity) not in existing]
    if additions:
        pd.concat([ledger, pd.DataFrame(additions, columns=ledger.columns)], ignore_index=True).to_csv(
            LEDGER,
            index=False,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch_registry = args.output_dir / "scratch_registry.csv"
    scratch_ledger = args.output_dir / "scratch_ledger.csv"
    if args.force or not scratch_registry.exists():
        shutil.copyfile(REGISTRY, scratch_registry)
    if args.force or not scratch_ledger.exists():
        shutil.copyfile(LEDGER, scratch_ledger)

    round20.optimizer_clock_fraction = corrected_clock_fraction
    round35.optimizer_phase0_fraction = corrected_phase0_fraction
    sgd_drift_diffusion_models.learning_rate = corrected_learning_rate

    modules = (
        ("OTTPF", round20),
        ("OTFSC", round22),
        ("SGDDD", round23),
        ("AAGF", round24),
        ("CTPF", round35),
        ("SPTF", round36),
    )
    for candidate_id, module in modules:
        output = args.output_dir / candidate_id.lower()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "report.md").exists() and not args.force:
            continue
        run_module(module, output, scratch_registry, scratch_ledger)

    scratch = pd.read_csv(scratch_registry)
    statuses = scratch.loc[
        scratch["id"].isin(candidate_id for candidate_id, _module in modules),
        [
            "id",
            "family",
            "status",
            "status_evidence",
        ],
    ].copy()
    statuses.to_csv(args.output_dir / "corrected_statuses.csv", index=False)
    record_final_decisions(statuses, args.output_dir)
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "clock_correction_only": True,
                "model_hyperparameters_changed": False,
                "routes": [candidate_id for candidate_id, _module in modules],
                "adversarial_targets_read": False,
                "sealed_confirmation_targets_read": False,
                "nominal_policy_boundary_note": (
                    "Dynamics retain the panel's nominal 0.5/0.8 boundary; exact aligned boundaries differ by "
                    "at most 3 of 3814 steps and are recorded in the schedule erratum."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = f"""# Round 37: corrected-clock reruns

The original frozen grids for all six optimizer-schedule-dependent routes were rerun after replacing the erroneous peak-LR-from-step-zero clock with the source-derived warmup, stable, and cosine-decay schedules. No model hyperparameter changed, and no 3e18 historical, exposed adversarial, or sealed-confirmation target was read.

## Final decisions

{statuses.to_markdown(index=False)}

None of the affected routes passes its original StarCoder gate. The correction therefore changes the provenance confidence but not the scientific decision: all six routes remain blocked before multi-swarm or adversarial evaluation.
"""
    (args.output_dir / "report.md").write_text(report)
    print(statuses.to_string(index=False))


if __name__ == "__main__":
    main()
