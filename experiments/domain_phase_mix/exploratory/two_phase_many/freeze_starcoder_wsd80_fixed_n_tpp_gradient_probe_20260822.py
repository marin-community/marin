# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the fixed-N StarCoder WSD80 TPP gradient-onset extension."""

import csv
import hashlib
import json
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as repair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_plot_completion_20260822 as historical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as parent,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_fixed_n_tpp_gradient_probe_v1_20260822"
RUNTIME_PATH = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_fixed_n_tpp_gradient_probe.py"
ANALYSIS_CONTRACT_PATH = OUTPUT_DIR / "analysis_contract.json"
FULL_MANIFEST_PATH = OUTPUT_DIR / "full_probe_manifest.csv"
CHECKPOINT_PROVENANCE_PATH = OUTPUT_DIR / "checkpoint_object_provenance.csv"
PARENT_RESULT_PROVENANCE_PATH = OUTPUT_DIR / "parent_result_object_provenance.csv"
RELEASE_PATH = OUTPUT_DIR / "release.json"
FULL_LAUNCH_AUTHORIZATION_PATH = OUTPUT_DIR / "full_launch_authorization.json"
PREFLIGHT_AUDIT_PATH = OUTPUT_DIR / "preflight_wave_audit.json"
FINAL_AUDIT_PATH = OUTPUT_DIR / "final_whole_panel_audit.json"
CC_REVIEW_PATH = REPO_ROOT / ".agents/handoffs/starcoder_wsd80_fixed_n_tpp_gradient_probe_cc_review_20260822.md"

PARENT_RELEASE_PATH = parent.OUTPUT_DIR / "release.json"
PARENT_FULL_MANIFEST_PATH = parent.OUTPUT_DIR / "full_probe_manifest.csv"
TRAJECTORY_MANIFEST_PATH = parent.DESIGN_DIR / "trajectory_manifest.csv"
SOURCE_DESIGN_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
)
HISTORICAL_RELEASE_PATH = historical.RELEASE_PATH

MARIN_PREFIX = "gs://marin-us-central1"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_fixed_n_tpp_gradient_probe_v1_20260822"
)
RELEASE_VERSION = "2026-08-22-fixed-n-tpp-gradient-probe-v1"
SCIENTIFIC_STATUS = "prospective_fixed_n_tpp_onset_extension_lower_rungs_uninspected"
PARENT_ARTIFACT_VERSION = repair.PARENT_ARTIFACT_VERSION
TASK_IMAGE = historical.TASK_IMAGE
EXPECTED_PYTHON_VERSION = historical.EXPECTED_PYTHON_VERSION
EXPECTED_RUNTIME_VERSIONS = historical.EXPECTED_RUNTIME_VERSIONS
EXPECTED_DEVICE_COUNT = historical.EXPECTED_DEVICE_COUNT
HISTORICAL_RUNTIME_COMMIT = historical.HISTORICAL_RUNTIME_COMMIT
GLOBAL_STARCODER = repair.GLOBAL_STARCODER
SUPPORT_STARCODER = repair.SUPPORT_STARCODER
NEMOTRON = repair.NEMOTRON
REMOTE_ADAPTER_CANARY_PATH = f"{RESULT_ROOT}/runtime_adapter_canary.json"
RUNTIME_ENVIRONMENT_BASELINE_PATH = OUTPUT_DIR / "runtime_environment_baseline.json"

FIXED_N_CELLS = (
    "r0_shared_h0640_s03820",
    "r1_increase_d_h0640_s07320",
    "r2_increase_d_h0640_s14960",
    "r3_increase_d_h0640_s28260",
)
CELL_STAGE = {cell: index + 1 for index, cell in enumerate(FIXED_N_CELLS)}
STAGE_ROW_COUNTS = {0: 16, **{stage: 64 for stage in CELL_STAGE.values()}}
CHECKPOINT_STATES = (
    "fraction_0p55",
    "fraction_0p70",
    "decay_minus_256",
    "decay_minus_64",
    "decay_onset",
    "decay_plus_64",
    "decay_plus_256",
    "fraction_0p90",
)
TRAINING_SEEDS = tuple(range(2026081000, 2026081008))
SOURCE_DISTRIBUTIONS = (repair.GLOBAL_STARCODER, repair.NEMOTRON)
TARGET_DISTRIBUTIONS = repair.TARGET_DISTRIBUTIONS
STANDARD_BLOCK_COUNTS = {
    repair.GLOBAL_STARCODER: 16,
    repair.NEMOTRON: 16,
    "paloma_programming_languages": 16,
    "paloma_c4_en": 7,
    "uncheatable_github_python": 7,
    "uncheatable_wikipedia_english": 4,
}
STANDARD_UPDATE_DRAWS = {
    repair.GLOBAL_STARCODER: 8,
    repair.NEMOTRON: 8,
    **{target: 0 for target in TARGET_DISTRIBUTIONS},
}
EXPECTED_ROWS = len(FIXED_N_CELLS) * len(TRAINING_SEEDS) * len(CHECKPOINT_STATES)
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_FIXED_N_TPP_GRADIENT_ONSET_PROBES"

ANALYSIS_CONTRACT: dict[str, Any] = {
    "contract_version": "2026-08-22-fixed-n-tpp-gradient-onset-v1",
    "scientific_status": SCIENTIFIC_STATUS,
    "endpoint_metrics_read_by_runner": False,
    "question": (
        "At fixed model size and a fixed tied 35% StarCoder policy, does increasing materialized-token TPP move "
        "the training time at which StarCoder and Nemotron gradients lose alignment?"
    ),
    "design": {
        "cells": list(FIXED_N_CELLS),
        "support_id": "full",
        "policy_role": "common_tied_035",
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_states": list(CHECKPOINT_STATES),
        "source_distributions": list(SOURCE_DISTRIBUTIONS),
        "target_distributions": list(TARGET_DISTRIBUTIONS),
        "standard_block_counts": STANDARD_BLOCK_COUNTS,
        "standard_update_draw_counts": STANDARD_UPDATE_DRAWS,
        "paired_reference_batches": (
            "The same frozen sequence-set identities and training seeds are used in every TPP cell. The first 16 "
            "blocks are recomputed at every rung, including r3, to avoid confounding TPP with estimator precision."
        ),
    },
    "primary_estimand": {
        "statistic": "projected-trunk raw-gradient cosine between global-heldout StarCoder and Nemotron",
        "inferential_unit": "training seed paired across all four TPP cells",
        "time_axis": "actual checkpoint_step / total_steps; LR decay begins at 0.80 in every cell",
        "plateau_reference": "within-seed mean of fraction_0p55 and fraction_0p70",
        "decline": (
            "Primary: plateau_reference minus cosine at fraction_0p90. This comparison uses the same normalized "
            "times at every rung; positive values mean less alignment."
        ),
        "tpp_comparison": (
            "Compare the four paired 0.90T declines. A monotone trend is descriptive with four cells; do not choose "
            "between total- and non-embedding-parameter TPP because they differ only by a constant factor at fixed N."
        ),
    },
    "secondary_estimands": {
        "onset": (
            "Fit C(t)=alpha-gamma*max(t-tau,0), gamma>=0, only on the common normalized-time subgrid "
            "{0.55, 0.70, 0.80, 0.90}; report tau as weakly identified with four time points. Separately show all "
            "eight probes on absolute steps from LR-decay onset. Do not compare full-grid normalized-time tau values "
            "across cells because the +/-64/256-step probes contract toward 0.80T as total steps increase."
        ),
        "other_statistics": (
            "Projected-trunk optimizer-update cosine and target-source utility cosines use the same rows. They are "
            "secondary because optimizer updates depend on optimizer state and the LR schedule, while the primary "
            "raw-gradient direction does not multiply the gradient by learning rate."
        ),
    },
    "interpretation_limits": (
        "The r3 trajectory motivated this extension, so the four-cell trend is not untouched global confirmation. "
        "The eight states largely reuse the r3 H2 mid/pre-decay/post-decay windows; r3 onset estimates are therefore "
        "post-selection, while lower-rung gradient outcomes were not inspected before freezing. This fixed-N ladder "
        "varies D, TPP, absolute update count, and exposure together; it tests whether decline tracks that bundle, "
        "not which member of the bundle is causal."
    ),
}


def canonical_json(value: Any) -> str:
    return historical.canonical_json(value)


def canonical_sha256(value: Any) -> str:
    return historical.canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen release artifact already exists with different content: {path}") from error


def _write_json(path: Path, value: Any) -> None:
    _write_create_only(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _write_create_only(path, buffer.getvalue().encode())


def _review_verdict(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    verdicts = [line for line in lines if line.startswith("VERDICT:")]
    if len(verdicts) != 1 or lines[-1] != verdicts[0]:
        raise ValueError("CC review must end with exactly one VERDICT line")
    return verdicts[0].partition(":")[2].strip()


def _parent_release() -> dict[str, Any]:
    release = repair._parent_release()
    if release != json.loads(PARENT_RELEASE_PATH.read_text()):
        raise ValueError("Validated parent release does not match the configured parent path")
    return release


def _cell_metadata() -> dict[str, dict[str, Any]]:
    payload = json.loads(SOURCE_DESIGN_PATH.read_text())
    result = {row["cell_id"]: row for row in payload["cells"] if row["cell_id"] in FIXED_N_CELLS}
    if set(result) != set(FIXED_N_CELLS):
        raise ValueError("Fixed-N source-cell inventory drifted")
    return result


def _probe_identity(payload: dict[str, Any]) -> tuple[str, str]:
    identity = {"scope": "full", **payload, "parent_release_sha256": _parent_release()["release_sha256"]}
    digest = canonical_sha256(identity)[:24]
    return f"fixed_n_tpp_{digest}", f"fixed_n_tpp_group_{digest}"


def _probe_row(group: list[dict[str, str]], trajectory: dict[str, str]) -> dict[str, Any]:
    representative = group[0]
    parent_blocks, parent_update_draws, sequence_sets, probe_row_ids = repair._distribution_contract(
        group,
        source_ids=SOURCE_DISTRIBUTIONS,
        target_ids=TARGET_DISTRIBUTIONS,
    )
    payload = {
        "trajectory_id": representative["trajectory_id"],
        "checkpoint_label": representative["checkpoint_label"],
        "checkpoint_step": int(representative["checkpoint_step"]),
        "expected_restored_state_step": int(representative["expected_restored_state_step"]),
        "checkpoint_uri": representative["checkpoint_uri"],
        "parent_probe_group_id": representative["group_id"],
        "train_config_sha256": representative["train_config_sha256"],
        "analysis_role": "fixed_n_tpp_onset",
        "source_distribution_ids_json": canonical_json(SOURCE_DISTRIBUTIONS),
        "target_distribution_ids_json": canonical_json(TARGET_DISTRIBUTIONS),
        "distribution_block_counts_json": canonical_json(STANDARD_BLOCK_COUNTS),
        "distribution_update_draw_counts_json": canonical_json(STANDARD_UPDATE_DRAWS),
        "parent_distribution_block_counts_json": canonical_json(parent_blocks),
        "parent_distribution_update_draw_counts_json": canonical_json(parent_update_draws),
        "distribution_sequence_set_ids_json": canonical_json(sequence_sets),
        "distribution_probe_row_ids_json": canonical_json(probe_row_ids),
        "scientific_status": SCIENTIFIC_STATUS,
        "endpoint_metrics_read_by_runner": False,
        "arm": trajectory["arm"],
        "cell_id": trajectory["cell_id"],
        "support_id": trajectory["support_id"],
        "training_seed": int(trajectory["training_seed"]),
        "policy_role": trajectory["policy_role"],
    }
    row_id, group_id = _probe_identity(payload)
    return {
        **payload,
        "scope": "full",
        "row_id": row_id,
        "group_id": group_id,
        "launch_stage": CELL_STAGE[trajectory["cell_id"]],
        "preflight_wave": int(
            int(trajectory["training_seed"]) == TRAINING_SEEDS[0]
            and (CHECKPOINT_STATES.index(representative["checkpoint_label"]) + CELL_STAGE[trajectory["cell_id"]]) % 2
            == 0
        ),
    }


def _rows() -> list[dict[str, Any]]:
    trajectories = {row["trajectory_id"]: row for row in _read_csv(TRAJECTORY_MANIFEST_PATH)}
    rows = []
    for group in repair._manifest_groups(PARENT_FULL_MANIFEST_PATH).values():
        representative = group[0]
        trajectory = trajectories[representative["trajectory_id"]]
        selected = (
            trajectory["cell_id"] in FIXED_N_CELLS
            and trajectory["support_id"] == "full"
            and trajectory["policy_role"] == "common_tied_035"
            and int(trajectory["training_seed"]) in TRAINING_SEEDS
            and representative["checkpoint_label"] in CHECKPOINT_STATES
        )
        if selected:
            rows.append(_probe_row(group, trajectory))
    return sorted(rows, key=lambda row: (int(row["launch_stage"]), row["group_id"]))


def _matches_parent_precision(row: dict[str, Any]) -> bool:
    return json.loads(row["distribution_block_counts_json"]) == json.loads(
        row["parent_distribution_block_counts_json"]
    ) and json.loads(row["distribution_update_draw_counts_json"]) == json.loads(
        row["parent_distribution_update_draw_counts_json"]
    )


def _parent_precision_reductions(row: dict[str, Any]) -> dict[str, dict[str, int]]:
    blocks = json.loads(row["distribution_block_counts_json"])
    parent_blocks = json.loads(row["parent_distribution_block_counts_json"])
    draws = json.loads(row["distribution_update_draw_counts_json"])
    parent_draws = json.loads(row["parent_distribution_update_draw_counts_json"])
    return {
        "blocks": {name: int(parent_blocks[name]) - int(blocks[name]) for name in blocks},
        "optimizer_draws": {name: int(parent_draws[name]) - int(draws[name]) for name in draws},
    }


def _validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_ROWS or len({row["row_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError(f"Fixed-N TPP row inventory drifted: {len(rows)} != {EXPECTED_ROWS}")
    stage_counts = Counter(int(row["launch_stage"]) for row in rows)
    if stage_counts != Counter({stage: 64 for stage in CELL_STAGE.values()}):
        raise ValueError(f"Fixed-N TPP stage inventory drifted: {stage_counts}")
    preflight_rows = [row for row in rows if int(row["preflight_wave"]) == 1]
    if (
        len(preflight_rows) != STAGE_ROW_COUNTS[0]
        or Counter(row["cell_id"] for row in preflight_rows) != Counter({cell: 4 for cell in FIXED_N_CELLS})
        or Counter(row["checkpoint_label"] for row in preflight_rows)
        != Counter({state: 2 for state in CHECKPOINT_STATES})
    ):
        raise ValueError("Fixed-N TPP preflight wave does not span every cell and temporal state")
    cells = _cell_metadata()
    total_parameters = {int(cells[cell]["total_parameters"]) for cell in FIXED_N_CELLS}
    non_embedding_parameters = {int(cells[cell]["non_embedding_parameters"]) for cell in FIXED_N_CELLS}
    if len(total_parameters) != 1 or len(non_embedding_parameters) != 1:
        raise ValueError("Fixed-N TPP ladder does not hold model size fixed")
    tpp = {
        cell: {
            "materialized_tokens": int(cells[cell]["materialized_tokens"]),
            "total_parameters": int(cells[cell]["total_parameters"]),
            "non_embedding_parameters": int(cells[cell]["non_embedding_parameters"]),
            "total_parameter_tpp": int(cells[cell]["materialized_tokens"]) / int(cells[cell]["total_parameters"]),
            "non_embedding_parameter_tpp": (
                int(cells[cell]["materialized_tokens"]) / int(cells[cell]["non_embedding_parameters"])
            ),
        }
        for cell in FIXED_N_CELLS
    }
    sequence_sets: dict[tuple[int, str], set[str]] = {}
    coverage = Counter((row["cell_id"], int(row["training_seed"])) for row in rows)
    if coverage != Counter({(cell, seed): len(CHECKPOINT_STATES) for cell in FIXED_N_CELLS for seed in TRAINING_SEEDS}):
        raise ValueError("Fixed-N TPP cell/seed/state coverage drifted")
    reduced_rows = []
    for row in rows:
        reductions = _parent_precision_reductions(row)
        if any(value < 0 for family in reductions.values() for value in family.values()):
            raise ValueError(f"Requested probe precision exceeds its frozen parent row: {row['row_id']}")
        if any(value > 0 for family in reductions.values() for value in family.values()):
            reduced_rows.append(row)
        for distribution, sequence_set in json.loads(row["distribution_sequence_set_ids_json"]).items():
            sequence_sets.setdefault((int(row["training_seed"]), distribution), set()).add(sequence_set)
    expected_reduced_states = {
        "fraction_0p55",
        "decay_minus_256",
        "decay_minus_64",
        "decay_plus_64",
        "decay_plus_256",
        "fraction_0p90",
    }
    if (
        len(reduced_rows) != 48
        or {row["cell_id"] for row in reduced_rows} != {FIXED_N_CELLS[-1]}
        or {row["checkpoint_label"] for row in reduced_rows} != expected_reduced_states
    ):
        raise ValueError("Fixed-N TPP parent-precision reduction inventory drifted")
    mismatched = {str(key): sorted(value) for key, value in sequence_sets.items() if len(value) != 1}
    if mismatched:
        raise ValueError(f"Reference sequence sets are not paired across TPP cells: {mismatched}")
    time_grid = {}
    for cell in FIXED_N_CELLS:
        steps_by_label = {
            label: {
                int(row["checkpoint_step"])
                for row in rows
                if row["cell_id"] == cell and row["checkpoint_label"] == label
            }
            for label in CHECKPOINT_STATES
        }
        if any(len(steps) != 1 for steps in steps_by_label.values()):
            raise ValueError(f"Fixed-N TPP checkpoint steps vary across seeds in {cell}: {steps_by_label}")
        cell_rows = {row["checkpoint_label"]: row for row in rows if row["cell_id"] == cell}
        total_steps = int(cells[cell]["total_steps"])
        boundary_step = int(cells[cell]["boundary_step"])
        time_grid[cell] = {
            label: {
                "checkpoint_step": int(cell_rows[label]["checkpoint_step"]),
                "normalized_time": int(cell_rows[label]["checkpoint_step"]) / total_steps,
                "steps_from_lr_decay_onset": int(cell_rows[label]["checkpoint_step"]) - boundary_step,
            }
            for label in CHECKPOINT_STATES
        }
    common_grid = ("fraction_0p55", "fraction_0p70", "decay_onset", "fraction_0p90")
    for label in common_grid:
        normalized_times = {round(time_grid[cell][label]["normalized_time"], 12) for cell in FIXED_N_CELLS}
        if len(normalized_times) != 1:
            raise ValueError(f"Primary common normalized-time grid drifted at {label}: {normalized_times}")
    exact_by_stage = {
        "0": sum(_matches_parent_precision(row) for row in preflight_rows),
        **{
            str(stage): sum(_matches_parent_precision(row) for row in rows if int(row["launch_stage"]) == stage)
            for stage in CELL_STAGE.values()
        },
        "all": sum(_matches_parent_precision(row) for row in rows),
    }
    return {
        "row_count": len(rows),
        "cell_count": len(FIXED_N_CELLS),
        "rows_per_cell": 64,
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_states": list(CHECKPOINT_STATES),
        "standard_block_counts": STANDARD_BLOCK_COUNTS,
        "standard_update_draw_counts": STANDARD_UPDATE_DRAWS,
        "paired_sequence_set_count": len(sequence_sets),
        "preflight_row_count": len(preflight_rows),
        "parent_precision_exact_row_count": sum(_matches_parent_precision(row) for row in rows),
        "parent_precision_exact_by_stage": exact_by_stage,
        "parent_precision_reduced_row_count": len(reduced_rows),
        "parent_precision_reduced_cells": sorted({row["cell_id"] for row in reduced_rows}),
        "parent_precision_reduced_states": sorted({row["checkpoint_label"] for row in reduced_rows}),
        "time_grid": time_grid,
        "common_normalized_time_grid": list(common_grid),
        "cell_tpp": tpp,
    }


def _report(validation: dict[str, Any]) -> str:
    lines = [
        "# Fixed-N TPP gradient-onset probe",
        "",
        "This endpoint-blind extension restores sealed checkpoints from four fixed-model-size StarCoder WSD80 cells. "
        "It asks whether the loss of StarCoder-Nemotron gradient alignment moves as D and TPP increase.",
        "",
        f"- {validation['row_count']} checkpoint groups: 4 cells x 8 seeds x 8 temporal states.",
        "- Full-support tied-35% trajectories only, avoiding finite-support repetition and policy changes.",
        "- All cells use the same frozen reference sequence sets, 16 source blocks, and 8 optimizer draws.",
        f"- {validation['parent_precision_exact_row_count']} rows reproduce their sealed parent precision exactly. "
        f"The other {validation['parent_precision_reduced_row_count']} r3 rows deliberately recompute the first "
        "16 of 64 parent blocks and first 8 of 32 optimizer draws so precision is comparable across rungs.",
        "- The primary comparison is the paired decline at 0.90T. Full-grid normalized-time change points are not "
        "compared because the absolute +/-64/256-step probes contract toward 0.80T as runs lengthen.",
        "- Raw projected-trunk gradient cosine is primary; optimizer-update and target-source quantities are secondary.",
        "- One adapter canary and one 16-row cross-cell preflight must pass before the four independent 64-row cell "
        "parents may launch concurrently.",
        "- The runner does not read endpoint metrics.",
        "",
        "| Cell | D | Total TPP | Non-embedding TPP |",
        "|---|---:|---:|---:|",
    ]
    for cell in FIXED_N_CELLS:
        item = validation["cell_tpp"][cell]
        lines.append(
            f"| {cell} | {item['materialized_tokens']:,} | {item['total_parameter_tpp']:.3f} | "
            f"{item['non_embedding_parameter_tpp']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _parent_result_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return repair._parent_result_inventory({"full": rows})


def freeze() -> dict[str, Any]:
    """Materialize the hash-pinned release after the required CC review."""
    if not CC_REVIEW_PATH.exists():
        raise FileNotFoundError(f"CC review is missing: {CC_REVIEW_PATH}")
    review_verdict = _review_verdict(CC_REVIEW_PATH.read_text())
    if review_verdict != "PASS":
        raise ValueError(f"CC review did not pass: {review_verdict}")
    rows = _rows()
    validation = _validate_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(ANALYSIS_CONTRACT_PATH, ANALYSIS_CONTRACT)
    _write_csv(FULL_MANIFEST_PATH, rows)
    checkpoint_inventory = repair._checkpoint_inventory({"full": rows})
    parent_result_inventory = _parent_result_inventory(rows)
    _write_csv(CHECKPOINT_PROVENANCE_PATH, checkpoint_inventory)
    _write_csv(PARENT_RESULT_PROVENANCE_PATH, parent_result_inventory)
    report_path = OUTPUT_DIR / "report.md"
    _write_create_only(report_path, _report(validation).encode())

    parent_release = _parent_release()
    historical_release = json.loads(HISTORICAL_RELEASE_PATH.read_text())
    implementation_paths = (
        Path(__file__).resolve(),
        RUNTIME_PATH,
        Path(historical.__file__).resolve(),
        historical.RUNTIME_PATH,
        Path(repair.__file__).resolve(),
        repair.RUNTIME_PATH,
    )
    implementation_files = {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in implementation_paths}
    parent_implementation_files = parent_release["implementation_files"]
    recovery_sha256 = canonical_sha256(
        {
            "implementation_files": implementation_files,
            "parent_implementation_files": parent_implementation_files,
        }
    )
    required_preauthorization_paths = sorted(
        {
            *historical_release["submission_contract"]["required_stage2_plus_workspace_paths"],
            *(str(path.relative_to(REPO_ROOT)) for path in implementation_paths),
            str(ANALYSIS_CONTRACT_PATH.relative_to(REPO_ROOT)),
            str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
            str(CHECKPOINT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
            str(PARENT_RESULT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
            str(report_path.relative_to(REPO_ROOT)),
            str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
            str(RELEASE_PATH.relative_to(REPO_ROOT)),
        }
    )
    required_preflight_paths = sorted(
        {*required_preauthorization_paths, str(FULL_LAUNCH_AUTHORIZATION_PATH.relative_to(REPO_ROOT))}
    )
    required_full_launch_paths = sorted({*required_preflight_paths, str(PREFLIGHT_AUDIT_PATH.relative_to(REPO_ROOT))})
    execution_acceptance = dict(historical_release["execution_acceptance"])
    execution_acceptance.pop("stage_promotion_requires_exact_prior_stage_audit", None)
    execution_acceptance.update(
        {
            "remote_adapter_canary_required": True,
            "preflight_wave_requires_exact_audit": True,
        }
    )
    release = {
        "release_version": RELEASE_VERSION,
        "release_sha256": "",
        "scientific_status": SCIENTIFIC_STATUS,
        "endpoint_metrics_read_by_runner": False,
        "parent_release_path": str(PARENT_RELEASE_PATH.relative_to(REPO_ROOT)),
        "parent_release_file_sha256": file_sha256(PARENT_RELEASE_PATH),
        "parent_release_sha256": parent_release["release_sha256"],
        "historical_release_path": str(HISTORICAL_RELEASE_PATH.relative_to(REPO_ROOT)),
        "historical_release_file_sha256": file_sha256(HISTORICAL_RELEASE_PATH),
        "historical_release_sha256": historical_release["release_sha256"],
        "parent_result_artifact_version": PARENT_ARTIFACT_VERSION,
        "result_root": RESULT_ROOT,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": MARIN_PREFIX,
        "analysis_contract": {
            "path": str(ANALYSIS_CONTRACT_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(ANALYSIS_CONTRACT_PATH),
        },
        "manifests": {
            "full": {
                "path": str(FULL_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(FULL_MANIFEST_PATH),
                "row_count": len(rows),
            },
            "checkpoint_provenance": {
                "path": str(CHECKPOINT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(CHECKPOINT_PROVENANCE_PATH),
                "row_count": len(checkpoint_inventory),
                "checkpoint_count": len({row["checkpoint_uri"] for row in checkpoint_inventory}),
            },
            "parent_result_provenance": {
                "path": str(PARENT_RESULT_PROVENANCE_PATH.relative_to(REPO_ROOT)),
                "sha256": file_sha256(PARENT_RESULT_PROVENANCE_PATH),
                "row_count": len(parent_result_inventory),
                "object_count": len({row["object_uri"] for row in parent_result_inventory}),
            },
        },
        "implementation_files": implementation_files,
        "parent_implementation_files": parent_implementation_files,
        "historical_runtime": {
            **historical_release["historical_runtime"],
            "recovery_implementation_manifest_sha256": recovery_sha256,
        },
        "execution_acceptance": execution_acceptance,
        "design_validation": validation,
        "external_review": {
            "path": str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(CC_REVIEW_PATH),
            "account": "plambdafour@proton.me",
            "model": "claude-opus-5[1m]",
            "verdict": review_verdict,
        },
        "authorization_contract": {
            "confirmation": FULL_LAUNCH_CONFIRMATION,
            "remote_adapter_canary_required": True,
            "remote_adapter_canary_path": REMOTE_ADAPTER_CANARY_PATH,
        },
        "reporting_contract": {
            "final_whole_panel_audit_required": True,
            "final_whole_panel_audit_path": str(FINAL_AUDIT_PATH.relative_to(REPO_ROOT)),
            "reason": (
                "Only the whole-panel audit proves cross-cell runtime uniformity and frozen first-batch identity for "
                "all eight seeds before scientific analysis."
            ),
        },
        "submission_contract": {
            "command_prefix": "UV_FROZEN=1 uv run python -m marin.run.iris_run",
            "required_environment": {"UV_FROZEN": "1"},
            "required_preauthorization_workspace_paths": required_preauthorization_paths,
            "required_preflight_workspace_paths": required_preflight_paths,
            "required_full_launch_workspace_paths": required_full_launch_paths,
            "reason": (
                "The historical uv.lock is part of the numerical source manifest. UV_FROZEN prevents the nested "
                "Iris invocation from rewriting it before workspace bundling."
            ),
        },
        "full_launch_stages": {
            "0": {"role": "cross_cell_preflight", "row_count": 16, "max_concurrent": 16},
            **{
                str(stage): {"role": "full_cell", "cell_id": cell, "row_count": 64, "max_concurrent": 64}
                for cell, stage in CELL_STAGE.items()
            },
        },
    }
    release["release_sha256"] = canonical_sha256({**release, "release_sha256": ""})
    _write_json(RELEASE_PATH, release)
    return release


def main() -> None:
    print(json.dumps(freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
