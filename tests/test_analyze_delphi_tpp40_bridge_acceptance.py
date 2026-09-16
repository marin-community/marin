# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from marin.evaluation.olmo_base_eval.aggregate import table9_macro
from marin.evaluation.olmo_base_eval.components import scored_tasks, table9_components

from experiments.domain_phase_mix import analyze_delphi_tpp40_bridge_acceptance as acceptance
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_same_region_east5_eval as same_region_eval
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval

IDEMPOTENCE_SHA256 = "e" * 64
INVENTORY_SHA256 = "d" * 64


def _path_manifest() -> dict:
    sides = {}
    for side in bridge_eval.BRIDGE_SIDES:
        checkpoint_source = "audited_europe_mirror" if side == "east5" else "native_training_output"
        uncheatable_cells = []
        table9_cells = []
        training_output_paths = []
        for run_order in bridge_eval.BRIDGE_RUN_ORDERS:
            run_name = f"fit_{run_order:03d}_test"
            source_run_name = f"run_{run_order:05d}"
            training_output_path = f"gs://{side}/{run_name}"
            training_output_paths.append(training_output_path)
            for checkpoint_step in bridge_eval.CHECKPOINT_STEPS:
                output_path = f"{training_output_path}/uncheatable/{checkpoint_step}"
                uncheatable_cells.append(
                    {
                        "side": side,
                        "run_order": run_order,
                        "run_name": run_name,
                        "source_run_name": source_run_name,
                        "data_seed": run_order + 1,
                        "trainer_seed": run_order + 2,
                        "checkpoint_step": checkpoint_step,
                        "checkpoint_path": f"{training_output_path}/checkpoints/step-{checkpoint_step}",
                        "canonical_checkpoint_path": f"{training_output_path}/checkpoints/step-{checkpoint_step}",
                        "reference_checkpoint_source": checkpoint_source,
                        "evaluator_region": same_region_eval.EUROPE_REGION,
                        "evaluator_zone": same_region_eval.EUROPE_ZONE,
                        "validation_payload_sha256": acceptance.EXPECTED_UNCHEATABLE_VALIDATION_PAYLOAD_SHA256,
                        "output_path": output_path,
                        "result_path": f"{output_path}/{bridge_eval.RESULT_FILE}",
                    }
                )
            output_path = f"{training_output_path}/table9"
            table9_cells.append(
                {
                    "side": side,
                    "run_order": run_order,
                    "run_name": run_name,
                    "source_run_name": source_run_name,
                    "panel_source": "frozen-panel",
                    "checkpoint_step": bridge_eval.CHECKPOINT_STEPS[-1],
                    "checkpoint_path": f"{training_output_path}/hf/step-{bridge_eval.CHECKPOINT_STEPS[-1]}",
                    "canonical_checkpoint_path": f"{training_output_path}/hf/step-{bridge_eval.CHECKPOINT_STEPS[-1]}",
                    "reference_checkpoint_source": checkpoint_source,
                    "evaluator_region": same_region_eval.EUROPE_REGION,
                    "evaluator_zone": same_region_eval.EUROPE_ZONE,
                    "request_set_dir": f"gs://{side}/table9",
                    "request_set_payload_sha256": acceptance.EXPECTED_TABLE9_REQUEST_SET_PAYLOAD_SHA256,
                    "output_path": output_path,
                    "result_path": f"{output_path}/results.json",
                }
            )
        sides[side] = {
            "training_output_paths": training_output_paths,
            "mirror_trees": (
                [
                    {
                        "relative_path": relative_path,
                        "source_path": f"gs://east5-source/{relative_path}",
                        "destination_path": f"gs://europe-mirror/{relative_path}",
                    }
                    for relative_path in (
                        "checkpoints/step-21855",
                        "checkpoints/step-27335",
                        "hf/step-27335",
                    )
                ]
                if side == "east5"
                else []
            ),
            "uncheatable_cells": uncheatable_cells,
            "table9_cells": table9_cells,
        }
    return {
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "east5_reference_mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "run_orders": list(bridge_eval.BRIDGE_RUN_ORDERS),
        "checkpoint_steps": list(bridge_eval.CHECKPOINT_STEPS),
        "sides": sides,
    }


def _uncheatable_payload(cell: dict, *, macro: float) -> dict:
    offsets = np.linspace(-0.003, 0.003, len(bridge_eval.EXPECTED_UNCHEATABLE_NAMES), dtype=np.float32)
    component_bpb = {
        name: float(np.float32(macro) + offset)
        for name, offset in zip(bridge_eval.EXPECTED_UNCHEATABLE_NAMES, offsets, strict=True)
    }
    reported_macro = float(np.mean(np.asarray(tuple(component_bpb.values()), dtype=np.float32)))
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "evaluator_tpu_type": bridge_eval.EVALUATOR_TPU_TYPE,
        "eval_batch_size": bridge_eval.EVAL_BATCH_SIZE,
        **{key: cell[key] for key in ("side", "run_order", "run_name", "source_run_name")},
        "data_seed": cell["data_seed"],
        "trainer_seed": cell["trainer_seed"],
        "checkpoint_step": cell["checkpoint_step"],
        "checkpoint_path": cell["checkpoint_path"],
        "checkpoint_metadata": {"step": cell["checkpoint_step"]},
        "checkpoint_metadata_sha256": "a" * 64,
        "model_config_sha256": "b" * 64,
        "total_trainable_params": 75_000_000_000,
        "validation_payload_sha256": cell["validation_payload_sha256"],
        "component_bpb": component_bpb,
        "macro_bpb": reported_macro,
    }


def _table9_payload(cell: dict, *, macro: float) -> dict:
    components = {name: macro + index * 0.0001 for index, name in enumerate(table9_components())}
    return {
        "name": f"t9_{cell['run_name']}",
        "checkpoint_path": cell["checkpoint_path"],
        "request_set_dir": cell["request_set_dir"],
        "request_set_version": "frozen-v2",
        "olmo_eval_git_sha": "c" * 40,
        "num_instances": {name: 1 for name in scored_tasks()},
        "task_bpb": {name: macro + index * 0.00001 for index, name in enumerate(scored_tasks())},
        "table9_components": components,
        "table9_macro_bpb": table9_macro(components),
        "provenance": {
            "evaluator": "marin-native-table9-bpb",
            "panel": "delphi_tpp40_augmented_fit_swarm",
            "scale": "fixed_n_total_tpp40",
            "source_run_name": cell["source_run_name"],
            "swarm_run_name": cell["run_name"],
            "panel_source": cell["panel_source"],
        },
    }


def _payloads(path_manifest: dict, *, europe_delta: float = 0.001) -> dict[str, dict]:
    payloads = {}
    for side_name, side in path_manifest["sides"].items():
        macro = 1.0 + (europe_delta if side_name == "europe" else 0.0)
        for cell in side["uncheatable_cells"]:
            key = f"uncheatable:{side_name}:{cell['run_order']}:{cell['checkpoint_step']}"
            payloads[key] = _uncheatable_payload(cell, macro=macro)
        for cell in side["table9_cells"]:
            key = f"table9:{side_name}:{cell['run_order']}:{cell['checkpoint_step']}"
            payloads[key] = _table9_payload(cell, macro=macro)
    return payloads


def _set_uncheatable_delta(
    payloads: dict[str, dict],
    path_manifest: dict,
    *,
    run_order: int,
    checkpoint_step: int,
    delta: float,
) -> None:
    cell = next(
        cell
        for cell in path_manifest["sides"]["europe"]["uncheatable_cells"]
        if cell["run_order"] == run_order and cell["checkpoint_step"] == checkpoint_step
    )
    key = f"uncheatable:europe:{run_order}:{checkpoint_step}"
    payloads[key] = _uncheatable_payload(cell, macro=1.0 + delta)


def _current_inventory() -> dict:
    return {
        "sides": {
            side: {
                "inventory_sha256": INVENTORY_SHA256,
                "unit_counts": {
                    "training": len(bridge_eval.BRIDGE_RUN_ORDERS),
                    "mirror": 3 if side == "east5" else 0,
                    "uncheatable": len(bridge_eval.BRIDGE_RUN_ORDERS) * len(bridge_eval.CHECKPOINT_STEPS),
                    "table9": len(bridge_eval.BRIDGE_RUN_ORDERS),
                },
            }
            for side in bridge_eval.BRIDGE_SIDES
        }
    }


def _idempotence_payload() -> dict:
    sides = {
        "east5": {
            "reference_eval_command_sha256": acceptance.EXPECTED_COMMAND_SHA256["east5"]["reference_eval"],
            "result_inventory_sha256_before": INVENTORY_SHA256,
            "result_inventory_sha256_after": INVENTORY_SHA256,
            "mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
            "reference_eval_rerun": {
                "state": "succeeded",
                "exit_code": 0,
                "child_job_count": 0,
                "finished_at_ms": 3_000,
            },
        },
        "europe": {
            "training_command_sha256": acceptance.EXPECTED_COMMAND_SHA256["europe"]["training"],
            "uncheatable_command_sha256": acceptance.EXPECTED_COMMAND_SHA256["europe"]["uncheatable"],
            "result_inventory_sha256_before": INVENTORY_SHA256,
            "result_inventory_sha256_after": INVENTORY_SHA256,
            "training_rerun": {
                "state": "succeeded",
                "exit_code": 0,
                "child_job_count": 0,
                "finished_at_ms": 3_000,
            },
            "uncheatable_rerun": {
                "state": "succeeded",
                "exit_code": 0,
                "child_job_count": 0,
                "finished_at_ms": 3_000,
            },
        },
    }
    return {
        "schema_version": 3,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "path_manifest_sha256": acceptance.EXPECTED_PATH_MANIFEST_SHA256,
        "evaluation_audit_sha256": acceptance.EXPECTED_EVALUATION_AUDIT_SHA256,
        "east5_reference_mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "after_inventory_started_at_ms": 4_000,
        "after_inventory_captured_at_ms": 5_000,
        "sides": sides,
    }


def _analyze(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path_manifest: dict,
    payloads: dict[str, dict],
    idempotence_payload: dict | None = None,
) -> dict:
    monkeypatch.setattr(acceptance, "EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256", IDEMPOTENCE_SHA256)
    return acceptance.analyze_payloads(
        contract=bridge_eval._load_acceptance_contract(),
        path_manifest=path_manifest,
        payloads=payloads,
        idempotence_payload=idempotence_payload,
        idempotence_sha256=IDEMPOTENCE_SHA256 if idempotence_payload is not None else None,
        current_inventory=_current_inventory() if idempotence_payload is not None else None,
        evaluation_data_identity={"passed": True},
        training_data_identity={"passed": True},
        observed_contract_sha256=bridge_eval.EXPECTED_CONTRACT_SHA256,
        observed_path_manifest_sha256=acceptance.EXPECTED_PATH_MANIFEST_SHA256,
    )


def test_complete_bridge_within_threshold_authorizes_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    report = _analyze(
        monkeypatch,
        path_manifest=path_manifest,
        payloads=_payloads(path_manifest),
        idempotence_payload=_idempotence_payload(),
    )

    assert report["numerical_acceptance_passed"] is True
    assert report["idempotence"]["passed"] is True
    assert report["production_launch_authorized"] is True
    assert report["blocking_errors"] == []


def test_uncheatable_payload_round_trip_accepts_sorted_json_keys(tmp_path: Path) -> None:
    cell = _path_manifest()["sides"]["east5"]["uncheatable_cells"][0]
    path = tmp_path / "bridge_result.json"
    bridge_eval._write_json(str(path), _uncheatable_payload(cell, macro=1.0))

    validated = acceptance._validate_uncheatable_payload(acceptance._read_json(path), cell)

    assert tuple(validated["component_bpb"]) == bridge_eval.EXPECTED_UNCHEATABLE_NAMES


def test_allow_incomplete_does_not_mask_complete_numerical_failure() -> None:
    assert (
        acceptance._allow_incomplete_failure(
            loading_errors=[],
            numerical_acceptance_passed=False,
            idempotence_payload=None,
        )
        is False
    )


def test_allow_incomplete_accepts_only_missing_outputs_or_idempotence() -> None:
    assert acceptance._allow_incomplete_failure(
        loading_errors=["uncheatable:europe:2:27335 executor status is None, not success"],
        numerical_acceptance_passed=False,
        idempotence_payload=None,
    )
    assert acceptance._allow_incomplete_failure(
        loading_errors=[],
        numerical_acceptance_passed=True,
        idempotence_payload=None,
    )


def test_mean_threshold_fails_even_when_pair_is_below_any_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_manifest = _path_manifest()
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=_payloads(path_manifest, europe_delta=0.0021))

    threshold = report["uncheatable"]["phase_0"]["threshold"]
    assert threshold["mean_absolute_paired_delta"] > 0.002
    assert threshold["maximum_absolute_paired_delta"] < 0.005
    assert threshold["passed"] is False


def test_one_pair_mean_threshold_is_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_manifest = _path_manifest()
    run_order = bridge_eval.BRIDGE_RUN_ORDERS[0]
    payloads = _payloads(path_manifest, europe_delta=0.0)
    _set_uncheatable_delta(
        payloads,
        path_manifest,
        run_order=run_order,
        checkpoint_step=bridge_eval.CHECKPOINT_STEPS[0],
        delta=0.0019,
    )
    below = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)
    assert below["uncheatable"]["phase_0"]["threshold"]["passed"] is True

    _set_uncheatable_delta(
        payloads,
        path_manifest,
        run_order=run_order,
        checkpoint_step=bridge_eval.CHECKPOINT_STEPS[0],
        delta=0.0021,
    )
    above = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)
    threshold = above["uncheatable"]["phase_0"]["threshold"]
    assert threshold["mean_absolute_paired_delta"] > 0.002
    assert threshold["maximum_absolute_paired_delta"] < 0.005
    assert threshold["passed"] is False


def test_phase_0_and_endpoint_thresholds_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    payloads = _payloads(path_manifest, europe_delta=0.0)
    _set_uncheatable_delta(
        payloads,
        path_manifest,
        run_order=bridge_eval.BRIDGE_RUN_ORDERS[0],
        checkpoint_step=bridge_eval.CHECKPOINT_STEPS[0],
        delta=0.006,
    )
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)

    assert report["uncheatable"]["phase_0"]["threshold"]["passed"] is False
    assert report["uncheatable"]["endpoint"]["threshold"]["passed"] is True


def test_missing_endpoint_pair_fails_pair_count(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    payloads = _payloads(path_manifest, europe_delta=0.0)
    payloads.pop(f"uncheatable:europe:{bridge_eval.BRIDGE_RUN_ORDERS[0]}:{bridge_eval.CHECKPOINT_STEPS[-1]}")
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)

    threshold = report["uncheatable"]["endpoint"]["threshold"]
    assert threshold["observed_pair_count"] == len(bridge_eval.BRIDGE_RUN_ORDERS) - 1
    assert threshold["passed"] is False
    assert report["numerical_acceptance_passed"] is False


@pytest.mark.parametrize("field", ["model_config_sha256", "data_seed", "trainer_seed"])
def test_cross_side_uncheatable_identity_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    path_manifest = _path_manifest()
    payloads = _payloads(path_manifest, europe_delta=0.0)
    key = f"uncheatable:europe:{bridge_eval.BRIDGE_RUN_ORDERS[0]}:{bridge_eval.CHECKPOINT_STEPS[0]}"
    payloads[key][field] = "f" * 64 if field == "model_config_sha256" else -1
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)

    assert report["numerical_acceptance_passed"] is False
    assert any("identity mismatch" in error for error in report["blocking_errors"])


def test_table9_request_payload_identity_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    path_manifest["sides"]["europe"]["table9_cells"][0]["request_set_payload_sha256"] = "f" * 64
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=_payloads(path_manifest, europe_delta=0.0))

    assert report["numerical_acceptance_passed"] is False
    assert any("unfrozen request-set payload" in error for error in report["blocking_errors"])


def test_missing_table9_result_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    payloads = _payloads(path_manifest, europe_delta=0.0)
    payloads.pop(f"table9:europe:{bridge_eval.BRIDGE_RUN_ORDERS[0]}:{bridge_eval.CHECKPOINT_STEPS[-1]}")
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)

    assert report["table9"]["threshold"]["observed_pair_count"] == len(bridge_eval.BRIDGE_RUN_ORDERS) - 1
    assert report["production_launch_authorized"] is False


def test_table9_macro_one_ulp_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    payloads = _payloads(path_manifest, europe_delta=0.0)
    key = f"table9:europe:{bridge_eval.BRIDGE_RUN_ORDERS[0]}:{bridge_eval.CHECKPOINT_STEPS[-1]}"
    macro = payloads[key]["table9_macro_bpb"]
    payloads[key]["table9_macro_bpb"] = float(np.nextafter(macro, np.inf))
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=payloads)

    assert report["numerical_acceptance_passed"] is False
    assert any("macro mismatch" in error for error in report["blocking_errors"])


def test_idempotence_rerun_with_child_job_blocks_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    idempotence = _idempotence_payload()
    idempotence["sides"]["europe"]["uncheatable_rerun"]["child_job_count"] = 1
    report = _analyze(
        monkeypatch,
        path_manifest=path_manifest,
        payloads=_payloads(path_manifest),
        idempotence_payload=idempotence,
    )

    assert report["numerical_acceptance_passed"] is True
    assert report["idempotence"]["passed"] is False
    assert report["production_launch_authorized"] is False


def test_east5_reference_idempotence_rerun_with_child_job_blocks_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_manifest = _path_manifest()
    idempotence = _idempotence_payload()
    idempotence["sides"]["east5"]["reference_eval_rerun"]["child_job_count"] = 1
    report = _analyze(
        monkeypatch,
        path_manifest=path_manifest,
        payloads=_payloads(path_manifest),
        idempotence_payload=idempotence,
    )

    assert report["idempotence"]["passed"] is False
    assert report["production_launch_authorized"] is False


def test_idempotence_inventory_captured_before_rerun_finished_blocks_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_manifest = _path_manifest()
    idempotence = _idempotence_payload()
    idempotence["sides"]["europe"]["training_rerun"]["finished_at_ms"] = 4_001
    report = _analyze(
        monkeypatch,
        path_manifest=path_manifest,
        payloads=_payloads(path_manifest),
        idempotence_payload=idempotence,
    )

    assert report["idempotence"]["passed"] is False
    assert report["production_launch_authorized"] is False
    assert any("finished after" in error for error in report["blocking_errors"])


def test_missing_idempotence_blocks_only_production_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    path_manifest = _path_manifest()
    report = _analyze(monkeypatch, path_manifest=path_manifest, payloads=_payloads(path_manifest))

    assert report["numerical_acceptance_passed"] is True
    assert report["idempotence"]["passed"] is False
    assert report["production_launch_authorized"] is False


def test_frozen_path_manifest_rejects_any_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(_path_manifest(), sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(acceptance, "EXPECTED_PATH_MANIFEST_SHA256", digest)
    acceptance._load_frozen_path_manifest(path)

    path.write_text(path.read_text().replace("fit_002_test", "fit_002_changed", 1))
    with pytest.raises(ValueError, match="path manifest changed"):
        acceptance._load_frozen_path_manifest(path)
