# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import random
import threading
import time
from argparse import Namespace
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tensorstore as ts
from fray.iris_backend import convert_constraints
from fray.types import ResourceConfig
from iris.cluster.constraints import preemptible_constraint, region_constraint, zone_constraint
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from levanter.distributed import DistributedConfig
from levanter.main import train_lm as train_lm_module
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import compare_starcoder_wsd80_gradient_conflict_canary as compare
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict as canary
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_resume_canary as resume
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_stress as stress
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_resume_canary_20260813 as resume_analyzer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_conflict_runtime_gate_20260811 as runtime_gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_starcoder_wsd80_gradient_conflict_outputs_20260811 as output_inventory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_conflict_training_release_20260813 as training_release,
)


def test_canary_manifest_matches_reviewed_design_contract():
    design = canary.load_canary_design()

    assert tuple(row.trajectory_id for row in design.trajectories) == canary.EXPECTED_TRAJECTORY_IDS
    all_steps = sorted((*design.scientific_checkpoint_steps, *design.diagnostic_checkpoint_steps))
    assert canary._checkpoint_keep(design, fork=False) == [
        {"every": step, "until": None if step == max(all_steps) else step} for step in all_steps
    ]
    assert canary._checkpoint_keep(design, fork=True) == [{"every": design.fork_reference_step, "until": None}]
    assert canary._checkpoint_keep(design, fork=True, decay_fork=True) == [
        {"every": canary.DECAY_FORK_REFERENCE_STEP, "until": None}
    ]
    assert design.fork_trainer_num_train_steps == design.fork_reference_step + 1
    assert canary.DECAY_FORK_TRAINER_NUM_STEPS == canary.DECAY_FORK_REFERENCE_STEP + 1
    assert canary.DECAY_FORK_REFERENCE_STEP - canary.DECAY_FORK_SOURCE_STEP == 128


def test_exact_checkpoint_comparison_reports_value_and_key_mismatches(tmp_path):
    parent = tmp_path / "parent"
    identical = tmp_path / "identical"
    changed = tmp_path / "changed"

    async def write_checkpoint(path, values):
        store = await ts.KvStore.open(compare._checkpoint_kvstore_spec(str(path)))
        for key, value in values.items():
            await store.write(key.encode(), value)

    async def run_test():
        await write_checkpoint(parent, {"state/a": b"one", "state/b": b"two"})
        await write_checkpoint(identical, {"state/a": b"one", "state/b": b"two"})
        await write_checkpoint(changed, {"state/a": b"ONE", "state/c": b"three"})

        exact = await compare.compare_checkpoint_pair("exact", str(parent), str(identical))
        mismatch = await compare.compare_checkpoint_pair("mismatch", str(parent), str(changed))

        assert exact["exact"] is True
        assert exact["value_mismatch_count"] == 0
        assert mismatch["exact"] is False
        assert mismatch["value_mismatch_count"] == 1
        assert mismatch["value_mismatches"][0]["key"] == "state/a"
        assert mismatch["missing_from_parent"] == ["state/c"]
        assert mismatch["missing_from_fork"] == ["state/b"]

    asyncio.run(run_test())


def test_exact_checkpoint_comparison_expectations_fail_closed():
    comparison = {
        "label": "seed0",
        "parent_key_count": 175,
        "fork_key_count": 175,
        "compared_bytes": 2_337_162_393,
        "parent_root": "gs://bucket/parent/step-22544",
        "fork_root": "gs://bucket/fork/step-22672",
    }
    args = Namespace(
        expected_key_count=175,
        expected_compared_bytes=2_337_162_393,
        expected_parent_step=22_544,
        expected_fork_step=22_672,
    )

    compare._assert_expected_comparison(comparison, args)
    for field, value in (
        ("parent_key_count", 174),
        ("fork_key_count", 176),
        ("compared_bytes", 2_337_162_392),
        ("parent_root", "gs://bucket/parent/step-22543"),
        ("fork_root", "gs://bucket/fork/step-22671"),
    ):
        mismatched = {**comparison, field: value}
        with pytest.raises(ValueError):
            compare._assert_expected_comparison(mismatched, args)


def test_full_training_manifest_matches_review_v9_contract(monkeypatch):
    cells, trajectories, keep_by_id = full.load_design()

    assert set(cells) == {
        "r0_shared_h0640_s03820",
        "r1_increase_d_h0640_s07320",
        "r2_increase_d_h0640_s14960",
        "r3_increase_d_h0640_s28260",
        full.H5_CELL_ID,
    }
    assert len(trajectories) == full.EXPECTED_TRAJECTORY_COUNT
    assert len(keep_by_id) == full.EXPECTED_TRAJECTORY_COUNT
    assert Counter(row.arm for row in trajectories) == {"p1": 112, "p2": 16, "p3": 16, "p4": 16, "b": 96}
    assert sum(len(keep_by_id[row.trajectory_id]) for row in trajectories) == full.EXPECTED_CHECKPOINT_COUNT
    assert all(row.train_holdout_sequences_per_component == 4096 for row in trajectories)
    assert {row.train_holdout_seed for row in trajectories} == {full.EXPECTED_TRAIN_HOLDOUT_SEED}
    assert {row.train_holdout_partition for row in trajectories} == {full.EXPECTED_TRAIN_HOLDOUT_PARTITION}
    assert set(full.EXPECTED_NEMOTRON_LEDGERS) == set(full.EXPECTED_TRAINING_COMPONENT_NAMES) - {"dolma/starcoder"}
    wandb_run_ids = {full._wandb_run_id(row) for row in trajectories}
    assert len(wandb_run_ids) == len(trajectories)
    assert wandb_run_ids.isdisjoint(full.CANARY_WANDB_RUN_IDS)
    assert max(map(len, wandb_run_ids)) < 128

    colliding_trajectory = next(row for row in trajectories if row.trajectory_id in full.CANARY_WANDB_RUN_IDS)
    monkeypatch.setattr(full, "WANDB_RUN_ID_PREFIX", "")
    with pytest.raises(ValueError, match="require a nonempty prefix"):
        full._wandb_run_id(colliding_trajectory)
    monkeypatch.setattr(full, "WANDB_RUN_ID_PREFIX", "gcf")
    suffix_only_trajectory = replace(
        colliding_trajectory,
        trajectory_id=colliding_trajectory.trajectory_id.removeprefix("gcf_"),
    )
    with pytest.raises(ValueError, match="W&B identity collides"):
        full._wandb_run_id(suffix_only_trajectory)


def test_full_training_h5_schedule_and_nearest_tied_comparator_are_explicit():
    _, trajectories, keep_by_id = full.load_design()
    h5_rows = [row for row in trajectories if row.cell_id == full.H5_CELL_ID]
    p4_rows = [row for row in trajectories if row.arm == "p4"]
    p3_rows = [row for row in trajectories if row.arm == "p3"]

    assert len(h5_rows) == 96
    assert {row.total_steps for row in h5_rows} == {full.H5_TOTAL_STEPS}
    assert {row.optimizer_decay_step for row in h5_rows} == {full.H5_DECAY_STEP}
    assert any(row.boundary_step < row.optimizer_decay_step for row in h5_rows)
    assert any(row.boundary_step > row.optimizer_decay_step for row in h5_rows)
    assert len(p4_rows) == len(p3_rows) == 16
    assert {row.support_id for row in p4_rows} == {"m100a", "full"}
    assert {row.training_seed for row in p4_rows} == {row.training_seed for row in p3_rows}
    assert all(row.phase_0_starcoder == row.phase_1_starcoder for row in p4_rows)
    assert all(len(keep_by_id[row.trajectory_id]) > 0 for row in h5_rows)


def test_full_training_support_halves_are_adjacent_and_disjoint():
    _, trajectories, _ = full.load_design()
    finite_by_id = {
        support_id: [row for row in trajectories if row.support_id == support_id] for support_id in ("m100a", "m100b")
    }

    for rows in finite_by_id.values():
        assert len({row.support_pool_seed for row in rows}) == 1
        assert len({row.support_batches for row in rows}) == 1
    support_batches = finite_by_id["m100a"][0].support_batches
    assert support_batches is not None
    assert {row.support_start_batches for row in finite_by_id["m100a"]} == {0}
    assert {row.support_start_batches for row in finite_by_id["m100b"]} == {support_batches}
    assert {row.support_pool_seed for row in finite_by_id["m100a"]} == {
        row.support_pool_seed for row in finite_by_id["m100b"]
    }


def _candidate_full_release() -> dict:
    _, trajectories, _ = full.load_design()
    release = {
        "release_version": full.EXPECTED_RELEASE_VERSION,
        "release_sha256": "",
        "design_version": full.EXPECTED_DESIGN_VERSION,
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
        "design_manifest_sha256": full.EXPECTED_DESIGN_MANIFEST_SHA256,
        "training_fanout_allowed": True,
        "probe_fanout_allowed": False,
        "maximum_trajectory_count": full.EXPECTED_TRAJECTORY_COUNT,
        "maximum_concurrent_trajectories": full.MAX_RELEASE_CONCURRENCY,
        "allowed_trajectory_ids": [row.trajectory_id for row in trajectories],
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "trajectory_count": full.EXPECTED_TRAJECTORY_COUNT,
        "checkpoint_count": full.EXPECTED_CHECKPOINT_COUNT,
        "runtime_source_sha256": full._runtime_source_sha256(),
        "train_holdout_seed": full.EXPECTED_TRAIN_HOLDOUT_SEED,
        "train_holdout_partition": full.EXPECTED_TRAIN_HOLDOUT_PARTITION,
        "support_partition_audit_sha256": full.EXPECTED_SUPPORT_PARTITION_AUDIT_SHA256,
        "validated_evidence": {
            "starcoder_flat_field_token_count": full.EXPECTED_STARCODER_SOURCE_TOKENS,
            "starcoder_packed_sequence_count": full.EXPECTED_STARCODER_SOURCE_SEQUENCES,
            "starcoder_trailing_token_count": full.EXPECTED_STARCODER_TRAILING_TOKENS,
            "finite_support_required_tokens": full.EXPECTED_FINITE_SUPPORT_REQUIRED_TOKENS,
            "runtime_config_count": full.EXPECTED_TRAJECTORY_COUNT,
            "support_audit_source_sha256": full.EXPECTED_SUPPORT_AUDIT_SOURCE_SHA256,
            "long_gate": {"status": "pass", "endpoint_metrics_read": False},
            "decoupled_switch_canary": {
                "status": "pass",
                "data_switch_step": 768,
                "optimizer_decay_step": 1_024,
            },
            "recovery_gates": [
                {
                    "maximum_concurrent": full.REQUIRED_RECOVERY_CONCURRENCIES[0],
                    "generation": full.GEN19_RECOVERY_GENERATION,
                    "status": "pass",
                    "report_sha256": full.GEN19_RECOVERY_REPORT_SHA256,
                    "preregistration_sha256": full.GEN19_RECOVERY_PREREGISTRATION_SHA256,
                    "analyzer_revision_sha256": full.GEN19_RECOVERY_ANALYZER_REVISION_SHA256,
                    "independent_review_verdict": full.GEN19_RECOVERY_REVIEW_VERDICT,
                    "independent_review_session_id": full.GEN19_RECOVERY_REVIEW_SESSION_ID,
                    "endpoint_metrics_read": False,
                },
                {
                    "maximum_concurrent": full.REQUIRED_RECOVERY_CONCURRENCIES[1],
                    "generation": full.GEN24_RECOVERY_GENERATION,
                    "status": "pass",
                    "report_sha256": full.GEN24_RECOVERY_REPORT_SHA256,
                    "preregistration_sha256": full.GEN24_RECOVERY_PREREGISTRATION_SHA256,
                    "prior_gate_report_sha256": full.GEN19_RECOVERY_REPORT_SHA256,
                    "independent_review_verdict": full.GEN24_RECOVERY_REVIEW_VERDICT,
                    "independent_review_session_id": full.GEN24_RECOVERY_REVIEW_SESSION_ID,
                    "endpoint_metrics_read": False,
                },
            ],
            "orchestration_scope": {
                "c64_gate_scope": "Levanter run-local checkpoint recovery under 64-way child preemption",
                "c64_gate_exercises_step_runner_fanout": False,
                "production_fanout": "StepRunner dispatches one independent Fray/Iris child per trajectory",
                "production_fanout_live_gate": False,
                "production_fanout_source_hash_pinned": True,
                "parent_failure_policy": "fail closed; resubmit the exact command against owned resumable roots",
                "child_application_failure_retries": 0,
                "child_preemption_retries": 100,
                "child_wall_timeout_seconds": None,
            },
            "output_inventory": {
                "expected_root_count": full.EXPECTED_TRAJECTORY_COUNT,
                "empty_root_count": full.EXPECTED_TRAJECTORY_COUNT,
                "bookkeeping_root_count": 0,
                "resumable_root_count": 0,
                "completed_root_count": 0,
                "partial_root_count": 0,
                "unexpected_root_count": 0,
            },
            "long_gate_report_sha256": full.LONG_GATE_REPORT_SHA256,
            "decoupled_switch_canary_report_sha256": full.DECOUPLED_SWITCH_REPORT_SHA256,
            "checkpoint_recovery_report_sha256": full.GEN19_RECOVERY_REPORT_SHA256,
            "operational_threshold_report_sha256": full.GEN24_RECOVERY_REPORT_SHA256,
            "output_inventory_report_sha256": full.OUTPUT_INVENTORY_REPORT_SHA256,
        },
        "independent_review": {"verdict": "PASS_FULL_TRAINING"},
    }
    release["release_sha256"] = full._canonical_sha256({**release, "release_sha256": ""})
    return release


def test_full_training_release_authorizes_exact_frozen_panel():
    release = full._validate_training_release(_candidate_full_release())
    _, trajectories, _ = full.load_design(selected_runs=frozenset(release["allowed_trajectory_ids"]))

    assert release["training_fanout_allowed"] is True
    assert release["probe_fanout_allowed"] is False
    assert release["maximum_trajectory_count"] == full.EXPECTED_TRAJECTORY_COUNT
    assert release["maximum_concurrent_trajectories"] == full.MAX_RELEASE_CONCURRENCY
    assert release["independent_review"]["verdict"] == "PASS_FULL_TRAINING"
    full._validate_training_selection(trajectories, release)


def test_full_training_release_pin_is_excluded_from_launcher_source_hash():
    prefix = "EXPECTED_RELEASE_MANIFEST_SHA256: str | None = "
    release_hash = "a" * 64
    unpinned = f"before\n{prefix}None\nafter\n"
    pinned = f'before\n{prefix}"{release_hash}"\nafter\n'

    assert full._normalize_launcher_release_pin(unpinned) == full._normalize_launcher_release_pin(pinned)


def test_full_training_release_rejects_runtime_source_drift():
    release = _candidate_full_release()
    launcher = "experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict_full.py"
    release["runtime_source_sha256"][launcher] = "0" * 64
    release["release_sha256"] = full._canonical_sha256({**release, "release_sha256": ""})

    with pytest.raises(ValueError, match="runtime source hashes drifted"):
        full._validate_training_release(release)


def test_final_release_requires_exact_reviewed_candidate_bytes(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text('{"release": "candidate"}\n')
    expected_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()

    assert training_release._load_reviewed_candidate(candidate_path, expected_sha256) == {"release": "candidate"}
    candidate_path.write_text('{"release":"candidate"}\n')
    with pytest.raises(ValueError, match="Reviewed candidate file hash drifted"):
        training_release._load_reviewed_candidate(candidate_path, expected_sha256)


def test_full_training_release_uses_reviewed_c6_c64_recovery_path():
    assert full.REQUIRED_RECOVERY_CONCURRENCIES == (6, 64)


def test_full_training_release_rejects_broken_recovery_report_chain():
    release = _candidate_full_release()
    release["validated_evidence"]["recovery_gates"][1]["prior_gate_report_sha256"] = "z" * 64
    release["release_sha256"] = full._canonical_sha256({**release, "release_sha256": ""})

    with pytest.raises(ValueError, match="recovery-gate evidence drifted"):
        full._validate_training_release(release)


def test_full_training_starcoder_token_domain_is_exact():
    assert full._validate_starcoder_token_domain(216_567_300_822) == (105_745_752, 726)
    with pytest.raises(ValueError, match="packed sequence domain drifted"):
        full._validate_starcoder_token_domain(216_567_300_821)


@pytest.mark.data_integration
def test_full_training_starcoder_support_identity_is_exact():
    observed = full._validate_starcoder_support_identity()

    assert observed["support_ordered_sequence_sha256"] == full.EXPECTED_STARCODER_SUPPORT_DIGESTS
    assert observed["shared_sequence_count"] == 0
    assert observed["holdout_overlap_sequence_count"] == {"m100a": 0, "m100b": 0}


@pytest.mark.parametrize("invalid_limit", [None, 0, "1", True])
def test_full_training_release_rejects_invalid_trajectory_limit(invalid_limit):
    release = _candidate_full_release()
    release["maximum_trajectory_count"] = invalid_limit
    release["release_sha256"] = full._canonical_sha256({**release, "release_sha256": ""})

    with pytest.raises(ValueError, match="exact v9 trajectory count"):
        full._validate_training_release(release)


def test_full_training_release_rejects_over_limit_and_wrong_identity():
    release = full._validate_training_release(_candidate_full_release())
    _, trajectories, _ = full.load_design()
    disallowed = replace(trajectories[0], trajectory_id="not-in-the-frozen-panel")

    with pytest.raises(RuntimeError, match=f"at most {full.EXPECTED_TRAJECTORY_COUNT} trajectories"):
        full._validate_training_selection((*trajectories, disallowed), release)
    with pytest.raises(RuntimeError, match="does not authorize trajectories"):
        full._validate_training_selection((disallowed,), release)


def test_full_training_release_gate_is_fail_closed_without_pinned_hash(monkeypatch):
    monkeypatch.setattr(full, "EXPECTED_RELEASE_MANIFEST_SHA256", None)
    with pytest.raises(RuntimeError, match="Training fanout is not released"):
        full._load_training_release()


@pytest.mark.parametrize("stage", stress.STAGE_CONCURRENCIES)
def test_stress_stage_is_exactly_concurrent_and_decouples_switch_from_decay(stage):
    rows = stress.rows_for_stage(stage)

    assert len(rows) == stage
    assert Counter(row.support_id for row in rows) == stress.STAGE_SUPPORT_COUNTS[stage]
    assert len({row.trajectory_id for row in rows}) == stage
    assert len({row.training_seed for row in rows}) == stage
    assert {row.boundary_step for row in rows} == {stress.DATA_SWITCH_STEP}
    if stage == 6:
        expected_decay_steps = set(stress.C6_OPTIMIZER_DECAY_STEPS)
    elif stage == 12:
        expected_decay_steps = set(stress.C12_OPTIMIZER_DECAY_STEPS)
    else:
        expected_decay_steps = {stress.OPTIMIZER_DECAY_STEP}
    assert {row.optimizer_decay_step for row in rows} == expected_decay_steps
    assert stress.DATA_SWITCH_STEP != stress.OPTIMIZER_DECAY_STEP
    assert {row.phase_0_starcoder for row in rows} == {stress.PHASE_0_STARCODER}
    assert {row.phase_1_starcoder for row in rows} == {stress.PHASE_1_STARCODER}
    assert {row.support_start_batches for row in rows if row.support_id == "m100a"} == {0}
    assert {row.support_start_batches for row in rows if row.support_id == "m100b"} == {stress.SUPPORT_BATCHES}


def test_c12_primary_decay_onsets_are_seeded_and_balanced():
    expected = list(stress.C12_PRIMARY_ONSET_MULTISET)
    random.Random(stress.C12_ONSET_ASSIGNMENT_SEED).shuffle(expected)

    assert tuple(expected) == stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS
    assert Counter(stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS) == {
        1_920: 2,
        2_304: 2,
        2_688: 2,
        3_072: 2,
    }
    unique_onsets = sorted(set(expected))
    minimum_separation = runtime_gate.C12_LEAD_PLACEBO_OFFSET_STEPS + 2 * runtime_gate.C12_ASSIGNMENT_WINDOW_STEPS
    assert all(right - left >= minimum_separation for left, right in pairwise(unique_onsets))


def test_stress_predecessor_chain_is_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="must not cite"):
        stress._validate_previous_stage(6, "a" * 64)
    with pytest.raises(ValueError, match="requires the SHA-256"):
        stress._validate_previous_stage(12, None)

    monkeypatch.setattr(stress.full, "_remote_sha256", lambda _: "b" * 64)
    with pytest.raises(ValueError, match="report drifted"):
        stress._validate_previous_stage(12, "a" * 64, 7)


def test_stress_predecessor_is_bound_to_preregistration(tmp_path, monkeypatch):
    report_path = tmp_path / "stage-c06.json"
    report_path.write_text('{"stage":6,"status":"pass"}\n')
    digest = "a" * 64
    preregistration = {
        "predecessor": {
            "generation": 11,
            "runtime_report_sha256": digest,
            "runtime_report_remote_generation": None,
        }
    }
    monkeypatch.setattr(stress, "report_path", lambda *_: str(report_path))
    monkeypatch.setattr(stress.full, "_remote_sha256", lambda _: digest)

    stress._validate_previous_stage(12, digest, 11, preregistration=preregistration)
    with pytest.raises(ValueError, match="generation does not match"):
        stress._validate_previous_stage(12, digest, 10, preregistration=preregistration)
    with pytest.raises(ValueError, match="hash does not match"):
        stress._validate_previous_stage(12, "b" * 64, 11, preregistration=preregistration)

    monkeypatch.setattr(stress, "_object_generation", lambda *_: "unexpected")
    with pytest.raises(ValueError, match="object generation drifted"):
        stress._validate_previous_stage(12, digest, 11, preregistration=preregistration)


def test_stress_predecessor_can_skip_preregistered_intermediate_stages(tmp_path, monkeypatch):
    report_path = tmp_path / "stage-c06.json"
    report_path.write_text('{"stage":6,"status":"pass"}\n')
    digest = "a" * 64
    preregistration = {
        "predecessor": {
            "stage": 6,
            "generation": 15,
            "runtime_report_sha256": digest,
            "runtime_report_remote_generation": None,
        }
    }
    observed: list[tuple[int, int]] = []

    def direct_report_path(stage, generation):
        observed.append((stage, generation))
        return str(report_path)

    monkeypatch.setattr(stress, "report_path", direct_report_path)
    monkeypatch.setattr(stress.full, "_remote_sha256", lambda _: digest)

    stress._validate_previous_stage(64, digest, 15, preregistration=preregistration)

    assert observed == [(6, 15)]


def test_stress_predecessor_rejects_nonpreceding_or_noninteger_stage():
    base_predecessor = {
        "generation": 15,
        "runtime_report_sha256": "a" * 64,
        "runtime_report_remote_generation": None,
    }
    with pytest.raises(ValueError, match="invalid predecessor stage"):
        stress._validate_previous_stage(
            64,
            "a" * 64,
            15,
            preregistration={"predecessor": {**base_predecessor, "stage": 64}},
        )
    with pytest.raises(ValueError, match="must be an integer"):
        stress._validate_previous_stage(
            64,
            "a" * 64,
            15,
            preregistration={"predecessor": {**base_predecessor, "stage": "6"}},
        )


def test_stress_rendezvous_releases_only_after_every_row_is_ready(tmp_path):
    row_ids = ("row-a", "row-b")
    errors: list[BaseException] = []

    def wait(row_id: str) -> None:
        try:
            stress._wait_for_stage_rendezvous(
                root=str(tmp_path),
                rendezvous_id="unit-test",
                row_id=row_id,
                row_ids=row_ids,
                worker_claim_id=f"/test/rendezvous/{row_id}",
                timeout_seconds=2.0,
                poll_seconds=0.005,
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=wait, args=(row_id,)) for row_id in row_ids]
    threads[0].start()
    time.sleep(0.05)
    assert threads[0].is_alive()
    threads[1].start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    fs, _, ready_paths, release_path = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    assert all(fs.exists(path) for path in ready_paths)
    assert fs.exists(release_path)


def test_stress_rendezvous_rejects_invalid_membership(tmp_path):
    with pytest.raises(ValueError, match="not in the expected stage rows"):
        stress._wait_for_stage_rendezvous(
            root=str(tmp_path),
            rendezvous_id="unit-test",
            row_id="row-c",
            row_ids=("row-a", "row-b"),
            worker_claim_id="/test/rendezvous/row-c",
            timeout_seconds=0.01,
        )


def test_stress_rendezvous_rejects_invalid_id_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="nonempty path segment"):
        stress._validate_empty_rendezvous(str(tmp_path), "nested/id", ("row-a",))


def test_stress_rendezvous_rejects_stale_state(tmp_path):
    row_ids = ("row-a", "row-b")
    stress._validate_empty_rendezvous(str(tmp_path), "unit-test", row_ids)
    fs, ready_dir, ready_paths, _ = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    with fs.open(ready_paths[0], "w") as handle:
        handle.write("{}")

    with pytest.raises(ValueError, match="already contains state"):
        stress._validate_empty_rendezvous(str(tmp_path), "unit-test", row_ids)


def test_stress_cohort_attempt_preflight_rejects_stale_completion_state(tmp_path):
    row_ids = ("row-a", "row-b")
    stress._validate_empty_cohort_attempt_rendezvous(str(tmp_path), 12, 14, 0)
    start_id = stress.cohort_rendezvous_id(12, 14, cohort_attempt=0, iris_attempt=3)
    fs, _, _, release_path = stress._rendezvous_paths(str(tmp_path), f"complete-{start_id}", row_ids)
    fs.makedirs(str(Path(release_path).parent), exist_ok=True)
    with fs.open(release_path, "w") as handle:
        handle.write("{}")

    with pytest.raises(ValueError, match="already contains rendezvous state"):
        stress._validate_empty_cohort_attempt_rendezvous(str(tmp_path), 12, 14, 0)


def test_stress_rendezvous_rejects_duplicate_worker_claim(tmp_path):
    row_ids = ("row-a", "row-b")
    fs, ready_dir, ready_paths, _ = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    with fs.open(ready_paths[0], "w") as handle:
        json.dump(
            stress._ready_marker_payload(
                worker_claim_id="/test/rendezvous/first-owner",
                row_id="row-a",
                rendezvous_id="unit-test",
                row_ids=row_ids,
            ),
            handle,
        )

    with pytest.raises(RuntimeError, match="already claimed"):
        stress._wait_for_stage_rendezvous(
            root=str(tmp_path),
            rendezvous_id="unit-test",
            row_id="row-a",
            row_ids=row_ids,
            worker_claim_id="/test/rendezvous/second-owner",
            timeout_seconds=0.1,
            poll_seconds=0.001,
        )


def test_stress_attempt_scoped_rendezvous_requires_one_complete_cohort():
    markers = {f"row-{index}": {"worker_claim_id": f"/calvinxu/root/cohort/{index}:1"} for index in range(3)}
    stress._validate_attempt_scoped_claims(markers, rendezvous_id="c03-retry11-20260812-attempt002-iris001")

    wrong_attempt = {**markers, "row-2": {"worker_claim_id": "/calvinxu/root/cohort/2:2"}}
    with pytest.raises(RuntimeError, match="one complete cohort attempt"):
        stress._validate_attempt_scoped_claims(
            wrong_attempt,
            rendezvous_id="c03-retry11-20260812-attempt002-iris001",
        )


def test_stress_rendezvous_create_is_idempotent_only_for_the_same_owner(tmp_path):
    fs, path = stress.fsspec.core.url_to_fs(str(tmp_path / "marker.json"))
    payload = {"marker_nonce": "owner-a"}

    stress._write_json_once(fs, path, payload)
    stress._write_json_once(fs, path, payload)
    with pytest.raises(RuntimeError, match="already claimed"):
        stress._write_json_once(fs, path, {"marker_nonce": "owner-b"})


def test_stress_rendezvous_waits_for_an_incomplete_visible_marker(tmp_path):
    row_ids = ("row-a", "row-b")
    fs, ready_dir, ready_paths, _ = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    with fs.open(ready_paths[1], "w"):
        pass

    def finish_marker() -> None:
        time.sleep(0.05)
        with fs.open(ready_paths[1], "w") as handle:
            json.dump(
                stress._ready_marker_payload(
                    worker_claim_id="/test/rendezvous/row-b",
                    row_id="row-b",
                    rendezvous_id="unit-test",
                    row_ids=row_ids,
                ),
                handle,
            )

    thread = threading.Thread(target=finish_marker)
    thread.start()
    stress._wait_for_stage_rendezvous(
        root=str(tmp_path),
        rendezvous_id="unit-test",
        row_id="row-a",
        row_ids=row_ids,
        worker_claim_id="/test/rendezvous/row-a",
        timeout_seconds=1.0,
        poll_seconds=0.005,
    )
    thread.join(timeout=1.0)

    assert not thread.is_alive()


@pytest.mark.parametrize("retry_row_id", ["row-a", "row-b"])
def test_stress_rendezvous_reuses_claim_after_logical_worker_retry(tmp_path, retry_row_id):
    row_ids = ("row-a", "row-b")
    claims = {row_id: f"/test/rendezvous/{row_id}" for row_id in row_ids}
    errors: list[BaseException] = []

    def wait(row_id: str) -> None:
        try:
            stress._wait_for_stage_rendezvous(
                root=str(tmp_path),
                rendezvous_id="unit-test",
                row_id=row_id,
                row_ids=row_ids,
                worker_claim_id=claims[row_id],
                timeout_seconds=1.0,
                poll_seconds=0.001,
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=wait, args=(row_id,)) for row_id in row_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)

    stress._wait_for_stage_rendezvous(
        root=str(tmp_path),
        rendezvous_id="unit-test",
        row_id=retry_row_id,
        row_ids=row_ids,
        worker_claim_id=claims[retry_row_id],
        timeout_seconds=0.1,
        poll_seconds=0.001,
    )


def test_stress_worker_claim_identifies_the_exact_cohort_attempt(monkeypatch):
    monkeypatch.setenv("IRIS_TASK_ID", "/calvinxu/root/child/0:0")
    first_attempt = stress._current_task_attempt()
    monkeypatch.setenv("IRIS_TASK_ID", "/calvinxu/root/child/0:1")
    second_attempt = stress._current_task_attempt()

    assert first_attempt.to_wire() == "/calvinxu/root/child/0:0"
    assert second_attempt.to_wire() == "/calvinxu/root/child/0:1"


@dataclass(frozen=True)
class _MinimalTrainConfig:
    trainer: TrainerConfig


def _minimal_stress_config(name: str = "stress-row") -> TrainLmOnPodConfig:
    return TrainLmOnPodConfig(
        train_config=_MinimalTrainConfig(
            trainer=TrainerConfig(
                id=name,
                tracker=WandbConfig(name=name, replicate_path="gs://bucket/base"),
                distributed=DistributedConfig(initialize_jax_distributed=False),
            )
        ),
        resources=ResourceConfig.with_cpu(),
        output_path="gs://bucket/base",
        env_vars={},
    )


def test_stress_retry_gets_fresh_checkpoint_and_wandb_namespaces():
    scoped = stress._attempt_scoped_config(_minimal_stress_config(), 3)
    train_config = scoped.train_config

    assert scoped.output_path == "gs://bucket/base/attempt-003"
    assert scoped.env_vars == {"RUN_ID": "stress-row_a003"}
    assert train_config.trainer.id == "stress-row_a003"
    assert train_config.trainer.load_checkpoint is False
    assert train_config.trainer.distributed.initialize_jax_distributed is False
    assert train_config.trainer.tracker.id == "stress-row_a003"
    assert train_config.trainer.tracker.name == "stress-row_a003"
    assert train_config.trainer.tracker.resume == "never"
    assert train_config.trainer.tracker.replicate_path == "gs://bucket/base/attempt-003"


def test_resume_canary_keeps_stable_output_and_wandb_identity():
    row = SimpleNamespace(trajectory_id="row-a")
    config = replace(_minimal_stress_config(), resources=replace(_minimal_stress_config().resources, preemptible=True))

    resumable = resume._resumable_config(config, row)
    train_config = resumable.train_config

    assert resumable.output_path == "gs://bucket/base/attempt-000"
    assert resumable.env_vars == {"RUN_ID": "gcfresumeg19_row-a"}
    assert train_config.trainer.id == "gcfresumeg19_row-a"
    assert train_config.trainer.load_checkpoint is None
    assert train_config.trainer.load_checkpoint_path is None
    assert train_config.trainer.checkpointer.save_interval == resume.CHECKPOINT_INTERVAL
    assert train_config.trainer.tracker.id == "gcfresumeg19_row-a"
    assert train_config.trainer.tracker.name == "gcfresumeg19_row-a"
    assert train_config.trainer.tracker.resume == "allow"
    assert train_config.trainer.tracker.replicate_path == resumable.output_path


def test_resume_checkpoint_claim_hashes_exact_metadata(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "step-123"
    checkpoint.mkdir(parents=True)
    metadata = b'{"step": 123, "timestamp": "2026-08-13T00:00:00", "is_temporary": true}'
    (checkpoint / "metadata.json").write_bytes(metadata)
    trainer = SimpleNamespace(id="row-a", checkpoint_search_paths=lambda _: [str(checkpoint_root)])
    runtime_config = SimpleNamespace(trainer=trainer)
    config = SimpleNamespace(output_path=str(tmp_path), train_config=runtime_config)
    monkeypatch.setattr(resume, "apply_output_path", lambda *_: runtime_config)

    claim = resume._checkpoint_claim(config)

    assert claim.path == str(checkpoint)
    assert claim.step == 123
    assert claim.metadata_sha256 == hashlib.sha256(metadata).hexdigest()


def test_resume_analyzer_requires_parent_and_worker_checkpoint_agreement():
    signature = {
        "checkpoint_path": "gs://bucket/checkpoints/step-123",
        "checkpoint_step": 123,
        "checkpoint_metadata_sha256": "abc",
    }
    attempts = {
        0: [
            {"attempt": 0, "checkpoint_path": None, "checkpoint_step": 0, "checkpoint_metadata_sha256": None},
            {"attempt": 1, **signature},
        ]
    }
    faults = {0: {"source_attempt": 0, **signature}}
    state_evidence = {0: {1: {"state_step": 123}}}

    matched, failures = resume_analyzer._forced_retry_evidence(attempts, faults, state_evidence)
    _, mismatched_failures = resume_analyzer._forced_retry_evidence(
        attempts,
        {0: {**faults[0], "checkpoint_step": 124}},
        state_evidence,
    )

    assert failures == []
    assert matched[0]["attempt"] == 1
    assert len(mismatched_failures) == 1


def test_resume_analyzer_allows_a_newer_checkpoint_during_queued_preemption():
    parent = {
        "checkpoint_path": "gs://bucket/checkpoints/step-123",
        "checkpoint_step": 123,
        "checkpoint_metadata_sha256": "abc",
        "source_attempt": 0,
    }
    worker = {
        "attempt": 1,
        "checkpoint_path": "gs://bucket/checkpoints/step-130",
        "checkpoint_step": 130,
        "checkpoint_metadata_sha256": "def",
    }

    matched, failures = resume_analyzer._forced_retry_evidence(
        {0: [worker]},
        {0: parent},
        {0: {1: {"state_step": 130}}},
    )

    assert failures == []
    assert matched[0] == worker


def test_resume_analyzer_rejects_discovery_without_a_loaded_state(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    path = resume._initial_state_evidence_path(0, 1)
    attempts = {
        0: [
            {
                "attempt": 1,
                "checkpoint_path": "gs://bucket/checkpoints/step-123",
                "checkpoint_step": 123,
                "checkpoint_metadata_sha256": "abc",
                "initial_state_evidence_path": path,
            }
        ]
    }
    monkeypatch.setattr(
        resume_analyzer,
        "_read_json",
        lambda _: {"run_id": resume._run_id(row), "state_step": 0},
    )

    _, failures = resume_analyzer._state_evidence((row,), attempts)

    assert failures == ["task 0 attempt 1 initialized at step 0, expected 123 from its checkpoint claim"]


def test_resume_analyzer_tolerates_preinitialization_preemption_but_requires_final_state(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    attempts = {
        0: [
            {"attempt": 0, "initial_state_evidence_path": "missing-attempt-0"},
            {"attempt": 1, "initial_state_evidence_path": "state-attempt-1", "checkpoint_step": 0},
        ]
    }

    def read_json(path):
        if path.startswith("missing-"):
            raise FileNotFoundError
        return {"run_id": resume._run_id(row), "state_step": 0}

    monkeypatch.setattr(resume_analyzer, "_read_json", read_json)

    observed, failures = resume_analyzer._state_evidence((row,), attempts)

    assert failures == []
    assert observed == {0: {1: {"run_id": resume._run_id(row), "state_step": 0}}}

    attempts[0].append({"attempt": 2, "initial_state_evidence_path": "missing-attempt-2"})
    _, failures = resume_analyzer._state_evidence((row,), attempts)
    assert failures == ["task 0 final attempt 2 lacks state evidence"]


def test_train_lm_initial_state_evidence_is_create_only(tmp_path):
    path = tmp_path / "state.json"

    train_lm_module._write_initial_state_evidence(
        str(path),
        checkpoint_search_paths=["gs://bucket/checkpoints"],
        run_id="run-a",
        state_step=123,
    )
    train_lm_module._write_initial_state_evidence(
        str(path),
        checkpoint_search_paths=["gs://bucket/checkpoints"],
        run_id="run-a",
        state_step=123,
    )

    evidence = json.loads(path.read_text())
    assert {key: evidence[key] for key in ("checkpoint_search_paths", "run_id", "state_step")} == {
        "checkpoint_search_paths": ["gs://bucket/checkpoints"],
        "run_id": "run-a",
        "state_step": 123,
    }
    assert evidence["written_at"].endswith("+00:00")
    with pytest.raises(RuntimeError, match="already claimed"):
        train_lm_module._write_initial_state_evidence(
            str(path),
            checkpoint_search_paths=["gs://bucket/checkpoints"],
            run_id="run-a",
            state_step=124,
        )


def test_resume_wandb_progress_flushes_cached_run_before_polling(monkeypatch):
    class FakeApi:
        def __init__(self, *, timeout):
            assert timeout == 60
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1

        def run(self, path):
            assert path == "marin-community/marin/run-a"
            return SimpleNamespace(summary={"global_step": self.flush_count})

    monkeypatch.setattr(resume.wandb, "Api", FakeApi)
    progress = resume.WandbProgress()

    assert progress.global_step("run-a") == 1
    assert progress.global_step("run-a") == 2


def test_resume_worker_tolerates_a_natural_preemption_before_first_checkpoint(monkeypatch):
    trainer = TrainerConfig(
        id="row-a",
        tracker=WandbConfig(name="row-a", replicate_path="gs://bucket/base"),
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )
    config = TrainLmOnPodConfig(
        train_config=TrainLmConfig(trainer=trainer),
        resources=ResourceConfig.with_cpu(),
        output_path="gs://bucket/base",
        env_vars={"RUN_ID": "row-a"},
    )
    row = SimpleNamespace(trajectory_id="row-a")
    attempts: list[dict[str, Any]] = []
    launched: list[TrainLmOnPodConfig] = []
    monkeypatch.setenv("IRIS_TASK_ID", "/calvinxu/parent/child/0:1")
    monkeypatch.setattr(resume, "_optional_checkpoint_claim", lambda _: None)
    monkeypatch.setattr(
        resume,
        "_write_attempt_evidence",
        lambda **kwargs: attempts.append(kwargs),
    )
    monkeypatch.setattr(resume, "run_levanter_train_lm", launched.append)

    resume._run_row((config,), (row,))

    assert attempts[0]["checkpoint"] is None
    assert attempts[0]["initial_state_evidence_path"].endswith("/state/task-000/attempt-001.json")
    assert launched[0].train_config.initial_state_evidence_path == attempts[0]["initial_state_evidence_path"]


def test_resume_fresh_state_preflight_rejects_evidence_and_checkpoints(tmp_path, monkeypatch):
    evidence_root = tmp_path / "evidence"
    output_root = tmp_path / "output"
    temporary_root = tmp_path / "temporary"
    trainer = SimpleNamespace(id="run-a", checkpoint_search_paths=lambda _: [str(output_root), str(temporary_root)])
    train_config = SimpleNamespace(trainer=trainer)
    config = SimpleNamespace(output_path=str(output_root), train_config=train_config)

    class FakeApi:
        def __init__(self, *, timeout):
            assert timeout == 60

        @staticmethod
        def flush():
            return None

        @staticmethod
        def runs(path, *, filters, per_page):
            assert path == "marin-community/marin"
            assert filters == {"name": {"$in": ["run-a"]}}
            assert per_page == 1
            return ()

    monkeypatch.setattr(resume, "EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.setattr(resume, "apply_output_path", lambda *_: train_config)
    monkeypatch.setattr(resume.wandb, "Api", FakeApi)

    resume._validate_fresh_state((config,))

    evidence_root.mkdir()
    (evidence_root / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence namespace is not empty"):
        resume._validate_fresh_state((config,))

    (evidence_root / "receipt.json").unlink()
    output_root.mkdir()
    (output_root / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="checkpoint namespace is not empty"):
        resume._validate_fresh_state((config,))

    (output_root / "metadata.json").unlink()
    temporary_root.mkdir()
    (temporary_root / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="checkpoint namespace is not empty"):
        resume._validate_fresh_state((config,))


def test_resume_fresh_state_preflight_rejects_existing_wandb_identity(tmp_path, monkeypatch):
    trainer = SimpleNamespace(id="run-a", checkpoint_search_paths=lambda _: [])
    train_config = SimpleNamespace(trainer=trainer)
    config = SimpleNamespace(output_path=str(tmp_path / "output"), train_config=train_config)

    class FakeApi:
        def __init__(self, *, timeout):
            assert timeout == 60

        @staticmethod
        def flush():
            return None

        @staticmethod
        def runs(*_, **__):
            return (SimpleNamespace(id="run-a"),)

    monkeypatch.setattr(resume, "EVIDENCE_ROOT", str(tmp_path / "evidence"))
    monkeypatch.setattr(resume, "apply_output_path", lambda *_: train_config)
    monkeypatch.setattr(resume.wandb, "Api", FakeApi)

    with pytest.raises(ValueError, match="W&B identities already exist"):
        resume._validate_fresh_state((config,))


def test_resume_wandb_history_stitches_complete_resumed_run_and_deduplicates_page_boundary():
    rows = [
        {"_step": 0, "global_step": 0, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
        {"_step": 1, "global_step": 1, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
        {"_step": 1, "global_step": 1, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
        {"_step": 2, "global_step": 2, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
    ]
    run = SimpleNamespace(id="run-a", scan_history=lambda **_: iter(rows))

    history = resume_analyzer._history(run)

    assert [item["global_step"] for item in history] == [0, 1, 2]


def test_resume_wandb_evidence_reports_conflicting_history_without_raising(monkeypatch):
    row = SimpleNamespace(trajectory_id="row-a")
    rows = [
        {"_step": 0, "global_step": 0, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
        {"_step": 0, "global_step": 1, "throughput/tokens_per_second": 1.0, "throughput/loading_time": 0.1},
    ]

    class FakeApi:
        def __init__(self, *, timeout):
            assert timeout == 60

        @staticmethod
        def flush():
            return None

        @staticmethod
        def run(_):
            return SimpleNamespace(id="run-a", scan_history=lambda **_: iter(rows))

    monkeypatch.setattr(resume_analyzer.wandb, "Api", FakeApi)

    evidence, failures = resume_analyzer._wandb_evidence((row,), {0: []}, {})

    assert evidence[row.trajectory_id]["status"] == "fail"
    assert len(failures) == 1
    assert "Conflicting duplicate W&B history step 0" in failures[0]


def test_resume_submission_uses_production_per_task_retry_semantics(monkeypatch):
    submitted: dict[str, Any] = {}
    injections: list[tuple[Any, ...]] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c06-resume-g19")

        @staticmethod
        def wait(*, timeout, raise_on_failure):
            assert timeout == resume.JOB_TIMEOUT_SECONDS
            assert raise_on_failure is False
            return SimpleNamespace(
                state=job_pb2.JOB_STATE_SUCCEEDED,
                failure_count=0,
                preemption_count=3,
            )

    class FakeClient:
        def submit(self, **kwargs):
            submitted.update(kwargs)
            return FakeJob()

    config = replace(_minimal_stress_config(), resources=replace(_minimal_stress_config().resources, preemptible=True))
    configs = (config,) * resume.STAGE
    rows = tuple(SimpleNamespace(trajectory_id=f"row-{index}") for index in range(resume.STAGE))
    monkeypatch.setattr(
        resume,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    monkeypatch.setattr(resume, "convert_resources", lambda resources: resources)
    monkeypatch.setattr(resume, "convert_constraints", lambda resources: ())
    monkeypatch.setattr(resume, "_inject_faults", lambda *args: injections.append(args))

    child = resume._submit_canary(configs, rows)

    assert child == "/calvinxu/test-parent/stage-c06-resume-g19"
    assert submitted["replicas"] == resume.STAGE
    assert submitted["coscheduling"] is None
    assert submitted["max_retries_failure"] == 0
    assert submitted["max_retries_preemption"] == resume.MAX_PREEMPTION_RETRIES
    assert submitted["max_task_failures"] == 0
    assert submitted["existing_job_policy"] == job_pb2.EXISTING_JOB_POLICY_ERROR
    assert len(injections) == 1
    assert isinstance(injections[0][0], FakeJob)
    assert injections[0][1:] == (rows, configs)


def test_resume_fault_injection_is_idempotent_after_receipts_exist(monkeypatch):
    monkeypatch.setattr(resume, "_existing_fault_tasks", lambda: {0, 1, 2})
    monkeypatch.setattr(resume, "iris_ctx", lambda: pytest.fail("Iris must not be consulted"))

    resume._inject_faults(SimpleNamespace(), (), ())


def test_resume_fault_injection_waits_for_a_complete_checkpoint(monkeypatch):
    fault = resume.FaultInjection(task_index=0, trigger_step=10, phase="unit_test")
    checkpoint = resume.CheckpointClaim(
        path="gs://bucket/checkpoints/step-8",
        step=8,
        metadata_sha256="abc",
    )
    checkpoint_calls = 0
    receipts: list[dict[str, Any]] = []

    def checkpoint_claim(_):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls == 1:
            raise FileNotFoundError
        return checkpoint

    class FakeClient:
        def kick_tasks(self, task_ids, *, desired_state, reason):
            assert task_ids == ["/calvinxu/test-parent/child/0"]
            assert desired_state == job_pb2.TASK_STATE_PREEMPTED
            assert "unit_test" in reason
            return [SimpleNamespace(queued=True)]

    class FakeProgress:
        @staticmethod
        def global_step(_):
            return 10

    job = SimpleNamespace(
        job_id=JobName.from_wire("/calvinxu/test-parent/child"),
        state_only=lambda: job_pb2.JOB_STATE_RUNNING,
    )
    monkeypatch.setattr(resume, "FAULT_INJECTIONS", (fault,))
    monkeypatch.setattr(resume, "_existing_fault_tasks", lambda: set())
    monkeypatch.setattr(resume, "_checkpoint_claim", checkpoint_claim)
    monkeypatch.setattr(resume, "_attempt_indices", lambda _: (0,))
    monkeypatch.setattr(resume, "WandbProgress", FakeProgress)
    monkeypatch.setattr(resume, "iris_ctx", lambda: SimpleNamespace(client=FakeClient()))
    monkeypatch.setattr(resume.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        resume,
        "_write_fault_receipt",
        lambda fault, **kwargs: receipts.append({"fault": fault, **kwargs}),
    )

    resume._inject_faults(job, (SimpleNamespace(trajectory_id="row-a"),), (SimpleNamespace(),))

    assert checkpoint_calls == 2
    assert len(receipts) == 1
    assert receipts[0]["checkpoint"] == checkpoint
    assert receipts[0]["source_attempt"] == 0


def test_stress_child_waits_at_start_and_completion_barriers(monkeypatch):
    config = replace(
        _minimal_stress_config(),
        env_vars={
            stress._RENDEZVOUS_ROOT_ENV: "gs://bucket/rendezvous",
            stress._RENDEZVOUS_ROW_ENV: "row-a",
            stress._RENDEZVOUS_ROWS_ENV: json.dumps(("row-a",)),
            stress._STRESS_GENERATION_ENV: "11",
            stress._STRESS_STAGE_ENV: "6",
        },
    )
    rendezvous_ids: list[str] = []
    monkeypatch.setenv("IRIS_TASK_ID", "/calvinxu/parent/child/0:7")
    monkeypatch.setattr(
        stress,
        "_wait_for_stage_rendezvous",
        lambda **kwargs: rendezvous_ids.append(kwargs["rendezvous_id"]),
    )
    monkeypatch.setattr(
        stress,
        "get_job_info",
        lambda: SimpleNamespace(
            worker_id="marin-tpu-v5p-preemptible-8-us-central1-test-worker-0", worker_region="us-central1"
        ),
    )
    monkeypatch.setattr(stress, "run_levanter_train_lm", lambda _: None)

    stress._run_stress_cohort((config,), 3)

    assert rendezvous_ids == [
        stress.cohort_rendezvous_id(6, 11, 3, 7),
        stress.completion_rendezvous_id(6, 11, 3, 7),
    ]


def test_stress_parent_runtime_contract_is_strict(monkeypatch):
    expected_env = {
        stress._PARENT_PREEMPTIBLE_ENV: "false",
        stress._PARENT_MAX_RETRIES_FAILURE_ENV: "0",
        stress._PARENT_MAX_RETRIES_PREEMPTION_ENV: "0",
    }
    inherited_placement = [region_constraint(["us-central1"]), zone_constraint("us-central1-a")]
    info = SimpleNamespace(constraints=inherited_placement, env=expected_env)
    monkeypatch.setattr(stress, "get_job_info", lambda: info)
    monkeypatch.setenv("IRIS_TASK_ID", "/calvinxu/parent/0:0")

    stress._validate_parent_runtime_contract()

    info.constraints = [*inherited_placement, preemptible_constraint(True)]
    with pytest.raises(RuntimeError, match="preemptible=true"):
        stress._validate_parent_runtime_contract()

    info.constraints = [zone_constraint("us-central1-a")]
    with pytest.raises(RuntimeError, match="placement metadata drifted"):
        stress._validate_parent_runtime_contract()

    info.constraints = inherited_placement
    info.env = {**expected_env, stress._PARENT_MAX_RETRIES_PREEMPTION_ENV: "1000"}
    with pytest.raises(RuntimeError, match="retry attestation drifted"):
        stress._validate_parent_runtime_contract()


def test_stress_submission_replaces_the_whole_preempted_cohort(monkeypatch, tmp_path):
    submissions: list[dict[str, object]] = []
    waits: list[tuple[float, bool]] = []

    class FakeJob:
        def __init__(self, attempt: int):
            self.attempt = attempt
            self.job_id = JobName.from_wire(f"/calvinxu/test-parent/{stress.cohort_child_name(6, 11, attempt)}")

        def wait(self, timeout: float, *, raise_on_failure: bool):
            waits.append((timeout, raise_on_failure))
            if self.attempt == 0:
                return job_pb2.JobStatus(
                    state=job_pb2.JOB_STATE_WORKER_FAILED,
                    failure_count=0,
                    preemption_count=1,
                )
            return job_pb2.JobStatus(
                state=job_pb2.JOB_STATE_SUCCEEDED,
                failure_count=0,
                preemption_count=0,
            )

        def terminate(self) -> None:
            raise AssertionError("successful wait path must not terminate the child")

    class FakeClient:
        def submit(self, **kwargs):
            submissions.append(kwargs)
            return FakeJob(len(submissions) - 1)

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(6))

    child_job = stress._submit_stress_cohort(
        configs,
        stage=6,
        generation=11,
        rendezvous_root=str(tmp_path),
    )

    assert child_job == "/calvinxu/test-parent/stage-c06-cohort-g11-attempt-001"
    assert len(submissions) == 2
    assert [submission["name"] for submission in submissions] == [
        "stage-c06-cohort-g11-attempt-000",
        "stage-c06-cohort-g11-attempt-001",
    ]
    for submission in submissions:
        assert submission["replicas"] == 6
        assert submission["coscheduling"].group_by == stress.COHORT_COSCHEDULING_GROUP
        assert submission["max_retries_failure"] == 0
        assert submission["max_retries_preemption"] == 0
        assert submission["max_task_failures"] == 0
        assert submission["existing_job_policy"] == job_pb2.EXISTING_JOB_POLICY_ERROR
        constraints = submission["constraints"]
        assert isinstance(constraints, list)
        assert preemptible_constraint(True) in constraints
    assert waits == [(stress.COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS, False)] * 2


def test_stress_submission_supports_independent_target_concurrency(monkeypatch, tmp_path):
    submissions: list[dict[str, object]] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c64-cohort-g17-attempt-000")

        def wait(self, timeout: float, *, raise_on_failure: bool):
            assert timeout == stress.COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS
            assert raise_on_failure is False
            return job_pb2.JobStatus(
                state=job_pb2.JOB_STATE_SUCCEEDED,
                failure_count=0,
                preemption_count=0,
            )

        def terminate(self) -> None:
            raise AssertionError("successful independent cohort must not be terminated")

    class FakeClient:
        def submit(self, **kwargs):
            submissions.append(kwargs)
            return FakeJob()

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(64))

    child_job = stress._submit_stress_cohort(
        configs,
        stage=64,
        generation=17,
        rendezvous_root=str(tmp_path),
        coscheduling_group_by=None,
        start_barrier_timeout_seconds=5_400.0,
        completion_barrier_timeout_seconds=1_800.0,
        parent_managed_preemption_retries=0,
    )

    assert child_job == "/calvinxu/test-parent/stage-c64-cohort-g17-attempt-000"
    assert len(submissions) == 1
    submission = submissions[0]
    assert submission["replicas"] == 64
    assert submission["coscheduling"] is None
    assert submission["max_retries_failure"] == 0
    assert submission["max_retries_preemption"] == 0
    assert submission["max_task_failures"] == 0


def test_stress_independent_placement_contract_is_fail_closed():
    preregistration = {
        "design": {
            "cohort": {
                "placement_mode": "independent",
                "coscheduling_group_by": None,
                "start_barrier_timeout_seconds": 5_400.0,
                "completion_barrier_timeout_seconds": 1_800.0,
                "parent_managed_whole_cohort_retries": 0,
            }
        }
    }

    assert stress._cohort_runtime_contract(preregistration) == (None, 5_400.0, 1_800.0, 0)

    preregistration["design"]["cohort"]["coscheduling_group_by"] = stress.COHORT_COSCHEDULING_GROUP
    with pytest.raises(ValueError, match="must not specify"):
        stress._cohort_runtime_contract(preregistration)

    preregistration["design"]["cohort"] = {
        "placement_mode": "independent",
        "coscheduling_group_by": None,
        "start_barrier_timeout_seconds": stress.COHORT_ATTEMPT_WAIT_TIMEOUT_SECONDS,
        "completion_barrier_timeout_seconds": 1_800.0,
        "parent_managed_whole_cohort_retries": 0,
    }
    with pytest.raises(ValueError, match="shorter than"):
        stress._cohort_runtime_contract(preregistration)

    preregistration["design"]["cohort"] = {
        "placement_mode": "independent",
        "coscheduling_group_by": None,
        "start_barrier_timeout_seconds": 5_400.0,
        "completion_barrier_timeout_seconds": 1_800.0,
        "parent_managed_whole_cohort_retries": 1,
    }
    with pytest.raises(ValueError, match="cannot safely use"):
        stress._cohort_runtime_contract(preregistration)


def test_stress_independent_submission_does_not_replace_a_preempted_task(monkeypatch, tmp_path):
    submissions: list[dict[str, object]] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c64-cohort-g18-attempt-000")

        def wait(self, timeout: float, *, raise_on_failure: bool):
            del timeout, raise_on_failure
            return job_pb2.JobStatus(
                state=job_pb2.JOB_STATE_WORKER_FAILED,
                failure_count=0,
                preemption_count=1,
            )

        def terminate(self) -> None:
            raise AssertionError("terminal preemption must not need termination")

    class FakeClient:
        def submit(self, **kwargs):
            submissions.append(kwargs)
            return FakeJob()

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(64))

    with pytest.raises(RuntimeError, match="exhausted 0 parent-managed preemption retries"):
        stress._submit_stress_cohort(
            configs,
            stage=64,
            generation=18,
            rendezvous_root=str(tmp_path),
            coscheduling_group_by=None,
            parent_managed_preemption_retries=0,
        )

    assert len(submissions) == 1


def test_runtime_gate_enforces_independent_zero_attempt_contract():
    preregistration = {
        "design": {
            "cohort": {
                "placement_mode": "independent",
                "coscheduling_group_by": None,
                "coscheduling_enabled": False,
                "parent_managed_whole_cohort_retries": 0,
            }
        },
        "release_gate": {
            "concurrency_and_integrity": {
                "iris_topology_coscheduling_forbidden": True,
                "parent_managed_whole_cohort_replacement_forbidden": True,
                "zero_parent_and_iris_attempt_required": True,
            }
        },
    }

    assert (
        runtime_gate._frozen_cohort_contract_failures(
            preregistration,
            cohort_attempt=0,
            iris_attempt=0,
        )
        == []
    )
    assert runtime_gate._frozen_cohort_contract_failures(
        preregistration,
        cohort_attempt=0,
        iris_attempt=1,
    ) == ["independent fail-closed gate requires parent attempt 0 and Iris attempt 0: parent=0, iris=1"]


def test_stress_submission_does_not_retry_an_application_failure(monkeypatch, tmp_path):
    submissions: list[dict[str, object]] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c06-cohort-g11-attempt-000")

        def wait(self, timeout: float, *, raise_on_failure: bool):
            del timeout, raise_on_failure
            return job_pb2.JobStatus(
                state=job_pb2.JOB_STATE_FAILED,
                failure_count=1,
                preemption_count=0,
            )

        def terminate(self) -> None:
            raise AssertionError("terminal application failure must not need termination")

    class FakeClient:
        def submit(self, **kwargs):
            submissions.append(kwargs)
            return FakeJob()

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(6))

    with pytest.raises(RuntimeError, match="without a retryable infrastructure preemption"):
        stress._submit_stress_cohort(
            configs,
            stage=6,
            generation=11,
            rendezvous_root=str(tmp_path),
        )

    assert len(submissions) == 1


def test_stress_submission_reattaches_without_replacing_a_finished_child(monkeypatch, tmp_path):
    submissions: list[str] = []
    parent_id = JobName.from_wire("/calvinxu/test-parent")

    class FakeJob:
        def __init__(self, job_id: JobName, status: job_pb2.JobStatus):
            self.job_id = job_id
            self.status = status

        def wait(self, timeout: float, *, raise_on_failure: bool):
            del timeout, raise_on_failure
            return self.status

        def terminate(self) -> None:
            raise AssertionError("terminal child must not need termination")

    preempted = job_pb2.JobStatus(
        state=job_pb2.JOB_STATE_WORKER_FAILED,
        failure_count=0,
        preemption_count=1,
    )
    succeeded = job_pb2.JobStatus(
        state=job_pb2.JOB_STATE_SUCCEEDED,
        failure_count=0,
        preemption_count=0,
    )

    class FakeClient:
        def submit(self, **kwargs):
            name = kwargs["name"]
            submissions.append(name)
            if name.endswith("attempt-000"):
                raise stress.JobAlreadyExists("existing attempt")
            return FakeJob(parent_id.child(name), succeeded)

    client = FakeClient()
    monkeypatch.setattr(stress, "iris_ctx", lambda: SimpleNamespace(client=client, job_id=parent_id))
    monkeypatch.setattr(stress, "Job", lambda attached_client, job_id: FakeJob(job_id, preempted))

    def validate_rendezvous(_root, _stage, _generation, cohort_attempt):
        if cohort_attempt == 0:
            raise ValueError("legitimate state from existing child")

    monkeypatch.setattr(stress, "_validate_empty_cohort_attempt_rendezvous", validate_rendezvous)
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(6))

    child_job = stress._submit_stress_cohort(
        configs,
        stage=6,
        generation=11,
        rendezvous_root=str(tmp_path),
    )

    assert child_job == "/calvinxu/test-parent/stage-c06-cohort-g11-attempt-001"
    assert submissions == [
        "stage-c06-cohort-g11-attempt-000",
        "stage-c06-cohort-g11-attempt-001",
    ]


def test_stress_submission_rejects_stale_rendezvous_for_a_new_child(monkeypatch, tmp_path):
    terminated: list[bool] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c06-cohort-g11-attempt-000")

        def terminate(self) -> None:
            terminated.append(True)

    class FakeClient:
        def submit(self, **kwargs):
            del kwargs
            return FakeJob()

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )

    def reject_stale_rendezvous(*_):
        raise ValueError("stale rendezvous")

    monkeypatch.setattr(stress, "_validate_empty_cohort_attempt_rendezvous", reject_stale_rendezvous)
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(6))

    with pytest.raises(RuntimeError, match="rendezvous namespace is stale"):
        stress._submit_stress_cohort(
            configs,
            stage=6,
            generation=11,
            rendezvous_root=str(tmp_path),
        )

    assert terminated == [True]


def test_stress_submission_terminates_a_timed_out_child(monkeypatch, tmp_path):
    terminated: list[bool] = []

    class FakeJob:
        job_id = JobName.from_wire("/calvinxu/test-parent/stage-c06-cohort-g11-attempt-000")

        def wait(self, timeout: float, *, raise_on_failure: bool):
            del timeout, raise_on_failure
            raise TimeoutError("test timeout")

        def terminate(self) -> None:
            terminated.append(True)

    class FakeClient:
        def submit(self, **kwargs):
            del kwargs
            return FakeJob()

    monkeypatch.setattr(
        stress,
        "iris_ctx",
        lambda: SimpleNamespace(client=FakeClient(), job_id=JobName.from_wire("/calvinxu/test-parent")),
    )
    configs = tuple(_minimal_stress_config(f"stress-row-{index}") for index in range(6))

    with pytest.raises(TimeoutError, match="test timeout"):
        stress._submit_stress_cohort(
            configs,
            stage=6,
            generation=11,
            rendezvous_root=str(tmp_path),
        )

    assert terminated == [True]


def test_stress_child_resources_are_explicitly_preemptible():
    resources = stress._stress_resources(
        tpu_type=stress.base.DEFAULT_TPU_TYPE,
        tpu_region=stress.base.DEFAULT_TPU_REGION,
        tpu_zone=stress.base.DEFAULT_TPU_ZONE,
    )

    assert resources.preemptible is True
    assert preemptible_constraint(False) not in convert_constraints(resources)


def test_stress_generation_scopes_all_external_identities():
    row = stress.rows_for_stage(6)[0]

    assert "retry8_20260811" in stress.namespace_name(8)
    assert stress.namespace_version(8) == "2026.08.11.8"
    assert stress.wandb_run_id(row, 8).startswith("gcfstressr8_")
    assert stress.wandb_run_id(row, 8, 3).endswith("_a003")
    assert "retry8_20260811" in stress.report_path(6, 8)
    assert stress.stage_rendezvous_id(6, 8) == "c06-retry8-20260811"
    assert stress.cohort_attempt_id(6, 8, 3) == "c06-retry8-20260811-attempt003"
    assert stress.cohort_rendezvous_id(6, 8, 3, 2) == "c06-retry8-20260811-attempt003-iris002"


def test_stress_preregistration_authorizes_only_its_frozen_stage(tmp_path, monkeypatch):
    launcher_path = Path(stress.__file__)
    rows = stress.rows_for_stage(6)
    preregistration = {
        "generation": 9,
        "analysis_scope": "operational_only_no_endpoint_metrics",
        "design": {
            "stage": 6,
            "optimizer_decay_steps": [row.optimizer_decay_step for row in rows],
            "support_ids": [row.support_id for row in rows],
            "training_seeds": [row.training_seed for row in rows],
            "cohort": {"coscheduling_group_by": stress.COHORT_COSCHEDULING_GROUP},
        },
        "implementation_sha256": {
            "stress_launcher": hashlib.sha256(launcher_path.read_bytes()).hexdigest(),
            "gradient_conflict_full": hashlib.sha256(Path(stress.full.__file__).read_bytes()).hexdigest(),
            "wsd80_surface": hashlib.sha256(Path(stress.base.__file__).read_bytes()).hexdigest(),
            "tests": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(preregistration))
    monkeypatch.setitem(stress.PREREGISTRATION_PATHS, 9, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert stress.validate_preregistration(digest, 9, 6) == preregistration
    with pytest.raises(ValueError, match="authorize this concurrency stage"):
        stress.validate_preregistration(digest, 9, 12)


def test_stress_overlap_measures_full_span_contention():
    overlap = runtime_gate._all_row_overlap(
        [
            {"first_host_timestamp": 0.0, "last_host_timestamp": 100.0, "runtime_span_seconds": 100.0},
            {"first_host_timestamp": 5.0, "last_host_timestamp": 100.0, "runtime_span_seconds": 95.0},
        ]
    )

    assert overlap == {
        "start": 5.0,
        "end": 100.0,
        "seconds": 95.0,
        "fraction_of_longest_span": 0.95,
        "longest_span_seconds": 100.0,
        "timeline": "host",
    }


def test_stress_rendezvous_worker_rejects_mismatched_release(tmp_path):
    row_ids = ("row-a",)
    fs, ready_dir, _, release_path = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    with fs.open(release_path, "w") as handle:
        json.dump(
            {
                "protocol_version": stress.RENDEZVOUS_PROTOCOL_VERSION,
                "ready_markers": {},
                "rendezvous_id": "wrong-id",
                "row_ids": list(row_ids),
            },
            handle,
        )

    with pytest.raises(RuntimeError, match="does not match"):
        stress._wait_for_stage_rendezvous(
            root=str(tmp_path),
            rendezvous_id="unit-test",
            row_id="row-a",
            row_ids=row_ids,
            worker_claim_id="/test/rendezvous/row-a",
            timeout_seconds=0.1,
            poll_seconds=0.001,
        )


def test_stress_rendezvous_times_out_without_every_row(tmp_path):
    with pytest.raises(TimeoutError, match="1/2 rows ready"):
        stress._wait_for_stage_rendezvous(
            root=str(tmp_path),
            rendezvous_id="unit-test",
            row_id="row-a",
            row_ids=("row-a", "row-b"),
            worker_claim_id="/test/rendezvous/row-a",
            timeout_seconds=0.02,
            poll_seconds=0.001,
        )


def test_stress_stage_allows_only_parent_managed_attempt_outputs(tmp_path, monkeypatch):
    marin_prefix = str(tmp_path / "marin")
    report_root = str(tmp_path / "reports")
    generation = 99
    monkeypatch.setattr(stress, "report_root", lambda _: report_root)
    stress._validate_resumable_stage_outputs(marin_prefix, 6, generation)

    valid = (
        tmp_path
        / "marin/tmp/ttl=14d/checkpoints"
        / stress.namespace_name(generation)
        / "stage-c06/row/version/attempt-003/checkpoints/step-1/file"
    )
    valid.parent.mkdir(parents=True)
    valid.write_text("partial attempt")
    stress._validate_resumable_stage_outputs(marin_prefix, 6, generation)

    invalid = valid.parents[3] / "unscoped-output.txt"
    invalid.write_text("invalid")
    with pytest.raises(ValueError, match="contains non-resumable state"):
        stress._validate_resumable_stage_outputs(marin_prefix, 6, generation)

    invalid.unlink()
    report_path = tmp_path / "reports/stage-c06.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}")
    with pytest.raises(ValueError, match="contains non-resumable state"):
        stress._validate_resumable_stage_outputs(marin_prefix, 6, generation)


def test_stress_cohort_tracks_start_and_completion_attempts(tmp_path):
    stage = 6
    generation = 99
    release = tmp_path / stress.cohort_rendezvous_id(stage, generation, 2, 1) / "release.json"
    release.parent.mkdir(parents=True)
    release.write_text("{}")
    completion_release = tmp_path / stress.completion_rendezvous_id(stage, generation, 2, 1) / "release.json"
    completion_release.parent.mkdir(parents=True)
    completion_release.write_text("{}")

    assert stress.released_cohort_executions(str(tmp_path), stage, generation) == ((2, 1),)
    assert stress.released_cohort_executions(str(tmp_path), stage, generation, completion=True) == ((2, 1),)


def test_stress_analyzer_resolves_the_latest_complete_parent_and_iris_attempt(monkeypatch):
    def releases(root, stage, generation, *, completion=False):
        del root, stage, generation
        return ((0, 0), (1, 0), (1, 1)) if not completion else ((1, 1),)

    monkeypatch.setattr(stress, "released_cohort_executions", releases)

    assert runtime_gate._resolve_cohort_execution(6, 11, None) == (1, 1)
    assert runtime_gate._resolve_cohort_execution(6, 11, 1) == (1, 1)


def test_stress_analyzer_rejects_a_newer_incomplete_execution(monkeypatch):
    def releases(root, stage, generation, *, completion=False):
        del root, stage, generation
        return ((0, 0), (1, 0)) if not completion else ((0, 0),)

    monkeypatch.setattr(stress, "released_cohort_executions", releases)

    with pytest.raises(ValueError, match="not the latest complete execution"):
        runtime_gate._resolve_cohort_execution(6, 11, None)


def test_stress_runtime_report_validates_rendezvous_evidence(tmp_path, monkeypatch):
    specs = runtime_gate._stress_specs(6)
    row_ids = tuple(spec.trajectory_id for spec in specs)
    monkeypatch.setattr(stress, "RENDEZVOUS_ROOT", str(tmp_path))
    fs, ready_dir, ready_paths, release_path = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    for index, (row_id, ready_path) in enumerate(zip(row_ids, ready_paths, strict=True)):
        with fs.open(ready_path, "w") as handle:
            json.dump(
                stress._ready_marker_payload(
                    worker_claim_id=f"/test/rendezvous/host-{index}",
                    row_id=row_id,
                    rendezvous_id="unit-test",
                    row_ids=row_ids,
                    physical_worker_id=(f"marin-tpu-v5p-preemptible-8-us-central1-20260813-0515-{index:08x}-worker-0"),
                    worker_region="us-central1",
                ),
                handle,
            )
    marker_identities = stress._ready_marker_identities(
        fs,
        ready_paths,
        rendezvous_id="unit-test",
        row_ids=row_ids,
    )
    with fs.open(release_path, "w") as handle:
        json.dump(
            {
                "protocol_version": stress.RENDEZVOUS_PROTOCOL_VERSION,
                "ready_markers": marker_identities,
                "rendezvous_id": "unit-test",
                "row_ids": list(row_ids),
            },
            handle,
        )

    evidence, failures = runtime_gate._rendezvous_evidence(
        "unit-test",
        specs,
        distinct_physical_workers_required=True,
        required_worker_region="us-central1",
        required_worker_id_regex=(r"^marin-tpu-v5p-preemptible-8-us-central1-[0-9]{8}-[0-9]{4}-[0-9a-f]+-worker-0$"),
    )

    assert not failures
    assert len(evidence["markers"]) == 6
    assert len({marker["physical_worker_id"] for marker in evidence["markers"]}) == 6
    assert evidence["release_after_last_ready_seconds"] is None


def test_stress_runtime_report_rejects_invalid_realized_placement(tmp_path, monkeypatch):
    specs = runtime_gate._stress_specs(6)
    row_ids = tuple(spec.trajectory_id for spec in specs)
    monkeypatch.setattr(stress, "RENDEZVOUS_ROOT", str(tmp_path))
    fs, ready_dir, ready_paths, release_path = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    for index, (row_id, ready_path) in enumerate(zip(row_ids, ready_paths, strict=True)):
        with fs.open(ready_path, "w") as handle:
            json.dump(
                stress._ready_marker_payload(
                    worker_claim_id=f"/test/rendezvous/host-{index}",
                    row_id=row_id,
                    rendezvous_id="unit-test",
                    row_ids=row_ids,
                    physical_worker_id="duplicate-worker" if index < 2 else f"worker-{index}",
                    worker_region="us-east5" if index == 5 else "us-central1",
                ),
                handle,
            )
    marker_identities = stress._ready_marker_identities(
        fs,
        ready_paths,
        rendezvous_id="unit-test",
        row_ids=row_ids,
    )
    with fs.open(release_path, "w") as handle:
        json.dump(
            {
                "protocol_version": stress.RENDEZVOUS_PROTOCOL_VERSION,
                "ready_markers": marker_identities,
                "rendezvous_id": "unit-test",
                "row_ids": list(row_ids),
            },
            handle,
        )

    _, failures = runtime_gate._rendezvous_evidence(
        "unit-test",
        specs,
        distinct_physical_workers_required=True,
        required_worker_region="us-central1",
        required_worker_id_regex=r"^marin-tpu-v5p-preemptible-8-us-central1-.+-worker-0$",
    )

    assert "rendezvous physical worker IDs are not unique" in failures
    assert "rendezvous workers landed outside us-central1: ['us-east5']" in failures
    assert any(failure.startswith("rendezvous worker identities do not match") for failure in failures)


def test_stress_runtime_report_rejects_duplicate_worker_claims(tmp_path, monkeypatch):
    specs = runtime_gate._stress_specs(6)
    row_ids = tuple(spec.trajectory_id for spec in specs)
    monkeypatch.setattr(stress, "RENDEZVOUS_ROOT", str(tmp_path))
    fs, ready_dir, ready_paths, release_path = stress._rendezvous_paths(str(tmp_path), "unit-test", row_ids)
    fs.makedirs(ready_dir, exist_ok=True)
    for row_id, ready_path in zip(row_ids, ready_paths, strict=True):
        with fs.open(ready_path, "w") as handle:
            json.dump(
                stress._ready_marker_payload(
                    worker_claim_id="/test/rendezvous/duplicate-child",
                    row_id=row_id,
                    rendezvous_id="unit-test",
                    row_ids=row_ids,
                ),
                handle,
            )
    marker_identities = stress._ready_marker_identities(
        fs,
        ready_paths,
        rendezvous_id="unit-test",
        row_ids=row_ids,
    )
    with fs.open(release_path, "w") as handle:
        json.dump(
            {
                "protocol_version": stress.RENDEZVOUS_PROTOCOL_VERSION,
                "ready_markers": marker_identities,
                "rendezvous_id": "unit-test",
                "row_ids": list(row_ids),
            },
            handle,
        )

    _, failures = runtime_gate._rendezvous_evidence("unit-test", specs)

    assert "rendezvous worker claim IDs are not unique" in failures


def test_output_inventory_separates_completed_partial_and_unexpected_roots():
    expected_terminal_steps = {"run-a": 9, "run-b": 19, "run-c": 29, "run-d": 39}
    clean = output_inventory.classify_objects((), expected_terminal_steps=expected_terminal_steps, version="v9")
    occupied = output_inventory.classify_objects(
        (
            "run-a/v9/.executor_info",
            "run-a/v9/checkpoints/step-1/manifest.ocdbt",
            "run-b/v9/.executor_info",
            "run-b/v9/checkpoints/step-19/metadata.json",
            "run-c/v9/.executor_info",
            "run-b/old/checkpoints/step-1/manifest.ocdbt",
            "not-frozen/v9/executor_state.json",
        ),
        expected_terminal_steps=expected_terminal_steps,
        version="v9",
        owned_expected_roots=frozenset({"run-a/v9", "run-b/v9", "run-c/v9"}),
        completed_expected_roots=frozenset({"run-b/v9"}),
    )

    assert asdict(clean) == {
        "expected_root_count": 4,
        "empty_root_count": 4,
        "bookkeeping_root_count": 0,
        "resumable_root_count": 0,
        "completed_root_count": 0,
        "partial_root_count": 0,
        "unexpected_root_count": 0,
        "nonempty_expected_roots": (),
        "bookkeeping_expected_roots": (),
        "resumable_expected_roots": (),
        "completed_expected_roots": (),
        "partial_expected_roots": (),
        "unexpected_roots": (),
    }
    assert occupied.empty_root_count == 1
    assert occupied.partial_root_count == 1
    assert occupied.bookkeeping_root_count == 1
    assert occupied.completed_root_count == 1
    assert occupied.unexpected_root_count == 2
    assert occupied.nonempty_expected_roots == ("run-a/v9", "run-b/v9", "run-c/v9")
    assert occupied.bookkeeping_expected_roots == ("run-c/v9",)
    assert occupied.completed_expected_roots == ("run-b/v9",)
    assert occupied.partial_expected_roots == ("run-a/v9",)


def test_output_inventory_admits_only_owned_checkpointed_roots_as_resumable():
    paths = (
        "run-owned/v9/.executor_info",
        "run-owned/v9/__temporary_state__",
        "run-foreign/v9/.executor_info",
        "run-foreign/v9/__temporary_state__",
    )

    inventory = output_inventory.classify_objects(
        paths,
        expected_terminal_steps={"run-owned": 9, "run-foreign": 9},
        version="v9",
        owned_expected_roots=frozenset({"run-owned/v9"}),
        resumable_expected_roots=frozenset({"run-owned/v9", "run-foreign/v9"}),
    )

    assert inventory.resumable_expected_roots == ("run-owned/v9",)
    assert inventory.partial_expected_roots == ("run-foreign/v9",)


def test_output_inventory_ignores_incomplete_temporary_checkpoint_after_owned_bookkeeping():
    inventory = output_inventory.classify_objects(
        ("run-owned/v9/.executor_info",),
        expected_terminal_steps={"run-owned": 9},
        version="v9",
        owned_expected_roots=frozenset({"run-owned/v9"}),
    )

    assert inventory.bookkeeping_expected_roots == ("run-owned/v9",)
    assert inventory.partial_root_count == 0


def test_full_training_emits_exact_output_owner_identity():
    _, all_trajectories, _ = full.load_design()
    trajectory_id = all_trajectories[0].trajectory_id
    trajectories, steps = full.build_training_steps(
        marin_prefix="gs://marin-us-central1",
        tpu_type=full.base.DEFAULT_TPU_TYPE,
        tpu_region=full.base.DEFAULT_TPU_REGION,
        tpu_zone=full.base.DEFAULT_TPU_ZONE,
        selected_runs=frozenset({trajectory_id}),
    )

    owners = full.expected_output_owners(trajectories, steps, marin_prefix="gs://marin-us-central1")
    owner = owners[f"{trajectory_id}/{full.VERSION}"]

    assert owner["executor_info"]["executor_version"] == "step_runner"
    assert owner["artifact_record"]["version"] == full.VERSION
    assert owner["artifact_record"]["output_path"].endswith(f"/{trajectory_id}/{full.VERSION}")
    assert owner["artifact_record"]["result_type"]
    assert owner["artifact_record"]["fingerprint_payload"]


def test_output_inventory_rejects_partial_artifact_records():
    expected = {
        "name": "trajectory",
        "version": "review-v9",
        "fingerprint": "frozen",
        "result_type": "LevanterCheckpoint",
        "output_path": "gs://example/trajectory/review-v9",
        "deps": [],
        "dep_paths": [],
        "source": None,
        "result": None,
        "fingerprint_payload": {"scientific": "identity"},
    }
    complete = {**expected, "config": {"num_train_steps": 100}, "provenance": {"git": "sha"}}

    assert output_inventory._artifact_record_matches(complete, expected)
    assert not output_inventory._artifact_record_matches(expected, expected)
    assert not output_inventory._artifact_record_matches({**complete, "config": None}, expected)
    assert not output_inventory._artifact_record_matches({**complete, "fingerprint": "foreign"}, expected)


def test_output_inventory_rejects_success_without_terminal_checkpoint():
    assert (
        output_inventory._root_state(
            terminal_step=9,
            permanent_steps={5},
            temporary_steps=set(),
            artifact_valid=True,
            status_success=True,
        )
        is output_inventory.RootState.INVALID
    )
    assert (
        output_inventory._root_state(
            terminal_step=9,
            permanent_steps={5},
            temporary_steps={7},
            artifact_valid=False,
            status_success=False,
        )
        is output_inventory.RootState.RESUMABLE
    )
    assert (
        output_inventory._root_state(
            terminal_step=9,
            permanent_steps={9},
            temporary_steps=set(),
            artifact_valid=True,
            status_success=True,
        )
        is output_inventory.RootState.COMPLETED
    )


def test_runtime_gate_uses_dense_throughput_only_history(monkeypatch):
    class FakeRun:
        id = "runtime-only"
        state = "finished"
        url = "https://wandb.ai/runtime-only"
        created_at = "2026-08-11T00:00:00Z"

        def __init__(self):
            self.summary = {"global_step": 1_279}
            total_nemotron = sum(stress.base.NEMOTRON_TOKEN_COUNTS.values())

            def phase_weights(starcoder_weight):
                weights = {
                    f"nemotron_cc/{split}-llama3": (1.0 - starcoder_weight) * count / total_nemotron
                    for split, count in stress.base.NEMOTRON_TOKEN_COUNTS.items()
                }
                weights["dolma/starcoder"] = starcoder_weight
                return weights

            phase_0 = phase_weights(0.02)
            phase_1 = phase_weights(0.82)
            self.config = {
                "data_seed": 123,
                "optimizer_schedule_num_train_steps": None,
                "optimizer": {"decay": 256},
                "trainer": {
                    "num_train_steps": 1_280,
                    "seed": 123,
                    "distributed": {"initialize_jax_distributed": False},
                },
                "data": {
                    "train_weights": [[0, phase_0], [768, phase_1]],
                    "mixture_block_size": 2_048,
                    "max_train_batches": {"dolma/starcoder": 1_068},
                    "max_train_batches_start": {"dolma/starcoder": 0},
                    "max_train_batches_subset_seed": 456,
                    "train_holdout_sequences": {name: 4_096 for name in full.EXPECTED_TRAINING_COMPONENT_NAMES},
                    "train_holdout_seed": 789,
                    "train_holdout_partition": "random_sparse_swap",
                    "permutation_type": "feistel",
                    "experiment_budget": None,
                    "target_budget": None,
                    "simulated_epoch_subset_seed": None,
                },
            }

        def scan_history(self, *, keys, page_size):
            assert tuple(keys) == runtime_gate.HISTORY_KEYS
            assert not any("eval" in key for key in keys)
            assert page_size == 10_000
            for step in range(1_280):
                yield {
                    "_runtime": 100.0 + step * 0.351,
                    "_timestamp": 1_700_000_000.0 + step,
                    "global_step": step,
                    "throughput/loading_time": 0.001,
                    "throughput/duration": 0.35,
                    "throughput/tokens_per_second": 730_000.0,
                    "throughput/mfu": 35.0,
                }
                if step == 100:
                    yield {"global_step": step, "some_eval_metric": 1.0}

    class FakeApi:
        def run(self, path):
            assert path == "marin-community/marin/runtime-only"
            return FakeRun()

    monkeypatch.setattr(runtime_gate, "_checkpoint_steps", lambda _: (768, 1_279))
    monkeypatch.setattr(runtime_gate, "_gcloud_size", lambda _: 2_337_100_000)
    spec = runtime_gate.RunSpec(
        trajectory_id="runtime-only",
        wandb_run_id="runtime-only",
        support_id="m100a",
        terminal_step=1_279,
        checkpoint_root="gs://marin-us-central1/checkpoints/runtime-only",
        expected_checkpoint_steps=(768, 1_279),
        event_steps=(("data_switch", 768), ("optimizer_decay", 1_024)),
        event_window_steps=128,
        total_steps=1_280,
        data_switch_step=768,
        optimizer_decay_step=1_024,
        phase_0_starcoder=0.02,
        phase_1_starcoder=0.82,
        training_seed=123,
        support_batches=1_068,
        support_start_batches=0,
        support_pool_seed=456,
        train_holdout_sequences_per_component=4_096,
        train_holdout_seed=789,
        train_holdout_partition="random_sparse_swap",
    )

    result = runtime_gate._evaluate_run(spec, FakeApi())

    assert result["status"] == "pass"
    assert result["history_coverage_fraction"] == 1.0
    assert all(event["pre_history_coverage_fraction"] == 1.0 for event in result["events"])
    assert all(event["post_history_coverage_fraction"] == 1.0 for event in result["events"])


def test_runtime_gate_deduplicates_exact_wandb_page_boundary_rows():
    row = {
        "_step": 0,
        "_runtime": 1.0,
        "_timestamp": 1.0,
        "global_step": 0,
        "throughput/loading_time": 0.001,
        "throughput/duration": 0.36,
        "throughput/tokens_per_second": 720_000.0,
        "throughput/mfu": 34.0,
    }

    class PaginatedRun:
        id = "paginated"
        created_at = "2026-08-11T00:00:00Z"

        @staticmethod
        def scan_history(**_):
            return iter((row, dict(row)))

    assert runtime_gate._wandb_history(PaginatedRun())["global_step"].tolist() == [0]


def test_runtime_gate_rejects_conflicting_duplicate_training_steps():
    class RestartedRun:
        id = "restarted"
        created_at = "2026-08-11T00:00:00Z"

        def scan_history(self, *, keys, page_size):
            del keys, page_size
            for index, step in enumerate((0, 1, 0)):
                yield {
                    "_runtime": 10.0 + index,
                    "_timestamp": 1_700_000_000.0 + step + index,
                    "global_step": step,
                    "throughput/loading_time": 0.001,
                    "throughput/duration": 0.35,
                    "throughput/tokens_per_second": 730_000.0,
                    "throughput/mfu": 35.0,
                }

    with pytest.raises(ValueError, match="Conflicting duplicate global_step 0"):
        runtime_gate._wandb_history(RestartedRun())


def test_runtime_gate_rejects_iris_preemption():
    clean = {
        "job": "/calvinxu/test",
        "state": "succeeded",
        "exit": "0",
        "failures": 0,
        "preemptions": 0,
        "completed_tasks": 1,
        "total_tasks": 1,
        "running_tasks": 0,
    }
    assert runtime_gate._iris_failures(clean) == []
    assert runtime_gate._iris_failures({**clean, "preemptions": 1})
    assert runtime_gate._iris_failures({**clean, "preemptions": 1}, allow_preemptions=True) == []


def test_runtime_gate_accepts_only_preemption_only_abandoned_cohorts():
    base = {
        "state": "failed",
        "exit": "1",
        "failures": 0,
        "preemptions": 1,
        "completed_tasks": 0,
        "total_tasks": 6,
        "running_tasks": 0,
    }
    children = [
        {
            **base,
            "job": "/calvinxu/parent/stage-c06-cohort-g11-attempt-000",
        },
        {
            **base,
            "job": "/calvinxu/parent/stage-c06-cohort-g11-attempt-001",
            "state": "succeeded",
            "exit": "0",
            "preemptions": 0,
            "completed_tasks": 6,
        },
    ]

    final, failures = runtime_gate._stress_cohort_child_evidence(
        children,
        stage=6,
        generation=11,
        final_attempt=1,
        replicas=6,
    )

    assert failures == []
    assert final == children[1]
    _, failures = runtime_gate._stress_cohort_child_evidence(
        [{**children[0], "failures": 1}, children[1]],
        stage=6,
        generation=11,
        final_attempt=1,
        replicas=6,
    )
    assert any("not preemption-only" in failure for failure in failures)


def test_forced_preemption_gate_requires_an_abandoned_attempt():
    assert runtime_gate._forced_preemption_recovery_failures(required=False, final_attempt=0) == []
    assert runtime_gate._forced_preemption_recovery_failures(required=True, final_attempt=1) == []
    assert runtime_gate._forced_preemption_recovery_failures(required=True, final_attempt=0) == [
        "forced-preemption gate did not replace an abandoned cohort attempt"
    ]


def test_runtime_gate_requires_every_event_window_to_run_at_full_concurrency():
    rows: list[dict[str, Any]] = [
        {
            "trajectory_id": "early",
            "events": [
                {
                    "event": "data_switch",
                    "pre_wallclock_start": 100.0,
                    "pre_wallclock_end": 119.0,
                    "post_wallclock_start": 120.0,
                    "post_wallclock_end": 180.0,
                }
            ],
        },
        {
            "trajectory_id": "late",
            "events": [
                {
                    "event": "optimizer_decay",
                    "pre_wallclock_start": 200.0,
                    "pre_wallclock_end": 249.0,
                    "post_wallclock_start": 250.0,
                    "post_wallclock_end": 310.0,
                }
            ],
        },
    ]

    assert runtime_gate._event_overlap_failures(rows, 100.0, 300.0) == [
        "late optimizer_decay post-window [250.000, 310.000] is outside all-row overlap [100.000, 300.000]"
    ]
    assert runtime_gate._event_overlap_failures(rows, 100.0, 320.0) == []

    rows[0]["events"][0]["pre_wallclock_start"] = 99.0
    assert runtime_gate._event_overlap_failures(rows, 100.0, 320.0) == [
        "early data_switch pre-window [99.000, 119.000] is outside all-row overlap [100.000, 320.000]"
    ]


def _stress_gate_rows(
    *,
    below_floor_positions,
    minimum_steps=None,
    decay_steps=None,
    analyzable_fractions=None,
):
    minimum_steps = minimum_steps or [1_000] * len(below_floor_positions)
    decay_steps = decay_steps or [1_000] * len(below_floor_positions)
    analyzable_fractions = analyzable_fractions or [1.0] * len(below_floor_positions)
    return [
        {
            "trajectory_id": f"row-{index}",
            "short_window_tokens_per_second_min": 650_000.0,
            "short_window_minimum_center_step": minimum_step,
            "short_window_longest_below_floor_positions": positions,
            "short_window_analyzable_fraction": analyzable_fraction,
            "causal_window_minimum_center_step": minimum_step,
            "optimizer_decay_step": decay_step,
        }
        for index, (positions, minimum_step, decay_step, analyzable_fraction) in enumerate(
            zip(
                below_floor_positions,
                minimum_steps,
                decay_steps,
                analyzable_fractions,
                strict=True,
            )
        )
    ]


def test_stress_short_window_gate_accepts_one_sustained_depression():
    rows = _stress_gate_rows(below_floor_positions=[97, 0, 0, 0, 0, 0])

    diagnostics, failures = runtime_gate._stress_short_window_gate(rows)

    assert failures == []
    assert diagnostics["status"] == "pass"
    assert diagnostics["exceedance_count"] == 1


def test_stress_short_window_gate_rejects_multiple_sustained_depressions():
    rows = _stress_gate_rows(below_floor_positions=[97, 97, 0, 0, 0, 0])

    diagnostics, failures = runtime_gate._stress_short_window_gate(rows)

    assert diagnostics["status"] == "fail"
    assert len(failures) == 1


def test_stress_short_window_gate_rejects_fragmented_history():
    rows = _stress_gate_rows(
        below_floor_positions=[0] * 6,
        analyzable_fractions=[0.94, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    diagnostics, failures = runtime_gate._stress_short_window_gate(rows)

    assert diagnostics["status"] == "fail"
    assert len(failures) == 1


def test_stress_decay_alignment_classifies_event_following_and_common_step():
    rows = _stress_gate_rows(
        below_floor_positions=[0] * 6,
        decay_steps=list(stress.C6_OPTIMIZER_DECAY_STEPS),
    )
    for row, support_id in zip(rows, ["m100a"] * 4 + ["full", "m100b"], strict=True):
        row["support_id"] = support_id

    diagnostic = runtime_gate._stress_decay_alignment_diagnostic(rows, stage=6)

    assert diagnostic["classification"] == "underpowered"
    assert diagnostic["primary_row_count"] == 4


def _c12_decay_rows(*, aligned: bool) -> list[dict[str, Any]]:
    steps = np.arange(stress.TOTAL_STEPS)
    rows: list[dict[str, Any]] = []
    for index, onset in enumerate(stress.C12_PRIMARY_OPTIMIZER_DECAY_STEPS):
        throughput = np.full(stress.TOTAL_STEPS, 700_000.0)
        if aligned:
            throughput[onset : onset + runtime_gate.C12_ASSIGNMENT_WINDOW_STEPS] = 600_000.0
        rows.append(
            {
                "trajectory_id": f"c12_{index:03d}_m100a",
                "support_id": "m100a",
                "optimizer_decay_step": onset,
                "_runtime_series": {
                    "global_step": steps,
                    "throughput/tokens_per_second": throughput,
                },
            }
        )
    return rows


def test_c12_exact_onset_assignment_detects_aligned_slowdown():
    diagnostic = runtime_gate._stress_decay_alignment_diagnostic(_c12_decay_rows(aligned=True), stage=12)

    assert diagnostic["classification"] == "decay_aligned"
    assert diagnostic["primary"]["assignment_count"] == 2_520
    assert diagnostic["primary"]["exact_p_value"] == pytest.approx(1 / 2_520)
    assert diagnostic["pretrend_falsification"]["exact_p_value"] > runtime_gate.C12_ASSIGNMENT_ALPHA
    assert diagnostic["lead_placebo_falsification"]["exact_p_value"] > runtime_gate.C12_ASSIGNMENT_ALPHA


def test_c12_exact_onset_assignment_reports_no_alignment_on_flat_histories():
    diagnostic = runtime_gate._stress_decay_alignment_diagnostic(_c12_decay_rows(aligned=False), stage=12)

    assert diagnostic["classification"] == "no_detectable_decay_alignment"
    assert diagnostic["primary"]["exact_p_value"] == 1.0


def test_c12_exact_onset_assignment_reports_no_alignment_for_row_specific_noise():
    rows = _c12_decay_rows(aligned=False)
    for index, row in enumerate(rows):
        rng = np.random.default_rng(10_000 + index)
        row["_runtime_series"]["throughput/tokens_per_second"] += rng.normal(0.0, 2_000.0, stress.TOTAL_STEPS)

    diagnostic = runtime_gate._stress_decay_alignment_diagnostic(rows, stage=12)

    assert diagnostic["classification"] == "no_detectable_decay_alignment"
    assert diagnostic["primary"]["exact_p_value"] > runtime_gate.C12_ASSIGNMENT_ALPHA


def test_c12_exact_onset_assignment_prioritizes_falsification_failure():
    rows = _c12_decay_rows(aligned=False)
    for row in rows:
        onset = row["optimizer_decay_step"]
        row["_runtime_series"]["throughput/tokens_per_second"][onset - 64 : onset] = 600_000.0

    diagnostic = runtime_gate._stress_decay_alignment_diagnostic(rows, stage=12)

    assert diagnostic["classification"] == "falsification_failed"
    assert diagnostic["pretrend_falsification"]["exact_p_value"] <= runtime_gate.C12_ASSIGNMENT_ALPHA


def test_c12_diagnostic_unavailability_does_not_raise_on_incomplete_history():
    rows = _c12_decay_rows(aligned=False)
    series = rows[0]["_runtime_series"]
    keep = series["global_step"] != stress.C12_PRIMARY_ONSET_MULTISET[0]
    series["global_step"] = series["global_step"][keep]
    series["throughput/tokens_per_second"] = series["throughput/tokens_per_second"][keep]

    diagnostic = runtime_gate._safe_stress_decay_alignment_diagnostic(rows, stage=12)

    assert diagnostic["classification"] == "unavailable"
    assert "complete contiguous histories" in diagnostic["reason"]
    assert "never overrides" in diagnostic["gate_role"]


def test_runtime_report_upload_is_create_only_and_byte_verified(tmp_path, monkeypatch):
    local = tmp_path / "report.json"
    remote = tmp_path / "remote.json"
    local.write_text('{"status":"pass"}\n')

    def copy_report(command, *, check):
        assert check is True
        assert command[:4] == ["gcloud", "storage", "cp", "--if-generation-match=0"]
        Path(command[-1]).write_bytes(Path(command[-2]).read_bytes())

    monkeypatch.setattr(runtime_gate.subprocess, "run", copy_report)

    evidence = runtime_gate._upload(local, str(remote))

    assert evidence["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert evidence["generation"] is None


def _synchronized_gate_rows(depressed_bins: int):
    timestamps = np.arange(400, dtype=float) * runtime_gate.STRESS_SYNCHRONIZED_BIN_SECONDS
    throughput = np.full(400, 700_000.0)
    throughput[:depressed_bins] = 600_000.0
    return [
        {
            "trajectory_id": f"row-{index}",
            "first_host_timestamp": 0.0,
            "last_host_timestamp": float(timestamps[-1]),
            "runtime_span_seconds": float(timestamps[-1]),
            "tokens_per_second_p50": 700_000.0,
            "pause_accounting": {"intervals": []},
            "_runtime_series": {
                "host_timestamp": timestamps,
                "throughput/tokens_per_second": throughput,
            },
        }
        for index in range(6)
    ]


def test_stress_synchronized_depression_gate_uses_strict_duration_boundary():
    passing, passing_failures = runtime_gate._stress_synchronized_depression_gate(
        _synchronized_gate_rows(depressed_bins=3)
    )
    failing, failing_failures = runtime_gate._stress_synchronized_depression_gate(
        _synchronized_gate_rows(depressed_bins=4)
    )

    assert passing_failures == []
    assert passing["longest_depressed_seconds"] == runtime_gate.STRESS_SYNCHRONIZED_MAX_SECONDS
    assert failing["longest_depressed_seconds"] == runtime_gate.STRESS_SYNCHRONIZED_MAX_SECONDS + 12.0
    assert len(failing_failures) == 1


def test_pause_accounting_separates_declared_checkpoint_and_unexplained_gaps():
    series = {
        "global_step": np.array([1, 2, 3, 4]),
        "host_timestamp": np.array([0.0, 3.0, 4.0, 11.0]),
        "throughput/duration": np.ones(4),
        "throughput/loading_time": np.zeros(4),
    }

    pauses = runtime_gate._pause_intervals(series, expected_checkpoint_steps=(2,))
    summary, failures = runtime_gate._pause_summary(pauses, expected_checkpoint_steps=(2,))

    assert [(pause.right_step, pause.classification, pause.seconds) for pause in pauses] == [
        (2, "checkpoint", 2.0),
        (4, "unexplained", 6.0),
    ]
    assert summary["checkpoint_seconds_by_step"] == {"2": 2.0}
    assert summary["unexplained_longest_seconds"] == 6.0
    assert failures == ["longest unexplained pause 6.000s > 5.000s"]


def test_synchronized_gate_excludes_declared_checkpoint_bins_without_compressing_time():
    rows = _synchronized_gate_rows(depressed_bins=0)
    for row in rows:
        row["pause_accounting"] = {
            "intervals": [
                {"start": 24.0, "stop": 36.0, "classification": "checkpoint"},
            ]
        }

    diagnostic, failures = runtime_gate._stress_synchronized_depression_gate(rows)

    assert failures == []
    assert diagnostic["excluded_checkpoint_bins"] == 1
    assert diagnostic["analyzed_bins"] == 398


def test_post_event_slowdown_does_not_cross_missing_history_steps(monkeypatch):
    series = {
        "global_step": np.array([0, 1, 2, 3, 10, 11, 12, 13]),
        "throughput/tokens_per_second": np.array([80.0] * 8),
    }
    monkeypatch.setattr(runtime_gate, "STRESS_EVENT_SLOWDOWN_ROLLING_WINDOW_STEPS", 2)
    slowdown = runtime_gate._post_event_slowdown_positions(
        series,
        event_step=0,
        stop_step=14,
        pre_tokens_per_second_p50=100.0,
    )

    assert slowdown.longest_positions == 3
    assert slowdown.analyzable_positions == 6
    assert slowdown.possible_positions == 13


@pytest.mark.parametrize(
    ("task_line", "completed_tasks", "running_tasks"),
    [
        ("Tasks: 1/1 completed  succeeded=1", 1, 0),
        ("Tasks: 0/1 completed  running=1", 0, 1),
    ],
)
def test_runtime_gate_parses_running_and_terminal_iris_summaries(task_line, completed_tasks, running_tasks):
    output = "\n".join(
        (
            "Job: /calvinxu/test (/calvinxu/test)",
            "State: succeeded  exit=0  failures=0  preemptions=0",
            task_line,
        )
    )

    summary = runtime_gate._parse_iris_summary("/calvinxu/test", output)

    assert summary["completed_tasks"] == completed_tasks
    assert summary["running_tasks"] == running_tasks
