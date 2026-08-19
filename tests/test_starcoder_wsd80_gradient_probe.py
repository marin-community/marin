# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import csv
import json
import math
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.schedule import BatchSchedule, ScheduleStep
from marin.run.iris_run import _should_stage

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as repair
from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as probe
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_starcoder_wsd80_gradient_mechanism_repair_20260818 as repair_analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_starcoder_wsd80_gradient_probe_canary_20260816 as canary_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as repair_freeze,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as freeze,
)


class IntegerDataset(AsyncDataset[int]):
    def __init__(self, length: int):
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices):
        return [int(index) for index in indices]


def _read_manifest(name: str) -> list[dict[str, str]]:
    with (freeze.OUTPUT_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _launch_command_patterns(name: str) -> tuple[list[str], list[str]]:
    path = freeze.OUTPUT_DIR.parent / "starcoder_wsd80_gradient_probe_preflight_20260816" / name
    tokens = shlex.split(path.read_text().replace("\\\n", " "))
    excludes = [tokens[index + 1] for index, token in enumerate(tokens) if token == "--working-dir-exclude"]
    includes = [tokens[index + 1] for index, token in enumerate(tokens) if token == "--working-dir-include"]
    return excludes, includes


def test_frozen_release_has_unique_expected_rows_and_hashes():
    release = json.loads((freeze.OUTPUT_DIR / "release.json").read_text())

    assert release["release_sha256"] == freeze.canonical_sha256({**release, "release_sha256": ""})
    assert release["endpoint_metrics_read"] is False
    assert release["artifact_triggered_async_readiness_implemented"] is False
    assert release["checkpoint_readiness_semaphore_limit"] == 64
    assert release["checkpoint_readiness_executor"] == "asyncio_default_thread_pool"
    assert release["full_launch_authorized"] is False
    assert freeze.RESULT_ROOT.endswith("starcoder_wsd80_gradient_probe_review_v9_release_v6_20260816")
    assert release["canary_checkpoint_count_reconciliation"] == {
        "source_design_expected_permanent_checkpoint_count": 13,
        "frozen_unique_permanent_checkpoint_count": 14,
        "explanation": (
            "The source design's count of 13 is stale. Two canary seeds each contribute seven distinct permanent "
            "checkpoint coordinates, so the execution manifests correctly require 14."
        ),
    }
    assert probe._artifact_name("canary", "probe", "group") in f"{freeze.RESULT_ROOT}/canary/probe/group"
    for relative_path, sha256 in release["implementation_files"].items():
        assert freeze.file_sha256(freeze.REPO_ROOT / relative_path) == sha256
    expected = {
        "canary_probe": (112, 14),
        "canary_optimizer": (42, 6),
        "canary_rollout": (14, 2),
        "full_probe": (19_264, 2_240),
        "full_optimizer": (448, 64),
        "full_rollout": (476, 68),
    }
    for name, (row_count, group_count) in expected.items():
        summary = release["manifests"][name]
        path = freeze.REPO_ROOT / summary["path"]
        rows = _read_manifest(f"{name}_manifest.csv")
        assert len(rows) == row_count
        assert len({row["row_id"] for row in rows}) == row_count
        assert len({row["group_id"] for row in rows}) == group_count
        assert freeze.file_sha256(path) == summary["sha256"]


@pytest.mark.parametrize(
    "command_name",
    [
        "canary_launch_command_v6_retry3.txt",
        "full_launch_command_v6.txt",
    ],
)
def test_launch_stages_every_local_provenance_dependency(command_name: str):
    excludes, includes = _launch_command_patterns(command_name)
    required = (
        "experiments/domain_phase_mix/starcoder_wsd80_gradient_probe.py",
        "experiments/domain_phase_mix/exploratory/two_phase_many/freeze_starcoder_wsd80_gradient_probe_20260816.py",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_gradient_conflict_design_20260810/design_manifest.json",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_gradient_conflict_design_20260811/support_partition_audit.json",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_gradient_conflict_design_20260811_v9/design_manifest.json",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_matched_nd_stage1_20260731/confirmation_design_20260801/design_manifest.json",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_matched_nd_stage1_20260731/confirmation_results_20260801/cell_confirmation_summary.csv",
        "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
        "starcoder_wsd80_gradient_probe_release_v6_20260816/release.json",
    )

    assert all(_should_stage(path, excludes, includes) for path in required)


def test_frozen_cache_provenance_pins_shard_inventory_and_completion_objects():
    rows = _read_manifest("cache_provenance.csv")
    expected_target_sequence_counts = {
        "paloma_programming_languages": 4_151,
        "paloma_c4_en": 472,
        "uncheatable_github_python": 457,
        "uncheatable_wikipedia_english": 283,
    }

    assert len(rows) == 11
    assert len({row["component_name"] for row in rows}) == 11
    assert all(row["shard_ledger_sha256"] for row in rows)
    assert all(row["completion_sha256"] for row in rows)
    for row in rows:
        if row["role"] == "training_source":
            assert row["split"] == "train"
            assert row["completion_uri"].endswith("/.executor_status")
        else:
            assert row["split"] == "validation"
            assert row["completion_uri"].endswith("/validation/.stats.json")
            assert int(row["total_elements"]) > 0
            assert int(row["total_tokens"]) > 0
            assert int(row["materialized_sequence_length"]) == 2_048
            assert (
                int(row["materialized_sequence_count"]) == expected_target_sequence_counts[row["target_distribution_id"]]
            )
            assert int(row["maximum_unique_full_probe_blocks"]) == int(row["materialized_sequence_count"]) // 64


def test_target_probe_rows_never_exceed_frozen_finite_population():
    release = json.loads((freeze.OUTPUT_DIR / "release.json").read_text())
    contract = release["target_sampling_contract"]

    for scope in ("canary", "full"):
        rows = _read_manifest(f"{scope}_probe_manifest.csv")
        for row in rows:
            distribution = row["distribution_id"]
            if distribution not in contract:
                continue
            required = int(row["replicate_blocks"]) * int(row["sequences_per_block"])
            assert row["target_sampling_mode"] == "seeded_feistel_shuffle_without_replacement"
            assert required <= int(row["target_available_sequence_count"])
            assert int(row["replicate_blocks"]) <= int(row["design_replicate_blocks"])


def test_target_reference_identities_are_shared_across_probe_kinds():
    for scope in ("canary", "full"):
        probe_rows = _read_manifest(f"{scope}_probe_manifest.csv")
        expected_by_trajectory: dict[str, dict[str, str]] = {}
        for row in probe_rows:
            if row["distribution_id"] not in probe.TARGET_DISTRIBUTIONS:
                continue
            expected_by_trajectory.setdefault(row["trajectory_id"], {})[row["distribution_id"]] = row[
                "probe_sequence_set_id"
            ]
        for kind in ("optimizer", "rollout"):
            for row in _read_manifest(f"{scope}_{kind}_manifest.csv"):
                assert (
                    json.loads(row["target_sequence_set_ids_json"])
                    == expected_by_trajectory[row["parent_trajectory_id"]]
                )


def test_all_probe_distributions_have_runtime_routes():
    rows = _read_manifest("full_probe_manifest.csv")
    observed = {row["distribution_id"] for row in rows}

    assert observed == probe.SOURCE_DISTRIBUTIONS | probe.TARGET_DISTRIBUTIONS
    assert set(probe.LEAF_DISTRIBUTIONS) <= probe.SOURCE_DISTRIBUTIONS


def test_shifted_restart_dataset_wraps_deterministically():
    dataset = probe.ShiftedRestartDataset(IntegerDataset(5), start=4, length=5)

    assert asyncio.run(dataset.get_batch([0, 1, 2, 6])) == [4, 0, 1, 0]
    assert asyncio.run(dataset.async_len()) == 5


def test_restart_nemotron_aggregate_is_iterable_without_outer_permutation():
    Pos = Axis("position", 1)
    aggregate = MixtureDataset(
        datasets={"a": IntegerDataset(8), "b": IntegerDataset(8)},
        weights={"a": 0.5, "b": 0.5},
        block_size=8,
        key=jrandom.PRNGKey(13),
    )

    wrapped = probe._distribution_dataset(
        "nemotron_aggregate",
        sequence_set_id="frozen-aggregate",
        train_config=SimpleNamespace(),
        Pos=Pos,
        sources={"nemotron_aggregate": aggregate},
    )

    assert wrapped.dataset is aggregate
    assert len(asyncio.run(wrapped.dataset.get_batch([0, 1, 2]))) == 3


def test_logical_component_offset_matches_materialized_mixture_blocks():
    mixture = MixtureDataset(
        datasets={"a": IntegerDataset(100), "b": IntegerDataset(100)},
        weights={"a": 0.35, "b": 0.65},
        key=jrandom.PRNGKey(17),
        block_size=16,
    )

    for offset in (0, 1, 15, 16, 17, 63, 64, 91):
        materialized = [mixture._get_block(block) for block in range((offset + 15) // 16)]
        prefix = [int(value) for block in materialized for value in block][:offset]
        expected = sum((value >> 16) == list(mixture.dataset_index).index("a") for value in prefix)
        assert probe._logical_component_offset(mixture, "a", offset) == expected


def test_logical_component_offset_respects_step_to_sequence_schedule_rescaling():
    batch_schedule = BatchSchedule([ScheduleStep(start=0, value=3), ScheduleStep(start=5, value=5)])
    step_weights = [(0, {"a": 0.9, "b": 0.1}), (4, {"a": 0.2, "b": 0.8})]
    sequence_weights = rescale_mixture_schedule_for_batch_schedule(step_weights, batch_schedule)
    mixture = MixtureDataset(
        datasets={"a": IntegerDataset(100), "b": IntegerDataset(100)},
        weights=sequence_weights,
        key=jrandom.PRNGKey(23),
        block_size=4,
    )
    sequence_offset = batch_schedule.global_data_offset_by_step(7)
    materialized = [mixture._get_block(block) for block in range((sequence_offset + 3) // 4)]
    prefix = [int(value) for block in materialized for value in block][:sequence_offset]
    expected = sum((value >> 16) == list(mixture.dataset_index).index("a") for value in prefix)

    assert sequence_weights[1][0] == batch_schedule.global_data_offset_by_step(4)
    assert probe._logical_component_offset(mixture, "a", sequence_offset) == expected


def test_create_only_rows_skip_identical_identity_and_reject_drift(tmp_path):
    path = str(tmp_path / "rows" / "row.json")

    assert probe._write_create_only_json(path, {"value": 1}, identity_sha256="abc") == "created"
    assert probe._write_create_only_json(path, {"value": 2}, identity_sha256="abc") == "skipped_existing"
    with pytest.raises(RuntimeError, match="another row identity"):
        probe._write_create_only_json(path, {"value": 1}, identity_sha256="different")


def test_partial_group_without_marker_recognizes_completed_rows(tmp_path):
    row = {
        "row_id": "row",
        "group_id": "group",
        "checkpoint_uri": "gs://bucket/step-7",
        "train_config_sha256": "config",
    }
    output_path = str(tmp_path / "group")
    release = "release"
    probe._write_create_only_json(
        probe._row_path(output_path, row["row_id"]),
        {"row": row},
        identity_sha256=probe._row_identity(row, release),
    )

    assert probe._assert_existing_row_identity(output_path, row, release)
    assert not (tmp_path / "group" / "group_complete.json").exists()


def test_checkpoint_metadata_must_match_step_and_be_permanent(tmp_path):
    checkpoint = tmp_path / "step-7"
    checkpoint.mkdir()
    metadata = checkpoint / "metadata.json"
    metadata.write_text(json.dumps({"step": 7, "is_temporary": False}))

    assert probe._read_checkpoint_metadata(str(checkpoint), 7)["step"] == 7
    with pytest.raises(RuntimeError, match="URI label"):
        probe._read_checkpoint_metadata(str(checkpoint).replace("step-7", "step-8"), 7)
    mislabeled_metadata = tmp_path / "step-8"
    mislabeled_metadata.mkdir()
    (mislabeled_metadata / "metadata.json").write_text(json.dumps({"step": 7, "is_temporary": False}))
    with pytest.raises(RuntimeError, match="step mismatch"):
        probe._read_checkpoint_metadata(str(mislabeled_metadata), 8)
    metadata.write_text(json.dumps({"step": 7, "is_temporary": True}))
    with pytest.raises(RuntimeError, match="not a permanent checkpoint"):
        probe._read_checkpoint_metadata(str(checkpoint), 7)


def test_prepare_train_config_retains_checkpointer_for_initialize_from():
    pod_config = next(iter(freeze._canary_configs().values()))
    prepared = probe._prepare_train_config(pod_config, "gs://bucket/checkpoint", "probe-group")

    assert prepared.trainer.checkpointer is not None
    assert prepared.trainer.initialize_from == "gs://bucket/checkpoint"
    assert prepared.trainer.load_checkpoint is False
    assert prepared.trainer.allow_partial_checkpoint is False
    assert prepared.optimizer_schedule_num_train_steps == pod_config.train_config.optimizer_schedule_num_train_steps
    assert probe._optimizer_schedule_summary(prepared) == {
        "configured_num_train_steps": None,
        "effective_num_train_steps": 28_260,
        "trainer_num_train_steps": 28_260,
        "matches_frozen_training_horizon": True,
    }


def test_group_contract_rejects_checkpoint_and_config_drift(monkeypatch):
    row = {
        "row_id": "row",
        "group_id": "group",
        "trajectory_id": "gcf_m100a_test",
        "checkpoint_uri": "gs://bucket/step-7",
        "checkpoint_step": "7",
        "expected_restored_state_step": "8",
        "train_config_sha256": "config",
        "sequences_per_block": "64",
        "training_sequences_per_update": "128",
    }
    config = probe.ProbeGroupConfig(
        scope="canary",
        group_id="group",
        checkpoint_uri="gs://bucket/step-7",
        checkpoint_step=7,
        expected_restored_state_step=8,
        rows=(row,),
        pod_config=SimpleNamespace(train_config=SimpleNamespace()),
        output_path=f"{freeze.RESULT_ROOT}/canary/probe/group/{probe.ARTIFACT_VERSION}",
        cache_provenance_sha256="cache",
        release_sha256="release",
    )
    monkeypatch.setattr(freeze, "_config_identity", lambda _: {"full_train_config_sha256": "config"})
    monkeypatch.setattr(probe, "_starcoder_support_contract", lambda _: {"support_id": "m100a"})

    probe._verify_group_contract(config)
    with pytest.raises(ValueError, match="different checkpoint"):
        probe._verify_group_contract(replace(config, checkpoint_uri="gs://bucket/step-8"))
    monkeypatch.setattr(freeze, "_config_identity", lambda _: {"full_train_config_sha256": "drift"})
    with pytest.raises(ValueError, match="configuration drifted"):
        probe._verify_group_contract(config)


def test_restored_optimizer_summary_requires_matching_counter():
    state = SimpleNamespace(step=8, opt_state={"scale": {"count": jnp.asarray(8, dtype=jnp.int32)}})

    summary = probe._restored_optimizer_summary(state, 7, 8, allow_partial_checkpoint=False)
    assert summary["checkpoint_label_step"] == 7
    assert summary["expected_restored_state_step"] == 8
    assert summary["optimizer_counter_matches_expected"] is True
    assert summary["trainer_state_step_matches_expected"] is True
    assert summary["allow_partial_checkpoint"] is False
    with pytest.raises(RuntimeError, match="Restored trainer state step"):
        probe._restored_optimizer_summary(
            SimpleNamespace(step=7, opt_state=state.opt_state),
            7,
            8,
            allow_partial_checkpoint=False,
        )
    counter_mismatch = SimpleNamespace(step=8, opt_state={"scale": {"count": jnp.asarray(7, dtype=jnp.int32)}})
    with pytest.raises(RuntimeError, match="do not contain expected restored state step"):
        probe._restored_optimizer_summary(counter_mismatch, 7, 8, allow_partial_checkpoint=False)
    with pytest.raises(RuntimeError, match="inconsistent"):
        probe._restored_optimizer_summary(state, 7, 9, allow_partial_checkpoint=False)
    with pytest.raises(RuntimeError, match="partial checkpoint restore"):
        probe._restored_optimizer_summary(state, 7, 8, allow_partial_checkpoint=True)


def test_finite_dataset_capacity_fails_before_iteration():
    summary = probe._dataset_capacity(IntegerDataset(128), 2, label="enough")

    assert summary["sequence_margin"] == 0
    with pytest.raises(RuntimeError, match="supplies 127 sequences but 128 are required"):
        probe._dataset_capacity(IntegerDataset(127), 2, label="short")


def test_direct_full_launch_requires_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "FULL_LAUNCH_AUTHORIZATION_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(probe, "_load_release", lambda _: {"release_sha256": "release"})

    with pytest.raises(ValueError, match="explicit reviewed confirmation token"):
        probe.launch(
            "full",
            release_sha256="release",
            max_concurrent=1,
            kinds=set(),
        )
    with pytest.raises(ValueError, match="user-authorized release sidecar"):
        probe.launch(
            "full",
            release_sha256="release",
            max_concurrent=1,
            kinds=set(),
            confirmation=probe.FULL_LAUNCH_CONFIRMATION,
        )


def test_output_identity_is_release_specific():
    row = {
        "row_id": "row",
        "group_id": "group",
        "checkpoint_uri": "gs://bucket/step-7",
        "train_config_sha256": "config",
    }

    assert probe._row_identity(row, "release-a") != probe._row_identity(row, "release-b")


def test_muon_projection_is_active_and_reports_stacked_layers():
    Layer = Axis("layer", 2)
    Out = Axis("out", 2)
    In = Axis("in", 2)
    model = {
        "weight": hax.named(
            jnp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0]]]),
            (Layer, Out, In),
        )
    }
    left = {
        "weight": hax.named(
            jnp.asarray([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]),
            (Layer, Out, In),
        )
    }
    right = {
        "weight": hax.named(
            jnp.asarray([[[0.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]]),
            (Layer, Out, In),
        )
    }
    mask = {"weight": "muonh"}

    coverage = probe._muon_projection_coverage(model, mask)
    raw = probe._tree_pair_statistics(left, right, model=model, optimizer_mask=mask, project_muon=False)
    projected = probe._tree_pair_statistics(left, right, model=model, optimizer_mask=mask, project_muon=True)

    assert coverage == {
        "muon_parameter_leaf_count": 1,
        "muon_layer_count": 2,
        "muon_matrix_axis_counts": [2],
        "muon_projection_active": True,
    }
    assert {"layer_00", "layer_01"} <= projected.keys()
    assert projected["full"]["dot"] != raw["full"]["dot"]


def test_muon_projection_flattens_multi_axis_linear_geometry():
    InA = Axis("in_a", 2)
    InB = Axis("in_b", 2)
    OutA = Axis("out_a", 2)
    OutB = Axis("out_b", 2)
    linear = hax.nn.Linear.init((InA, InB), (OutA, OutB), key=jrandom.PRNGKey(5), use_bias=False)
    model = {"linear": linear}
    left = {"linear": replace(linear, weight=linear.weight * 0.5)}
    right = {"linear": replace(linear, weight=linear.weight * -0.25)}
    mask = {"linear": replace(linear, weight="muonh")}

    coverage = probe._muon_projection_coverage(model, mask)
    projected = probe._tree_pair_statistics(
        left,
        right,
        model=model,
        optimizer_mask=mask,
        project_muon=True,
    )

    assert coverage["muon_matrix_axis_counts"] == [2]
    assert math.isfinite(projected["full"]["dot"])


def test_zero_norm_cosine_is_explicitly_undefined_without_nonfinite_json():
    Layer = Axis("layer", 1)
    Out = Axis("out", 2)
    In = Axis("in", 2)
    model = {"weight": hax.named(jnp.ones((1, 2, 2)), (Layer, Out, In))}
    zero = {"weight": hax.named(jnp.zeros((1, 2, 2)), (Layer, Out, In))}
    mask = {"weight": "muonh"}

    statistics = probe._tree_pair_statistics(zero, zero, model=model, optimizer_mask=mask, project_muon=True)

    assert statistics["full"]["cosine"] is None
    assert statistics["full"]["cosine_defined"] is False
    assert not probe._contains_nonfinite_number(statistics)


def test_runtime_muon_coverage_requires_named_transformer_layers():
    Out = Axis("out", 2)
    In = Axis("in", 2)
    model = {"weight": hax.named(jnp.ones((2, 2)), (Out, In))}
    mask = {"weight": "muonh"}

    with pytest.raises(RuntimeError, match="no named transformer layers"):
        probe._runtime_muon_projection_coverage(model, mask)


def test_probe_runtime_does_not_import_endpoint_outcomes():
    source = Path(probe.__file__).read_text().lower()

    for forbidden in ("wandb", "eval_metrics.jsonl", "tracker_metrics", "endpoint_predictions"):
        assert forbidden not in source


def test_nonfinite_output_detection_is_recursive():
    assert probe._contains_nonfinite_number({"nested": [1.0, float("nan")]})
    assert probe._contains_nonfinite_number({"nested": {"value": float("inf")}})
    assert not probe._contains_nonfinite_number({"nested": [1.0, 2, "nan"]})


def test_canary_acceptance_audit_validates_rollout_step_arithmetic():
    row = {
        "expected_restored_state_step": "101",
        "updates": "16",
        "readout_steps": "4|8|16",
    }
    document = {
        "final_state_step": 117,
        "readouts": [{"updates": 4}, {"updates": 8}, {"updates": 16}],
    }

    canary_audit._assert_rollout(document, row)
    document["final_state_step"] = 116
    with pytest.raises(RuntimeError, match="rollout final state step"):
        canary_audit._assert_rollout(document, row)


def test_canary_acceptance_audit_rejects_non_sha256_batch_identity():
    assert canary_audit._is_sha256("a" * 64)
    assert not canary_audit._is_sha256("a" * 63)
    assert not canary_audit._is_sha256("g" * 64)


def test_canary_acceptance_audit_rederives_state_and_source_offset():
    pod_config = next(iter(freeze._canary_configs().values()))
    train_config = pod_config.train_config
    checkpoint_step = 7
    expected_state_step = checkpoint_step + 1
    expected_sequence_offset = train_config.trainer.batch_schedule.global_data_offset_by_step(expected_state_step)
    row = {
        "checkpoint_step": str(checkpoint_step),
        "expected_restored_state_step": str(expected_state_step),
        "checkpoint_uri": f"gs://bucket/checkpoints/step-{checkpoint_step}",
    }
    document = {
        "checkpoint_metadata": {"step": checkpoint_step, "is_temporary": False},
        "restored_state_step": expected_state_step,
        "runtime_summary": {
            "restoration": {
                "checkpoint_label_step": checkpoint_step,
                "expected_restored_state_step": expected_state_step,
                "trainer_state_step": expected_state_step,
                "trainer_state_step_matches_expected": True,
                "optimizer_step_counters": {"count": expected_state_step},
                "optimizer_counter_matches_expected": True,
                "allow_partial_checkpoint": False,
            },
            "source_stream": {
                "restored_state_step": expected_state_step,
                "global_sequence_offset": expected_sequence_offset,
                "logical_component_offsets": {name: 0 for name in freeze.TRAINING_COMPONENTS},
                "source_sequence_counts": {name: 1 for name in freeze.TRAINING_COMPONENTS},
                "on_policy_stream_rule": "continue_exact_per_source_logical_offset",
                "step_schedule_rescaled_to_sequences": True,
            },
            "optimizer_schedule": {
                "configured_num_train_steps": train_config.optimizer_schedule_num_train_steps,
                "effective_num_train_steps": train_config.trainer.num_train_steps,
                "trainer_num_train_steps": train_config.trainer.num_train_steps,
                "matches_frozen_training_horizon": True,
            },
            "muon_projection": {
                "muon_projection_active": True,
                "muon_parameter_leaf_count": 1,
                "muon_layer_count": 1,
                "muon_matrix_axis_counts": [2],
            },
        },
    }

    canary_audit._assert_runtime_summary(document, row, pod_config)
    document["runtime_summary"]["source_stream"]["global_sequence_offset"] -= 128
    with pytest.raises(RuntimeError, match="source continuation sequence offset"):
        canary_audit._assert_runtime_summary(document, row, pod_config)


def test_canary_acceptance_audit_pins_gcs_object_generation():
    baseline = {
        "entries": [{"path": "object", "size": 3, "sha256": "abc", "generation": "1"}],
        "inventory_sha256": "inventory",
    }
    replay = json.loads(json.dumps(baseline))

    canary_audit._assert_idempotent_snapshot(replay, baseline)
    replay["entries"][0]["generation"] = "2"
    with pytest.raises(RuntimeError, match="idempotent replay output inventory"):
        canary_audit._assert_idempotent_snapshot(replay, baseline)


def test_mechanism_repair_persists_source_source_geometry_and_exact_target_update_contrast():
    Layer = Axis("layer", 1)
    Out = Axis("out", 2)
    In = Axis("in", 2)
    model = {"weight": hax.named(jnp.ones((1, 2, 2)), (Layer, Out, In))}
    mask = {"weight": "muonh"}
    target = {"weight": hax.named(jnp.asarray([[[1.0, -2.0], [0.5, 1.0]]]), (Layer, Out, In))}
    starcoder = {"weight": hax.named(jnp.asarray([[[0.5, 0.0], [1.5, -0.5]]]), (Layer, Out, In))}
    nemotron = {"weight": hax.named(jnp.asarray([[[-0.5, 1.0], [0.5, 0.25]]]), (Layer, Out, In))}
    gradients = {
        repair_freeze.GLOBAL_STARCODER: starcoder,
        repair_freeze.NEMOTRON: nemotron,
    }
    updates = dict(gradients)

    source_pairs = repair._source_pair_statistics(
        gradients,
        updates,
        model=model,
        optimizer_mask=mask,
    )
    utilities, contrasts = repair._target_statistics(
        {"target": target},
        updates,
        model=model,
        optimizer_mask=mask,
    )

    assert len(source_pairs) == 1
    key = f"{repair_freeze.GLOBAL_STARCODER}__minus__{repair_freeze.NEMOTRON}"
    observed = contrasts["target"][key]["statistic"]["raw"]["full"]
    expected_dot = (
        utilities["target"][repair_freeze.GLOBAL_STARCODER]["raw"]["full"]["dot"]
        - utilities["target"][repair_freeze.NEMOTRON]["raw"]["full"]["dot"]
    )
    direct = probe._tree_pair_statistics(
        probe._tree_scale(target, -1.0),
        probe._tree_subtract(starcoder, nemotron),
        model=model,
        optimizer_mask=mask,
        project_muon=False,
    )["full"]
    assert observed["dot"] == pytest.approx(expected_dot)
    assert observed == direct
    assert contrasts["target"][key]["interpretation"] == "dot_is_X_y_and_cosine_is_A_y"


def test_mechanism_repair_create_only_outputs_are_idempotent_and_release_specific(tmp_path):
    path = str(tmp_path / "row.json")

    assert repair._write_create_only(path, {"value": 1}, identity_sha256="identity") == "created"
    assert repair._write_create_only(path, {"value": 1}, identity_sha256="identity") == "skipped_existing"
    with pytest.raises(RuntimeError, match="payload differs"):
        repair._write_create_only(path, {"value": 2}, identity_sha256="identity")
    with pytest.raises(RuntimeError, match="another identity"):
        repair._write_create_only(path, {"value": 1}, identity_sha256="different")


def test_mechanism_repair_recovers_complete_row_without_recomputing(tmp_path):
    row = {
        "row_id": "row",
        "group_id": "group",
        "checkpoint_uri": "gs://bucket/checkpoint",
        "checkpoint_step": "1",
        "expected_restored_state_step": "2",
    }
    config = repair.MechanismGroupConfig(
        scope="canary",
        group_id="group",
        checkpoint_uri="gs://bucket/checkpoint",
        checkpoint_step=1,
        expected_restored_state_step=2,
        row=row,
        pod_config=None,
        output_path=str(tmp_path),
        parent_cache_provenance_sha256="parent",
        release_sha256="release",
    )
    row_path = repair._row_path(config.output_path, "row")
    payload = {
        "kind": "gradient_mechanism_repair",
        "scope": "canary",
        "group_id": "group",
        "row": row,
        "checkpoint_metadata": {"step": 1},
        "runtime_summary": {"restored": True},
        "execution_observation": {"wall_seconds": 123.0},
        "parent_cache_provenance_sha256": "parent",
        "release_sha256": "release",
        "scientific_status": repair_freeze.SCIENTIFIC_STATUS,
        "source_pair_statistics": {},
        "target_source_gradient_statistics": {},
        "target_source_utility_statistics": {},
        "target_source_choice_contrasts": {},
        "endpoint_metrics_read": False,
    }
    repair._write_create_only(row_path, payload, identity_sha256=repair._row_identity(row, "release"))

    assert repair._existing_group_complete(config) is True
    marker = repair._read_document(str(tmp_path / "group_complete.json"))
    persisted_row = repair._read_document(row_path)
    assert marker is not None and persisted_row is not None
    assert marker["row_document_sha256"] == persisted_row["payload_sha256"]
    assert marker["execution_observation"] == {"wall_seconds": 123.0}


def test_mechanism_repair_h1_selection_is_exactly_the_frozen_subset(monkeypatch):
    manifest = [
        {"row_id": "h1", "analysis_role": "h1_trajectory_extension", "checkpoint_label": "final"},
        {"row_id": "h2", "analysis_role": "h2_primary", "checkpoint_label": "final"},
    ]
    contract = {**repair_freeze.ANALYSIS_CONTRACT["estimands"]["h1"], "row_count": 1, "states": ["final"]}
    monkeypatch.setattr(repair_analysis, "_read_manifest", lambda: manifest)
    monkeypatch.setitem(repair_freeze.ANALYSIS_CONTRACT["estimands"], "h1", contract)
    documents = [
        {"row": {"row_id": "h1", "analysis_role": "h1_trajectory_extension", "checkpoint_label": "final"}},
        {"row": {"row_id": "h2", "analysis_role": "h2_primary", "checkpoint_label": "final"}},
    ]

    assert [document["row"]["row_id"] for document in repair_analysis.select_h1_documents(documents)] == ["h1"]
    with pytest.raises(RuntimeError, match="H1 document inventory drifted"):
        repair_analysis.select_h1_documents(documents[1:])


def test_mechanism_repair_contract_excludes_h4_and_marks_post_outcome_status():
    contract = repair_freeze.ANALYSIS_CONTRACT

    assert contract["outcomes_inspected_before_contract"] is True
    assert contract["scientific_status"] == repair_freeze.SCIENTIFIC_STATUS
    assert "H4 is excluded" in contract["h4_exclusion"]
    assert set(contract["estimands"]) == {"h1", "h2", "h3", "h5_profile"}
    assert contract["estimands"]["h2"]["states"]["late_post_decay"] == ["decay_plus_64", "decay_plus_256"]
    assert "unseen_utility_decline" in contract["estimands"]["h3"]
    assert "support_separation_growth" in contract["estimands"]["h3"]
    assert contract["estimands"]["h1"]["status"] == "restricted_descriptive_subset"
    assert contract["estimands"]["h1"]["trajectory_inventory"] == {
        "m100a": 24,
        "full": 24,
        "m100b": 8,
        "total": 56,
    }


def test_mechanism_repair_manifest_has_one_unique_row_per_checkpoint_group():
    release = json.loads(repair_freeze.RELEASE_PATH.read_text())
    rows = list(csv.DictReader(repair_freeze.FULL_MANIFEST_PATH.open()))
    canary_rows = list(csv.DictReader(repair_freeze.CANARY_MANIFEST_PATH.open()))

    assert release["scientific_status"] == repair_freeze.SCIENTIFIC_STATUS
    assert release["h4_included"] is False
    assert len(canary_rows) == release["manifests"]["canary"]["row_count"] == 14
    assert len(rows) == release["manifests"]["full"]["row_count"]
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len({row["group_id"] for row in rows}) == len(rows)
    assert all(row["endpoint_metrics_read_by_runner"] == "False" for row in rows)
    assert {row["analysis_role"] for row in rows} == {
        "h1_trajectory_extension",
        "h2_primary",
        "h3_full_support_pair",
        "h3_second_pool_sensitivity",
        "h5_preregistered_profile",
    }
    assert {stage: sum(int(row["launch_stage"]) == stage for row in rows) for stage in (1, 2, 3)} == {
        1: 28,
        2: 56,
        3: 876,
    }
    assert {row["analysis_role"] for row in rows if row["launch_stage"] == "1"} == {
        "h1_trajectory_extension",
        "h2_primary",
        "h3_full_support_pair",
        "h3_second_pool_sensitivity",
        "h5_preregistered_profile",
    }


def test_mechanism_repair_reuses_parent_v6_stochastic_row_identities():
    rows = list(csv.DictReader(repair_freeze.FULL_MANIFEST_PATH.open()))
    parent_probe = {
        (row["trajectory_id"], row["checkpoint_label"], row["distribution_id"]): row["row_id"]
        for row in _read_manifest("full_probe_manifest.csv")
    }
    for row in rows:
        probe_ids = json.loads(row["distribution_probe_row_ids_json"])
        for distribution, row_id in probe_ids.items():
            assert row_id == parent_probe[(row["trajectory_id"], row["checkpoint_label"], distribution)]


def test_mechanism_repair_uses_global_holdout_source_contrast_at_every_checkpoint():
    rows = list(csv.DictReader(repair_freeze.FULL_MANIFEST_PATH.open()))

    assert repair_freeze.PRIMARY_UPDATE_CONTRAST == (repair_freeze.GLOBAL_STARCODER, repair_freeze.NEMOTRON)
    for row in rows:
        sources = set(json.loads(row["source_distribution_ids_json"]))
        assert {repair_freeze.GLOBAL_STARCODER, repair_freeze.NEMOTRON} <= sources
        assert repair_freeze.ON_POLICY_STARCODER not in sources


def test_mechanism_repair_requires_source_invariant_no_data_update():
    tree = {"value": jnp.asarray([1.0, 2.0])}
    same = {"a": tree, "b": {"value": jnp.asarray([1.0, 2.0])}}
    summaries = {
        name: {"no_data_update_within_source_max_abs_diff": 0.0, "corrected_update_norm_mean": 2.0} for name in same
    }
    audit = repair._assert_common_no_data_update(same, summaries)
    assert audit["passed"] is True
    assert audit["relative_to_min_corrected_update_norm"] == 0.0

    drifted = {**same, "b": {"value": jnp.asarray([1.0, 2.01])}}
    with pytest.raises(RuntimeError, match="depends on source loss or RNG key"):
        repair._assert_common_no_data_update(drifted, summaries)


def test_mechanism_repair_launch_api_enforces_frozen_concurrency(monkeypatch):
    release = {
        "execution_acceptance": {"canary_max_concurrent": 14},
        "full_launch_stages": {
            "1": {"max_concurrent": 28},
            "2": {"max_concurrent": 56},
            "3": {"max_concurrent": 64},
        },
    }
    monkeypatch.setattr(repair, "_load_release", lambda _: release)

    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        repair.launch("full", release_sha256="release", max_concurrent=65, confirmation=None, stage=3)
    with pytest.raises(ValueError, match=r"\[1, 14\]"):
        repair.launch("canary", release_sha256="release", max_concurrent=15, confirmation=None, stage=None)


def test_mechanism_repair_release_freezes_execution_and_shape_gates():
    release = json.loads(repair_freeze.RELEASE_PATH.read_text())

    assert repair_analysis.ARTIFACT_VERSION == repair.ARTIFACT_VERSION
    assert "results_v8" in repair_analysis.DEFAULT_OUTPUT_DIR.name
    gate = release["execution_acceptance"]
    assert gate["stage_promotion_requires_exact_prior_stage_audit"] is True
    assert gate["allowed_resource_exhaustion_failures"] == 0
    assert gate["max_group_wall_seconds"] == 2_700
    assert gate["probe_batch_size"] == probe.PROBE_BATCH_SIZE
    assert release["design_validation"]["h1_trajectory_count"] == 56
    assert (
        release["design_validation"]["full_workload_shape_sha256"]
        == release["design_validation"]["stage1_workload_shape_sha256"]
    )


def test_mechanism_repair_pins_checkpoint_object_generations_and_checksums():
    release = json.loads(repair_freeze.RELEASE_PATH.read_text())
    rows = list(csv.DictReader(repair_freeze.CHECKPOINT_PROVENANCE_PATH.open()))

    assert len(rows) == release["manifests"]["checkpoint_provenance"]["row_count"]
    assert (
        len({row["checkpoint_uri"] for row in rows}) == release["manifests"]["checkpoint_provenance"]["checkpoint_count"]
    )
    assert all(row["generation"] and row["size"] and row["crc32c"] and row["etag"] for row in rows)


def test_mechanism_repair_pins_exact_parent_result_objects():
    release = json.loads(repair_freeze.RELEASE_PATH.read_text())
    rows = list(csv.DictReader(repair_freeze.PARENT_RESULT_PROVENANCE_PATH.open()))

    summary = release["manifests"]["parent_result_provenance"]
    assert len(rows) == summary["row_count"] == summary["object_count"]
    assert len({row["object_uri"] for row in rows}) == len(rows)
    assert all(
        row["generation"]
        and row["size"]
        and row["crc32c"]
        and row["etag"]
        and len(row["payload_sha256"]) == 64
        and row["parent_identity_sha256"]
        for row in rows
    )


def test_mechanism_repair_row_identity_binds_complete_frozen_row():
    row = {
        "row_id": "row",
        "group_id": "group",
        "checkpoint_uri": "gs://bucket/checkpoint",
        "train_config_sha256": "config",
        "analysis_role": "h2_primary",
    }
    identity = repair._row_identity(row, "release")

    assert repair._row_identity({**row, "analysis_role": "h1_trajectory_extension"}, "release") != identity


def test_mechanism_repair_freezer_is_create_only(tmp_path):
    path = tmp_path / "frozen.json"

    repair_freeze._write_create_only(path, b"same")
    repair_freeze._write_create_only(path, b"same")
    with pytest.raises(RuntimeError, match="different content"):
        repair_freeze._write_create_only(path, b"different")


def test_mechanism_repair_exact_inventory_rejects_missing_rows(monkeypatch):
    manifest = [
        {
            "row_id": "row-a",
            "analysis_role": "h2_primary",
            "checkpoint_label": "fraction_0p40",
            "target_distribution_ids_json": '["target"]',
            "source_distribution_ids_json": '["source"]',
        }
    ]
    monkeypatch.setattr(repair_analysis, "_read_manifest", lambda: manifest)
    frame = repair_analysis.pd.DataFrame(columns=["row_id", "target"])

    with pytest.raises(RuntimeError, match="inventory drifted"):
        repair_analysis._assert_exact_manifest_inventory(
            frame,
            roles={"h2_primary"},
            labels={"fraction_0p40"},
            include_sources=False,
            name="test",
        )


def _trunk(cosine, defined, left_norm, right_norm, dot=0.0):
    return {"cosine": cosine, "cosine_defined": defined, "dot": dot, "left_norm": left_norm, "right_norm": right_norm}


def _statistic(trunk):
    return {"raw": {"trunk": dict(trunk)}, "projected": {"trunk": dict(trunk)}}


def test_audit_accepts_an_undefined_cosine_only_when_a_norm_is_actually_zero():
    """At `final` the schedule has decayed the learning rate to zero, so the corrected optimizer update is
    identically zero and its cosine is undefined rather than faulty. That case must pass; an undefined
    cosine arising any other way must still fail closed, or the audit stops protecting anything."""
    repair._assert_defined_statistic(_statistic(_trunk(None, False, 0.0, 0.0)), label="zero update")
    repair._assert_defined_statistic(_statistic(_trunk(None, False, 0.0, 3.5)), label="one side zero")

    with pytest.raises(RuntimeError, match="undefined"):
        repair._assert_defined_statistic(_statistic(_trunk(None, False, 2.0, 3.0)), label="unjustified")
    with pytest.raises(RuntimeError, match="undefined"):
        repair._assert_defined_statistic(
            {"raw": {"trunk": {"cosine": None, "cosine_defined": False, "dot": 0.0}}, "projected": {"trunk": {}}},
            label="norms missing",
        )


def test_audit_still_rejects_non_finite_statistics():
    """The relaxation must not open a path for NaN to be waved through as a zero update."""
    with pytest.raises(RuntimeError, match="non-finite"):
        repair._assert_defined_statistic(_statistic(_trunk(None, False, 0.0, float("nan"))), label="nan norm")
    with pytest.raises(RuntimeError, match="non-finite"):
        repair._assert_defined_statistic(_statistic(_trunk(float("nan"), True, 1.0, 1.0)), label="nan cosine")


def test_flatten_h1_records_an_undefined_cosine_as_missing_rather_than_zero():
    """A cosine between zero vectors has no value. Imputing zero would fabricate an alignment reading in a
    descriptive statistic, so the row is kept with a NaN cosine and a flag saying it is undefined."""
    document = {
        "row": {
            "row_id": "r0",
            "trajectory_id": "t0",
            "training_seed": 1,
            "support_id": "full",
            "policy_role": "tied",
            "analysis_role": "h1_trajectory_extension",
            "checkpoint_label": "final",
        },
        "source_uri": "gs://example/row.json",
        "source_pair_statistics": {
            "starcoder__vs__nemotron": {
                "gradient": _statistic(_trunk(0.25, True, 1.0, 1.0, dot=0.25)),
                "optimizer_update": _statistic(_trunk(None, False, 0.0, 0.0)),
            }
        },
    }
    frame = repair_analysis.flatten_h1([document])

    updates = frame[frame["statistic"].eq("optimizer_update")]
    gradients = frame[frame["statistic"].eq("gradient")]
    assert len(updates) == 2 and not updates["cosine_defined"].any()
    assert updates["cosine"].isna().all()
    assert gradients["cosine_defined"].all() and gradients["cosine"].eq(0.25).all()

    document["source_pair_statistics"]["starcoder__vs__nemotron"]["optimizer_update"] = _statistic(
        _trunk(None, False, 2.0, 3.0)
    )
    with pytest.raises(RuntimeError, match="Undefined H1 cosine"):
        repair_analysis.flatten_h1([document])
