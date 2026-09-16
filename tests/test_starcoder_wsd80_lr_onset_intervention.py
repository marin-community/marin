# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from levanter.data.dataset import AsyncDataset

from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_intervention as intervention
from experiments.domain_phase_mix import starcoder_wsd80_lr_onset_gradient_probe as onset_probe


class IntegerDataset(AsyncDataset[int]):
    def __init__(self, length: int):
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices):
        return list(indices)


def _frozen_probe_rows():
    return [onset_probe.ProbeRow(**row) for row in json.loads(onset_probe.MANIFEST_PATH.read_text())]


def test_restored_hyperparameters_uses_preincrement_schedule_step_and_ignores_schedule_state():
    optimizer = intervention._optimizer(intervention.ARMS[0])
    restored_state_step = 2_102
    schedule_step = restored_state_step - 1
    schedule = optimizer.lr_scheduler(intervention.TOTAL_STEPS)
    adam_schedule = optimizer.lr_scheduler(intervention.TOTAL_STEPS, override_lr=optimizer.adam_lr)
    state = SimpleNamespace(
        step=restored_state_step,
        opt_state={
            "hyperparams": {
                "learning_rate": jnp.asarray(schedule(schedule_step)),
                "adam_lr": jnp.asarray(adam_schedule(schedule_step)),
            },
            "hyperparams_states": {
                "learning_rate": {"count": jnp.asarray(999_999)},
                "adam_lr": {"count": jnp.asarray(999_999)},
            },
        },
    )
    train_config = SimpleNamespace(
        optimizer=optimizer,
        optimizer_schedule_num_train_steps=None,
        trainer=SimpleNamespace(num_train_steps=intervention.TOTAL_STEPS),
    )

    result = intervention._restored_hyperparameters(state, train_config)

    assert result["schedule_step"] == schedule_step
    assert result["observed"] == result["expected"]
    assert result["matches_expected"] is True


def test_stage0_aggregate_validation_is_deterministic_and_checks_bitwise_prefix(tmp_path):
    release_sha256 = "a" * 64
    checkpoint_step = intervention._step(0.55)
    fingerprint = {
        "model_sha256": "b" * 64,
        "optimizer_state_sha256": "c" * 64,
        "training_key_sha256": "d" * 64,
    }
    arm_outputs = {}
    for arm in intervention.ARMS:
        arm_path = tmp_path / arm.arm
        arm_outputs[arm.arm] = str(arm_path)
        payload = {
            "schema_version": "test",
            "arm": arm.arm,
            "checkpoint_uri": f"gs://example/{arm.arm}",
            "checkpoint_step": checkpoint_step,
            "restored_state_step": checkpoint_step + 1,
            "release_sha256": release_sha256,
            "learning_rates": {
                "observed": {"learning_rate": 1.0, "adam_lr": 2.0},
                "matches_expected": True,
            },
            "state_fingerprint": fingerprint,
            "device_count": 4,
            "device_kinds": ["TPU v5"],
        }
        payload["payload_sha256"] = intervention._sha256_bytes(intervention._canonical_json(payload).encode())
        intervention._write_remote_json(f"{arm_path}/validation.json", payload)

    output_path = tmp_path / "aggregate"
    config = intervention.Stage0AggregateValidationConfig(
        arm_outputs=arm_outputs,
        expected_arms=tuple(arm.arm for arm in intervention.ARMS),
        expected_learning_rate=1.0,
        checkpoint_step=checkpoint_step,
        output_path=str(output_path),
        release_sha256=release_sha256,
    )

    intervention.run_stage0_aggregate_validation(config)
    first = (output_path / "validation.json").read_bytes()
    intervention.run_stage0_aggregate_validation(config)

    assert (output_path / "validation.json").read_bytes() == first
    assert json.loads(first)["status"] == "PASS"


def test_lr_onset_probe_rows_cover_every_trajectory_checkpoint_and_share_reference_panels():
    rows = _frozen_probe_rows()

    assert len(rows) == 32 * len(onset_probe.PROBE_CHECKPOINT_STEPS)
    assert len({row.row_id for row in rows}) == len(rows)
    for field in ("starcoder_sequence_set_id", "half_a_sequence_offset", "half_b_sequence_offset"):
        assert len({getattr(row, field) for row in rows}) == 1
    assert len({row.nemotron_sequence_set_id for row in rows}) == len(intervention.TRAINING_SEEDS)
    assert rows[0].half_a_sequence_offset == 0
    assert rows[0].half_b_sequence_offset == onset_probe.BLOCKS_PER_HALF * onset_probe.probe.PROBE_BATCH_SIZE
    base_dataset = IntegerDataset(onset_probe.PANEL_SEQUENCE_COUNT)
    half_a = onset_probe.probe.ShiftedRestartDataset(
        base_dataset,
        start=rows[0].half_a_sequence_offset,
        length=onset_probe.PANEL_SEQUENCE_COUNT,
    )
    half_b = onset_probe.probe.ShiftedRestartDataset(
        base_dataset,
        start=rows[0].half_b_sequence_offset,
        length=onset_probe.PANEL_SEQUENCE_COUNT,
    )
    assert (half_a.start, half_b.start) == (0, onset_probe.PANEL_SEQUENCE_COUNT // 2)
    half_a_indices = set(asyncio.run(half_a.get_batch(range(onset_probe.PANEL_SEQUENCE_COUNT // 2))))
    half_b_indices = set(asyncio.run(half_b.get_batch(range(onset_probe.PANEL_SEQUENCE_COUNT // 2))))
    assert not half_a_indices & half_b_indices
    assert len(half_a_indices | half_b_indices) == onset_probe.PANEL_SEQUENCE_COUNT
    expected_partitions = {
        intervention._step(0.60): {"peak_lr_prefix"},
        intervention._step(0.80): {"decay_0p60", "peak_lr_prefix"},
        intervention._step(0.90): {"decay_0p60", "decay_0p80", "peak_lr_prefix"},
    }
    trajectories, _ = intervention.build_training_steps()
    for checkpoint_step, expected in expected_partitions.items():
        observed = {
            onset_probe._expected_state_equivalence_class(trajectory, checkpoint_step)
            for trajectory in trajectories
            if trajectory.training_seed == intervention.TRAINING_SEEDS[0]
        }
        assert observed == expected


def test_noise_corrected_statistics_uses_split_half_signal_norms():
    combined = {"projected": {"trunk": {"dot": 6.0, "left_norm": 4.0, "right_norm": 3.0, "cosine": 0.5}}}
    left = {"projected": {"trunk": {"dot": 9.0, "left_norm": 4.0, "right_norm": 4.0, "cosine": 0.75}}}
    right = {"projected": {"trunk": {"dot": 4.0, "left_norm": 3.0, "right_norm": 3.0, "cosine": 0.5}}}

    result = onset_probe._noise_corrected_statistics(combined, left, right)["projected"]["trunk"]

    assert result["disattenuated_cosine"] == pytest.approx(1.0)
    assert result["left_spearman_brown_reliability"] == pytest.approx(6.0 / 7.0)
    assert result["right_spearman_brown_reliability"] == pytest.approx(2.0 / 3.0)


def test_noise_corrected_statistics_rejects_negative_split_half_signal():
    combined = {"projected": {"trunk": {"dot": 1.0, "left_norm": 1.0, "right_norm": 1.0}}}
    left = {"projected": {"trunk": {"dot": -1.0, "cosine": -0.5}}}
    right = {"projected": {"trunk": {"dot": -1.0, "cosine": -0.5}}}

    result = onset_probe._noise_corrected_statistics(combined, left, right)["projected"]["trunk"]

    assert result["defined"] is False
    assert result["disattenuated_cosine"] is None


def test_result_path_matches_versioned_artifact_layout():
    rows = _frozen_probe_rows()

    assert onset_probe._result_path(rows[0]) == (
        f"{onset_probe.RESULT_ROOT}/{rows[0].row_id}/{onset_probe.VERSION}/result.json"
    )


def test_lr_onset_probe_remote_document_hash_is_verified(tmp_path):
    path = str(tmp_path / "result.json")
    payload: dict[str, object] = {"identity_sha256": "a" * 64, "value": 1}
    payload["payload_sha256"] = onset_probe._sha256(onset_probe._canonical_json(payload).encode())

    onset_probe._write_remote_json(path, payload)

    assert onset_probe._read_remote_json(path) == payload
    (tmp_path / "result.json").write_text(json.dumps({**payload, "value": 2}))
    with pytest.raises(RuntimeError, match="payload hash"):
        onset_probe._read_remote_json(path)
