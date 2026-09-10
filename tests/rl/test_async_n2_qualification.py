# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from marin.execution.lazy import materialized_config
from marin.rl.skyrl import ArtifactDataSource

from experiments.post_training.async_n2_qualification import build_qualification
from experiments.post_training.math_eval.bucket_launcher import VerifiedBucketDataSource
from experiments.post_training.math_eval.launcher import VerifiedPoolDataSource

PREFIX = "s3://marin-us-east-02a/marin"
MARKER = PREFIX + "/users/ahmad/diagnostics/async-rl/async-n2-native-v1/measurement/fresh.json"
CHECKPOINT = PREFIX + "/users/ahmad/checkpoints/n2-qualification/global_step_7"


@pytest.fixture
def local_config_inputs(monkeypatch):
    # This tests configuration composition, never regional content qualification.
    # The production resolver retains its native-context and exact-byte checks.
    monkeypatch.setattr(VerifiedBucketDataSource, "resolve", ArtifactDataSource.resolve)
    monkeypatch.setattr(VerifiedPoolDataSource, "resolve", ArtifactDataSource.resolve)


@pytest.mark.parametrize("checkpoint", [None, CHECKPOINT])
def test_qualification_request_keeps_eight_optimizer_updates_and_explicit_resume(checkpoint, local_config_inputs):
    step = build_qualification(
        version="2026.09.09.259" if checkpoint is None else "2026.09.09.260",
        measurement_uri=MARKER if checkpoint is None else MARKER.replace("fresh", "continuation"),
        checkpoint_seven=checkpoint,
    )
    config = materialized_config(step, PREFIX)
    request = config.request
    recipe = yaml.safe_load(request.config_yaml)
    trainer = recipe["trainer"]
    assert trainer["max_steps"] == trainer["eval_interval"] == 8
    assert trainer["ckpt_interval"] == 7
    assert trainer["train_batch_size"] == 128 and trainer["policy_mini_batch_size"] == 64
    assert trainer["fully_async"]["first_token_admission"] is True
    assert trainer["fully_async"]["max_staleness_steps"] == 0
    assert trainer["fully_async"].get("weight_sync_interval", 1) == 1
    assert trainer["fully_async"]["num_parallel_generation_workers"] == 64
    assert trainer["algorithm"]["policy_loss_type"] == "regular"
    assert trainer["algorithm"]["use_tis"] is False
    assert trainer["algorithm"]["use_kl_loss"] is False
    assert trainer["dump_data_batch"] is True
    assert trainer["measurement_guard_resume_step"] == (None if checkpoint is None else 7)
    assert request.completion_mode.value == "checkpoint"
    assert "++trainer.hf_hub_repo_id=null" in request.overrides
    assert "++trainer.hf_save_interval=-1" in request.overrides
    mode = "none" if checkpoint is None else "from_path"
    assert f"++trainer.resume_mode={mode}" in request.overrides
    if checkpoint is not None:
        assert f"++trainer.resume_path={checkpoint}" in request.overrides
    assert config.execution.max_retries == 1
    assert config.execution.timeout_seconds == 1800
    assert config.execution.wandb_entity == "dogml"


def test_measurement_identity_changes_fingerprinted_request():
    first = build_qualification(version="2026.09.09.259", measurement_uri=MARKER)
    second = build_qualification(version="2026.09.09.259", measurement_uri=MARKER.replace("fresh", "other"))
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize("uri", ["s3://marin-us-west-01/test", "/tmp/marker", "s3://marin-us-east-02a"])
def test_qualification_rejects_unbound_or_wrong_region_marker(uri):
    with pytest.raises(ValueError, match="east S3"):
        build_qualification(version="2026.09.09.259", measurement_uri=uri)
