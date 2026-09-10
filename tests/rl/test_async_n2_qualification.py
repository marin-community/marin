# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fray.iris_backend import FrayIrisClient
from fray.local_backend import LocalClient
from hydra.core.override_parser.overrides_parser import OverridesParser
from marin.execution import artifact, lazy
from marin.execution.lazy import materialized_config, run
from marin.rl.skyrl import ArtifactDataSource

from experiments.post_training.async_n2_qualification import (
    build_qualification,
    local_model_dependency,
    preparation_dependency,
    resolve_qualification,
)
from experiments.post_training.curriculum_rl import launch
from experiments.post_training.math_eval.bucket_launcher import VerifiedBucketDataSource
from experiments.post_training.math_eval.launcher import VerifiedPoolDataSource

PREFIX = "s3://marin-us-east-02a/marin"
MARKER = PREFIX + "/users/ahmad/diagnostics/async-rl/async-n2-native-v1/measurement/fresh.json"
CHECKPOINT = "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/n2-qualification/global_step_7"


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
        parsed_overrides = OverridesParser.create().parse_overrides(list(request.overrides))
        resume = [item.value() for item in parsed_overrides if item.key_or_group == "trainer.resume_path"]
        assert resume == [checkpoint]
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


def test_local_model_dependency_preserves_identity_and_callable():
    step = build_qualification(version="2026.09.09.259", measurement_uri=MARKER)
    mirrors = [dep for dep in step.deps if callable(getattr(dep.run, "fn", None))]
    assert len(mirrors) == 1
    remote = mirrors[0]
    local = local_model_dependency(remote)
    assert local.fingerprint() == remote.fingerprint()
    assert local.path(PREFIX) == remote.path(PREFIX)
    assert local.build_config is remote.build_config
    assert local.run is remote.run.fn
    with pytest.raises(ValueError, match="Only the native model mirror"):
        local_model_dependency(step)


def test_local_mirror_uses_actual_step_runner_without_submissions(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    calls = []

    def download(*, repo_id, revision, local_dir, cache_dir):
        calls.append((repo_id, revision))
        path = Path(local_dir)
        path.mkdir(parents=True)
        (path / "config.json").write_bytes(b'{"test":true}')
        (path / "model.safetensors").write_bytes(b"bounded model fixture")
        return str(path)

    def submission_trap(*args, **kwargs):
        raise AssertionError("Dependency escaped the CPU preparation process")

    monkeypatch.setattr(launch, "snapshot_download", download)
    monkeypatch.setattr(FrayIrisClient, "submit", submission_trap)
    monkeypatch.setattr(LocalClient, "submit", submission_trap)
    original = launch.model_step("2026.01.01")
    local = local_model_dependency(original)
    assert materialized_config(local, str(tmp_path)) == materialized_config(original, str(tmp_path))
    run(local)
    output = Path(local.path())
    assert (output / "hf/model.safetensors").read_bytes() == b"bounded model fixture"
    assert (output / "hf/config.json").read_bytes() == b'{"test":true}'
    retained = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    run(local)
    assert len(calls) == 1
    assert retained == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_complete_fresh_and_continuation_graph_uses_no_nested_submission(tmp_path, monkeypatch, local_config_inputs):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    downloads = []

    def download(*, repo_id, revision, local_dir, cache_dir):
        downloads.append((repo_id, revision))
        local = Path(local_dir)
        local.mkdir(parents=True)
        (local / "config.json").write_text("{}")
        return str(local)

    def submission_trap(*args, **kwargs):
        raise AssertionError("Complete graph attempted a nested submission")

    fresh = build_qualification(version="2026.09.09.259", measurement_uri=MARKER)
    adopted = {dep.adopt_source for dep in fresh.deps if dep.adopt_source}
    assert len(fresh.deps) == 3 and len(adopted) == 2
    original_fs = lazy.url_to_fs
    original_record = artifact.read_record

    def metadata_fs(path, **kwargs):
        if path in adopted:
            return SimpleNamespace(exists=lambda value: value in adopted), path
        return original_fs(path, **kwargs)

    # Only retained input existence/record reads and input content validators are
    # stubbed. Real StepRunner, graph dispatch and request builders execute.
    monkeypatch.setattr(lazy, "url_to_fs", metadata_fs)
    monkeypatch.setattr(artifact, "read_record", lambda path: None if path in adopted else original_record(path))
    monkeypatch.setattr(launch, "snapshot_download", download)
    monkeypatch.setattr(FrayIrisClient, "submit", submission_trap)
    monkeypatch.setattr(LocalClient, "submit", submission_trap)
    seen = set()
    first = resolve_qualification(fresh, PREFIX, seen)
    checkpoint = first.request.output.checkpoint_root + "/global_step_7"
    continuation = build_qualification(
        version="2026.09.09.260", measurement_uri=MARKER.replace("fresh", "continuation"), checkpoint_seven=checkpoint
    )
    second = resolve_qualification(continuation, PREFIX, seen)
    assert len(seen) == 4 and len(downloads) == 2
    parsed = OverridesParser.create().parse_overrides(list(second.request.overrides))
    assert [item.value() for item in parsed if item.key_or_group == "trainer.resume_path"] == [checkpoint]
    assert yaml.safe_load(first.request.config_yaml)["trainer"]["max_steps"] == 8
    assert yaml.safe_load(second.request.config_yaml)["trainer"]["max_steps"] == 8
    wrong = replace(fresh.deps[1], adopt_source=PREFIX + "/unreviewed")
    with pytest.raises(ValueError, match="Unsupported adopted"):
        preparation_dependency(wrong)
    with pytest.raises(ValueError, match="Only the native model mirror"):
        preparation_dependency(fresh)


def test_rno_changes_only_execution_cluster(local_config_inputs):
    kwargs = dict(version="2026.09.09.259", measurement_uri=MARKER)
    east = materialized_config(build_qualification(**kwargs), PREFIX)
    rno = materialized_config(build_qualification(**kwargs, cluster="cw-rno2a"), PREFIX)
    assert rno.execution.cluster == "cw-rno2a"
    assert rno.execution.cluster_config == "lib/iris/config/cw-rno2a.yaml"
    assert (
        replace(rno.execution, cluster=east.execution.cluster, cluster_config=east.execution.cluster_config)
        == east.execution
    )
    assert replace(rno.request, attempt_id=east.request.attempt_id) == east.request
    assert rno.request.output.checkpoint_root.startswith("s3://marin-us-east-02a/")
    assert rno.request.model.uri.startswith("s3://marin-us-east-02a/")
    with pytest.raises(ValueError, match="Qwen qualification"):
        build_qualification(**kwargs, cluster="cw-us-east-08a")
