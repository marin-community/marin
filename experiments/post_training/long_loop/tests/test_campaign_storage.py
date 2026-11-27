# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the campaign final-checkpoint retention."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from marin.execution.artifact import write_artifact
from marin.rl.skyrl import SkyRLRun, skyrl_temporary_run_path
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop import campaign_storage
from experiments.post_training.long_loop.campaign_storage import retain_campaign_checkpoints
from vendor.skyrl.marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from vendor.skyrl.marinskyrl.model_manifest import ModelManifestFile, build_model_manifest

FINAL_STEP = 50
WEIGHTS = b"policy-weights"


@pytest.fixture(autouse=True)
def regional_cpu_job(monkeypatch, tmp_path):
    """Run inside a fake regional worker whose output prefix accepts the test's local output."""
    previous = get_job_info()
    set_job_info(JobInfo(task_id=SimpleNamespace(value="/test/retain-campaign-checkpoints")))
    monkeypatch.setattr(campaign_storage, "REGIONAL_OUTPUT_PREFIX", f"file://{tmp_path}/")
    yield
    set_job_info(previous)


def write_export(root) -> tuple[ModelManifestFile, ...]:
    """Write a minimal embedded-mode HF export; return the manifest entries backing its identity."""
    root = StoragePath(str(root))
    weights_index = json.dumps(
        {"metadata": {"total_size": len(WEIGHTS)}, "weight_map": {"w": "model-00001-of-00001.safetensors"}},
        indent=2,
        sort_keys=True,
    ).encode()
    files = {
        "config.json": b'{"model_type": "test"}',
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": weights_index,
        "model-00001-of-00001.safetensors": WEIGHTS,
    }
    entries = tuple(
        ModelManifestFile(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, data in sorted(files.items())
    )
    manifest = build_model_manifest(entries, None, None)
    root.mkdirs()
    for name, data in files.items():
        (root / name).write_bytes(data)
    (root / MODEL_MANIFEST_FILENAME).write_text(json.dumps(manifest.model_dump(mode="json")))
    return entries


def stage_run(tmp_path, *, export_under_temporary_root: bool):
    """Write one committed run record, its final native checkpoints and its HF export."""
    training = tmp_path / "rl-data-campaign" / "v3-mix-048" / "2026.11.26.532"
    temporary_run = StoragePath(skyrl_temporary_run_path(str(training), ttl_days=14))
    checkpoints = prefix_join(str(temporary_run), "checkpoints")
    root = StoragePath(checkpoints)
    root.mkdirs()
    (root / LATEST_CHECKPOINT_FILE).write_text(f"{FINAL_STEP}\n")
    (root / f"global_step_{FINAL_STEP}" / "trainer_state.pt").write_bytes(b"state")
    (root / f"global_step_{FINAL_STEP}" / "weights.safetensors").write_bytes(b"weights")
    (root / "global_step_48" / "trainer_state.pt").write_bytes(b"older-state")
    export = (
        StoragePath(prefix_join(str(temporary_run), "export/policy"))
        if export_under_temporary_root
        else training / "exports" / f"global_step_{FINAL_STEP}" / "policy"
    )
    entries = write_export(export)
    result = SkyRLRun(
        hf_model_uri=str(export),
        global_step=FINAL_STEP,
        tokenizer_uri="tokenizer",
        tokenizer_revision="rev",
        checkpoint_root=str(root),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(training / "terminal.json"),
        iris_job_id="/benfeuer/test",
    )
    write_artifact(result.result_payload(), str(training))
    return str(training), root, str(export), entries


def test_permanent_export_is_recorded_and_native_scratch_is_pruned(tmp_path):
    training, root, export, _ = stage_run(tmp_path, export_under_temporary_root=False)
    output = tmp_path / "storage" / "2026.11.26.532"

    retain_campaign_checkpoints(training=training, output=f"file://{output}")

    assert (output / "prune-plan.json").exists() and (output / "storage-receipt.json").exists()
    receipt = json.loads((output / "storage-receipt.json").read_text())
    assert receipt["hf_export"] == export
    assert receipt["hf_export_relocation"] is None
    assert receipt["final_native_copy"] is None
    assert receipt["final_step"] == FINAL_STEP
    # Every byte under the campaign-owned temporary root is accounted for, then removed.
    assert receipt["deleted_bytes"] == len(f"{FINAL_STEP}\n") + len(b"state") + len(b"weights") + len(b"older-state")
    assert not StoragePath(root).exists()
    # The delivered export is untouched: no re-hash, no second copy.
    assert (StoragePath(export) / MODEL_MANIFEST_FILENAME).exists()
    assert (StoragePath(export) / "model-00001-of-00001.safetensors").read_bytes() == WEIGHTS


def test_temporary_export_is_relocated_to_a_permanent_path_before_pruning(tmp_path):
    training, root, export, _ = stage_run(tmp_path, export_under_temporary_root=True)
    output = tmp_path / "storage" / "2026.11.26.532"

    retain_campaign_checkpoints(training=training, output=f"file://{output}")

    receipt = json.loads((output / "storage-receipt.json").read_text())
    assert receipt["hf_export_relocation"] == {"from": export, "to": receipt["hf_export"]}
    moved = StoragePath(receipt["hf_export"])
    assert str(moved).startswith(f"file://{output}/")
    assert (moved / MODEL_MANIFEST_FILENAME).exists()
    assert (moved / "model-00001-of-00001.safetensors").read_bytes() == WEIGHTS
    assert not StoragePath(root).exists()


def test_missing_completion_manifest_aborts_before_any_deletion(tmp_path):
    training, root, _export, _ = stage_run(tmp_path, export_under_temporary_root=False)
    Path(str(StoragePath(_export) / MODEL_MANIFEST_FILENAME)).unlink()
    output = tmp_path / "storage" / "2026.11.26.532"

    with pytest.raises(ValueError, match="completion manifest"):
        retain_campaign_checkpoints(training=training, output=f"file://{output}")

    assert StoragePath(root).exists()
    assert not (output / "prune-plan.json").exists()


def test_foreign_checkpoint_root_is_never_pruned(tmp_path):
    training, root, _export, _ = stage_run(tmp_path, export_under_temporary_root=False)
    output = tmp_path / "storage" / "2026.11.26.532"
    record = json.loads((Path(training) / ".artifact.json").read_text())
    record["result"]["checkpoint_root"] = f"file://{tmp_path}/someone-else/checkpoints"
    (Path(training) / ".artifact.json").write_text(json.dumps(record))

    with pytest.raises(ValueError, match="campaign-owned"):
        retain_campaign_checkpoints(training=training, output=f"file://{output}")

    assert StoragePath(root).exists()


def test_uncommitted_final_checkpoint_aborts_before_any_deletion(tmp_path):
    training, root, _export, _ = stage_run(tmp_path, export_under_temporary_root=False)
    StoragePath(root / f"global_step_{FINAL_STEP}" / "trainer_state.pt").rm()
    output = tmp_path / "storage" / "2026.11.26.532"

    with pytest.raises(ValueError, match="not committed"):
        retain_campaign_checkpoints(training=training, output=f"file://{output}")

    assert StoragePath(root).exists()
