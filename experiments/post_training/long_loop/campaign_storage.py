# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote one campaign final checkpoint and remove its verified scratch payloads."""

import hashlib
import json

from iris.cluster.client.job_info import get_job_info
from marin.rl.skyrl import SkyRLRun, skyrl_temporary_run_path
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.fsutil.verified_copy import verified_copy_prefix

from vendor.skyrl.marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from vendor.skyrl.marinskyrl.model_manifest import ModelManifest


def retain_campaign_checkpoints(*, training: str, output: str) -> None:
    """Verify the delivered HF bytes, promote final native state, then prune run-owned scratch."""
    if get_job_info() is None or not output.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Campaign checkpoint housekeeping requires an 08a CPU worker")
    result = SkyRLRun.raw_load(training)
    if result.hf_model_uri is None:
        raise ValueError("Training did not commit its delivered HF export")
    hf = StoragePath(result.hf_model_uri)
    manifest = ModelManifest.from_mapping(json.loads((hf / MODEL_MANIFEST_FILENAME).read_text()), str(hf))
    for entry in manifest.files:
        fs, key = filesystem_for(str(hf / entry.path))
        if fs.size(key) != entry.size:
            raise ValueError(f"Best export file size changed: {entry.path}")
        with fs.open(key, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry.sha256:
                raise ValueError(f"Best export file checksum changed: {entry.path}")
    root = StoragePath(result.checkpoint_root)
    expected = prefix_join(skyrl_temporary_run_path(training, ttl_days=14), "checkpoints")
    if str(root) != expected or "rl-data-campaign" not in root.segments:
        raise ValueError("Only campaign-owned temporary native checkpoint roots can be pruned")
    final_step = int((root / LATEST_CHECKPOINT_FILE).read_text().strip())
    final = root / f"global_step_{final_step}"
    if not (final / "trainer_state.pt").exists():
        raise ValueError("Final checkpoint is not committed")
    permanent = prefix_join(output, f"final-native/global_step_{final_step}")
    copied = verified_copy_prefix(str(final), permanent)
    if copied.total_files <= 1:
        raise ValueError("Final checkpoint has no weights payload beyond its trainer state")
    fs, key = filesystem_for(str(root))
    inventory = fs.find(key, detail=True)
    deleted_bytes = sum(item["size"] for item in inventory.values() if item["type"] == "file")
    receipt = {
        "hf_export": str(hf),
        "hf_export_identity": manifest.identity,
        "final_native": permanent,
        "final_step": final_step,
        "final_copy": copied.manifest_url,
        "pruned_root": str(root),
        "deleted_bytes": deleted_bytes,
    }
    target = StoragePath(output)
    target.mkdirs(exist_ok=True)
    # Preserve the exact deletion plan before removing any payload. Source and target
    # are disjoint; all original permanent SFT/draft/other-run paths are outside root.
    (target / "prune-plan.json").write_text(json.dumps(receipt, indent=2))
    fs.rm(key, recursive=True)
    (target / "storage-receipt.json").write_text(json.dumps(receipt, indent=2))
