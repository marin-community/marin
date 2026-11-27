# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep one permanent copy of a campaign run's final HF export and prune its native scratch."""

import json

from iris.cluster.client.job_info import get_job_info
from marin.rl.skyrl import SkyRLRun, skyrl_temporary_run_path
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.filesystem.transfer import copy as transfer_copy

from vendor.skyrl.marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from vendor.skyrl.marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from vendor.skyrl.marinskyrl.model_manifest import ModelManifest

REGIONAL_OUTPUT_PREFIX = "s3://marin-us-east-08a/"


def retain_campaign_checkpoints(*, training: str, output: str) -> None:
    """Keep the delivered HF export at a permanent path, then prune the run's native scratch.

    User-approved 2026-11-27 protocol: a campaign run's deliverable is its final HF export alone.
    The export's completion manifest must exist, but its bytes are not re-hashed, and the final
    native checkpoint is no longer copied to the run output. An export still under the run's
    14-day temporary root is copied — only the export, never the native state — to a permanent
    path first. The campaign-owned temporary native checkpoint root is then deleted, keeping the
    prune-plan and storage receipts.
    """
    if get_job_info() is None or not output.startswith(REGIONAL_OUTPUT_PREFIX):
        raise ValueError("Campaign checkpoint housekeeping requires an 08a CPU worker")
    result = SkyRLRun.raw_load(training)
    if result.hf_model_uri is None:
        raise ValueError("Training did not commit its delivered HF export")
    hf = StoragePath(result.hf_model_uri)
    manifest_path = hf / MODEL_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError("Delivered HF export has no completion manifest")
    manifest = ModelManifest.from_mapping(json.loads(manifest_path.read_text()), str(hf))
    temporary_run = StoragePath(skyrl_temporary_run_path(training, ttl_days=14))
    relocation = None
    if hf.is_relative_to(temporary_run):
        # Move only the export out of the TTL root; the native state stays for the prune below.
        permanent = StoragePath(prefix_join(output, "final-hf-export/policy"))
        transfer_copy(str(hf), str(permanent), recursive=True)
        relocation = {"from": str(hf), "to": str(permanent)}
        hf = permanent
    root = StoragePath(result.checkpoint_root)
    expected = prefix_join(skyrl_temporary_run_path(training, ttl_days=14), "checkpoints")
    if str(root) != expected or "rl-data-campaign" not in root.segments:
        raise ValueError("Only campaign-owned temporary native checkpoint roots can be pruned")
    final_step = int((root / LATEST_CHECKPOINT_FILE).read_text().strip())
    final = root / f"global_step_{final_step}"
    if not (final / "trainer_state.pt").exists():
        raise ValueError("Final checkpoint is not committed")
    fs, key = filesystem_for(str(root))
    inventory = fs.find(key, detail=True)
    deleted_bytes = sum(item["size"] for item in inventory.values() if item["type"] == "file")
    receipt = {
        "hf_export": str(hf),
        "hf_export_identity": manifest.identity,
        "hf_export_relocation": relocation,
        "final_native_copy": None,
        "final_step": final_step,
        "pruned_root": str(root),
        "deleted_bytes": deleted_bytes,
        "protocol": "final-hf-export-only, user-approved 2026-11-27",
    }
    target = StoragePath(output)
    target.mkdirs(exist_ok=True)
    # Preserve the exact deletion plan before removing any payload. Source and target
    # are disjoint; all original permanent SFT/draft/other-run paths are outside root.
    (target / "prune-plan.json").write_text(json.dumps(receipt, indent=2))
    fs.rm(key, recursive=True)
    (target / "storage-receipt.json").write_text(json.dumps(receipt, indent=2))
