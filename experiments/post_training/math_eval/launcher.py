# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned pool inputs with verification at the coordinator's data-resolution boundary."""

import hashlib
import json
import os
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import ArtifactDataSource
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.math_eval.audit_overlay import validated_statuses
from experiments.post_training.math_eval.contract import QWEN, prompt_messages
from experiments.post_training.math_eval.pool import canonical_json

POOL_ARGUMENT = "math-eval-pool@1.0.0-candidate1"
POOL_URI = "s3://marin-us-east-02a/marin/users/ahmad/documents/math-eval-pool/1.0.0-candidate1"
MANIFEST_SHA256 = "d39aa170289c9637427bb2f6a829b49321875aa1ebc054fbcda7150bd2482cc1"
OVERLAY_SHA256 = "6446f8764259723e6c3e56577745f2c6b636401016adeff5e4c49bca68e7874c"
DEV_SHA256 = "c9554313800cc9305bbd46737dfa207f4841552ab357fc6c1d731c1ae0bbef76"
BATTERY_PATH = "batteries/mechanical-v1"
TRAIN_ROWS = 14431
DEV_ROWS = 256


def validate_view(rows, manifest, selection, overlay, *, split, expected_ids):
    """Check every consumed question, gold, template, bin and acceptance verdict."""
    if hashlib.sha256(canonical_json(manifest).encode()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Pinned candidate manifest changed")
    if selection["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Candidate selection points to another manifest")
    statuses, overlay_sha = validated_statuses(manifest, selection, overlay)
    if overlay_sha != OVERLAY_SHA256:
        raise ValueError("Pinned audit overlay changed")
    lookup = {row["prompt_sha256"]: row for row in manifest}
    ids = [row["extra_info"]["prompt_sha256"] for row in rows]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError("Dataset membership or order changed")
    for row in rows:
        digest = row["extra_info"]["prompt_sha256"]
        item = lookup[digest]
        if (
            item["split"] != split
            or statuses[digest] != "accept"
            or row["prompt"] != prompt_messages(item["problem"], item["env_class"])
            or row["env_class"] != item["env_class"]
            or row["data_source"] != item["bin"]
            or row["reward_model"]["ground_truth"] != item["gold"]
            or row["reward_spec"]["ground_truth"] != item["gold"]
            or row["extra_info"]["prompt_template_id"] != QWEN.template_id
            or row["extra_info"].get("contract") != item["contract"]
        ):
            raise ValueError(f"Dataset row changed or lacks acceptance: {digest}")


@dataclass(frozen=True)
class VerifiedPoolDataSource(ArtifactDataSource):
    """Check S3-returned bytes before a real training config can consume this source."""

    def resolve(self, ctx: StepContext):
        resolved = super().resolve(ctx)
        if ctx.is_fingerprint:
            return resolved
        if not os.environ.get("IRIS_TASK_ID"):
            raise ValueError("Resolve the pool inside an Iris CPU coordinator; bulk reads cannot run on the devbox")
        if resolved.uri != POOL_URI:
            raise ValueError("Pinned pool URI changed")
        selection = json.loads(StoragePath(POOL_URI + "/selection.json").read_bytes())
        manifest = pq.read_table(pa.BufferReader(StoragePath(POOL_URI + "/manifest.parquet").read_bytes())).to_pylist()
        overlay = json.loads(StoragePath(f"{POOL_URI}/{BATTERY_PATH}/audit-overlay.json").read_bytes())
        content = StoragePath(POOL_URI + "/" + self.relative_path).read_bytes()
        rows = pq.read_table(pa.BufferReader(content)).to_pylist()
        if self.relative_path == "qwen/train.parquet":
            split = "train"
            expected_ids = [row["prompt_sha256"] for row in manifest if row["split"] == split]
            expected_rows = TRAIN_ROWS
        elif self.relative_path == BATTERY_PATH + "/qwen/dev.parquet":
            if hashlib.sha256(content).hexdigest() != DEV_SHA256:
                raise ValueError("Pinned development parquet changed")
            battery = json.loads(StoragePath(f"{POOL_URI}/{BATTERY_PATH}/selection.json").read_bytes())
            split = "dev"
            expected_ids = battery["protocols"]["qwen/dev"]["prompt_sha256"]
            expected_rows = DEV_ROWS
        else:
            raise ValueError("Unregistered pool data view")
        if len(rows) != expected_rows:
            raise ValueError("Pinned dataset size changed")
        validate_view(rows, manifest, selection, overlay, split=split, expected_ids=expected_ids)
        return resolved


def pool_inputs(argument: str):
    """Only the audited MVP is selectable; None remains the launcher's legacy path."""
    if argument != POOL_ARGUMENT:
        raise ValueError(f"Unknown frozen pool; expected {POOL_ARGUMENT}")
    artifact = ArtifactStep.adopt(
        user_owned_name("documents/math-eval-pool"),
        "2026.09.08.1",
        POOL_URI,
        kind=Artifact,
        config={
            "manifest_sha256": MANIFEST_SHA256,
            "audit_overlay_sha256": OVERLAY_SHA256,
            "dev_sha256": DEV_SHA256,
            "train_rows": TRAIN_ROWS,
            "validation_rows": DEV_ROWS,
        },
    )
    artifact = replace(artifact, expected_fingerprint="ab36432a")
    return (
        VerifiedPoolDataSource(artifact, "qwen/train.parquet"),
        VerifiedPoolDataSource(artifact, BATTERY_PATH + "/qwen/dev.parquet"),
    )
