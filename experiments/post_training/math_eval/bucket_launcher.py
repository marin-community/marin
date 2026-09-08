# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable measured Bucket S inputs, retaining the frozen development panel."""

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import ArtifactDataSource
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.math_eval import launcher
from experiments.post_training.math_eval.pool import canonical_json

BUCKET_ARGUMENT = "math-eval-pool@1.0.0-qwen-bucket-s1"
BUCKET_URI = launcher.POOL_URI + "/buckets/qwen-s-v1"
RATINGS_SHA256 = "8705243f5dcec88fd10664f471730ede0f8471e69f092ff27d0e39f8841bd806"
SELECTION_SHA256 = "0654080a14b7ad73eeffaa0c01e7692f30154664c1d82e62b78835b08bbf113b"
SELECTION_BYTES_SHA256 = "86b4638ea50cd971d217d673042258d11aecf69c4a4eb1c555368bfd8d752e31"
TRAIN_SHA256 = "6c6c289c0d06346cc76229038e8efe46428f0a493d7a911a36e9f0c6980a1cef"
IDS_SHA256 = "cab7e27370ef70ac7c7cba83f859ee448553efeb9b9d0b1284eb67ae3d8c8d8b"
TRAIN_ROWS = 1918


def validate_bucket_bytes(content, selection_content):
    """Bind exact returned bytes and membership before decoding training inputs."""
    if hashlib.sha256(content).hexdigest() != TRAIN_SHA256:
        raise ValueError("Pinned Bucket S training parquet changed")
    if hashlib.sha256(selection_content).hexdigest() != SELECTION_BYTES_SHA256:
        raise ValueError("Pinned Bucket S selection bytes changed")
    selection = json.loads(selection_content)
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        hashlib.sha256(canonical_json(unsigned).encode()).hexdigest() != SELECTION_SHA256
        or selection["selection_sha256"] != SELECTION_SHA256
        or selection["ratings_sha256"] != RATINGS_SHA256
        or selection["selected_ids_sha256"] != IDS_SHA256
        or hashlib.sha256(canonical_json(selection["prompt_sha256"]).encode()).hexdigest() != IDS_SHA256
        or selection["manifest_sha256"] != launcher.MANIFEST_SHA256
        or selection["audit_overlay_sha256"] != launcher.OVERLAY_SHA256
        or selection["adoptable"] is not True
        or selection["combined"]["meets_thresholds"] is not True
        or len(selection["prompt_sha256"]) != TRAIN_ROWS
        or len(set(selection["prompt_sha256"])) != TRAIN_ROWS
    ):
        raise ValueError("Pinned Bucket S selection provenance changed")
    return selection


@dataclass(frozen=True)
class VerifiedBucketDataSource(ArtifactDataSource):
    """Verify measured membership and original prompt/gold/contract at resolution."""

    def resolve(self, ctx: StepContext):
        resolved = super().resolve(ctx)
        if ctx.is_fingerprint:
            return resolved
        if not os.environ.get("IRIS_TASK_ID"):
            raise ValueError("Resolve Bucket S inside an Iris CPU coordinator")
        if resolved.uri != BUCKET_URI or self.relative_path != "train.parquet":
            raise ValueError("Pinned Bucket S URI changed")
        content = StoragePath(BUCKET_URI + "/train.parquet").read_bytes()
        selected = validate_bucket_bytes(content, StoragePath(BUCKET_URI + "/selection.json").read_bytes())
        rows = pq.read_table(pa.BufferReader(content)).to_pylist()
        manifest = pq.read_table(
            pa.BufferReader(StoragePath(launcher.POOL_URI + "/manifest.parquet").read_bytes())
        ).to_pylist()
        source_selection = json.loads(StoragePath(launcher.POOL_URI + "/selection.json").read_bytes())
        overlay = json.loads(StoragePath(launcher.POOL_URI + "/batteries/mechanical-v1/audit-overlay.json").read_bytes())
        launcher.validate_view(
            rows, manifest, source_selection, overlay, split="train", expected_ids=selected["prompt_sha256"]
        )
        if Counter(row["data_source"] for row in rows) != {"g02-rg-sum-hard": 854, "g05-math-l12": 1064}:
            raise ValueError("Pinned Bucket S source quotas changed")
        return resolved


def bucket_inputs():
    artifact = ArtifactStep.adopt(
        user_owned_name("documents/math-eval-qwen-bucket-s"),
        "2026.09.08.1",
        BUCKET_URI,
        kind=Artifact,
        config={
            "ratings_sha256": RATINGS_SHA256,
            "selection_sha256": SELECTION_SHA256,
            "train_sha256": TRAIN_SHA256,
            "train_ids_sha256": IDS_SHA256,
            "train_rows": TRAIN_ROWS,
            "dev_sha256": launcher.DEV_SHA256,
        },
    )
    artifact = replace(artifact, expected_fingerprint="7e3208db")
    _train, dev = launcher.pool_inputs(launcher.POOL_ARGUMENT)
    return VerifiedBucketDataSource(artifact, "train.parquet"), dev
