# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import pytest
from marin.execution.artifact import FingerprintMismatchError
from marin.execution.lazy import StepContext

from experiments.post_training.math_eval import bucket_launcher as bucket
from experiments.post_training.math_eval import launcher
from experiments.post_training.math_eval.pool import canonical_json


def test_bucket_fingerprint_preserves_original_dev_and_does_not_read_storage(monkeypatch):
    train, dev = launcher.pool_inputs(bucket.BUCKET_ARGUMENT)
    monkeypatch.setattr(bucket.StoragePath, "read_bytes", lambda _self: pytest.fail("Fingerprint read storage"))
    ctx = StepContext.for_fingerprint(deps=train.deps())
    assert train.resolve(ctx).relative_path == "train.parquet"
    assert dev.step.fingerprint() == "ab36432a"
    assert train.step.expected_fingerprint == train.step.fingerprint()
    with pytest.raises(FingerprintMismatchError):
        replace(train.step, adopt_source="s3://marin-us-east-02a/foreign").lower()


def fixture_selection(monkeypatch):
    ids = ["a" * 64, "b" * 64]
    digest = hashlib.sha256(canonical_json(ids).encode()).hexdigest()
    monkeypatch.setattr(bucket, "TRAIN_ROWS", 2)
    monkeypatch.setattr(bucket, "IDS_SHA256", digest)
    selection = {
        "ratings_sha256": bucket.RATINGS_SHA256,
        "selected_ids_sha256": digest,
        "prompt_sha256": ids,
        "manifest_sha256": launcher.MANIFEST_SHA256,
        "audit_overlay_sha256": launcher.OVERLAY_SHA256,
        "adoptable": True,
        "combined": {"meets_thresholds": True},
    }
    digest = hashlib.sha256(canonical_json(selection).encode()).hexdigest()
    monkeypatch.setattr(bucket, "SELECTION_SHA256", digest)
    selection["selection_sha256"] = digest
    content = canonical_json(selection).encode()
    monkeypatch.setattr(bucket, "SELECTION_BYTES_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(bucket, "TRAIN_SHA256", hashlib.sha256(b"parquet-fixture").hexdigest())
    return selection, content


def test_exact_bytes_bind_before_deserialization(monkeypatch):
    selection, content = fixture_selection(monkeypatch)
    assert bucket.validate_bucket_bytes(b"parquet-fixture", content) == selection
    with pytest.raises(ValueError, match="parquet"):
        bucket.validate_bucket_bytes(b"PARQUET-fixture", content)
    with pytest.raises(ValueError, match="selection bytes"):
        bucket.validate_bucket_bytes(b"parquet-fixture", content + b" ")


@pytest.mark.parametrize("poison", ["ratings", "ids", "acceptance", "manifest", "overlay", "threshold"])
def test_rehashed_selection_must_still_bind_pinned_sources(monkeypatch, poison):
    selection, _content = fixture_selection(monkeypatch)
    if poison == "ratings":
        selection["ratings_sha256"] = "c" * 64
    elif poison == "ids":
        selection["prompt_sha256"].reverse()
    elif poison == "acceptance":
        selection["adoptable"] = False
    elif poison == "manifest":
        selection["manifest_sha256"] = "c" * 64
    elif poison == "overlay":
        selection["audit_overlay_sha256"] = "c" * 64
    else:
        selection["combined"]["meets_thresholds"] = False
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    digest = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    selection["selection_sha256"] = digest
    monkeypatch.setattr(bucket, "SELECTION_SHA256", digest)
    content = canonical_json(selection).encode()
    monkeypatch.setattr(bucket, "SELECTION_BYTES_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match="provenance"):
        bucket.validate_bucket_bytes(b"parquet-fixture", content)


def test_live_bucket_resolution_requires_regional_coordinator(monkeypatch):
    train, _dev = bucket.bucket_inputs()
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.setattr(bucket.StoragePath, "read_bytes", lambda _self: pytest.fail("Read before coordinator guard"))
    ctx = StepContext(
        output_path="unused",
        prefix="unused",
        region="us-east-02a",
        is_fingerprint=False,
        _dep_ref=lambda _step: bucket.BUCKET_URI,
        _runtime_args={},
        _deps=train.deps(),
    )
    with pytest.raises(ValueError, match="CPU coordinator"):
        train.resolve(ctx)
