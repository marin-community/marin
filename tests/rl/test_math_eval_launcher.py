# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual datasource boundary, including inert fingerprints."""

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
from marin.execution.artifact import FingerprintMismatchError
from marin.execution.lazy import StepContext

import experiments.post_training.math_eval.launcher as launcher
from experiments.post_training.curriculum_rl.pool import GSM8K_BIN, _pool_record
from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.pool import SourceRows, build_pool, canonical_json


def test_pinned_datasource_fingerprint_performs_no_storage_reads(monkeypatch):
    train, dev = launcher.pool_inputs(launcher.POOL_ARGUMENT)
    monkeypatch.setattr(launcher.StoragePath, "read_bytes", lambda _self: pytest.fail("Fingerprint read storage"))
    ctx = StepContext.for_fingerprint(deps=train.deps())
    assert train.resolve(ctx).relative_path == "qwen/train.parquet"
    assert dev.resolve(ctx).relative_path.endswith("qwen/dev.parquet")
    assert train.step.fingerprint() == dev.step.fingerprint()
    with pytest.raises(ValueError, match="Unknown frozen pool"):
        launcher.pool_inputs("math-eval-pool@mutable")


@pytest.mark.parametrize("alteration", ["prompt", "gold", "acceptance", "order"])
def test_dataset_boundary_rejects_changed_content_after_a_valid_view(monkeypatch, alteration):
    records = [
        _pool_record(question=f"Compute {i} plus {i * i}.", answer="5", pool_bin=GSM8K_BIN, split="train", index=i)
        for i in range(10)
    ]
    built = build_pool(
        [SourceRows("fixture", "a" * 40, "MIT", records, "hash")],
        {"qwen": lambda _text: [0], "snowball": lambda _text: [0]},
        version="fixture",
        code_sha="b" * 40,
        tokenizer_hashes={"qwen": "c" * 64, "snowball": "d" * 64},
    )
    overlay = {
        "manifest_sha256": built.selection["manifest_sha256"],
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": "e" * 64,
        "statuses": {row["prompt_sha256"]: "accept" for row in built.manifest},
    }
    monkeypatch.setattr(launcher, "MANIFEST_SHA256", built.selection["manifest_sha256"])
    monkeypatch.setattr(launcher, "OVERLAY_SHA256", hashlib.sha256(canonical_json(overlay).encode()).hexdigest())
    rows = deepcopy(built.records["qwen"]["train"])
    kwargs = {"split": "train", "expected_ids": built.selection["rows"]["train"]}
    launcher.validate_view(rows, built.manifest, built.selection, overlay, **kwargs)
    if alteration == "prompt":
        rows[0]["prompt"][-1]["content"] += "Changed question."
    elif alteration == "gold":
        rows[0]["reward_model"]["ground_truth"] = "6"
    elif alteration == "acceptance":
        overlay["statuses"][rows[0]["extra_info"]["prompt_sha256"]] = "reject"
    else:
        rows.reverse()
    with pytest.raises(ValueError, match=r"changed|membership"):
        launcher.validate_view(rows, built.manifest, built.selection, overlay, **kwargs)


def test_live_pool_resolution_refuses_devbox_bulk_reads(monkeypatch):
    train, _dev = launcher.pool_inputs(launcher.POOL_ARGUMENT)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.setattr(launcher.StoragePath, "read_bytes", lambda _self: pytest.fail("Read before coordinator guard"))
    ctx = StepContext(
        output_path="unused",
        prefix="unused",
        region="us-east-02a",
        is_fingerprint=False,
        _dep_ref=lambda _step: launcher.POOL_URI,
        _runtime_args={},
        _deps=train.deps(),
    )
    with pytest.raises(ValueError, match="CPU coordinator"):
        train.resolve(ctx)


def test_pool_alias_is_pinned_against_repointing():
    train, _dev = launcher.pool_inputs(launcher.POOL_ARGUMENT)
    assert train.step.expected_fingerprint == train.step.fingerprint() == "ab36432a"
    changed = replace(train.step, adopt_source="s3://marin-us-east-02a/another-pool")
    with pytest.raises(FingerprintMismatchError):
        changed.lower()
