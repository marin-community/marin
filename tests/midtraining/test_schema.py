# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stable JSON manifest schema in ``marin.midtraining.schema``."""


import pytest
from marin.midtraining.schema import (
    SCHEMA_VERSION,
    is_run_manifest,
    read_run_manifest,
    write_run_manifest,
)


def _valid_row() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "written_at": "2026-05-15T00:00:00+00:00",
        "logical_cell_id": "cell-x",
        "attempt": 1,
        "run_id": "cell-x-a001",
        "mode": "cpt",
        "output_path": "gs://marin-us-east5/checkpoints/cell-x-a001",
        "wandb_project": "delphi-midtraining",
        "wandb_entity": "marin-community",
        "base_flops_key": "1e21",
        "tpu_type": "v5p-64",
        "train_batch_size": 512,
        "per_device_parallelism": -1,
        "max_task_failures": 100,
        "data_manifest_uri": "gs://marin-us-east5/midtrain-manifests/data/p33m67/abc.json",
        "data_manifest_fingerprint": "sha256:abc",
        "tokenizer": {
            "key": "llama3",
            "hf_repo": "meta-llama/Meta-Llama-3-8B",
            "revision": "62bd457b6fe961a42a631306577e622c83876cb6",
            "bos_token_id": 128_000,
            "eos_token_id": 128_001,
            "vocab_size": 128_256,
            "fingerprint": None,
        },
        "seq_len": 4096,
        "num_train_steps": 4416,
        "actual_tokens": 9_260_000_000,
        "train_config_uri": "gs://marin-us-east5/checkpoints/cell-x-a001/train_lm_config.yaml",
        "permanent_checkpoints_uri": "gs://marin-us-east5/checkpoints/cell-x-a001/checkpoints",
        "temp_checkpoints_uri": "gs://marin-us-east5/tmp/ttl=14d/...",
        "init_checkpoint_uri": "gs://marin-us-central2/.../step-21979",
        "staged_checkpoint_uri": None,
        "cooldown_stage_record": None,
        "preflight_failures": [],
        "preflight_warnings": [],
        "preflight_notes": ["permanent checkpoints search path: ..."],
        "extra_tags": [],
        "status": "planned",
    }


def test_valid_row_passes_typeguard():
    assert is_run_manifest(_valid_row())


def test_invalid_mode_fails_typeguard():
    row = _valid_row()
    row["mode"] = "sft"
    assert not is_run_manifest(row)


def test_missing_required_key_fails_typeguard():
    row = _valid_row()
    del row["run_id"]
    assert not is_run_manifest(row)


def test_roundtrip_via_local_file(tmp_path):
    row = _valid_row()
    path = tmp_path / "midtrain_manifest.json"
    write_run_manifest(row, path)
    loaded = read_run_manifest(path)
    assert loaded["run_id"] == row["run_id"]
    assert loaded["tokenizer"]["bos_token_id"] == 128_000


def test_read_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a manifest"}', encoding="utf-8")
    with pytest.raises(TypeError, match="Invalid run manifest"):
        read_run_manifest(path)


def test_write_refuses_malformed_row(tmp_path):
    bad = {"hello": "world"}
    with pytest.raises(TypeError, match="malformed RunManifestRow"):
        write_run_manifest(bad, tmp_path / "x.json")  # type: ignore[arg-type]
