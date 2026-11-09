# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.execution.artifact import write_artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for

from experiments.post_training.long_loop.archive import (
    ArchiveInput,
    checkpoint_cleanup,
    seal_archive,
    verify_archive,
)
from experiments.post_training.long_loop.archive_stage import REQUIRED_CATEGORIES, archive_selected_run
from experiments.post_training.long_loop.graph import execute_local_graph
from experiments.post_training.long_loop.hf_archive import file_identity, upload_checkpoint
from experiments.post_training.long_loop.tests.test_hf_archive import upload_config


def test_archive_survives_source_loss_and_detects_corruption(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_bytes(b"literal: 1\n# exact comment\n")
    root = str(tmp_path / "archive")
    inputs = [ArchiveInput(str(source), "configs/consumed.yaml", "config")]
    first = seal_archive(root, inputs, {"marin_commit": "a" * 40})
    assert seal_archive(root, inputs, {"marin_commit": "a" * 40}) == first
    source.unlink()
    assert verify_archive(root) == first
    Path(first["objects"][0]["destination"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="failed verification"):
        verify_archive(root)


def test_cleanup_is_verified_scoped_and_preserves_selected(tmp_path, hub):
    checkpoints = tmp_path / "checkpoints"
    keep = checkpoints / "global_step_3"
    old = checkpoints / "global_step_2"
    keep.mkdir(parents=True)
    old.mkdir()
    (keep / "weight").write_bytes(b"best")
    item = old / "weight"
    item.write_bytes(b"old")
    root = str(tmp_path / "archive")

    def inventory(path):
        fs, key = filesystem_for(str(path))
        return {"source": str(path), "storage_identity": file_identity(fs.info(key))}

    receipt = upload_checkpoint(
        upload_config(tmp_path, keep), timing_destination=str(tmp_path / "timing" / "upload.jsonl")
    )
    seal_archive(
        root,
        [],
        {
            "hf_checkpoint": receipt,
            "checkpoint_inventory": {
                "root": str(checkpoints),
                "selected": str(keep),
                "ttl": "14d",
                "disposable_files": [inventory(item)],
            },
        },
    )
    with pytest.raises(ValueError, match="selected"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(keep / "weight")])
    original_stat = item.stat()
    item.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    item.write_bytes(b"old")
    os.utime(item, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    assert not item.exists()
    assert (keep / "weight").read_bytes() == b"best"


def test_archive_postscript_runs_without_training_and_keeps_verbatim_inputs(tmp_path, monkeypatch, hub):

    report = tmp_path / "report"
    report.mkdir()
    (report / "evaluation.json").write_text('{"all_verified": true}')
    selected = tmp_path / "checkpoints" / "global_step_5"
    selected.mkdir(parents=True)
    weight = selected / "weight"
    weight.write_bytes(b"selected-native-weights")
    inputs = []
    for category in sorted(REQUIRED_CATEGORIES):
        path = tmp_path / f"{category}.txt"
        path.write_bytes(b"verbatim evidence\n")
        inputs.append({"source": str(path), "name": path.name, "category": category})
    plan = {
        "inputs": inputs,
        "hf_upload": asdict(upload_config(tmp_path, selected)),
        "provenance": {
            "marin_commit": "a" * 40,
            "evaluation_complete": True,
            "checkpoint_inventory": {
                "root": str(selected.parent),
                "selected": str(selected),
                "ttl": "14d",
                "disposable_files": [],
            },
        },
    }
    native = tmp_path / "native-eval"
    native.mkdir()
    (native / "bench.jsonl").write_text(
        json.dumps(
            {
                "uid": "task1",
                "data_source": "bench",
                "env_class": "taskcompendium",
                "verification": {"status": "verified", "score": 1, "passed": True},
            }
        )
        + "\n"
    )
    plan["evaldash"] = {
        "source": str(native),
        "records_prefix": str(tmp_path / "evals"),
        "record": {
            "run_id": "test",
            "group_id": "test",
            "created_at": "2026-11-09T00:00:00Z",
            "user": "test",
            "model": {"name": "test", "location": "test", "backend": "vllm"},
            "eval": {"name": "smoke", "mechanism": "skyrl"},
            "hardware": {"platform": "local", "accelerator": "none", "region_or_cluster": None},
            "status": "succeeded",
            "error": None,
            "results_path": "resolved-by-archive",
            "metrics": {},
            "jobs": {},
            "log_tails": {},
            "provenance": {"git_sha": "a" * 40, "launch_host": "local"},
        },
    }
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    (selected.parent / "best_checkpoint.json").write_text(
        json.dumps({"step": 5, "model_uri": str(selected / "policy"), "ema_window": 5})
    )
    (selected.parent / "validation_selection_history.json").write_text("[]")
    prior = selected.parent / "global_step_4"
    prior.mkdir()
    (prior / "weights.distcp").write_bytes(b"unused native")
    plan["native_checkpoint_ttl"] = "14d"
    trained_path = str(tmp_path / "rl-artifact")
    write_artifact(
        SkyRLRun(
            hf_model_uri=plan["hf_upload"]["source"],
            global_step=5,
            tokenizer_uri="test",
            tokenizer_revision="a" * 40,
            checkpoint_root=str(selected.parent),
            draft_checkpoint_root=None,
            terminal_manifest_uri="test",
            iris_job_id="test",
        ).result_payload(),
        trained_path,
    )
    trained = ArtifactStep.adopt("test/rl", "2026.11.09", trained_path, kind=SkyRLRun)
    completed = ArtifactStep.adopt("test/completed-report", "2026.11.09", str(report))
    stage = apply(
        "test/archive",
        archive_selected_run,
        version="2026.11.09",
        output=OUT,
        plan=plan,
        trained=trained,
        completed_report=completed,
    )
    execute_local_graph(stage)
    root = stage.path()
    archive_selected_run(output=root, plan=plan, trained=trained_path, completed_report=str(report))
    assert {item["category"] for item in verify_archive(root)["objects"]} == REQUIRED_CATEGORIES
    manifest = verify_archive(root)
    assert manifest["provenance"]["hf_checkpoint"]["revision"] == hub.revision
    assert manifest["provenance"]["checkpoint_inventory"]["ttl"] == "14d"
    assert not any("weight" in item["destination"] for item in manifest["objects"])
    assert weight.read_bytes() == b"selected-native-weights"
    assert not (prior / "weights.distcp").exists()
