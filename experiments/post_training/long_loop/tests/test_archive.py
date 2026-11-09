# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from marin.execution.lazy import ArtifactStep

from experiments.post_training.long_loop.archive import (
    ArchiveInput,
    checkpoint_cleanup,
    digest,
    seal_archive,
    verify_archive,
)
from experiments.post_training.long_loop.archive_stage import REQUIRED_CATEGORIES, archive_run, archive_step
from experiments.post_training.long_loop.graph import execute_local_graph


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


def test_cleanup_is_verified_scoped_and_preserves_selected(tmp_path):
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
        size, checksum = digest(str(path))
        return {"source": str(path), "size": size, "sha256": checksum}

    seal_archive(
        root,
        [ArchiveInput(str(keep / "weight"), "best/weight", "checkpoint")],
        {
            "checkpoint_inventory": {
                "root": str(checkpoints),
                "selected": str(keep),
                "selected_files": [inventory(keep / "weight")],
                "disposable_files": [inventory(item)],
            },
        },
    )
    with pytest.raises(ValueError, match="selected"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(keep / "weight")])
    item.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    item.write_bytes(b"old")
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    assert not item.exists()
    assert (keep / "weight").read_bytes() == b"best"


def test_archive_postscript_runs_without_training_and_keeps_verbatim_inputs(tmp_path, monkeypatch):

    report = tmp_path / "report"
    report.mkdir()
    (report / "evaluation.json").write_text('{"all_verified": true}')
    selected = tmp_path / "checkpoints" / "global_step_5"
    selected.mkdir(parents=True)
    weight = selected / "weight"
    weight.write_bytes(b"selected-native-weights")
    inputs = []
    for category in sorted(REQUIRED_CATEGORIES - {"checkpoint"}):
        path = tmp_path / f"{category}.txt"
        path.write_bytes(b"verbatim evidence\n")
        inputs.append({"source": str(path), "name": path.name, "category": category})
    inputs.append({"source": str(weight), "name": "best/weight", "category": "checkpoint"})
    size, sha = digest(str(weight))
    plan = {
        "inputs": inputs,
        "provenance": {
            "marin_commit": "a" * 40,
            "evaluation_complete": True,
            "checkpoint_inventory": {
                "root": str(selected.parent),
                "selected": str(selected),
                "selected_files": [{"source": str(weight), "size": size, "sha256": sha}],
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
    completed = ArtifactStep.adopt("test/completed-report", "2026.11.09", str(report))
    stage = archive_step("2026.11.09", plan, completed)
    execute_local_graph(stage)
    root = stage.path()
    archive_run(output=root, plan=plan, completed_report=str(report))
    assert {item["category"] for item in verify_archive(root)["objects"]} == REQUIRED_CATEGORIES
    assert weight.read_bytes() == b"selected-native-weights"
