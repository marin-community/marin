# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from marin.execution.artifact import RECORD_FILENAME, write_artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from wandb.apis.public.runhistory.downloads import IncompleteRunHistoryError

from experiments.post_training.long_loop import archive_stage
from experiments.post_training.long_loop.archive import (
    ArchiveInput,
    checkpoint_cleanup,
    seal_archive,
    verify_archive,
)
from experiments.post_training.long_loop.archive_stage import (
    REQUIRED_CATEGORIES,
    archive_selected_run,
    derive_archive_plan,
)
from experiments.post_training.long_loop.graph import execute_local_graph
from experiments.post_training.long_loop.hf_archive import file_identity, upload_checkpoint
from experiments.post_training.long_loop.tests.test_hf_archive import upload_config


def test_archive_survives_source_loss_and_detects_corruption(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_bytes(b"literal: 1\n# exact comment\n")
    root = str(tmp_path / "archive")
    inputs = [ArchiveInput(str(source), "configs/consumed.yaml", "config")]
    first = seal_archive(root, inputs, {"marin_commit": "a" * 40})
    assert seal_archive(root + "/", inputs, {"marin_commit": "a" * 40}) == first
    source.unlink()
    assert verify_archive(root + "/") == first
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


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("unscored", [False, True])
def test_archive_postscript_runs_without_training_and_keeps_verbatim_inputs(
    tmp_path, monkeypatch, hub, incomplete, unscored
):

    report = tmp_path / "report"
    report.mkdir()
    accepted = [{"data_source": "bench", "uid": "task1", "exception_type": "AddTestsDirError"}]
    (report / "evaluation.json").write_text(
        json.dumps(
            {
                "all_verified": not unscored,
                "expected_rollouts": 1,
                "completed_rollouts": 1,
                "source_coverage_matches": True,
                "identity_coverage_matches": True,
                "unscored_attempts": accepted if unscored else [],
                "accepted_unscored_attempts": accepted if unscored else [],
            }
        )
    )
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
            "evaluation_complete": not unscored,
            "checkpoint_inventory": {
                "root": str(selected.parent),
                "selected": str(selected),
                "ttl": "14d",
                "disposable_files": [],
            },
        },
    }
    if incomplete:
        plan["provenance"]["missing_evidence"] = {"timing": "Native journal coverage incomplete"}
        plan["provenance"]["cleanup_blockers"] = ["Native journal coverage incomplete"]
    if unscored:
        plan["provenance"].setdefault("missing_evidence", {})["evaluation"] = "One explicitly unscored attempt"
        plan["provenance"].setdefault("cleanup_blockers", []).append("One explicitly unscored attempt")
    native = tmp_path / "native-eval"
    native.mkdir()
    (native / "bench.jsonl").write_text(
        json.dumps(
            {
                "uid": "task1",
                "data_source": "bench",
                "env_class": "taskcompendium",
                "verification": (
                    {"status": "error", "score": None, "passed": None}
                    if unscored
                    else {"status": "verified", "score": 1, "passed": True}
                ),
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
            "status": "infra_failed" if unscored else "succeeded",
            "error": "One unscored attempt" if unscored else None,
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
    assert manifest["provenance"]["evaluation_complete"] is not unscored
    assert (prior / "weights.distcp").exists() == (incomplete or unscored)
    assert Path(root, "cleanup-blocked.json").exists() == (incomplete or unscored)


@pytest.mark.parametrize("incomplete_wandb_history", [False, True])
@pytest.mark.parametrize("rl_state", ["succeeded", "failed"])
@pytest.mark.parametrize("unscored", [False, True])
def test_runtime_plan_uses_consumed_paths_and_preserves_missing_evidence(
    tmp_path, monkeypatch, incomplete_wandb_history, rl_state, unscored
):
    def document(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    settings = yaml.safe_load(Path("experiments/post_training/long_loop/specs/archive.yaml").read_text())
    roots = {stage: str(tmp_path / stage) for stage in settings["stage_files"]}
    trained = roots["rl"]
    run = SkyRLRun(
        hf_model_uri=str(tmp_path / "hf"),
        global_step=5,
        tokenizer_uri="test",
        tokenizer_revision="a" * 40,
        checkpoint_root=str(tmp_path / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri="test",
        iris_job_id="test",
    )
    write_artifact(run.result_payload(), trained)
    record = json.loads((Path(trained) / RECORD_FILENAME).read_text())
    document(Path(trained) / RECORD_FILENAME, {**record, "version": "2026.11.10"})
    for stage, entries in settings["stage_files"].items():
        for entry in entries:
            document(Path(roots[stage]) / entry["path"], {})
    terminal = {"state": rl_state, "failure": "GroupAdmissionStalledError" if rl_state == "failed" else None}
    document(Path(trained) / "terminal.json", {"result": terminal})
    tracker = {"entity": None, "project": "test", "id": None}
    rows = [{"_step": 4, "loss": 0.4, "nested": {"key": "é"}}, {"other": 12}, {"_step": 2, "loss": 0.6}]
    if incomplete_wandb_history:
        tracker = {"entity": "test-entity", "project": "test", "id": "sft-id"}

        class Run:
            def download_history_exports(self, directory, require_complete_history):
                assert require_complete_history
                raise IncompleteRunHistoryError()

            def scan_history(self):
                yield from rows

        class Client:
            def __init__(self, *, api_key):
                assert api_key == "fake-for-offline-test"

            def run(self, path):
                assert path == "test-entity/test/sft-id"
                return Run()

        monkeypatch.setattr(archive_stage.wandb, "Api", Client)
        monkeypatch.setattr(
            archive_stage.subprocess,
            "run",
            lambda args, **kwargs: subprocess.CompletedProcess(
                args, 1, stdout=b"", stderr=b"wandb.apis.public.runhistory.downloads.IncompleteRunHistoryError\n"
            ),
        )
        monkeypatch.setenv("WANDB_API_KEY", "fake-for-offline-test")
    document(
        Path(roots["sft"]) / "consumed-config.json",
        {
            "trainer": {"trainer": {"tracker": tracker}},
            "resources": {"replicas": 2},
            "processes_per_task": 4,
        },
    )
    for stage in ("rl", "eval"):
        retention = tmp_path / ("retention-" + stage)
        document(retention / "_retention_ledger.json", {"archives": {"trace.zip": {"bytes": 3}}, "records": {"id": {}}})
        (retention / "trace.zip").write_bytes(b"zip")
        document(
            Path(roots[stage]) / "resolved-launch.yaml",
            {
                "config": {
                    "skyrl": {
                        "generator": {"trajectory_retention": {"output_path": str(retention)}},
                        "trainer": {"export_path": str(tmp_path / ("exports-" + stage)), "project_name": "test"},
                    },
                }
            },
        )
    document(Path(roots["eval"]) / "exports/dumped_evals/eval_only/bench.jsonl", {"uid": "id"})
    document(Path(roots["report"]) / "timing-sources.json", [])
    document(
        Path(roots["report"]) / "evaluation.json",
        {
            "all_verified": not unscored,
            "result_uri": roots["eval"],
            "expected_rollouts": 44,
            "completed_rollouts": 44,
            "verified_rollouts": 43 if unscored else 44,
            "source_coverage_matches": True,
            "identity_coverage_matches": True,
            "unscored_attempts": (
                [{"data_source": "bench", "uid": "id", "exception_type": "AddTestsDirError"}] if unscored else []
            ),
            "accepted_unscored_attempts": (
                [{"data_source": "bench", "uid": "id", "exception_type": "AddTestsDirError"}] if unscored else []
            ),
        },
    )
    spec = {"marin_revision": "a" * 40, "name": "test", "datasets": {"rows": "pinned"}, "rl": {"execution": {}}}
    plan = derive_archive_plan(
        output=str(tmp_path / "archive"),
        settings=settings,
        resolved_spec=spec,
        artifacts=roots,
        trained=trained,
        evaluated=roots["eval"],
        completed_report=roots["report"],
    )
    assert plan["hf_upload"]["repo_id"].endswith("-2026.11.10")
    assert plan["provenance"]["rl_terminal"] == terminal
    assert set(plan["provenance"]["missing_evidence"]) == (
        {"timing", "wandb", "evaluation"} if unscored else {"timing", "wandb"}
    )
    assert plan["provenance"]["evaluation_complete"] is not unscored
    assert plan["evaldash"]["record"]["status"] == ("infra_failed" if unscored else "succeeded")
    assert any(item["name"] == "eval_inputs/selection.json" for item in plan["inputs"])
    assert sum(item["name"].endswith("trace.zip") for item in plan["inputs"]) == 2
    evidence = json.loads((tmp_path / "archive/derived-inputs/evidence-status.json").read_text())
    assert evidence["native_sft_timing_coverage"]["native-sft-journal"]["expected"] == 8
    if incomplete_wandb_history:
        manifest = seal_archive(
            str(tmp_path / "sealed"), [ArchiveInput(**item) for item in plan["inputs"]], plan["provenance"]
        )
        history = next(item for item in manifest["objects"] if item["destination"].endswith("scan-history.jsonl"))
        raw = Path(history["destination"]).read_bytes()
        assert [json.loads(line) for line in raw.splitlines()] == rows
        receipt_object = next(
            item for item in manifest["objects"] if item["destination"].endswith("scan-history-receipt.json")
        )
        receipt = json.loads(Path(receipt_object["destination"]).read_text())
        assert receipt["sha256"] == hashlib.sha256(raw).hexdigest() == history["sha256"]
        assert receipt["bytes"] == len(raw)
        status = manifest["provenance"]["wandb_runs"][0]
        assert (status["observed_rows"], status["step_min"], status["step_max"]) == (3, 2, 4)
        assert status["provider_completeness_certified"] is False
        assert status["deletion_ready"] is False
        assert status["status"] == "scan_history_fallback_verified"
        assert "wandb" in manifest["provenance"]["missing_evidence"]
        assert manifest["provenance"]["cleanup_blockers"]


def test_wandb_export_propagates_unrelated_export_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-for-offline-test")
    monkeypatch.setattr(archive_stage, "require_secret", lambda name: "never-log-this-value")
    monkeypatch.setattr(
        archive_stage.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, stdout=b"", stderr=b"PermissionError: api_key=never-log-this-value\n"
        ),
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        archive_stage._wandb_export(
            [{"stage": "sft", "entity": "test", "project": "test", "id": "run"}],
            {"package": "fake-exporter", "include": {}},
            str(tmp_path),
        )
    details = "\n".join(caught.value.__notes__)
    assert "PermissionError" in details
    assert "never-log-this-value" not in details
    assert not list(tmp_path.glob("*.json"))


def test_wandb_history_fallback_propagates_provider_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-for-offline-test")

    class Run:
        def download_history_exports(self, directory, require_complete_history):
            raise PermissionError("provider denied history")

    class Client:
        def __init__(self, *, api_key):
            assert api_key == "fake-for-offline-test"

        def run(self, path):
            return Run()

    monkeypatch.setattr(archive_stage.wandb, "Api", Client)
    with pytest.raises(PermissionError, match="provider denied history"):
        archive_stage._scan_wandb_history("test/project/run", tmp_path)
    assert not list(tmp_path.glob("scan-history*"))
