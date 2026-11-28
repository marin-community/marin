# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml
from levanter.checkpoint import save_checkpoint
from marin.execution.artifact import RECORD_FILENAME, write_artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.buckets import filesystem_for
from wandb.apis.public.runhistory.downloads import IncompleteRunHistoryError

from experiments.post_training.long_loop import archive_stage
from experiments.post_training.long_loop.archive import (
    ArchiveInput,
    assert_best_rl_checkpoint,
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
from experiments.post_training.long_loop.recipe_bundle import runtime_source_manifest, verify_recipe_bundle
from experiments.post_training.long_loop.tests.test_hf_archive import upload_config
from vendor.skyrl.marinskyrl.checkpoint_paths import (
    BEST_CHECKPOINT_FILENAME,
    LATEST_CHECKPOINT_FILE,
    VALIDATION_RECEIPTS_FILENAME,
    ValidationReceipt,
)


def write_native_selection(root: Path, step: int = 5, reward: float = 0.5, attempted: int = 2) -> None:
    """Write a committed first-eligible native selection at the requested step."""
    root.mkdir(parents=True, exist_ok=True)
    history = [{"step": step, "reward": reward, "attempted": attempted, "scored": attempted}]
    (root / "validation_selection_history.json").write_text(json.dumps(history))
    (root / BEST_CHECKPOINT_FILENAME).write_text(
        json.dumps(
            {
                "model_format": "native",
                "step": step,
                "model_uri": str(root / f"global_step_{step}" / "policy"),
                "ema_window": 5,
                "ema_alpha": 2 / 6,
                "attempted": attempted,
                "scored": attempted,
                "expected_tasks": attempted,
                "minimum_scored_tasks": attempted,
                "eligible": True,
                "accuracy": reward,
                "evaluation_interval": 1,
            }
        )
    )
    checkpoint = root / f"global_step_{step}"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "trainer_state.pt").write_bytes(b"committed-fixture-state")
    (root / LATEST_CHECKPOINT_FILE).write_text(str(step))
    receipt = ValidationReceipt(
        step,
        attempted,
        attempted,
        attempted,
        attempted,
        0,
        True,
        "best_checkpoint_committed",
        reward,
        ema_score=reward,
        improved=True,
        checkpoint_uri=str(checkpoint / "policy"),
    )
    (root / VALIDATION_RECEIPTS_FILENAME).write_text(json.dumps([asdict(receipt)]))


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
    recent = checkpoints / "global_step_4" / "weight"
    recent.parent.mkdir()
    recent.write_bytes(b"recent")
    root = str(tmp_path / "archive")

    def inventory(path):
        fs, key = filesystem_for(str(path))
        return {"source": str(path), "storage_identity": file_identity(fs.info(key))}

    receipt = upload_checkpoint(
        upload_config(tmp_path, keep), timing_destination=str(tmp_path / "timing" / "upload.jsonl")
    )
    protected = checkpoints / "sft-final"
    protected.mkdir()
    (protected / "weight").write_bytes(b"final-sft")
    seal_archive(
        root,
        [],
        {
            "hf_checkpoint": receipt,
            "checkpoint_inventory": {
                "root": str(checkpoints),
                "selected": str(keep),
                "retained": [str(keep), str(recent.parent)],
                "ttl": "14d",
                "disposable_files": [inventory(item), inventory(recent)],
            },
        },
    )
    with pytest.raises(ValueError, match="selected"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(keep / "weight")])
    with pytest.raises(ValueError, match="retained"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(recent)])
    with pytest.raises(ValueError, match="protected checkpoint"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(protected / "weight")], protected=[str(protected)])
    original_stat = item.stat()
    item.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)], protected=[str(protected)])
    item.write_bytes(b"old")
    os.utime(item, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)], protected=[str(protected)])
    checkpoint_cleanup(root, str(checkpoints), str(keep), [str(item)])
    assert not item.exists()
    assert (keep / "weight").read_bytes() == b"best"
    assert recent.read_bytes() == b"recent"
    assert (protected / "weight").read_bytes() == b"final-sft"


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("unscored", [False, True])
@pytest.mark.parametrize("max_checkpoints", [3, -1])
def test_archive_postscript_runs_without_training_and_keeps_verbatim_inputs(
    permanent_path, monkeypatch, hub, incomplete, unscored, max_checkpoints
):
    tmp_path = permanent_path

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
    (selected / "trainer_state.pt").write_bytes(b"selected-native-state")
    sft_checkpoint = tmp_path / "sft" / "checkpoints" / "step-1"
    save_checkpoint(
        {"model": np.asarray([0.64], np.float32), "opt_state": {"count": np.asarray(1, np.float32)}},
        step=1,
        checkpoint_path=str(sft_checkpoint),
        is_temporary=False,
    )
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
            "final_sft_checkpoint": {"uri": str(sft_checkpoint), "step": 1},
            "checkpoint_inventory": {
                "root": str(selected.parent),
                "selected": str(selected),
                "retained": [str(selected)],
                "ttl": "14d",
                "disposable_files": [],
            },
        },
    }
    exported = tmp_path / "exports" / "global_step_5" / "policy"
    exported.parent.mkdir(parents=True)
    Path(plan["hf_upload"]["source"]).rename(exported)
    plan["hf_upload"]["source"] = str(exported)
    workspace = tmp_path / "runtime"
    workspace.mkdir()
    (workspace / "recipe.py").write_text("BATCH = 256\n")
    monkeypatch.setattr(archive_stage, "RUNTIME_WORKSPACE", workspace)
    plan["runtime_source"] = runtime_source_manifest(workspace, "a" * 40)
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
    write_native_selection(selected.parent)
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
    (Path(trained_path) / "resolved-launch.yaml").write_text(
        json.dumps({"config": {"skyrl": {"trainer": {"max_ckpts_to_keep": max_checkpoints}}}})
    )
    final = selected.parent / "global_step_6"
    final.mkdir()
    (final / "trainer_state.pt").write_bytes(b"committed-final-state")
    (final / "weights.distcp").write_bytes(b"final-native-weights")
    (selected.parent / LATEST_CHECKPOINT_FILE).write_text("6")
    # Failed uploads with higher step numbers must not displace the committed final.
    recent = []
    for step in (8, 10, 12):
        checkpoint = selected.parent / f"global_step_{step}"
        checkpoint.mkdir()
        path = checkpoint / "weights.distcp"
        path.write_bytes(f"native-{step}".encode())
        recent.append(path)
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
    sealed = verify_archive(root)
    assert {item["category"] for item in sealed["objects"]} == REQUIRED_CATEGORIES | {"recipe"}
    recipe = sealed["provenance"]["recipe_bundle"]
    assert verify_recipe_bundle(str(Path(root) / "objects/recipe"), recipe["manifest_sha256"]) == recipe
    manifest = verify_archive(root)
    assert manifest["provenance"]["hf_checkpoint"]["revision"] == hub.revision
    assert manifest["provenance"]["checkpoint_inventory"]["ttl"] == "14d"
    assert not any("weight" in item["destination"] for item in manifest["objects"])
    assert weight.read_bytes() == b"selected-native-weights"
    assert (final / "weights.distcp").read_bytes() == b"final-native-weights"
    assert (final / "trainer_state.pt").read_bytes() == b"committed-final-state"
    inventory = manifest["provenance"]["checkpoint_inventory"]
    assert inventory["final"] == str(final)
    assert str(final) in inventory["retained"]
    evidence = {item["destination"] for item in manifest["objects"]}
    assert {
        str(Path(root) / "objects" / "selection" / filename)
        for filename in (LATEST_CHECKPOINT_FILE, VALIDATION_RECEIPTS_FILENAME)
    } <= evidence
    assert [path.read_bytes() for path in recent] == [b"native-8", b"native-10", b"native-12"]
    assert manifest["provenance"]["checkpoint_inventory"]["max_ckpts_to_keep"] == max_checkpoints
    assert manifest["provenance"]["evaluation_complete"] is not unscored
    assert (prior / "weights.distcp").exists() == (incomplete or unscored or max_checkpoints < 0)
    assert Path(root, "cleanup-blocked.json").exists() == (incomplete or unscored)
    assert json.loads(Path(root, "final-sft-checkpoint.json").read_text())["verdict"] == "full"
    assert (sft_checkpoint / "metadata.json").exists()


@pytest.mark.parametrize("incomplete_wandb_history", [False, True])
@pytest.mark.parametrize("rl_state", ["succeeded", "failed"])
@pytest.mark.parametrize("unscored", [False, True])
@pytest.mark.parametrize("sft_format", ["GrugRunConfig", "SFTSpec"])
def test_runtime_plan_uses_consumed_paths_and_preserves_missing_evidence(
    tmp_path, monkeypatch, incomplete_wandb_history, rl_state, unscored, sft_format
):
    def document(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    settings = yaml.safe_load(Path("experiments/post_training/long_loop/specs/archive.yaml").read_text())
    roots = {stage: str(tmp_path / stage) for stage in settings["stage_files"]}
    trained = roots["rl"]
    run = SkyRLRun(
        hf_model_uri=str(tmp_path / "checkpoints" / "global_step_5" / "policy"),
        global_step=5,
        tokenizer_uri="test",
        tokenizer_revision="a" * 40,
        checkpoint_root=str(tmp_path / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri="test",
        iris_job_id="test",
    )
    write_artifact(run.result_payload(), trained)
    write_native_selection(Path(run.checkpoint_root))
    record = json.loads((Path(trained) / RECORD_FILENAME).read_text())
    document(Path(trained) / RECORD_FILENAME, {**record, "version": "2026.11.10"})
    for stage, entries in settings["stage_files"].items():
        for entry in entries:
            document(Path(roots[stage]) / entry["path"], {})
    document(Path(roots["sft"]) / "selected.json", {"checkpoint": str(tmp_path / "sft-final"), "step": 12})
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
            def __init__(self, *, api_key, overrides):
                assert api_key == "fake-for-offline-test"
                assert overrides == {"base_url": settings["wandb"]["base_url"]}

            def run(self, path):
                assert path == "test-entity/test/sft-id"
                return Run()

        monkeypatch.setattr(archive_stage.wandb, "Api", Client)

        def exporter(args, *, env, **kwargs):
            config = yaml.safe_load(Path(args[args.index("backup") + 1]).read_text())
            assert "WANDB_SERVICE" not in env
            assert os.environ["WANDB_SERVICE"] == "parent-sdk-service"
            assert args[args.index("--with") + 1] == "wandb==" + settings["wandb"]["sdk_version"]
            assert config["source"]["base_url"] == env["WANDB_BASE_URL"] == settings["wandb"]["base_url"]
            return subprocess.CompletedProcess(
                args, 1, stdout=b"", stderr=b"wandb.apis.public.runhistory.downloads.IncompleteRunHistoryError\n"
            )

        monkeypatch.setattr(archive_stage.subprocess, "run", exporter)
        monkeypatch.setenv("WANDB_API_KEY", "fake-for-offline-test")
        monkeypatch.setenv("WANDB_BASE_URL", "https://api.forge.coreweave.com")
        monkeypatch.setenv("WANDB_SERVICE", "parent-sdk-service")
    document(
        Path(roots["sft"]) / "consumed-config.json",
        (
            {
                "trainer": {"trainer": {"tracker": tracker}},
                "resources": {"replicas": 2},
                "processes_per_task": 4,
            }
            if sft_format == "GrugRunConfig"
            else {
                "format": "SFTSpec",
                "train_config": {"trainer": {"tracker": tracker}},
                "resources": {"replicas": 2},
            }
        ),
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
    spec = {
        "marin_revision": "a" * 40,
        "name": "test",
        "datasets": {"rows": "pinned"},
        "rl": {"execution": {}},
        "runtime_source": {"schema": 1, "marin_revision": "a" * 40, "files": []},
    }
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
    assert plan["provenance"]["final_sft_checkpoint"] == {"uri": str(tmp_path / "sft-final"), "step": 12}
    assert plan["provenance"]["rl_terminal"] == terminal
    assert set(plan["provenance"]["missing_evidence"]) == (
        {"timing", "wandb", "metric_history", "evaluation"} if unscored else {"timing", "wandb", "metric_history"}
    )
    assert plan["provenance"]["evaluation_complete"] is not unscored
    assert plan["evaldash"]["record"]["status"] == ("infra_failed" if unscored else "succeeded")
    assert any(item["name"] == "eval_inputs/selection.json" for item in plan["inputs"])
    assert sum(item["name"].endswith("trace.zip") for item in plan["inputs"]) == 2
    evidence = json.loads((tmp_path / "archive/derived-inputs/evidence-status.json").read_text())
    assert evidence["native_sft_timing_coverage"]["native-sft-journal"]["expected"] == (
        8 if sft_format == "GrugRunConfig" else 2
    )
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
            {
                "package": "fake-exporter",
                "include": {},
                "base_url": "https://api.wandb.ai",
                "sdk_version": "0.31.0",
            },
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
        def __init__(self, *, api_key, overrides):
            assert api_key == "fake-for-offline-test"

        def run(self, path):
            return Run()

    monkeypatch.setattr(archive_stage.wandb, "Api", Client)
    with pytest.raises(PermissionError, match="provider denied history"):
        archive_stage._scan_wandb_history("test/project/run", tmp_path, "https://api.wandb.ai")
    assert not list(tmp_path.glob("scan-history*"))


def test_best_rl_checkpoint_assertion(permanent_path):
    tmp_path = permanent_path
    best = tmp_path / "checkpoints" / "global_step_5"
    best.mkdir(parents=True)
    (best / "trainer_state.pt").write_bytes(b"best-state")
    (best / "policy").mkdir()
    (best / "policy" / "weights.distcp").write_bytes(b"best-weights")
    final = tmp_path / "checkpoints" / "global_step_6"
    final.mkdir(parents=True)
    (final / "trainer_state.pt").write_bytes(b"final-state")
    (final / "policy").mkdir()
    (final / "policy" / "weights.distcp").write_bytes(b"final-weights")
    inventory = {"selected": str(best), "final": str(final)}
    receipt = assert_best_rl_checkpoint(str(tmp_path / "archive"), inventory)
    assert receipt["verdict"] == "full"
    assert receipt["selected"]["step"] == 5 and receipt["final"]["step"] == 6
    sealed = json.loads((tmp_path / "archive" / "best-rl-checkpoint.json").read_text())
    assert sealed == receipt

    scratch = {"selected": str(tmp_path / "tmp" / "global_step_5"), "final": str(final)}
    with pytest.raises(ValueError, match="permanent path"):
        assert_best_rl_checkpoint(str(tmp_path / "archive2"), scratch)
    missing_state = tmp_path / "checkpoints" / "global_step_7"
    missing_state.mkdir()
    (missing_state / "policy").mkdir()
    broken = {"selected": str(missing_state), "final": str(final)}
    with pytest.raises(ValueError, match=r"trainer_state\.pt"):
        assert_best_rl_checkpoint(str(tmp_path / "archive3"), broken)
    empty = tmp_path / "checkpoints" / "global_step_8"
    empty.mkdir()
    (empty / "trainer_state.pt").write_bytes(b"state")
    hollow = {"selected": str(empty), "final": str(final)}
    with pytest.raises(ValueError, match="no weights payload"):
        assert_best_rl_checkpoint(str(tmp_path / "archive4"), hollow)
    with pytest.raises(ValueError, match="final RL checkpoint"):
        assert_best_rl_checkpoint(str(tmp_path / "archive5"), {"selected": str(best)})


def _externally_stopped_sft_fixture(tmp_path, *, consumed_config, selection, rl_telemetry):
    """Build the minimal archive inputs for an externally stopped SFT stage.

    The SFT selection carries no native completion receipt (the stage was stopped before
    its own selection step ran), and the adopted artifact preserves the launcher's
    submitted request config instead of an SFTSpec consumed config.
    """

    def document(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    settings = yaml.safe_load(Path("experiments/post_training/long_loop/specs/archive.yaml").read_text())
    roots = {stage: str(tmp_path / stage) for stage in settings["stage_files"]}
    trained = roots["rl"]
    run = SkyRLRun(
        hf_model_uri=str(tmp_path / "checkpoints" / "global_step_5" / "policy"),
        global_step=5,
        tokenizer_uri="test",
        tokenizer_revision="a" * 40,
        checkpoint_root=str(tmp_path / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri="test",
        iris_job_id="test",
    )
    write_artifact(run.result_payload(), trained)
    write_native_selection(Path(run.checkpoint_root))
    record = json.loads((Path(trained) / RECORD_FILENAME).read_text())
    document(Path(trained) / RECORD_FILENAME, {**record, "version": "2026.11.10"})
    for stage, entries in settings["stage_files"].items():
        for entry in entries:
            document(Path(roots[stage]) / entry["path"], {})
    document(Path(roots["sft"]) / "selected.json", selection)
    document(Path(roots["sft"]) / "consumed-config.json", consumed_config)
    document(Path(trained) / "terminal.json", {"result": {"state": "succeeded", "failure": None}})
    rl_trainer = {"export_path": str(tmp_path / "exports-rl"), "project_name": "test"}
    if rl_telemetry:
        rl_trainer["native_telemetry_root"] = str(tmp_path / "rl-telemetry")
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
                        "trainer": (
                            rl_trainer
                            if stage == "rl"
                            else {"export_path": str(tmp_path / "exports-eval"), "project_name": "test"}
                        ),
                    },
                }
            },
        )
    document(Path(roots["eval"]) / "exports/dumped_evals/eval_only/bench.jsonl", {"uid": "id"})
    document(Path(roots["report"]) / "timing-sources.json", [])
    document(
        Path(roots["report"]) / "evaluation.json",
        {
            "all_verified": True,
            "result_uri": roots["eval"],
            "expected_rollouts": 44,
            "completed_rollouts": 44,
            "verified_rollouts": 44,
            "source_coverage_matches": True,
            "identity_coverage_matches": True,
            "unscored_attempts": [],
            "accepted_unscored_attempts": [],
        },
    )
    spec = {
        "marin_revision": "a" * 40,
        "name": "test",
        "datasets": {"rows": "pinned"},
        "rl": {"execution": {}},
        "runtime_source": {"schema": 1, "marin_revision": "a" * 40, "files": []},
    }
    return settings, roots, trained, spec


LAUNCHER_REQUEST_CONSUMED_CONFIG = {
    "train_config": {"trainer": {"tracker": {"entity": None, "project": "test", "id": None}}},
    "resources": {"replicas": 2},
}
EXTERNALLY_STOPPED_SELECTION = {"checkpoint": "s3://bucket/sft-final", "step": 12}


def test_archive_records_launcher_request_consumed_config(tmp_path):
    """An externally stopped SFT preserves the launcher's request config; record the shape."""
    settings, roots, trained, spec = _externally_stopped_sft_fixture(
        tmp_path,
        consumed_config=LAUNCHER_REQUEST_CONSUMED_CONFIG,
        selection=EXTERNALLY_STOPPED_SELECTION,
        rl_telemetry=False,
    )
    plan = derive_archive_plan(
        output=str(tmp_path / "archive"),
        settings=settings,
        resolved_spec=spec,
        artifacts=roots,
        trained=trained,
        evaluated=roots["eval"],
        completed_report=roots["report"],
    )
    assert plan["provenance"]["sft_consumed_config_shape"] == "launcher-request"
    assert plan["provenance"]["sft_externally_stopped"] is True
    assert "metric_history" in plan["provenance"]["missing_evidence"]
    assert plan["provenance"]["final_sft_checkpoint"] == {"uri": "s3://bucket/sft-final", "step": 12}


def test_archive_falls_back_for_externally_stopped_sft_telemetry(tmp_path):
    """Telemetry on the RL side alone no longer fails when the SFT was stopped externally."""
    settings, roots, trained, spec = _externally_stopped_sft_fixture(
        tmp_path,
        consumed_config=LAUNCHER_REQUEST_CONSUMED_CONFIG,
        selection=EXTERNALLY_STOPPED_SELECTION,
        rl_telemetry=True,
    )
    plan = derive_archive_plan(
        output=str(tmp_path / "archive"),
        settings=settings,
        resolved_spec=spec,
        artifacts=roots,
        trained=trained,
        evaluated=roots["eval"],
        completed_report=roots["report"],
    )
    assert plan["provenance"]["sft_consumed_config_shape"] == "launcher-request"
    assert plan["provenance"]["sft_externally_stopped"] is True
    assert "metric_history" in plan["provenance"]["missing_evidence"]
    assert any("externally stopped" in limitation for limitation in plan["provenance"]["limitations"])


def test_archive_records_omitted_sft_accounting_as_missing_evidence(tmp_path):
    """A structurally unsupported accounting step is recorded, with the future pointer."""
    settings, roots, trained, spec = _externally_stopped_sft_fixture(
        tmp_path,
        consumed_config=LAUNCHER_REQUEST_CONSUMED_CONFIG,
        selection=EXTERNALLY_STOPPED_SELECTION,
        rl_telemetry=False,
    )
    revision = {
        "reason": (
            "the recovery plan inherited the exp6 round6 accounting settings (the V9 source_loss metadata); "
            "the consumed V10.1 mixture adds sft/source/balanced_reasoning/v1"
        ),
        "loss_metadata": {
            "source": (
                "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.25-v10-1-balanced-toolconv/release-v10.1/source_loss.json"
            ),
            "sha256": "6bae0724b8083d06bbebbfbfcc54fdbb42af2c522f0c58e73c9a967c0ff0d4fa",
        },
    }
    spec["archive"] = {
        "sft_data_accounting": {
            "source_loss_metadata": revision["loss_metadata"],
            "pinned_evidence": {},
        }
    }
    spec["accounting_pin_revision"] = revision
    plan = derive_archive_plan(
        output=str(tmp_path / "archive"),
        settings=settings,
        resolved_spec=spec,
        artifacts=roots,
        trained=trained,
        evaluated=roots["eval"],
        completed_report=roots["report"],
    )
    missing = plan["provenance"]["missing_evidence"]["sft_data_accounting"]
    assert "balanced_reasoning/v1" in missing
    assert "no measured step seconds was pinned" in missing
    assert revision["loss_metadata"]["source"] in missing
    assert revision["loss_metadata"]["sha256"] in missing
    assert plan["provenance"]["sft_data_accounting_future"] == revision["loss_metadata"]
