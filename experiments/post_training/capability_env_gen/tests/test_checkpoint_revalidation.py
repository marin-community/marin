from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from capability_pipeline.checkpoint_revalidation import (
    CheckpointRevalidationError,
    _source_inventory,
    prepare_checkpoint_revalidation,
    run_prepared_checkpoint_revalidation,
    running_controller_provenance,
    validate_checkpoint_bundle,
)
from capability_pipeline.synthesis import _safe_name

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = json.loads((ROOT / "data/c05-portable-runtime-006/accepted.json").read_text())[0]
PROVENANCE = {"schema_version": "capability-controller-provenance-v1", **running_controller_provenance()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _files(root: Path, omit: Path | None = None) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != omit
    }


def _bundle(tmp_path: Path, repair_rounds: tuple[int, ...] = ()) -> tuple[Path, str]:
    bundle = tmp_path / "bundle"
    seed = bundle / "restore-seed"
    proposal = ACCEPTED["proposal"]
    key = f"{proposal['capability_id']}:{proposal['slot']}"
    item_name = f"{_safe_name(key)}-{ACCEPTED['proposal_hash'][:12]}"
    construction = seed / "construction"
    item = construction / "items" / item_name
    _dump(item / "contract/accepted.json", ACCEPTED)
    workspace = item / "workspace"
    (workspace / "artifact.txt").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "artifact.txt").write_text("completed builder output\n")
    sessions = []
    for row in proposal["builder_plan"]:
        name = row["session"]
        sessions.append({"session": name, "status": "complete"})
        _dump(item / "sessions" / name / "status.json", {"session": name, "status": "complete"})
        _dump(workspace / "handoffs" / f"{_safe_name(name)}.json", {
            "session": name, "status": "complete", "artifacts": ["artifact.txt"],
            "checks": [{"name": "fixture", "command": "true", "exit_code": 0}],
        })
    for round_ in repair_rounds:
        _dump(construction / "repairs" / item_name / f"attempt-{round_}" / "result.json", {"round": round_})
        _dump(construction / "repair-history" / item_name / f"attempt-{round_}" / "status.json", {"round": round_})
        _dump(construction / "repair-budget" / item_name / f"attempt-{round_}" / "reservation.json", {"round": round_})
    _dump(item / "status.json", {
        "key": f"{proposal['capability_id']}:{proposal['slot']}",
        "proposal_hash": ACCEPTED["proposal_hash"], "state": "failed",
        "sessions": sessions, "repairs": [{"round": round_} for round_ in repair_rounds],
        "repair_budget": {"used": max(repair_rounds, default=0), "max": 2, "exhausted": bool(repair_rounds)},
    })
    _dump(construction / "report.json", {"state": "needs_continuation"})
    _dump(seed / "report.json", {"state": "complete", "generation_state": "complete_with_rejections"})
    files = _files(seed)
    remote = {"snapshot_id": "fixture", "final": True, "files": files, "omitted": {}}
    raw = json.dumps(remote)
    remote_sha = hashlib.sha256(raw.encode()).hexdigest()
    _dump(bundle / "source-metadata/pull-manifest.json", {
        "schema_version": "capability-snapshot-pull-v1", "complete_manifest": True,
        "complete_snapshot": True, "remote_final": True, "files": files,
        "remote_snapshot_id": "fixture", "remote_manifest_sha256": remote_sha,
    })
    _dump(bundle / "source-metadata/snapshot-capture.json", {
        "remote_manifest_json": raw, "remote_manifest_sha256": remote_sha,
    })
    pull = bundle / "source-metadata/pull-manifest.json"
    capture = bundle / "source-metadata/snapshot-capture.json"
    tree = hashlib.sha256("".join(f"{name}\0{files[name]}\n" for name in sorted(files)).encode()).hexdigest()
    request = {
        "schema_version": "capability-completed-checkpoint-revalidation-request-v1",
        "state": "approved", "item_name": item_name, "construction_root": "construction",
        "source": {"pull_manifest_sha256": _sha(pull), "snapshot_capture_sha256": _sha(capture), "source_tree_sha256": tree},
        "source_controller": PROVENANCE,
    }
    _dump(bundle / "request.json", request)
    _dump(bundle / "accepted.json", [ACCEPTED])
    manifest = {"schema_version": "capability-completed-checkpoint-revalidation-bundle-v1", "state": "prepared_pending_maintained_validation", "files": _files(bundle)}
    _dump(bundle / "manifest.json", manifest)
    return bundle, item_name


def _plan(bundle: Path):
    return validate_checkpoint_bundle(bundle, expected_manifest_sha256=_sha(bundle / "manifest.json"), expected_request_sha256=_sha(bundle / "request.json"))


def _refresh_source_and_manifest(bundle: Path) -> None:
    seed = bundle / "restore-seed"
    files = _files(seed)
    pull = bundle / "source-metadata/pull-manifest.json"
    pull_value = json.loads(pull.read_text())
    pull_value["files"] = files
    capture = bundle / "source-metadata/snapshot-capture.json"
    captured = json.loads(capture.read_text())
    remote = json.loads(captured["remote_manifest_json"])
    remote["files"] = files
    raw = json.dumps(remote)
    captured["remote_manifest_json"] = raw
    captured["remote_manifest_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    _dump(capture, captured)
    pull_value["remote_manifest_sha256"] = captured["remote_manifest_sha256"]
    _dump(pull, pull_value)
    request = bundle / "request.json"
    request_value = json.loads(request.read_text())
    request_value["source"]["pull_manifest_sha256"] = _sha(pull)
    request_value["source"]["snapshot_capture_sha256"] = _sha(capture)
    request_value["source"]["source_tree_sha256"] = hashlib.sha256(
        "".join(f"{name}\0{files[name]}\n" for name in sorted(files)).encode()
    ).hexdigest()
    _dump(request, request_value)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"] = _files(bundle, bundle / "manifest.json")
    _dump(bundle / "manifest.json", manifest)


def test_valid_checkpoint_preserves_completed_builder_and_budget(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path)
    plan = _plan(bundle)
    assert plan.item_name == item_name
    assert plan.repair_rounds == ()
    prepared = prepare_checkpoint_revalidation(plan, tmp_path / "fresh", new_controller=PROVENANCE)
    archived = json.loads((prepared.history_root / "status.json").read_text())
    assert archived["state"] == "failed"
    calls = []

    def attempt(item, root):
        calls.append(item["proposal_hash"])
        return {"key": "fixture", "proposal_hash": item["proposal_hash"], "sessions": [], "state": "pending_runtime", "item_root": str(root / "items" / item_name)}

    result = run_prepared_checkpoint_revalidation(prepared, attempt)
    assert calls == [ACCEPTED["proposal_hash"]]
    assert result["checkpoint_revalidation"]["semantic_repair_performed"] is False
    assert result["repair_budget"]["used"] == 0
    assert (prepared.item_root / "workspace/artifact.txt").read_text() == "completed builder output\n"
    assert not (prepared.root / "construction/report.json").exists()
    assert not (prepared.root / "report.json").exists()
    assert (prepared.history_root / "controller/construction-report.json").is_file()


def test_standalone_synthesis_checkpoint_revalidates_at_snapshot_root(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path)
    seed = bundle / "restore-seed"
    shutil.move(seed / "construction/items", seed / "items")
    shutil.rmtree(seed / "construction")
    request_path = bundle / "request.json"
    request = json.loads(request_path.read_text())
    request["construction_root"] = "."
    _dump(request_path, request)
    _refresh_source_and_manifest(bundle)

    plan = _plan(bundle)
    assert plan.construction_root == Path(".")
    prepared = prepare_checkpoint_revalidation(
        plan, tmp_path / "fresh", new_controller=PROVENANCE
    )
    assert prepared.item_root == prepared.root / "items" / item_name

    def attempt(item, root):
        return {"state": "pending_quality_review", "proposal_hash": item["proposal_hash"]}

    result = run_prepared_checkpoint_revalidation(prepared, attempt)
    assert result["state"] == "pending_quality_review"
    assert result["checkpoint_revalidation"]["maintained_gate_attempts"] == 1
    assert (prepared.history_root / "status.json").is_file()


def test_rejects_missing_handoff_even_with_rehashed_bundle(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    next((bundle / "restore-seed/construction/items").glob("*/workspace/handoffs/*.json")).unlink()
    _refresh_source_and_manifest(bundle)
    with pytest.raises(CheckpointRevalidationError, match="complete builder handoff"):
        _plan(bundle)


def test_rejects_maintained_gate_workspace_mutation(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path)
    prepared = prepare_checkpoint_revalidation(_plan(bundle), tmp_path / "fresh", new_controller=PROVENANCE)

    def attempt(item, root):
        (root / "construction/items" / item_name / "workspace/artifact.txt").write_text("changed\n")
        return {"state": "failed", "proposal_hash": item["proposal_hash"]}

    with pytest.raises(CheckpointRevalidationError, match="protected workspace"):
        run_prepared_checkpoint_revalidation(prepared, attempt)
    status = json.loads((prepared.item_root / "status.json").read_text())
    assert status["state"] == "checkpoint_revalidation_error"


def test_rejects_added_workspace_file(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path, (1,))
    prepared = prepare_checkpoint_revalidation(
        _plan(bundle), tmp_path / "fresh", new_controller=PROVENANCE
    )

    def attempt(item, root):
        (root / "construction/items" / item_name / "workspace/new.txt").write_text("extra\n")
        return {"state": "failed", "proposal_hash": item["proposal_hash"]}

    with pytest.raises(CheckpointRevalidationError, match="protected workspace inventory"):
        run_prepared_checkpoint_revalidation(prepared, attempt)


def test_rejects_added_repair_round(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path, (1,))
    prepared = prepare_checkpoint_revalidation(
        _plan(bundle), tmp_path / "fresh", new_controller=PROVENANCE
    )

    def attempt(item, root):
        (root / "construction/repairs" / item_name / "attempt-2").mkdir(parents=True)
        return {"state": "failed", "proposal_hash": item["proposal_hash"]}

    with pytest.raises(CheckpointRevalidationError, match="protected repair rounds"):
        run_prepared_checkpoint_revalidation(prepared, attempt)


def test_quarantines_export_when_postcheck_rejects_result(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path)
    prepared = prepare_checkpoint_revalidation(
        _plan(bundle), tmp_path / "fresh", new_controller=PROVENANCE
    )

    def attempt(item, root):
        export = root / "construction/validated" / item_name
        export.mkdir(parents=True)
        (export / "manifest.json").write_text("candidate export\n")
        (root / "construction/items" / item_name / "workspace/extra.txt").write_text("extra\n")
        return {"state": "quality_accepted", "proposal_hash": item["proposal_hash"]}

    with pytest.raises(CheckpointRevalidationError):
        run_prepared_checkpoint_revalidation(prepared, attempt)
    assert not (prepared.root / "construction/validated" / item_name).exists()
    assert (prepared.history_root / "quarantined-export/manifest.json").is_file()
    assert json.loads((prepared.item_root / "status.json").read_text())["state"] == "checkpoint_revalidation_error"


def test_preserves_consumed_repair_history_without_reset(tmp_path: Path) -> None:
    bundle, item_name = _bundle(tmp_path, (1, 2))
    prepared = prepare_checkpoint_revalidation(
        _plan(bundle), tmp_path / "fresh", new_controller=PROVENANCE
    )

    result = run_prepared_checkpoint_revalidation(
        prepared,
        lambda item, root: {"state": "pending_runtime", "proposal_hash": item["proposal_hash"]},
    )

    assert result["repair_budget"]["used"] == 2
    assert [record["round"] for record in result["repairs"]] == [1, 2]
    assert (prepared.root / "construction/repairs" / item_name / "attempt-2/result.json").is_file()
    assert (prepared.root / "construction/repair-history" / item_name / "attempt-2/status.json").is_file()


def test_rejects_claimed_new_controller_hash_that_differs_from_running_source(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    incorrect = dict(PROVENANCE)
    incorrect["synthesis_sha256"] = "0" * 64
    with pytest.raises(CheckpointRevalidationError, match="running source"):
        prepare_checkpoint_revalidation(_plan(bundle), tmp_path / "fresh", new_controller=incorrect)


def test_controller_source_inventory_ignores_bytecode_cache(tmp_path: Path) -> None:
    package = tmp_path / "capability_pipeline"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n")
    first = _source_inventory(package)
    (package / "__pycache__").mkdir()
    (package / "__pycache__/module.cpython-313.pyc").write_bytes(b"cache")
    assert _source_inventory(package) == first == {"module.py": _sha(package / "module.py")}
