"""ops/export_accepted.py: renders task.md for accepted healthcare items, idempotently."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_accepted as ex  # noqa: E402
from test_ops_conveyor import FakeRun, FakeStore  # noqa: E402


def accepted_item(run: FakeRun, name: str, task_id: str, *, via_harbor: bool = False) -> None:
    run.item(name, {"state": "quality_accepted", "runtime_validated": True,
                    "taskcompendium": {"specification_sha256": "ab" * 32},
                    "repair_budget": {"used": 0, "max": 4, "exhausted": False}})
    run.put(f"items/{name}/contract/accepted.json", {"proposal": {"capability_id": "d18.allied.nutrition", "slot": 1,
                                                                   "title": f"Title {name}", "task_family": "fam",
                                                                   "environment": "reasoning"}})
    prefix = f"items/{name}/harbor" if via_harbor else f"validated/{name}"
    spec = {"id": task_id, "schema_version": "x", "difficulty": 2, "success_policy": "all", "steps": [
        {"verifier": {"kind": "exact", "parameters": {}}, "context_requirement": "c", "answer_requirements": []}],
        "metadata": {"task_shape": "single"}, "requirements": {"state": {"image": None}}}
    run.put(f"{prefix}/specification.json", spec)
    run.put(f"{prefix}/binding.json", {"b": 1})
    run.put(f"{prefix}/renderings.json", [{"r": 1}])
    run.put(f"{prefix}/task.toml", b'version = "1"\n')
    run.put(f"{prefix}/instruction.md", f"Do the thing for {name}.\n".encode())


def test_exports_new_accepted_items_once(tmp_path):
    run = FakeRun("hc1")
    accepted_item(run, "d18.a-1-aaaaaaaaaaaa", "synthetic/d18-a/001")
    accepted_item(run, "d18.b-2-bbbbbbbbbbbb", "synthetic/d18-b/002", via_harbor=True)
    run.item("d18.c-3-cccccccccccc", {"state": "pending_quality_review"})
    run.put("submission.json", {"run_name": "cap-construct-003-hc1-s2"})
    store = FakeStore([run])
    out = tmp_path / "out"
    logs: list[str] = []
    summary = ex.export(store, out, bases=("hc1",), log=logs.append)
    assert summary["accepted_seen"] == 2 and not summary["errors"], logs
    slugs = sorted(e["slug"] for e in summary["exported"])
    assert slugs == ["d18-a-001", "d18-b-002"]
    text = (out / "d18-a-001" / "task.md").read_text()
    assert "# Task `synthetic/d18-a/001`" in text and "`validated/d18.a-1-aaaaaaaaaaaa/`" in text
    assert "/muchanem/cap-construct-003-hc1-s2" in text and "Do the thing" in text
    assert not list(out.glob("*/build.md"))
    ledger = json.loads((out / ".export-ledger.json").read_text())
    assert set(ledger["items"]) == {"d18.a-1-aaaaaaaaaaaa", "d18.b-2-bbbbbbbbbbbb"}
    before = (out / "d18-a-001" / "task.md").stat().st_mtime_ns
    again = ex.export(store, out, bases=("hc1",), log=logs.append)
    assert again["exported"] == [] and len(again["skipped"]) == 2
    assert (out / "d18-a-001" / "task.md").stat().st_mtime_ns == before


def test_items_rendered_elsewhere_are_skipped(tmp_path):
    run = FakeRun("hc2")
    accepted_item(run, "d18.a-1-aaaaaaaaaaaa", "synthetic/d18-a/001")
    elsewhere = tmp_path / "docs-exports" / "some-slug"
    elsewhere.mkdir(parents=True)
    (elsewhere / "task.md").write_text("The authoritative bytes are `validated/d18.a-1-aaaaaaaaaaaa/` (9 files).\n")
    summary = ex.export(FakeStore([run]), tmp_path / "out", bases=("hc2",), skip_dirs=[tmp_path / "docs-exports"], log=lambda m: None)
    assert summary["exported"] == [] and summary["skipped"][0]["item"] == "d18.a-1-aaaaaaaaaaaa"


def test_dry_run_writes_nothing(tmp_path):
    run = FakeRun("hc3")
    accepted_item(run, "d18.a-1-aaaaaaaaaaaa", "synthetic/d18-a/001")
    out = tmp_path / "out"
    summary = ex.export(FakeStore([run]), out, bases=("hc3",), dry_run=True, log=lambda m: None)
    assert summary["exported"][0]["dry_run"] is True
    assert not list(out.glob("*/task.md"))


def test_refuses_to_write_into_docs_exports():
    import pytest

    with pytest.raises(SystemExit):
        ex.main(["--out", str(ex.LIVE_EXPORTS / "x"), "--dry-run"])
