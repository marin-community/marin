"""Metadata-only no-tool reset tests; no generated task executes locally."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from capability_pipeline import non_docker_reset as reset


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def item(tmp_path: Path, kind: str = "none") -> Path:
    root = tmp_path / "item"
    harbor = root / "harbor"
    harbor.mkdir(parents=True)
    (harbor / "binding.json").write_text(json.dumps({
        "environment": {"kind": kind}, "tools": [],
    }))
    (harbor / "manifest.json").write_text(json.dumps({"step_names": ["step-1"]}))
    (harbor / "task.toml").write_text('version = "1.0"\n')
    (harbor / "specification.json").write_text("{}\n")
    (harbor / "renderings.json").write_text("[]\n")
    (harbor / "instruction.md").write_text("Do the task.\n")
    return root


def fake_runner(prompt: str = "Do the task.\n", *, missing: bool = False):
    calls = []

    def run(*, attempt: Path, **_kwargs) -> int:
        calls.append(attempt)
        source = json.loads((attempt / "binding.json").read_text())["source"]
        trials = attempt / "raw/trials"
        episodes = []
        for cycle in range(1, 6):
            name = f"reset-{cycle:02d}"
            root = trials / name
            (root / "agent").mkdir(parents=True)
            (root / "result.json").write_text(f'{{"cycle":{cycle}}}\n')
            observed = "different" if cycle == 3 and prompt == "different" else prompt
            (root / "agent/transcript.json").write_text(json.dumps([
                {"role": "user", "content": observed},
                {"role": "assistant", "content": ""},
            ]))
            episodes.append({
                "cycle": cycle, "trial": name,
                "prompts_sha256": [digest(observed.encode())],
                "result_sha256": reset.sha256(root / "result.json"),
                "exception": None,
            })
        if not missing:
            reset._write(attempt / "raw/report.json", {
                "schema_version": reset.SCHEMA, "source": source, "episodes": episodes,
            })
        return 0

    return run, calls


def test_five_no_tool_episodes_are_recomputed_and_reused(tmp_path):
    root = item(tmp_path)
    runner, calls = fake_runner()
    toolchain = SimpleNamespace(package_root=tmp_path)
    first = reset.run_frozen_non_docker_reset(root, toolchain, 300, runner=runner)
    assert first["state"] == "ready"
    assert first["summary"] == {"episodes": 5, "mismatch_cycles": [], "public_resources": "none"}
    assert len(calls) == 1
    second = reset.run_frozen_non_docker_reset(
        root, toolchain, 300, runner=lambda **_: (_ for _ in ()).throw(AssertionError("resampled"))
    )
    assert second["state"] == "ready"
    assert len(calls) == 1


def test_mismatched_prompt_is_semantic_and_report_forgery_is_pending(tmp_path):
    root = item(tmp_path)
    runner, _ = fake_runner("different")
    first = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path), 300, runner=runner)
    assert first["state"] == "semantic_failed"
    assert first["summary"]["mismatch_cycles"] == [1, 2, 3, 4, 5]
    attempt = Path(first["attempt"])
    report = attempt / "raw/report.json"
    value = json.loads(report.read_text())
    value["episodes"][0]["prompts_sha256"] = ["0" * 64]
    report.write_text(json.dumps(value))
    second = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path), 300)
    assert second["state"] == "pending"
    assert "summary differs" in second["issues"][0]


def test_incomplete_report_does_not_resample(tmp_path):
    root = item(tmp_path)
    runner, calls = fake_runner(missing=True)
    first = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path), 300, runner=runner)
    assert first["state"] == "pending"
    again = reset.run_frozen_non_docker_reset(
        root, SimpleNamespace(package_root=tmp_path), 300,
        runner=lambda **_: (_ for _ in ()).throw(AssertionError("resampled")),
    )
    assert again["state"] == "pending"
    assert len(calls) == 1


def test_shellsim_is_explicit_pending_until_full_vfs_snapshot(tmp_path):
    root = item(tmp_path, "shellsim")
    result = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path), 300)
    assert result["state"] == "pending"
    assert "snapshot" in result["issues"][0]
    assert not (root / "diagnostics").exists()


def test_no_tool_rejects_inputs_and_links(tmp_path):
    root = item(tmp_path)
    (root / "harbor/environment/inputs").mkdir(parents=True)
    result = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path), 300)
    assert result["state"] == "pending"
    assert "filesystem inputs" in result["issues"][0]


def test_source_drift_and_raw_link_are_not_accepted(tmp_path):
    root = item(tmp_path)
    runner, _ = fake_runner()
    toolchain = SimpleNamespace(package_root=tmp_path)
    first = reset.run_frozen_non_docker_reset(root, toolchain, 300, runner=runner)
    assert first["state"] == "ready"
    (root / "harbor/instruction.md").write_text("Changed.\n")
    drift = reset.run_frozen_non_docker_reset(root, toolchain, 300)
    assert drift["state"] == "pending"
    (root / "harbor/instruction.md").write_text("Do the task.\n")
    attempt = Path(first["attempt"])
    (attempt / "raw/alias").symlink_to("report.json")
    linked = reset.run_frozen_non_docker_reset(root, toolchain, 300)
    assert linked["state"] == "pending"
    assert "link" in linked["issues"][0]


def test_byte_only_restore_reuses_original_logical_attempt(tmp_path):
    root = item(tmp_path)
    empty = root / "harbor/unused-empty-dir"
    empty.mkdir()
    first = reset.run_frozen_non_docker_reset(
        root, SimpleNamespace(package_root=tmp_path), 300, runner=fake_runner()[0]
    )
    assert first["state"] == "ready"
    frozen_empty = Path(first["attempt"]) / "input/harbor/unused-empty-dir"
    frozen_empty.rmdir()
    empty.rmdir()
    again = reset.run_frozen_non_docker_reset(
        root, SimpleNamespace(package_root=tmp_path), 300,
        runner=lambda **_: (_ for _ in ()).throw(AssertionError("resampled")),
    )
    assert again["state"] == "ready"
    assert again["attempt"] == first["attempt"]
    assert frozen_empty.is_dir()


def _snapshot(nonce: str, pid: int, *, mutated: bool = False, mode: int = 0o755) -> dict:
    entries = [{"path": ".", "kind": "directory", "mode": mode,
                "size": None, "sha256": None, "target": None}]
    total = 0
    if mutated:
        payload = b"reset-mutation\n"
        total = len(payload)
        entries.append({"path": "__capability_reset_marker", "kind": "file", "mode": 0o644,
                        "size": total, "sha256": digest(payload), "target": None})
    limits = {"max_entries": 100_000, "max_file_bytes": 64 * 1024 * 1024,
              "max_response_bytes": 8 * 1024 * 1024}
    wire = {"schema_version": "taskcompendium-shellsim-vfs-snapshot-v1", "root": "/",
            "limits": limits, "entries": entries}
    return {"session_nonce": nonce, "bridge_pid": pid,
            "snapshot": {"snapshot": wire,
                         "snapshot_sha256": digest(json.dumps(wire, separators=(",", ":"), ensure_ascii=True).encode()),
                         "entry_count": len(entries), "total_file_bytes": total}}


def _fake_shellsim_runner(*, changed_cycle: int | None = None):
    calls = []

    def run(*, attempt: Path, **_kwargs) -> int:
        calls.append(attempt)
        source = json.loads((attempt / "binding.json").read_text())["source"]
        episodes = []
        for cycle in range(6):
            name = "baseline" if cycle == 0 else f"reset-{cycle:02d}"
            trial = attempt / "raw/trials" / name
            (trial / "agent").mkdir(parents=True)
            (trial / "result.json").write_text("{}\n")
            (trial / "agent/transcript.json").write_text(json.dumps([
                {"role": "user", "content": "Do the task.\n"},
            ]))
            nonce = f"{cycle + 1:032x}"
            reset._write(trial / "initial-snapshot.json", _snapshot(
                nonce, 1000 + cycle, mode=0o700 if cycle == changed_cycle else 0o755,
            ))
            reset._write(trial / "session-stop.json", {"session_nonce": nonce, "process_exited": True})
            if cycle == 0:
                reset._write(trial / "mutated-snapshot.json", _snapshot(nonce, 1000, mutated=True))
            row = {"cycle": cycle, "trial": name, "prompts_sha256": [digest(b"Do the task.\n")],
                   "result_sha256": reset.sha256(trial / "result.json"), "exception": None}
            if cycle == 0:
                baseline = row
            else:
                episodes.append(row)
        reset._write(attempt / "raw/report.json", {"schema_version": reset.SCHEMA,
                     "source": source, "baseline": baseline, "episodes": episodes})
        return 0

    return run, calls


def test_shellsim_candidate_receipt_requires_closed_unique_snapshot(tmp_path):
    root = tmp_path / "trial"
    root.mkdir()
    nonce = "a" * 32
    reset._write(root / "initial-snapshot.json", _snapshot(nonce, 101))
    reset._write(root / "session-stop.json", {"session_nonce": nonce, "process_exited": True})
    seen = set()
    receipt = reset.shellsim_candidate_record(root, seen, "b" * 64)
    assert receipt["process_exited"] is True
    assert receipt["session_nonce"] == nonce
    assert len(receipt["initial_snapshot_sha256"]) == 64
    import pytest

    with pytest.raises(ValueError, match="session or bridge"):
        reset.shellsim_candidate_record(root, seen, "b" * 64)
    (root / "session-stop.json").write_text(json.dumps({"session_nonce": nonce, "process_exited": False}))
    with pytest.raises(ValueError, match="not confirmed closed"):
        reset.shellsim_candidate_record(root, set(), "b" * 64)


def test_shellsim_five_fresh_vfs_snapshots_are_recomputed_without_resampling(tmp_path):
    root = item(tmp_path, "shellsim")
    bridge = tmp_path / "bridge"
    bridge.write_bytes(b"pinned bridge")
    runner, calls = _fake_shellsim_runner()
    toolchain = SimpleNamespace(package_root=tmp_path)
    first = reset.run_frozen_non_docker_reset(root, toolchain, 300, bridge, runner=runner)
    assert first["state"] == "ready"
    assert first["summary"]["fresh_sessions"] == 6
    assert first["summary"]["public_resources"] == "complete_shellsim_vfs"
    second = reset.run_frozen_non_docker_reset(root, toolchain, 300, bridge,
        runner=lambda **_: (_ for _ in ()).throw(AssertionError("resampled")))
    assert second["state"] == "ready" and len(calls) == 1
    snapshot = Path(first["attempt"]) / "raw/trials/reset-01/initial-snapshot.json"
    value = json.loads(snapshot.read_text())
    value["snapshot"]["snapshot"]["entries"][0]["mode"] = 0o777
    snapshot.write_text(json.dumps(value))
    tampered = reset.run_frozen_non_docker_reset(root, toolchain, 300, bridge)
    assert tampered["state"] == "pending"
    assert "digest" in tampered["issues"][0]


def test_shellsim_reset_mismatch_is_semantic(tmp_path):
    root = item(tmp_path, "shellsim")
    bridge = tmp_path / "bridge"
    bridge.write_bytes(b"pinned bridge")
    result = reset.run_frozen_non_docker_reset(root, SimpleNamespace(package_root=tmp_path),
                                               300, bridge, runner=_fake_shellsim_runner(changed_cycle=3)[0])
    assert result["state"] == "semantic_failed"
    assert result["summary"]["vfs_mismatch_cycles"] == [3]


def test_shellsim_snapshot_canonical_hash_uses_utf8_for_unicode_names(tmp_path):
    record = _snapshot("1" * 32, 1234)
    wire = record["snapshot"]["snapshot"]
    wire["entries"].append({"path": "café", "kind": "symlink", "mode": 0o777,
                            "size": None, "sha256": None, "target": "雪.txt"})
    record["snapshot"]["entry_count"] = 2
    expected = digest(json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    record["snapshot"]["snapshot_sha256"] = expected
    path = tmp_path / "snapshot.json"
    reset._write(path, record)

    assert reset._shellsim_snapshot(path)[0] == expected
    assert expected != digest(json.dumps(wire, separators=(",", ":"), ensure_ascii=True).encode())
