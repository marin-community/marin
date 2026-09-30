"""Metadata-only reset wrapper tests; no candidate code runs locally."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from capability_pipeline import reset_diagnostics, reset_runner


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _item(tmp_path: Path, *, policy: bool = True, additional: bool = False) -> Path:
    item = tmp_path / "item"
    harbor = item / "harbor"
    inputs = harbor / "environment/inputs"
    inputs.mkdir(parents=True)
    (inputs / "seed.txt").write_text("seed")
    image = "registry.example/reset@sha256:" + "a" * 64
    (harbor / "binding.json").write_text(json.dumps({
        "environment": {"kind": "docker", "image": image, "workdir": "/workspace",
                        "additional_directories": ["/other-root"] if additional else [],
                        "setup_commands": []},
        "tools": [],
    }))
    (harbor / "task.toml").write_text(
        'version = "1.0"\n[environment]\nallow_internet = false\n'
        f'docker_image = "{image}"\nworkdir = "/workspace"\n'
    )
    task = item / "workspace/task"
    task.mkdir(parents=True)
    (task / "candidate-resources.json").write_text(json.dumps({
        "cpu": 2, "memory_gb": 2, "disk_gb": 10,
    }))
    if policy:
        (task / "reset-policy.json").write_text(json.dumps({
            "schema_version": "capability-reset-policy-v1",
            "public_root": "/workspace",
            "readiness": {"command": "ready", "timeout_seconds": 5},
            "process_policy": {"allowed_comm": ["init"], "max_count": 2},
            "environment_name_policy": {
                "allowed_names": ["PATH", "READY"],
                "required_names": ["READY"],
                "forbidden_names": ["SECRET_TOKEN"],
            },
        }))
    tools = item / "workspace/tools/daytona"
    tools.mkdir(parents=True)
    (tools / "dt.py").write_text("# trusted test helper\n")
    return item


class _Adapter:
    def __init__(self):
        self.created = []

    def ensure_snapshot(self, _plan):
        return "cap-reset-fixture"

    def create_sandbox(self, _snapshot):
        sandbox = SimpleNamespace(id=f"reset-{len(self.created) + 1}")
        self.created.append(sandbox)
        return sandbox, [{"attempt": 1, "state": "created"}]

    def start_task(self, _sandbox, _plan):
        pass

    def run(self, _sandbox, command, _timeout):
        if command == "ready":
            return {"exit": 0, "timed_out": False}
        return {"exit": 0, "timed_out": False, "stdout": json.dumps({
            "files": {"seed.txt": _sha(b"seed")}, "symlinks": [],
            "process_comm_counts": {"init": 1},
            "environment_names": ["PATH", "READY"],
        })}

    def delete(self, _sandbox):
        return {"verified_absent": True, "observations": [{"state": "not_found"}]}


def _runner(adapter):
    def run(*, attempt, **_kwargs):
        source = attempt / "input"
        reset_diagnostics.run_reset_diagnostics(
            source / "reset-policy.json", source / "harbor",
            source / "resource-receipt.json", attempt / "raw", adapter=adapter,
        )
        return 0
    return run


def test_reset_wrapper_recomputes_six_fresh_sandboxes_and_reuses_raw(tmp_path):
    item = _item(tmp_path)
    adapter = _Adapter()
    toolchain = SimpleNamespace(package_root=tmp_path / "trusted-toolchain")
    first = reset_runner.run_frozen_reset(item, toolchain, 600, None, runner=_runner(adapter))
    assert first["state"] == "ready"
    assert first["full_quality_reset_gate"] == "unassessed"
    assert len(adapter.created) == 6
    assert len(set(first["summary"]["cycle_sandbox_ids"] + [first["summary"]["baseline_sandbox_id"]])) == 6
    assert all(Path(path).is_file() for path in first["extra_files"].values())
    second = reset_runner.run_frozen_reset(
        item, toolchain, 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not resample")),
    )
    assert second["state"] == "ready"
    assert len(adapter.created) == 6


def test_missing_policy_is_reviewable_semantic_authoring_failure(tmp_path):
    item = _item(tmp_path, policy=False)
    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert result["state"] == "semantic_failed"
    assert result["reviewable"] is True
    assert "reset-policy.json" in result["issues"][0]


def test_additional_roots_remain_pending_despite_public_reset_pass(tmp_path):
    item = _item(tmp_path, additional=True)
    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=_runner(_Adapter()),
    )
    assert result["state"] == "pending"
    assert result["summary"]["additional_roots"] == ["/other-root"]


def test_report_boolean_tamper_and_partial_output_do_not_resample(tmp_path):
    item = _item(tmp_path)
    toolchain = SimpleNamespace(package_root=tmp_path)
    first = reset_runner.run_frozen_reset(item, toolchain, 600, None, runner=_runner(_Adapter()))
    attempt = Path(first["attempt"])
    report = attempt / "raw/report.json"
    value = json.loads(report.read_text())
    value["reset_conformance"] = "semantic_failed"
    report.write_text(json.dumps(value))
    tampered = reset_runner.run_frozen_reset(
        item, toolchain, 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not rerun")),
    )
    assert tampered["state"] == "pending"
    report.unlink()
    partial = reset_runner.run_frozen_reset(
        item, toolchain, 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not rerun")),
    )
    assert partial["state"] == "pending"


def test_empty_input_directory_restored_from_frozen_logical_tree(tmp_path):
    item = _item(tmp_path)
    empty = item / "harbor/environment/inputs/empty"
    empty.mkdir()
    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=_runner(_Adapter()),
    )
    assert result["state"] == "ready"
    attempt = Path(result["attempt"])
    frozen_empty = attempt / "input/harbor/environment/inputs/empty"
    frozen_empty.rmdir()
    reset_runner._validate_frozen(attempt, reset_runner._json(attempt / "binding.json"))
    assert frozen_empty.is_dir()


def test_raw_probe_tamper_cannot_turn_incomplete_evidence_into_ready(tmp_path):
    item = _item(tmp_path)

    def tampering_runner(**kwargs):
        _runner(_Adapter())(**kwargs)
        trace = kwargs["attempt"] / "raw/five-cycle-probes.jsonl"
        lines = trace.read_text().splitlines()
        first = json.loads(lines[0])
        first["probe"]["files"]["seed.txt"] = "0" * 64
        lines[0] = json.dumps(first)
        trace.write_text("\n".join(lines) + "\n")
        return 0

    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=tampering_runner,
    )
    assert result["state"] == "pending"
    assert "raw probe" in result["issues"][0]


def test_baseline_sandbox_must_differ_from_each_reset_cycle(tmp_path):
    item = _item(tmp_path)

    class ReusedBaseline(_Adapter):
        def create_sandbox(self, snapshot):
            sandbox, attempts = super().create_sandbox(snapshot)
            if len(self.created) == 2:
                sandbox.id = self.created[0].id
            return sandbox, attempts

    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=_runner(ReusedBaseline()),
    )
    assert result["state"] == "pending"
    assert "sandbox identities" in result["issues"][0]


def test_mode_drift_resolves_existing_attempt_and_never_resamples(tmp_path):
    item = _item(tmp_path)
    toolchain = SimpleNamespace(package_root=tmp_path)
    first = reset_runner.run_frozen_reset(item, toolchain, 600, None, runner=_runner(_Adapter()))
    assert first["state"] == "ready"
    source = item / "harbor/environment/inputs/seed.txt"
    source.chmod(0o755)
    result = reset_runner.run_frozen_reset(
        item, toolchain, 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not resample")),
    )
    assert result["state"] == "pending"
    assert result["attempt"] == first["attempt"]


def test_started_marker_prevents_retry_after_runner_exception_before_output(tmp_path):
    item = _item(tmp_path)
    toolchain = SimpleNamespace(package_root=tmp_path)

    def fail(**_kwargs):
        raise RuntimeError("provider failed before creating raw output")

    first = reset_runner.run_frozen_reset(item, toolchain, 600, None, runner=fail)
    assert first["state"] == "pending"
    assert (Path(first["attempt"]) / "run-started.json").is_file()
    second = reset_runner.run_frozen_reset(
        item, toolchain, 600, None,
        runner=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not retry")),
    )
    assert second["state"] == "pending"
    assert "cannot be rerun" in second["issues"][0]


def _mutation_policy(item: Path):
    path = item / "workspace/task/reset-policy.json"
    value = json.loads(path.read_text())
    value["mutation"] = {"command": "mutate", "timeout_seconds": 5}
    path.write_text(json.dumps(value))


def test_no_change_mutation_is_reviewable_from_raw_baseline(tmp_path):
    item = _item(tmp_path)
    _mutation_policy(item)
    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=_runner(_Adapter()),
    )
    assert result["state"] == "semantic_failed"
    assert result["reviewable"] is True
    assert "mutation_did_not_change_public_state" in result["issues"][0]


def test_invalid_baseline_probe_stays_pending_without_retained_raw_bytes(tmp_path):
    item = _item(tmp_path)

    class LinkedProbe(_Adapter):
        def run(self, sandbox, command, timeout):
            value = super().run(sandbox, command, timeout)
            if command != "ready":
                raw = json.loads(value["stdout"])
                raw["symlinks"] = ["unsafe-link"]
                value["stdout"] = json.dumps(raw)
            return value

    result = reset_runner.run_frozen_reset(
        item, SimpleNamespace(package_root=tmp_path), 600, None,
        runner=_runner(LinkedProbe()),
    )
    assert result["state"] == "pending"
    assert result["reviewable"] is False


def test_default_runner_uses_pinned_runtime_and_remote_inner_guard(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "binding.json").write_text("{}")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-only")
    launched = {}

    class Child:
        pid = 1234

        def wait(self, timeout=None):
            launched["timeout"] = timeout
            return 0

    def popen(command, **kwargs):
        launched["command"] = command
        launched["environment"] = kwargs["env"]
        return Child()

    monkeypatch.setattr(reset_runner.subprocess, "Popen", popen)
    monkeypatch.setattr(reset_runner, "_stop", lambda _child: None)
    toolchain = SimpleNamespace(runtime_command=lambda: [
        "env", "-u", "VIRTUAL_ENV", "uv", "run", "--project", "pinned", "--frozen",
        "--extra", "harbor", "python", "legacy_driver.py",
    ])
    assert reset_runner._default_runner(toolchain=toolchain, attempt=attempt, timeout=30) == 0
    assert launched["command"][-2:] == ["--binding-sha256", reset_runner.sha256(attempt / "binding.json")]
    assert launched["command"][launched["command"].index("python") + 1:][:3] == [
        "-m", "capability_pipeline.reset_runner", "--inner",
    ]
    assert launched["environment"]["CAPABILITY_REMOTE_RESET"] == "1"
    assert launched["environment"]["CAPABILITY_DAYTONA_TOOLS"] == str((attempt / "input/tools").resolve())
    assert launched["timeout"] == 30
