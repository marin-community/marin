import hashlib
import json
from types import SimpleNamespace

import pytest

from capability_pipeline import reset_diagnostics as diagnostic


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path, *, mutation=False, empty=False, extra_root=False):
    package = tmp_path / "package"
    inputs = package / "environment" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "seed.txt").write_text("seed")
    image = "registry.example/task@sha256:" + "a" * 64
    (package / "binding.json").write_text(json.dumps({
        "environment": {"kind": "docker", "image": image, "workdir": "/workspace",
                        "additional_directories": ["/var/task-cache"] if extra_root else []},
        "tools": [],
    }))
    (package / "task.toml").write_text(
        'version = "1.0"\n[environment]\nallow_internet = false\n'
        f'docker_image = "{image}"\nworkdir = "/workspace"\n'
    )
    policy = {
        "schema_version": diagnostic.POLICY_SCHEMA,
        "public_root": "/workspace",
        "readiness": {"command": "ready", "timeout_seconds": 5},
        "process_policy": {"allowed_comm": ["init"], "max_count": 2},
        "environment_name_policy": {"allowed_names": ["PATH"],
                                    "required_names": ["PATH"], "forbidden_names": []},
    }
    if mutation:
        policy["mutation"] = {"command": "mutate", "timeout_seconds": 5}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({
        "cpu": 2, "memory_gb": 2, "disk_gb": 10,
        "provider_units": {"cpu": "cores", "memory": "GB", "disk": "GB"},
        "source": "adapter_kwargs", "effective_limits": "unverified",
    }))
    return policy_path, package, resources


class Adapter:
    def __init__(self, *, empty=False, fail_process=False, fail_start=False,
                 drift_cycle=None, unconfirmed_delete=None):
        self.empty = empty
        self.fail_process = fail_process
        self.fail_start = fail_start
        self.drift_cycle = drift_cycle
        self.unconfirmed_delete = unconfirmed_delete
        self.created = []
        self.deleted = []

    def ensure_snapshot(self, plan):
        return "frozen-snapshot"

    def create_sandbox(self, snapshot):
        sandbox = SimpleNamespace(id=f"sandbox-{len(self.created)+1}", mutated=False)
        self.created.append(sandbox.id)
        return sandbox, [{"attempt": 1, "state": "created"}]

    def start_task(self, sandbox, plan):
        if self.fail_start:
            class StartupError(RuntimeError):
                phase = "setup_command"
                return_code = 2
            raise StartupError("setup failed")
        sandbox.started = True

    def run(self, sandbox, command, timeout):
        if command == "ready":
            return {"exit": 0, "timed_out": False}
        if command == "mutate":
            sandbox.mutated = True
            return {"exit": 0, "timed_out": False}
        files = {} if self.empty and not sandbox.mutated else {
            "state.txt": _sha(b"mutated" if sandbox.mutated else b"initial")
        }
        if self.drift_cycle == sandbox.id:
            files["unexpected.txt"] = _sha(b"drift")
        process = {"init": 1}
        if self.fail_process:
            process["unapproved"] = 1
        return {"exit": 0, "timed_out": False, "stdout": json.dumps({
            "files": files, "symlinks": [], "process_comm_counts": process,
            "environment_names": ["PATH"],
        })}

    def delete(self, sandbox):
        self.deleted.append(sandbox.id)
        if self.unconfirmed_delete == sandbox.id:
            return {"verified_absent": False, "observations": [{"state": "present"}]}
        return {"verified_absent": True, "observations": [{"state": "not_found"}]}


def test_task_bound_empty_baseline_and_five_fresh_cycles(tmp_path):
    policy, package, resources = _fixture(tmp_path)
    adapter = Adapter(empty=True)
    output = tmp_path / "result"
    report = diagnostic.run_reset_diagnostics(policy, package, resources, output,
                                              adapter=adapter)
    assert report["reset_conformance"] == "passed"
    assert report["full_quality_reset_gate"] == "unassessed"
    assert json.loads((output / "plan.json").read_text())["public_files"] == {}
    assert adapter.created == [f"sandbox-{index}" for index in range(1, 7)]
    assert adapter.deleted == adapter.created
    assert len((output / "five-cycle-probes.jsonl").read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        diagnostic.run_reset_diagnostics(policy, package, resources, output,
                                         adapter=Adapter(empty=True))


def test_baseline_mutation_freezes_second_inventory_and_explicit_scope(tmp_path):
    policy, package, resources = _fixture(tmp_path, mutation=True, extra_root=True)
    report = diagnostic.run_reset_diagnostics(policy, package, resources,
                                              tmp_path / "result", adapter=Adapter())
    assert report["reset_conformance"] == "passed"
    assert report["unassessed_additional_roots"] == ["/var/task-cache"]
    assert report["full_quality_reset_gate"] == "unassessed"
    plan = json.loads((tmp_path / "result" / "plan.json").read_text())
    assert plan["public_files"] == {"state.txt": _sha(b"initial")}
    assert plan["mutation"]["expected_files"] == {"state.txt": _sha(b"mutated")}


def test_policy_scope_and_default_resources_rejected_before_provider(tmp_path):
    policy, package, resources = _fixture(tmp_path)
    value = json.loads(policy.read_text())
    value["public_root"] = "/workspace/subdir"
    policy.write_text(json.dumps(value))
    with pytest.raises(diagnostic.ResetDiagnosticsError, match="bound workdir"):
        diagnostic.validate_policy_and_build_plan(policy, package, resources)
    value["public_root"] = "/workspace"
    policy.write_text(json.dumps(value))
    receipt = json.loads(resources.read_text())
    receipt["source"] = "compatible_default_no_pinned_profile"
    resources.write_text(json.dumps(receipt))
    with pytest.raises(diagnostic.ResetDiagnosticsError, match="explicit requested"):
        diagnostic.validate_policy_and_build_plan(policy, package, resources)


def test_baseline_policy_and_setup_failures_are_not_resampled(tmp_path):
    policy, package, resources = _fixture(tmp_path)
    policy_report = diagnostic.run_reset_diagnostics(
        policy, package, resources, tmp_path / "policy-result",
        adapter=Adapter(fail_process=True),
    )
    assert policy_report["reset_conformance"] == "semantic_failed"
    assert not (tmp_path / "policy-result" / "plan.json").exists()
    startup_report = diagnostic.run_reset_diagnostics(
        policy, package, resources, tmp_path / "startup-result",
        adapter=Adapter(fail_start=True),
    )
    assert startup_report["reset_conformance"] == "semantic_failed"
    baseline = json.loads((tmp_path / "startup-result" / "baseline.json").read_text())
    assert baseline["startup_exit"] == 2
    assert not (tmp_path / "startup-result" / "five-cycle").exists()


def test_complete_cycle_mismatch_is_semantic_but_unconfirmed_cleanup_is_pending(tmp_path):
    policy, package, resources = _fixture(tmp_path)
    mismatch = diagnostic.run_reset_diagnostics(
        policy, package, resources, tmp_path / "mismatch",
        adapter=Adapter(drift_cycle="sandbox-4"),
    )
    assert mismatch["reset_conformance"] == "semantic_failed"
    pending = diagnostic.run_reset_diagnostics(
        policy, package, resources, tmp_path / "cleanup",
        adapter=Adapter(unconfirmed_delete="sandbox-4"),
    )
    assert pending["reset_conformance"] == "pending_infrastructure"
