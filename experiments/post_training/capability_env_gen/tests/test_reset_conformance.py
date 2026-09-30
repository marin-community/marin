import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_reset_conformance.py"
SPEC = importlib.util.spec_from_file_location("reset_conformance", SCRIPT)
reset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reset
SPEC.loader.exec_module(reset)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def payload(*, mutation=False):
    image = "registry.example/c32/candidate@sha256:" + "a" * 64
    recipe = f"FROM {image}\n"
    result = {
        "schema_version": "capability-reset-conformance-plan-v1",
        "image": image,
        "recipe": recipe,
        "recipe_sha256": _hash(recipe),
        "resource_profile": {
            "cpu": 2,
            "memory_gb": 1,
            "disk_gb": 10,
            "provider_units": {"cpu": "cores", "memory": "GB", "disk": "GB"},
            "source": "test_pinned_profile",
            "effective_limits": "unverified",
        },
        "public_root": "/workspace/public",
        "public_files": {"state.txt": _hash("initial")},
        "process_policy": {"allowed_comm": ["init"], "max_count": 2},
        "environment_name_policy": {
            "allowed_names": ["PATH", "READY"],
            "required_names": ["READY"],
            "forbidden_names": ["SECRET_TOKEN"],
        },
        "readiness": {"command": "ready", "timeout_seconds": 5},
        "cycles": 5,
    }
    if mutation:
        result["mutation"] = {
            "command": "mutate",
            "timeout_seconds": 5,
            "expected_files": {"state.txt": _hash("mutated")},
        }
    return result


class FakeAdapter:
    def __init__(self, *, unexpected_process=False, fail_delete=False):
        self.unexpected_process = unexpected_process
        self.fail_delete = fail_delete
        self.created = []
        self.deleted = []

    def ensure_snapshot(self, plan):
        return "cap-reset-frozen"

    def create_sandbox(self, snapshot):
        sandbox = SimpleNamespace(id=f"sandbox-{len(self.created) + 1}", mutated=False)
        self.created.append(sandbox)
        return sandbox, [{"attempt": 1, "state": "created"}]

    def run(self, sandbox, command, timeout):
        if command == "ready":
            return {"exit": 0, "timed_out": False}
        if command == "mutate":
            sandbox.mutated = True
            return {"exit": 0, "timed_out": False}
        files = {"state.txt": _hash("mutated" if sandbox.mutated else "initial")}
        processes = {"init": 1}
        if self.unexpected_process:
            processes["leaked-worker"] = 1
        return {
            "exit": 0,
            "timed_out": False,
            "stdout": json.dumps(
                {
                    "files": files,
                    "symlinks": [],
                    "process_comm_counts": processes,
                    "environment_names": ["PATH", "READY"],
                }
            ),
        }

    def delete(self, sandbox):
        self.deleted.append(sandbox.id)
        if self.fail_delete:
            raise RuntimeError("provider cleanup message must not be emitted")
        return {"verified_absent": True, "observations": [{"state": "not_found"}]}


def test_validator_binds_exact_recipe_profile_and_five_cycles():
    plan = reset.validate_plan_payload(payload())
    assert plan.profile.cpu == 2
    assert plan.profile.memory_gb == 1
    assert plan.profile.disk_gb == 10
    bad = payload()
    bad["cycles"] = 4
    with pytest.raises(ValueError, match="exactly five"):
        reset.validate_plan_payload(bad)
    bad = payload()
    bad["resource_profile"] = {"cpu": 2}
    with pytest.raises(ValueError, match="resource receipt"):
        reset.validate_plan_payload(bad)
    bad = payload()
    bad["public_files"] = {}
    with pytest.raises(ValueError, match="nonempty"):
        reset.validate_plan_payload(bad)


@pytest.mark.parametrize(
    "relative", ["./state.txt", "nested//state.txt", "nested/../state.txt"]
)
def test_validator_rejects_noncanonical_public_inventory_paths(relative):
    bad = payload()
    bad["public_files"] = {relative: _hash("initial")}
    with pytest.raises(ValueError, match="canonical relative"):
        reset.validate_plan_payload(bad)
    bad = payload()
    bad["public_root"] = "/proc/task"
    with pytest.raises(ValueError, match="safe public directory"):
        reset.validate_plan_payload(bad)


def test_probe_command_is_one_shell_argument_with_real_newlines():
    command = reset._probe_command("/workspace/public")
    argv = shlex.split(command)
    assert argv[:3] == ["python3", "-I", "-c"]
    assert "\\n" not in argv[3]
    assert "\n" in argv[3]


def test_malformed_symlink_inventory_cannot_pass_as_empty(tmp_path):
    class InvalidProbe(FakeAdapter):
        def run(self, sandbox, command, timeout):
            result = super().run(sandbox, command, timeout)
            if "stdout" in result:
                raw = json.loads(result["stdout"])
                raw["symlinks"] = ""
                result["stdout"] = json.dumps(raw)
            return result

    report = reset.run_plan(
        reset.validate_plan_payload(payload()), "a" * 64,
        tmp_path / "result", adapter=InvalidProbe(),
    )
    assert report["conformance"] == "failed"
    assert all(
        cycle["initial_state"]["status"] == "probe_invalid"
        for cycle in report["cycles"]
    )


def test_five_fresh_cycles_compare_to_frozen_plan_and_delete_each(tmp_path):
    plan_payload = payload(mutation=True)
    plan = reset.validate_plan_payload(plan_payload)
    adapter = FakeAdapter()
    report = reset.run_plan(
        plan, _hash(json.dumps(plan_payload)), tmp_path / "result", adapter=adapter
    )
    assert report["conformance"] == "passed"
    assert report["admission"] == "unassessed"
    assert [cycle["sandbox_id"] for cycle in report["cycles"]] == [
        "sandbox-1",
        "sandbox-2",
        "sandbox-3",
        "sandbox-4",
        "sandbox-5",
    ]
    assert all(
        cycle["initial_state"]["inventory_matches"] for cycle in report["cycles"]
    )
    assert all(cycle["mutation"]["matches"] for cycle in report["cycles"])
    assert all(
        cycle["provisioning_attempts"] == [{"attempt": 1, "state": "created"}]
        for cycle in report["cycles"]
    )
    assert adapter.deleted == [
        "sandbox-1",
        "sandbox-2",
        "sandbox-3",
        "sandbox-4",
        "sandbox-5",
    ]


def test_unknown_process_and_cleanup_failure_are_preserved_not_accepted(tmp_path):
    plan = reset.validate_plan_payload(payload())
    adapter = FakeAdapter(unexpected_process=True, fail_delete=True)
    report = reset.run_plan(plan, "f" * 64, tmp_path / "result", adapter=adapter)
    assert report["conformance"] == "failed"
    assert len(report["cycles"]) == 5
    assert all(cycle["status"] == "failed" for cycle in report["cycles"])
    assert all(
        cycle["initial_state"]["process"]["unexpected_comm"] == ["leaked-worker"]
        for cycle in report["cycles"]
    )
    assert all(
        cycle["cleanup"]
        == {"attempted": True, "succeeded": False, "error_type": "RuntimeError"}
        for cycle in report["cycles"]
    )
    serialized = json.dumps(report)
    assert "provider cleanup message" not in serialized


def test_snapshot_failure_preserves_all_cycles_and_fresh_output(tmp_path):
    class SnapshotFailure(FakeAdapter):
        def ensure_snapshot(self, plan):
            raise RuntimeError("provider response with secret-like text")

    plan = reset.validate_plan_payload(payload())
    output = tmp_path / "result"
    report = reset.run_plan(plan, "f" * 64, output, adapter=SnapshotFailure())
    assert report["conformance"] == "incomplete"
    assert report["snapshot"] == {"status": "failed", "error_type": "RuntimeError"}
    assert report["cycles"] == [
        {"cycle": number, "status": "not_started", "reason": "snapshot_unavailable"}
        for number in range(1, 6)
    ]
    assert "secret-like" not in (output / "report.json").read_text()
    with pytest.raises(FileExistsError):
        reset.run_plan(plan, "f" * 64, output, adapter=SnapshotFailure())


def test_validator_requires_meaningful_mutation_state():
    bad = payload(mutation=True)
    bad["mutation"]["expected_files"] = bad["public_files"].copy()
    with pytest.raises(ValueError, match="must differ"):
        reset.validate_plan_payload(bad)


def test_unconfirmed_provider_deletion_fails_each_cycle(tmp_path):
    class UnconfirmedDelete(FakeAdapter):
        def delete(self, sandbox):
            self.deleted.append(sandbox.id)
            return {"verified_absent": False, "observations": [{"state": "present"}]}

    plan = reset.validate_plan_payload(payload())
    report = reset.run_plan(
        plan, "f" * 64, tmp_path / "result", adapter=UnconfirmedDelete()
    )
    assert report["conformance"] == "failed"
    assert all(
        cycle["cleanup"]
        == {
            "attempted": True,
            "succeeded": False,
            "failure": "deletion_unconfirmed",
        }
        for cycle in report["cycles"]
    )


def _task_payload(tmp_path):
    data = payload()
    bundle = tmp_path / "bundle"
    inputs = bundle / "environment" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "seed.txt").write_text("frozen")
    (bundle / "binding.json").write_text(json.dumps({
        "environment": {
            "kind": "docker", "image": data["image"], "workdir": "/workspace",
            "additional_directories": ["/workspace/cache"],
            "setup_commands": ["touch /workspace/ready"],
        },
        "tools": [],
    }))
    (bundle / "task.toml").write_text(
        'version = "1.0"\n[environment]\nallow_internet = false\n'
        f'docker_image = "{data["image"]}"\nworkdir = "/workspace"\n'
    )
    data["schema_version"] = "capability-reset-conformance-plan-v2"
    data["task_binding"] = {
        "bundle_path": str(bundle),
        "binding_sha256": reset._sha256_bytes((bundle / "binding.json").read_bytes()),
        "inputs_sha256": reset._inputs_sha256(inputs),
        "task_toml_sha256": reset._sha256_bytes((bundle / "task.toml").read_bytes()),
    }
    return data, bundle


def test_task_bound_plan_rejects_changed_binding_inputs_and_image(tmp_path):
    data, bundle = _task_payload(tmp_path)
    plan = reset.validate_plan_payload(data)
    assert plan.task_bundle == bundle
    (bundle / "environment" / "inputs" / "seed.txt").write_text("changed")
    with pytest.raises(ValueError, match="inputs fingerprint mismatch"):
        reset.validate_plan_payload(data)
    (bundle / "environment" / "inputs" / "seed.txt").write_text("frozen")
    binding = json.loads((bundle / "binding.json").read_text())
    binding["environment"]["image"] = "registry.example/other@sha256:" + "b" * 64
    (bundle / "binding.json").write_text(json.dumps(binding))
    data["task_binding"]["binding_sha256"] = reset._sha256_bytes(
        (bundle / "binding.json").read_bytes()
    )
    with pytest.raises(ValueError, match="image differs"):
        reset.validate_plan_payload(data)


def test_task_bound_reset_runs_startup_on_each_fresh_sandbox(tmp_path):
    data, bundle = _task_payload(tmp_path)
    class BoundAdapter(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.started = []

        def start_task(self, sandbox, plan):
            self.started.append(sandbox.id)
            assert plan.task_bundle == bundle

    adapter = BoundAdapter()
    report = reset.run_plan(
        reset.validate_plan_payload(data), _hash(json.dumps(data)),
        tmp_path / "result", adapter=adapter,
    )
    assert report["conformance"] == "passed"
    assert report["startup_scope"] == "task_bound"
    assert report["task_bundle_unchanged_at_finish"] is True
    assert adapter.started == [f"sandbox-{index}" for index in range(1, 6)]
    assert all(cycle["task_startup"]["status"] == "completed" for cycle in report["cycles"])


def test_task_bound_plan_rejects_mode_change_and_injected_environment(tmp_path):
    data, bundle = _task_payload(tmp_path)
    input_file = bundle / "environment" / "inputs" / "seed.txt"
    input_file.chmod(0o755)
    with pytest.raises(ValueError, match="inputs fingerprint mismatch"):
        reset.validate_plan_payload(data)
    input_file.chmod(0o644)
    task_file = bundle / "task.toml"
    task_file.write_text(task_file.read_text() + '[environment.env]\nTOKEN = "secret"\n')
    data["task_binding"]["task_toml_sha256"] = reset._sha256_bytes(task_file.read_bytes())
    with pytest.raises(ValueError, match="injected task environment"):
        reset.validate_plan_payload(data)


def test_task_bound_reset_cannot_skip_startup(tmp_path):
    data, _ = _task_payload(tmp_path)
    report = reset.run_plan(
        reset.validate_plan_payload(data), _hash(json.dumps(data)),
        tmp_path / "result", adapter=FakeAdapter(),
    )
    assert report["conformance"] == "failed"
    assert all(cycle["error_type"] == "RuntimeError" for cycle in report["cycles"])


def test_task_bound_reset_does_not_inspect_after_failed_setup(tmp_path):
    data, _ = _task_payload(tmp_path)

    class FailedSetup(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.inspections = 0

        def start_task(self, sandbox, plan):
            raise RuntimeError("setup failed")

        def run(self, sandbox, command, timeout):
            self.inspections += 1
            return super().run(sandbox, command, timeout)

    adapter = FailedSetup()
    report = reset.run_plan(
        reset.validate_plan_payload(data), _hash(json.dumps(data)),
        tmp_path / "result", adapter=adapter,
    )
    assert report["conformance"] == "failed"
    assert adapter.inspections == 0
    assert len(adapter.deleted) == 5
