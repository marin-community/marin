import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from capability_pipeline import adversary, evaluation, synthesis
from capability_pipeline.runtime import sha256


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    package = tmp_path / "package"
    bundle = tmp_path / "bundle"
    package.mkdir(parents=True)
    bundle.mkdir(parents=True)
    specification = bundle / "specification.json"
    specification.write_text('{"task":"frozen"}\n')
    (bundle / "binding.json").write_text(json.dumps({"environment": {"kind": "none"}}))
    # Lowering re-serializes the specification, so the package holds the same
    # document in different bytes and the manifest binds the lowered bytes.
    lowered = package / "specification.json"
    lowered.write_text(json.dumps({"task": "frozen"}, indent=2) + "\n")
    (package / "manifest.json").write_text(
        json.dumps({"specification_sha256": sha256(lowered)})
    )
    controls = tmp_path / "controls.json"
    controls.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "gold",
                        "class": "positive",
                        "source_author": "author",
                        "category": "solver",
                        "expect": {"status": "graded", "reward_min": 1},
                    },
                    {
                        "id": "negative",
                        "class": "negative",
                        "source_author": "author",
                        "category": "control",
                        "expect": {"status": "graded", "reward_max": 0},
                    },
                ]
            }
        )
    )
    return package, bundle, controls


def _plan(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    package, bundle, controls = _inputs(tmp_path)
    plan = tmp_path / "plan.json"
    evaluation.make_plan(
        SimpleNamespace(
            package=str(package),
            bundle=str(bundle),
            controls=str(controls),
            out=str(plan),
            attempt_timeout=60,
        )
    )
    return plan, package, bundle, controls


def _valid_cell(attempt: int, *, solver=True, sandbox=None, negatives=True) -> dict:
    return {
        "attempt": attempt,
        "state": "valid",
        "oracle_passed": True,
        "solver_passed": solver,
        "authored_controls_passed": negatives,
        "independent_attacks_passed": True,
        "sandbox_ids": [sandbox or f"sandbox-{attempt}"],
    }


def test_repeated_plan_binds_primary_adversarial_review_without_new_attacks(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    gate = tmp_path / "primary-gate.json"
    gate.write_text(json.dumps({
        "schema_version": "capability-primary-adversarial-gate-v1",
        "state": "resolved",
        "runtime_evidence_sha256": "a" * 64,
        "adjudication_result_sha256": "b" * 64,
        "new_attacks_in_repeated_measurement": False,
    }))
    plan_path = tmp_path / "plan.json"
    evaluation.make_plan(SimpleNamespace(
        package=package, bundle=bundle, controls=controls,
        out=plan_path, attempt_timeout=60, primary_adversarial_gate=gate,
    ))
    plan, inputs = evaluation.validate_plan(plan_path, sha256(plan_path))
    assert plan["purpose"] == "primary_bound_repeatability"
    assert inputs["primary_adversarial_gate"] == gate
    cells = [
        {**_valid_cell(index), "independent_attacks_passed": False,
         "primary_adversarial_review_bound": True}
        for index in (1, 2, 3)
    ]
    matrix = evaluation.aggregate(plan, cells)
    assert matrix["state"] == "repeated_runtime_passed"
    assert "independent_attacks_all_attempts" not in matrix["gates"]
    assert matrix["gates"]["primary_adversarial_review_bound"] is True
    gate.write_text(gate.read_text().replace('"resolved"', '"uncertain"'))
    with pytest.raises(ValueError, match="input fingerprint mismatch"):
        evaluation.validate_plan(plan_path, sha256(plan_path))


def test_candidate_resources_are_docker_only_frozen_and_forwarded(
    tmp_path, monkeypatch
):
    package, bundle, controls = _inputs(tmp_path)
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"cpu": 2, "memory_gb": 2, "disk_gb": 10}))
    helper = tmp_path / "dt.py"
    helper.write_text("# helper fixture\n")
    plan_path = tmp_path / "plan.json"
    args = SimpleNamespace(
        package=package,
        bundle=bundle,
        controls=controls,
        out=plan_path,
        attempt_timeout=60,
        candidate_resources=resources,
        daytona_helper=helper,
    )
    with pytest.raises(ValueError, match="Docker"):
        evaluation.make_plan(args)
    (bundle / "binding.json").write_text(
        json.dumps({"environment": {"kind": "docker"}})
    )
    evaluation.make_plan(args)
    fingerprint = sha256(plan_path)
    plan, inputs = evaluation.validate_plan(plan_path, fingerprint)
    assert inputs["candidate_resources"] == resources
    assert plan["inputs"]["candidate_resources"]["sha256"] == sha256(resources)
    calls = []
    monkeypatch.setattr(
        synthesis,
        "OfficialToolchain",
        SimpleNamespace(
            resolve=lambda *_: SimpleNamespace(
                runtime_command=lambda: ["remote-runtime"]
            )
        ),
    )

    def run(argv, *, timeout, env):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(synthesis, "_run", run)
    monkeypatch.setattr(
        evaluation, "inspect_attempt", lambda root, *_: _valid_cell(int(root.name))
    )
    evaluation.run_evaluation(
        SimpleNamespace(
            plan=plan_path,
            plan_sha256=fingerprint,
            out=tmp_path / "results",
            taskcompendium_source=None,
            concurrency=3,
        )
    )
    assert len(calls) == 3
    assert all(
        argv[argv.index("--candidate-resources") + 1] == str(resources)
        for argv in calls
    )
    resources.write_text(json.dumps({"cpu": 4, "memory_gb": 2, "disk_gb": 10}))
    with pytest.raises(
        ValueError, match="input fingerprint mismatch: candidate_resources"
    ):
        evaluation.validate_plan(plan_path, fingerprint)


def test_private_daytona_helper_is_required_frozen_and_forwarded(tmp_path, monkeypatch):
    package, bundle, controls = _inputs(tmp_path)
    document = {"steps": [{"verifier": {"runtime": {"kind": "container"}}}]}
    spec = bundle / "specification.json"
    spec.write_text(json.dumps(document))
    lowered = package / "specification.json"
    lowered.write_text(json.dumps(document, indent=2) + "\n")
    (package / "manifest.json").write_text(
        json.dumps({"specification_sha256": sha256(lowered)})
    )
    plan_path = tmp_path / "plan.json"
    args = SimpleNamespace(
        package=package,
        bundle=bundle,
        controls=controls,
        out=plan_path,
        attempt_timeout=60,
        daytona_helper=None,
    )
    with pytest.raises(ValueError, match="frozen --daytona-helper"):
        evaluation.make_plan(args)
    helper = tmp_path / "tools" / "dt.py"
    helper.parent.mkdir()
    helper.write_text("# trusted helper fixture; never executed\n")
    args.daytona_helper = helper
    evaluation.make_plan(args)
    plan_sha = sha256(plan_path)
    plan, inputs = evaluation.validate_plan(plan_path, plan_sha)
    assert inputs["daytona_helper"] == helper
    assert plan["inputs"]["daytona_helper"]["sha256"] == sha256(helper)
    calls = []

    class Toolchain:
        @staticmethod
        def resolve(*_args):
            return SimpleNamespace(runtime_command=lambda: ["remote-runtime"])

    def run(argv, *, timeout, env):
        calls.append((argv, env))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(synthesis, "OfficialToolchain", Toolchain)
    monkeypatch.setattr(synthesis, "_run", run)
    monkeypatch.setattr(
        evaluation, "inspect_attempt", lambda root, *_: _valid_cell(int(root.name))
    )
    monkeypatch.setenv("CAPABILITY_DAYTONA_TOOLS", "/wrong/ambient/tools")
    evaluation.run_evaluation(
        SimpleNamespace(
            plan=plan_path,
            plan_sha256=plan_sha,
            out=tmp_path / "results",
            taskcompendium_source=None,
            concurrency=3,
        )
    )
    assert len(calls) == 3
    assert all(
        env["CAPABILITY_DAYTONA_TOOLS"] == str(helper.parent) for _, env in calls
    )
    assert all("--daytona-helper" not in argv for argv, _ in calls)
    helper.write_text("# drift\n")
    with pytest.raises(ValueError, match="input fingerprint mismatch: daytona_helper"):
        evaluation.validate_plan(plan_path, plan_sha)


def test_plan_validates_and_binds_relative_replay_workspaces(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    value = json.loads(controls.read_text())
    value["cases"][0]["workspace"] = "controls/gold"
    controls.write_text(json.dumps(value))
    plan = tmp_path / "plan.json"
    args = SimpleNamespace(
        package=package, bundle=bundle, controls=controls, out=plan, attempt_timeout=60
    )
    with pytest.raises(RuntimeError, match="control workspace is unavailable"):
        evaluation.make_plan(args)
    assert not plan.exists()
    workspace = tmp_path / "controls/gold"
    workspace.mkdir(parents=True)
    answer = workspace / "answer.txt"
    answer.write_text("fixed submission")
    evaluation.make_plan(args)
    fingerprint = sha256(plan)
    frozen, _ = evaluation.validate_plan(plan, fingerprint)
    assert frozen["replay_inventory"]["gold"]["files"][0]["sha256"] == sha256(answer)
    answer.write_text("changed submission outside package and task bundle")
    with pytest.raises(ValueError, match="control replay inventory differs"):
        evaluation.validate_plan(plan, fingerprint)


def test_plan_rejects_immutable_input_and_controller_drift(tmp_path, monkeypatch):
    plan, package, _bundle, _controls = _plan(tmp_path)
    fingerprint = sha256(plan)
    evaluation.validate_plan(plan, fingerprint)

    (package / "extra.txt").write_text("changed")
    with pytest.raises(ValueError, match="input fingerprint mismatch: package"):
        evaluation.validate_plan(plan, fingerprint)

    plan, _package, _bundle, _controls = _plan(tmp_path / "controller")
    fingerprint = sha256(plan)
    monkeypatch.setattr(evaluation, "controller_hashes", lambda: {"changed.py": "x"})
    with pytest.raises(ValueError, match="changed evaluation contract/controller"):
        evaluation.validate_plan(plan, fingerprint)


def test_aggregate_requires_exact_attempt_inventory_and_fresh_sandboxes(tmp_path):
    plan, *_ = _plan(tmp_path)
    frozen = evaluation.read(plan)

    missing = evaluation.aggregate(frozen, [_valid_cell(1), _valid_cell(3)])
    assert missing["complete_attempt_inventory"] is False
    assert missing["state"] == "needs_review"

    duplicate = evaluation.aggregate(
        frozen, [_valid_cell(1), _valid_cell(1), _valid_cell(3)]
    )
    assert duplicate["complete_attempt_inventory"] is False

    reused = evaluation.aggregate(
        frozen,
        [
            _valid_cell(1, sandbox="same"),
            _valid_cell(2, sandbox="same"),
            _valid_cell(3),
        ],
    )
    assert reused["reused_sandbox_ids"] == ["same"]
    assert reused["gates"]["solver_at_least_2_of_3"] is False


def test_aggregate_accepts_two_of_three_solvers_but_rejects_negative_failure(tmp_path):
    plan, *_ = _plan(tmp_path)
    frozen = evaluation.read(plan)
    acceptable = evaluation.aggregate(
        frozen, [_valid_cell(1), _valid_cell(2), _valid_cell(3, solver=False)]
    )
    assert acceptable["state"] == "repeated_runtime_passed"
    assert acceptable["gates"]["solver_at_least_2_of_3"] is True

    negatives = evaluation.aggregate(
        frozen,
        [
            _valid_cell(1),
            _valid_cell(2, negatives=False),
            _valid_cell(3, solver=False),
        ],
    )
    assert negatives["gates"]["solver_at_least_2_of_3"] is True
    assert negatives["gates"]["authored_controls_all_attempts"] is False
    assert negatives["state"] == "needs_review"


def test_inspect_attempt_rejects_retry_limit_and_nonzero_negative(
    tmp_path, monkeypatch
):
    package, bundle, controls = _inputs(tmp_path)
    root = tmp_path / "attempt"
    root.mkdir()
    for name in ("oracle.json", "solver.json", "adversary.json"):
        (root / name).write_text(json.dumps({"cases": []}))
    evidence = {
        "attestation": {
            "solver": {"retry_limit": 2, "state": "passed"},
            "adversarial": {"independent_attack_executed": True},
            "oracle_artifact": "oracle.json",
            "solver_artifact": "solver.json",
            "adversary_artifact": "adversary.json",
        },
        "cases": [
            {
                "id": "gold",
                "source_author": "author",
                "category": "solver",
                "result": {"status": "graded", "reward": 1},
            },
            {
                "id": "negative",
                "source_author": "author",
                "category": "control",
                "result": {"status": "graded", "reward": 1},
            },
        ],
    }
    (root / "runtime-evidence.json").write_text(json.dumps(evidence))
    monkeypatch.setattr(synthesis, "_attestation_issues", lambda *args, **kwargs: [])
    monkeypatch.setattr(adversary, "assess_attack_results", lambda *args, **kwargs: [])
    result = evaluation.inspect_attempt(
        root,
        {"package": package, "bundle": bundle, "controls": controls},
        {
            "specification_sha256": sha256(bundle / "specification.json"),
            "lowered_specification_sha256": sha256(bundle / "specification.json"),
            "critical_control_ids": ["negative"],
        },
    )
    assert result["state"] == "invalid_evidence"
    assert "exactly one solver attempt" in " ".join(result["issues"])
    assert result["authored_controls_passed"] is False


def test_run_uses_frozen_environment_and_never_overwrites_output(tmp_path, monkeypatch):
    plan, _package, _bundle, _controls = _plan(tmp_path)
    plan_sha = sha256(plan)
    output = tmp_path / "evaluation-output"
    calls = []

    class Toolchain:
        @staticmethod
        def resolve(*_args):
            return SimpleNamespace(runtime_command=lambda: ["remote-runtime"])

    def fake_run(argv, *, timeout, env):
        calls.append((argv, timeout, env))
        return subprocess.CompletedProcess(argv, 0, "remote stdout", "")

    def fake_inspect(root, _inputs, _plan):
        number = int(root.name)
        return _valid_cell(number, solver=number != 3)

    monkeypatch.setenv("CAPABILITY_SOLVER_RETRIES", "99")
    monkeypatch.setattr(synthesis, "OfficialToolchain", Toolchain)
    monkeypatch.setattr(synthesis, "_run", fake_run)
    monkeypatch.setattr(evaluation, "inspect_attempt", fake_inspect)
    args = SimpleNamespace(
        plan=str(plan),
        plan_sha256=plan_sha,
        out=str(output),
        taskcompendium_source=None,
        shellsim_bridge=None,
        concurrency=1,
    )
    assert evaluation.run_evaluation(args) == 0
    assert len(calls) == 3
    assert all(timeout == 60 for _, timeout, _ in calls)
    assert all(env["CAPABILITY_SOLVER_RETRIES"] == "1" for _, _, env in calls)
    assert all(env["CAPABILITY_SOLVER_MODEL"] == "glm-5.3" for _, _, env in calls)
    assert all("--output" in argv for argv, _, _ in calls)
    assert evaluation.read(output / "matrix.json")["state"] == "repeated_runtime_passed"

    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(args)


def test_policy_uses_the_real_adversary_remaining_context_parser(monkeypatch):
    # This imports only the parser with a minimal Harbor type stub; no agent,
    # generated code, or inference is created on the controller.
    taskcompendium = ModuleType("taskcompendium")
    harbor = ModuleType("taskcompendium.harbor")
    agents = ModuleType("taskcompendium.harbor.agents")
    agents.DirectChatAgent = type("DirectChatAgent", (), {})
    agents.ReplayAgent = type("ReplayAgent", (), {})
    agents.ShellToolAgent = type("ShellToolAgent", (), {})
    agents.TurnCapExhaustedError = type("TurnCapExhaustedError", (RuntimeError,), {})
    agents._record = lambda *args, **kwargs: None
    agents.shell_tool_definition = lambda binding: {}
    monkeypatch.setitem(sys.modules, "taskcompendium", taskcompendium)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor", harbor)
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.agents", agents)
    sys.modules.pop("capability_pipeline.runtime_agents", None)
    from capability_pipeline.runtime_agents import adversary_token_limits

    assert adversary_token_limits(
        evaluation.POLICY["CAPABILITY_ADVERSARY_TOKEN_LIMITS"]
    ) == (
        131072,
        None,
    )


def test_artifact_manifest_is_reproducible_and_excludes_its_own_outputs(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "payload.txt").write_text("payload")
    (root / "nested/trace.json").write_text("{}")
    first = evaluation.artifact_manifest(root)
    evaluation.write_new(root / "artifacts.manifest.json", first)
    evaluation.write_new(root / "receipt.json", {"state": "valid"})
    assert evaluation.artifact_manifest(root) == first
    assert set(first["files"]) == {"nested/trace.json", "payload.txt"}
    (root / "payload.txt").write_text("changed")
    assert (
        evaluation.artifact_manifest(root)["files"]["payload.txt"]["sha256"]
        != first["files"]["payload.txt"]["sha256"]
    )


def test_inspect_allows_noncritical_partial_credit_but_rejects_critical_credit(
    tmp_path, monkeypatch
):
    package, bundle, controls = _inputs(tmp_path)
    definition = evaluation.read(controls)
    definition["cases"][1]["expect"] = {"status": "graded", "reward_max": 0.2}
    controls.write_text(json.dumps(definition))
    root = tmp_path / "attempt"
    root.mkdir()
    for name in ("oracle.json", "solver.json", "adversary.json"):
        (root / name).write_text(json.dumps({"cases": []}))
    evidence = {
        "attestation": {
            "solver": {"retry_limit": 1, "state": "passed"},
            "adversarial": {"independent_attack_executed": True},
            "oracle_artifact": "oracle.json",
            "solver_artifact": "solver.json",
            "adversary_artifact": "adversary.json",
        },
        "cases": [
            {
                "id": "gold",
                "source_author": "author",
                "category": "solver",
                "control_type": "independent_solver",
                "result": {"status": "graded", "reward": 1},
            },
            {
                "id": "negative",
                "source_author": "author",
                "category": "control",
                "control_type": "authored_adversarial_control",
                "result": {"status": "graded", "reward": 0.1},
            },
        ],
    }
    (root / "runtime-evidence.json").write_text(json.dumps(evidence))
    monkeypatch.setattr(synthesis, "_attestation_issues", lambda *args, **kwargs: [])
    monkeypatch.setattr(adversary, "assess_attack_results", lambda *args, **kwargs: [])
    inputs = {"package": package, "bundle": bundle, "controls": controls}
    plan = {
        "specification_sha256": sha256(bundle / "specification.json"),
        "lowered_specification_sha256": sha256(bundle / "specification.json"),
        "critical_control_ids": [],
    }
    assert evaluation.inspect_attempt(root, inputs, plan)["state"] == "valid"

    plan["critical_control_ids"] = ["negative"]
    rejected = evaluation.inspect_attempt(root, inputs, plan)
    assert rejected["authored_controls_passed"] is False
    assert "negative received nonzero reward" in " ".join(
        rejected["control_results"]["authored_controls"]["issues"]
    )

    evidence["attestation"]["solver"]["state"] = "unknown"
    (root / "runtime-evidence.json").write_text(json.dumps(evidence))
    unknown = evaluation.inspect_attempt(root, inputs, plan)
    assert "unknown solver outcome state" in unknown["issues"]


def test_inspect_primary_bound_repeat_checks_fresh_controls_without_new_attack(
    tmp_path, monkeypatch
):
    package, bundle, controls = _inputs(tmp_path)
    root = tmp_path / "attempt"
    root.mkdir()
    for name in ("oracle.json", "solver.json"):
        (root / name).write_text(json.dumps({"cases": []}))
    (root / "runtime-evidence.json").write_text(json.dumps({
        "attestation": {
            "solver": {"retry_limit": 1, "state": "passed"},
            "adversarial": {"diagnostic_new_attacks": "not_run_primary_bound"},
            "oracle_artifact": "oracle.json",
            "solver_artifact": "solver.json",
            "adversary_artifact": None,
        },
        "cases": [
            {"id": "gold", "source_author": "author", "category": "solver",
             "control_type": "independent_solver",
             "result": {"status": "graded", "reward": 1}},
            {"id": "negative", "source_author": "author", "category": "control",
             "control_type": "authored_adversarial_control",
             "result": {"status": "graded", "reward": 0}},
        ],
    }))
    observed = []

    def attestation_issues(*_args, **kwargs):
        observed.append(kwargs["require_independent_adversary"])
        return [] if kwargs["require_independent_adversary"] is False else [
            "new adversary evidence required"
        ]

    monkeypatch.setattr(synthesis, "_attestation_issues", attestation_issues)
    inputs = {"package": package, "bundle": bundle, "controls": controls}
    plan = {
        "purpose": "primary_bound_repeatability",
        "specification_sha256": sha256(bundle / "specification.json"),
        "lowered_specification_sha256": sha256(bundle / "specification.json"),
        "critical_control_ids": ["negative"],
    }
    result = evaluation.inspect_attempt(root, inputs, plan)
    assert result["state"] == "valid"
    assert result["oracle_passed"] is True
    assert result["solver_passed"] is True
    assert result["authored_controls_passed"] is True
    assert result["primary_adversarial_review_bound"] is True
    assert result["independent_attacks_passed"] is False
    assert observed == [False]

    plan["purpose"] = "full_runtime"
    assert evaluation.inspect_attempt(root, inputs, plan)["state"] == "invalid_evidence"
    assert observed == [False, True]


def test_shellsim_bridge_is_required_and_frozen_with_the_plan(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    (bundle / "binding.json").write_text(
        json.dumps({"environment": {"kind": "shellsim"}})
    )
    plan = tmp_path / "shellsim-plan.json"
    common = {
        "package": str(package),
        "bundle": str(bundle),
        "controls": str(controls),
        "out": str(plan),
        "attempt_timeout": 60,
    }
    with pytest.raises(ValueError, match="pinned bridge"):
        evaluation.make_plan(SimpleNamespace(**common, shellsim_bridge=None))

    bridge = tmp_path / "bridge.py"
    bridge.write_text("# pinned bridge\n")
    assert (
        evaluation.make_plan(SimpleNamespace(**common, shellsim_bridge=str(bridge)))
        == 0
    )
    frozen, inputs = evaluation.validate_plan(plan, sha256(plan))
    assert frozen["inputs"]["shellsim_bridge"]["sha256"] == sha256(bridge)
    assert inputs["shellsim_bridge"] == bridge.resolve()


def test_critical_control_inventory_and_flags_fail_closed(tmp_path):
    plan_path, _package, _bundle, controls = _plan(tmp_path)
    document = evaluation.read(plan_path)
    document["critical_control_ids"] = []
    plan_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="critical control inventory"):
        evaluation.validate_plan(plan_path, sha256(plan_path))

    cases = evaluation.read(controls)["cases"]
    cases[1]["critical"] = "yes"
    with pytest.raises(ValueError, match="must be boolean"):
        evaluation.critical_control_ids(cases)
    cases[1]["critical"] = True
    cases[1]["expect"] = {"status": "extraction_error"}
    with pytest.raises(ValueError, match="graded exact-zero"):
        evaluation.critical_control_ids(cases)
    cases[1]["class"] = "positive"
    cases[1]["expect"] = {"status": "graded", "reward_min": 0, "reward_max": 0}
    with pytest.raises(ValueError, match="graded exact-zero"):
        evaluation.critical_control_ids(cases)


def test_lowered_specification_is_bound_by_bytes_and_checked_by_meaning(tmp_path):
    """Lowering re-serializes; the manifest binds lowered bytes, not authored ones."""
    package, bundle, controls = _inputs(tmp_path)
    authored = bundle / "specification.json"
    lowered = package / "specification.json"
    assert authored.read_bytes() != lowered.read_bytes()
    assert json.loads(authored.read_text()) == json.loads(lowered.read_text())

    plan_path = tmp_path / "plan.json"
    evaluation.make_plan(SimpleNamespace(
        package=package, bundle=bundle, controls=controls,
        out=plan_path, attempt_timeout=60,
    ))
    plan = json.loads(plan_path.read_text())
    # The plan still identifies the authored bundle it will re-verify at run time.
    assert plan["specification_sha256"] == sha256(authored)


def test_package_manifest_must_bind_its_own_lowered_specification(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    (package / "manifest.json").write_text(
        json.dumps({"specification_sha256": sha256(bundle / "specification.json")})
    )
    with pytest.raises(ValueError, match="does not bind the lowered specification"):
        evaluation.make_plan(SimpleNamespace(
            package=package, bundle=bundle, controls=controls,
            out=tmp_path / "plan.json", attempt_timeout=60,
        ))


def test_lowered_specification_must_still_mean_the_authored_one(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    lowered = package / "specification.json"
    lowered.write_text(json.dumps({"task": "substituted"}, indent=2) + "\n")
    (package / "manifest.json").write_text(
        json.dumps({"specification_sha256": sha256(lowered)})
    )
    with pytest.raises(ValueError, match="differs from the authored bundle"):
        evaluation.make_plan(SimpleNamespace(
            package=package, bundle=bundle, controls=controls,
            out=tmp_path / "plan.json", attempt_timeout=60,
        ))


def test_lowered_package_without_a_specification_is_refused(tmp_path):
    package, bundle, controls = _inputs(tmp_path)
    (package / "specification.json").unlink()
    with pytest.raises(ValueError, match="missing its specification"):
        evaluation.make_plan(SimpleNamespace(
            package=package, bundle=bundle, controls=controls,
            out=tmp_path / "plan.json", attempt_timeout=60,
        ))


def test_plan_records_the_lowered_hash_and_inspect_refuses_a_plan_without_it(tmp_path):
    """The runtime attests the lowered package; the plan must carry that hash."""
    package, bundle, controls = _inputs(tmp_path)
    plan_path = tmp_path / "plan.json"
    evaluation.make_plan(SimpleNamespace(
        package=package, bundle=bundle, controls=controls,
        out=plan_path, attempt_timeout=60,
    ))
    plan = json.loads(plan_path.read_text())
    assert plan["lowered_specification_sha256"] == sha256(package / "specification.json")
    assert plan["specification_sha256"] == sha256(bundle / "specification.json")
    assert plan["lowered_specification_sha256"] != plan["specification_sha256"]

    root = tmp_path / "attempt"
    root.mkdir()
    (root / "runtime-evidence.json").write_text(json.dumps({"attestation": {}}))
    inputs = {"package": package, "bundle": bundle, "controls": controls}
    del plan["lowered_specification_sha256"]
    with pytest.raises(ValueError, match="does not record the lowered specification hash"):
        evaluation.inspect_attempt(root, inputs, plan)


def test_runtime_policy_binds_the_sandbox_provider(monkeypatch):
    from capability_pipeline import evaluation

    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    policy = evaluation.runtime_policy()
    assert policy["CAPABILITY_SANDBOX_PROVIDER"] == "silo"
    # every frozen knob is still there, unchanged
    assert {k: v for k, v in policy.items() if k != "CAPABILITY_SANDBOX_PROVIDER"} == evaluation.POLICY


def test_a_plan_cannot_run_under_a_different_provider(monkeypatch):
    from capability_pipeline import evaluation

    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    silo_plan = evaluation.runtime_policy()
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "daytona")
    daytona_plan = evaluation.runtime_policy()
    assert evaluation.policy_matches(daytona_plan)
    assert not evaluation.policy_matches(silo_plan)
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    assert evaluation.policy_matches(silo_plan)
    assert not evaluation.policy_matches(daytona_plan)


def test_plans_without_a_provider_stay_valid_only_under_daytona(monkeypatch):
    from capability_pipeline import evaluation

    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "daytona")
    assert evaluation.policy_matches(dict(evaluation.POLICY))
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    assert not evaluation.policy_matches(dict(evaluation.POLICY))
