import asyncio
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import capability_pipeline.regrade as regrade_module
from capability_pipeline.regrade import (
    HARBOR_REVISION,
    _candidate_resource_request,
    _native_deterministic_verifier,
    _runtime_fields,
    actual_extraction_transport,
    attest_trial_isolation,
    build_plan,
    create_plan_bundle,
    summarize,
    validate_controls,
    validate_package_manifest,
    validate_persisted_plan,
    validate_plan_bundle,
)


def controls():
    return {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "response": "answer",
                "expect": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
            },
            {
                "id": "negative",
                "class": "negative",
                "response": "wrong",
                "expect": {"status": "graded", "reward_min": 0.0, "reward_max": 0.2},
            },
        ]
    }


def test_validate_controls_accepts_fixed_no_tool_responses():
    assert [case["id"] for case in validate_controls(controls())] == [
        "positive",
        "negative",
    ]


def test_validate_controls_accepts_exact_extraction_expectation_only_for_negative():
    value = controls()
    value["cases"][1]["expect"] = {"status": "extraction_error"}
    assert validate_controls(value) == value["cases"]
    value["cases"][1]["expect"]["reward_max"] = 0
    with pytest.raises(ValueError, match="cannot declare reward"):
        validate_controls(value)
    value["cases"][1]["expect"] = {"status": "extraction_error"}
    value["cases"][1]["class"] = "positive"
    with pytest.raises(ValueError, match="positive controls"):
        validate_controls(value)


def test_extraction_transport_is_expected_but_fixed_input_remains_unassessed():
    value = controls()
    value["cases"][1]["expect"] = {"status": "extraction_error"}
    plan = _summary_plan()
    rows = []
    for index, cell in enumerate(plan["cells"]):
        positive = cell["case_id"] == "positive"
        rows.append({
            **cell,
            "status": "graded" if positive else "extraction_error",
            "reward": 1.0 if positive else None,
            "outcome_class": "graded" if positive else "extraction_error",
            "private_verifier": {"sandbox_id": str(index)} if positive else None,
            "verifier_result_present": positive,
            "exception": None if positive else {"type": "ExtractionError", "message": "bad extraction"},
            "grading_sha256": "a" * 64,
        })
    report = summarize(plan, rows, value)
    assert report["state"] == "unassessed"
    assert report["cases"][1]["expectation_met"]
    assert report["cases"][1]["fixed_grading_input_assessment"] == "unassessed_missing_preverifier_fingerprint"
    rows[-1]["verifier_result_present"] = True
    assert summarize(plan, rows, value)["state"] == "failed"


def test_validate_controls_allows_container_replay_but_not_no_tool():
    value = controls()
    value["cases"][0]["workspace"] = "controls/positive"
    value["cases"][0]["commands"] = ["true"]
    assert validate_controls(value, binding_kind="docker") == value["cases"]
    assert validate_controls(value, binding_kind="shellsim") == value["cases"]
    with pytest.raises(ValueError, match="no-tool"):
        validate_controls(value)


def test_package_manifest_binds_harbor_specification_and_runtime():
    validate_package_manifest(
        {
            "harbor_revision": HARBOR_REVISION,
            "specification_sha256": "spec",
            "step_names": ["step-1"],
            "verifier_runtimes": [{"kind": "container", "image": "image"}],
        },
        specification_sha256="spec",
        runtime_image="image",
    )


def test_package_manifest_rejects_runtime_mismatch():
    with pytest.raises(RuntimeError, match="runtime"):
        validate_package_manifest(
            {
                "harbor_revision": HARBOR_REVISION,
                "specification_sha256": "spec",
                "step_names": ["step-1"],
                "verifier_runtimes": [{"kind": "container", "image": "other"}],
            },
            specification_sha256="spec",
            runtime_image="image",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"workspace": "control"},
        {"commands": ["true"]},
        {"response": None},
        {"transcript": []},
        {"step_index": 1},
    ],
)
def test_validate_controls_rejects_unsupported_replay_shapes(change):
    value = controls()
    value["cases"][0].update(change)
    with pytest.raises((TypeError, ValueError)):
        validate_controls(value)


def _write(path: Path, value="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_plan_freezes_matrix_before_outputs(tmp_path):
    package, bundle = tmp_path / "package", tmp_path / "bundle"
    tc, tools = tmp_path / "taskcompendium", tmp_path / "tools"
    controls_path = bundle / "controls.json"
    for path in (
        package / "manifest.json",
        bundle / "binding.json",
        bundle / "renderings.json",
        bundle / "specification.json",
        tc / "pyproject.toml",
        tc / "uv.lock",
        tools / "dt.py",
    ):
        _write(path)
    controls_path.write_text(json.dumps(controls()))
    args = SimpleNamespace(
        package=package,
        bundle=bundle,
        controls=controls_path,
        taskcompendium_source=tc,
        daytona_tools=tools,
        repeats=10,
        parallelism=8,
    )
    plan = build_plan(args, controls(), runtime_image="image@sha256:abc", supervisor_python="python3")
    assert plan["cell_count"] == 20
    assert len({cell["trial_name"] for cell in plan["cells"]}) == 20
    assert plan["cells"][0]["trial_name"] == "regrade-positive-run-01"
    assert plan["cells"][-1]["trial_name"] == "regrade-negative-run-10"
    _write(tc / "src" / "package" / "module.py", "code")
    baseline = build_plan(args, controls(), runtime_image="image@sha256:abc", supervisor_python="python3")
    _write(tc / "src" / "package" / "__pycache__" / "module.pyc", "cache")
    _write(tc / ".venv" / "state", "mutable")
    assert build_plan(args, controls(), runtime_image="image@sha256:abc", supervisor_python="python3") == baseline
    relocated = tmp_path / "relocated"
    for name in ("package", "bundle", "taskcompendium", "tools"):
        shutil.copytree(tmp_path / name, relocated / name)
    moved_args = SimpleNamespace(**{
        **vars(args),
        "package": relocated / "package",
        "bundle": relocated / "bundle",
        "controls": relocated / "bundle" / "controls.json",
        "taskcompendium_source": relocated / "taskcompendium",
        "daytona_tools": relocated / "tools",
    })
    assert build_plan(moved_args, controls(), runtime_image="image@sha256:abc", supervisor_python="python3") == baseline


def test_docker_plan_rebuild_preserves_binding_and_replay_sources(tmp_path):
    package, bundle = tmp_path / "package", tmp_path / "bundle"
    tc, tools = tmp_path / "taskcompendium", tmp_path / "tools"
    for path in (
        package / "manifest.json", bundle / "binding.json",
        bundle / "renderings.json", bundle / "specification.json",
        tc / "pyproject.toml", tc / "uv.lock", tools / "dt.py",
    ):
        _write(path)
    value = controls()
    value["cases"][0]["commands"] = ["printf answer > result.txt"]
    controls_path = bundle / "controls.json"
    controls_path.write_text(json.dumps(value))
    args = SimpleNamespace(
        package=package, bundle=bundle, controls=controls_path,
        taskcompendium_source=tc, daytona_tools=tools,
        repeats=10, parallelism=8,
    )
    first = build_plan(args, value, runtime_image="image@sha256:abc", supervisor_python="python3", binding_kind="docker")
    rebuilt = build_plan(args, value, runtime_image="image@sha256:abc", supervisor_python="python3", binding_kind="docker")
    assert first["grading_strategy"] == "capture_once"
    assert len(first["identities"]["fixed_grading_capture"]["sha256"]) == 64
    assert first == rebuilt
    assert first["binding_kind"] == "docker"
    assert "positive" in first["replay_inventory"]
    assert {"candidate_environment", "candidate_resources", "candidate_telemetry", "runtime_agents"} <= first["identities"].keys()


def test_shellsim_plan_binds_snapshot_overlay_and_fixed_workspace(tmp_path):
    package, bundle = tmp_path / "package", tmp_path / "bundle"
    tc, tools = tmp_path / "taskcompendium", tmp_path / "tools"
    for path in (
        package / "manifest.json", bundle / "binding.json",
        bundle / "renderings.json", bundle / "specification.json",
        tc / "pyproject.toml", tc / "uv.lock", tools / "dt.py",
    ):
        _write(path)
    value = controls()
    value["cases"][0]["workspace"] = "controls/positive"
    value["cases"][0]["commands"] = ["printf answer > answer.txt"]
    _write(bundle / "controls/positive/fixture.txt", "fixed")
    controls_path = bundle / "controls.json"
    controls_path.write_text(json.dumps(value))
    args = SimpleNamespace(
        package=package, bundle=bundle, controls=controls_path,
        taskcompendium_source=tc, daytona_tools=tools,
        repeats=10, parallelism=2,
    )
    plan = build_plan(args, value, runtime_image="image@sha256:abc", supervisor_python="python3", binding_kind="shellsim")
    assert plan["grading_strategy"] == "capture_once"
    assert plan["binding_kind"] == "shellsim"
    assert len(plan["identities"]["shellsim_overlay"]["patch_sha256"]) == 64
    assert len(plan["identities"]["shellsim_capture_adapter"]["sha256"]) == 64
    assert plan == build_plan(args, value, runtime_image="image@sha256:abc", supervisor_python="python3", binding_kind="shellsim")


def test_persisted_plan_requires_hash_and_exact_content(tmp_path):
    plan = {"cells": [{"case_id": "fixed"}]}
    path = tmp_path / "plan.json"
    data = (json.dumps(plan) + "\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert validate_persisted_plan(plan, path, digest) == data
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_persisted_plan(plan, path, "0" * 64)
    with pytest.raises(RuntimeError, match="frozen inputs"):
        validate_persisted_plan({"cells": []}, path, digest)


def test_real_c17_metadata_plan_is_seventy_cells_and_rejects_unsupported_binding(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/c17-repeated-evaluation-003"
    if not source.is_dir():
        pytest.skip("frozen c17 fixture unavailable")
    tc = tmp_path / "taskcompendium"
    _write(tc / "pyproject.toml")
    _write(tc / "uv.lock")
    result = create_plan_bundle(source, tc, tmp_path / "regrade")
    assert result["cells"] == 70
    assert validate_plan_bundle(tmp_path / "regrade", tc, result["plan_sha256"])[0]["cell_count"] == 70
    unsupported = tmp_path / "unsupported"
    shutil.copytree(source, unsupported)
    binding_path = unsupported / "bundle/binding.json"
    binding = json.loads(binding_path.read_text())
    binding["tools"] = [{"name": "shell"}]
    binding_path.write_text(json.dumps(binding))
    manifest_path = unsupported / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["bundle/binding.json"] = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no-tool"):
        create_plan_bundle(unsupported, tc, tmp_path / "rejected")


def test_real_c32_direct_verifier_metadata_creates_fixed_matrix(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/c32-evaluation-diagnostic-004"
    if not source.is_dir():
        pytest.skip("frozen c32 fixture unavailable")
    tc = tmp_path / "taskcompendium"
    _write(tc / "pyproject.toml")
    _write(tc / "uv.lock")
    image, supervisor, kind = _runtime_fields(source / "bundle")
    assert kind == "docker"
    assert image.endswith("@sha256:25cda6cf8b745cbc02614069ee165e9af4816f672ffae7b41ed3d640c707ce29")
    assert supervisor == "/opt/py312/bin/python3"
    result = create_plan_bundle(source, tc, tmp_path / "regrade", parallelism=8)
    frozen, runtime_args = validate_plan_bundle(
        tmp_path / "regrade", tc, result["plan_sha256"]
    )
    assert frozen["case_count"] == 7
    assert frozen["cell_count"] == 70
    assert frozen["parallelism"] == 8
    assert frozen["candidate_resource_request"] == {
        "path": "candidate-resources.json",
        "sha256": "03efc70b84d416004409e912a6784d47c4f6ff9ea506bd765b8fcdb340b3a626",
        "request": {"cpu": 2, "memory_gb": 2, "disk_gb": 10},
    }
    assert _candidate_resource_request(runtime_args, "docker")[0] == {
        "cpu": 2,
        "memory_gb": 2,
        "disk_gb": 10,
    }


def test_candidate_resources_reject_no_tool_binding_and_remain_optional(tmp_path):
    resources = tmp_path / "candidate-resources.json"
    resources.write_text(json.dumps({"cpu": 2, "memory_gb": 2, "disk_gb": 10}))
    args = SimpleNamespace(candidate_resources=resources)
    with pytest.raises(ValueError, match="Docker"):
        _candidate_resource_request(args, "none")
    assert _candidate_resource_request(SimpleNamespace(candidate_resources=None), "none") == (
        None,
        None,
    )


def test_runtime_fields_keeps_only_valid_nested_legacy_shape(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "binding.json").write_text(json.dumps({"environment": {"kind": "none"}, "tools": []}))
    direct = {
        "kind": "tasktrove",
        "mode": "script",
        "runtime": {"kind": "container", "image": "image", "supervisor_python": "python3"},
    }
    (bundle / "specification.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "verifier": {
                            "kind": "code_answer",
                            "output_path": "submission.json",
                            "verifier": direct,
                        }
                    }
                ]
            }
        )
    )
    assert _runtime_fields(bundle) == ("image", "python3", "none")
    # A malformed direct declaration cannot bypass validation through a nested
    # value with otherwise acceptable fields.
    (bundle / "specification.json").write_text(
        json.dumps({"steps": [{"verifier": {"kind": "other", "verifier": direct}}]})
    )
    with pytest.raises(ValueError, match="ContainerRuntime"):
        _runtime_fields(bundle)


@pytest.mark.parametrize("mode", ["mcq", "math", "numeric", "exact", "json-schema", "xml-elements", "csv-columns", "ifeval"])
def test_native_runtime_fields_require_direct_deterministic_mode(tmp_path, mode):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "binding.json").write_text(json.dumps({"environment": {"kind": "none"}, "tools": []}))
    (bundle / "specification.json").write_text(json.dumps({"steps": [{"verifier": {"kind": "tasktrove", "mode": mode, "parameters": {}}}]}))
    assert _runtime_fields(bundle) == (None, None, "none")
    (bundle / "binding.json").write_text(json.dumps({"environment": {"kind": "shellsim", "workdir": "/app"}, "tools": [{"name": "shell", "backend": "shellsim"}]}))
    assert _runtime_fields(bundle) == (None, None, "shellsim")
    (bundle / "specification.json").write_text(json.dumps({"steps": [{"verifier": {"kind": "tasktrove", "mode": mode, "parameters": {}, "runtime": {"kind": "container"}}}]}))
    with pytest.raises(ValueError, match="no runtime"):
        _runtime_fields(bundle)


@pytest.mark.parametrize("mode", sorted(regrade_module.NATIVE_DETERMINISTIC_MODES))
def test_native_code_answer_wrapper_keeps_extraction_contract(tmp_path, mode):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "binding.json").write_text(json.dumps({"environment": {"kind": "none"}, "tools": []}))
    inner = {"kind": "tasktrove", "mode": mode, "parameters": {}}
    wrapper = {"kind": "code_answer", "output_path": "submission.json", "verifier": inner}
    specification = bundle / "specification.json"
    specification.write_text(json.dumps({"steps": [{"verifier": wrapper}]}))
    assert _runtime_fields(bundle) == (None, None, "none")
    wrapper["output_path"] = ""
    specification.write_text(json.dumps({"steps": [{"verifier": wrapper}]}))
    with pytest.raises(ValueError, match="ContainerRuntime"):
        _runtime_fields(bundle)
    wrapper["output_path"] = "submission.json"
    for mixed in ("runtime", "judge"):
        wrapper[mixed] = {"kind": "container"}
        specification.write_text(json.dumps({"steps": [{"verifier": wrapper}]}))
        with pytest.raises(ValueError, match="wrapper cannot add"):
            _runtime_fields(bundle)
        del wrapper[mixed]
        inner[mixed] = {"kind": "container"}
        specification.write_text(json.dumps({"steps": [{"verifier": wrapper}]}))
        with pytest.raises(ValueError, match="no runtime or judge"):
            _runtime_fields(bundle)
        del inner[mixed]


def test_resolved_nested_native_mode_has_no_private_runtime():
    verifier = SimpleNamespace(mode=SimpleNamespace(value="exact"), judge=None)
    assert _native_deterministic_verifier(verifier, None)
    assert not _native_deterministic_verifier(verifier, object())
    verifier.judge = object()
    assert not _native_deterministic_verifier(verifier, None)


def test_native_package_manifest_requires_null_runtime():
    manifest = {"harbor_revision": HARBOR_REVISION, "specification_sha256": "abc", "step_names": ["step-1"], "verifier_runtimes": [None]}
    validate_package_manifest(manifest, specification_sha256="abc", runtime_image=None)
    manifest["verifier_runtimes"] = [{}]
    with pytest.raises(RuntimeError, match="runtime"):
        validate_package_manifest(manifest, specification_sha256="abc", runtime_image=None)


def test_native_plan_freezes_adapter_and_upstream_source(tmp_path, monkeypatch):
    package, bundle = tmp_path / "package", tmp_path / "bundle"
    tc, tools = tmp_path / "taskcompendium", tmp_path / "tools"
    for path in (
        package / "manifest.json", bundle / "binding.json", bundle / "renderings.json",
        bundle / "specification.json", tc / "pyproject.toml", tc / "uv.lock", tools / "dt.py",
    ):
        _write(path)
    upstream = tc / "src/taskcompendium/harbor/verifier.py"
    _write(upstream, "pinned upstream")
    monkeypatch.setattr(regrade_module, "BASE_VERIFIER_SHA256", hashlib.sha256(upstream.read_bytes()).hexdigest())
    control_path = bundle / "controls.json"
    control_path.write_text(json.dumps(controls()))
    args = SimpleNamespace(
        package=package, bundle=bundle, controls=control_path,
        taskcompendium_source=tc, daytona_tools=tools, repeats=10, parallelism=8,
    )
    plan = build_plan(args, controls(), runtime_image=None, supervisor_python=None)
    assert plan["verifier_surface"] == "native_deterministic"
    assert plan["runtime"] == {"image": None, "supervisor_python": None}
    assert plan["identities"]["native_semantic_verifier"]["sha256"] == hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert plan["identities"]["native_semantic_verifier_runtime_sha256"] == regrade_module.PATCHED_VERIFIER_SHA256
    assert len(plan["identities"]["native_verifier"]["sha256"]) == 64
    upstream.write_text("changed upstream")
    with pytest.raises(ValueError, match="pinned base"):
        build_plan(args, controls(), runtime_image=None, supervisor_python=None)


def test_native_summary_requires_bound_receipt_without_private_verifier():
    plan = {**_summary_plan(), "verifier_surface": "native_deterministic", "identities": {
        "native_verifier": {"sha256": "a" * 64},
        "native_semantic_verifier_runtime_sha256": "b" * 64,
    }}
    rows = [{
        **cell,
        "status": "graded",
        "reward": 1.0 if cell["case_id"] == "positive" else 0.0,
        "outcome_class": "graded",
        "verifier_result_present": True,
        "native_verifier_receipt": {
            "schema_version": "capability-native-verifier-receipt-v1",
            "adapter_sha256": "a" * 64,
            "semantic_verifier_sha256": "b" * 64,
        },
    } for cell in plan["cells"]]
    assert summarize(plan, rows, controls())["state"] == "passed"
    rows[0]["private_verifier"] = {"sandbox_id": "invented"}
    assert summarize(plan, rows, controls())["state"] == "failed"
    rows[0].pop("private_verifier")
    rows[0]["native_verifier_receipt"]["adapter_sha256"] = "c" * 64
    assert summarize(plan, rows, controls())["state"] == "failed"


def test_native_extraction_can_pass_only_with_bound_preverifier_fingerprint():
    value = controls()
    value["cases"][1]["expect"] = {"status": "extraction_error"}
    plan = {**_summary_plan(), "verifier_surface": "native_deterministic", "require_fixed_grading_input": True,
            "identities": {"native_verifier": {"sha256": "a" * 64},
                           "native_semantic_verifier_runtime_sha256": "b" * 64}}
    rows = []
    for cell in plan["cells"]:
        extraction = cell["case_id"] == "negative"
        rows.append({
            **cell,
            "status": "extraction_error" if extraction else "graded",
            "reward": None if extraction else 1.0,
            "outcome_class": "extraction_error" if extraction else "graded",
            "exception": {"type": "ExtractionError"} if extraction else None,
            "verifier_result_present": not extraction,
            "grading_sha256": "d" * 64,
            "native_verifier_receipt": {"schema_version": "capability-native-verifier-receipt-v1",
                                        "adapter_sha256": "a" * 64, "semantic_verifier_sha256": "b" * 64},
            "grading_input_fingerprint": {
                "schema_version": "capability-grading-input-fingerprint-v1",
                "submission_sha256": ("c" if extraction else "a") * 64,
                "grading_input_sha256": ("d" if extraction else "b") * 64,
                "specification_sha256": "e" * 64,
                "protocol_sha256": "f" * 64,
                "transcript_sha256": "0" * 64,
                "payload_sha256": "1" * 64,
                "step_index": 0,
                "workspace_file_count": 0,
            },
        })
    assert summarize(plan, rows, value)["state"] == "passed"
    rows[-1].pop("grading_input_fingerprint")
    assert summarize(plan, rows, value)["state"] == "failed"


@pytest.mark.parametrize("kind", ["none", "docker"])
def test_native_trial_attestation_skips_private_but_keeps_candidate(kind, tmp_path, monkeypatch):
    calls = []

    def candidate_record(path, trials, seen):
        calls.append(("candidate", path, trials))
        seen.add("candidate-1")
        return {"sandbox_id": "candidate-1"}

    def forbidden_private(*args, **kwargs):
        raise AssertionError("native verifier must not request a private sandbox receipt")

    monkeypatch.setattr(regrade_module, "provider_isolation_record", candidate_record)
    monkeypatch.setattr(regrade_module, "verifier_isolation_record", forbidden_private)
    trial_root = tmp_path / "native-01"
    (trial_root / "verifier").mkdir(parents=True)
    (trial_root / "verifier/taskcompendium-result.json").write_text('{"status":"graded","reward":1}')
    rows = [{"trial_name": "native-01", "outcome_class": "graded"}]
    attest_trial_isolation(rows, tmp_path, kind=kind, native=True, runtime_image=None, supervisor_python=None)
    assert rows[0]["outcome_class"] == "graded"
    assert "private_verifier" not in rows[0]
    assert (rows[0].get("candidate_environment") == {"sandbox_id": "candidate-1"}) is (kind == "docker")
    assert len(calls) == (1 if kind == "docker" else 0)


def test_actual_native_extraction_transport_is_independent_of_expected_label():
    trial = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="ExtractionError"),
        verifier_result=None,
    )
    assert actual_extraction_transport({"status": "extraction_error", "reward": None}, trial)
    assert not actual_extraction_transport({"status": "graded", "reward": 0}, trial)
    trial.verifier_result = object()
    assert not actual_extraction_transport({"status": "extraction_error", "reward": None}, trial)


def test_unexpected_complete_native_extraction_fails_graded_expectation():
    plan = {**_summary_plan(), "verifier_surface": "native_deterministic", "identities": {
        "native_verifier": {"sha256": "a" * 64},
        "native_semantic_verifier_runtime_sha256": "b" * 64,
    }}
    rows = []
    for cell in plan["cells"]:
        unexpected = cell["case_id"] == "negative"
        rows.append({
            **cell,
            "status": "extraction_error" if unexpected else "graded",
            "reward": None if unexpected else 1.0,
            "outcome_class": "extraction_error" if unexpected else "graded",
            "exception": {"type": "ExtractionError"} if unexpected else None,
            "verifier_result_present": not unexpected,
            "grading_sha256": "d" * 64,
            "native_verifier_receipt": {
                "schema_version": "capability-native-verifier-receipt-v1",
                "adapter_sha256": "a" * 64,
                "semantic_verifier_sha256": "b" * 64,
            },
        })
    report = summarize(plan, rows, controls())
    assert report["state"] == "failed"
    assert not report["cases"][1]["expectation_met"]


def test_summary_keeps_failed_cells_in_denominator_and_fails_determinism():
    plan = _summary_plan()
    cells = []
    for case_id, reward in (("positive", 1.0), ("negative", 0.0)):
        for repeat in range(1, 11):
            cells.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "ordinal": len(cells) + 1,
                    "trial_name": f"regrade-{case_id}-run-{repeat:02d}",
                    "status": "graded",
                    "reward": reward,
                    "outcome_class": "graded",
                    "private_verifier": {"sandbox_id": f"sandbox-{len(cells) + 1}"},
                }
            )
    cells[-1].update(status=None, reward=None, outcome_class="runner_exception")
    result = summarize(plan, cells, controls())
    assert result["denominator"] == 20
    assert result["recorded_cells"] == 20
    assert result["state"] == "failed"
    assert result["cases"][1]["cell_count"] == 10


def test_summary_passes_equal_rewards_with_declared_ranges():
    plan = _summary_plan()
    cells = [
        {
            "case_id": case_id,
            "repeat": repeat,
            "ordinal": ordinal,
            "trial_name": f"regrade-{case_id}-run-{repeat:02d}",
            "status": "graded",
            "reward": reward,
            "outcome_class": "graded",
            "private_verifier": {"sandbox_id": f"sandbox-{ordinal}"},
        }
        for ordinal, (case_id, reward, repeat) in enumerate(
            ((case_id, reward, repeat)
             for case_id, reward in (("positive", 1.0), ("negative", 0.0))
             for repeat in range(1, 11)),
            1,
        )
    ]
    assert summarize(plan, cells, controls())["state"] == "passed"


def test_fixed_grading_input_requires_all_ten_identical_complete_fingerprints():
    plan = {**_summary_plan(), "require_fixed_grading_input": True}
    rows = [
        {**cell, "status": "graded", "reward": 1.0 if cell["case_id"] == "positive" else 0.0,
         "outcome_class": "graded", "private_verifier": {"sandbox_id": str(index)},
         "grading_input_fingerprint": {
             "schema_version": "capability-grading-input-fingerprint-v1",
             "submission_sha256": "d" * 64,
             "grading_input_sha256": ("a" if cell["case_id"] == "positive" else "b") * 64,
             "specification_sha256": "e" * 64,
             "protocol_sha256": "f" * 64,
             "transcript_sha256": "0" * 64,
             "payload_sha256": "1" * 64,
             "step_index": 0,
             "workspace_file_count": 1,
         }}
        for index, cell in enumerate(plan["cells"])
    ]
    assert summarize(plan, rows, controls())["state"] == "passed"
    rows[0]["grading_input_fingerprint"].pop("payload_sha256")
    assert summarize(plan, rows, controls())["state"] == "failed"
    rows[0]["grading_input_fingerprint"]["payload_sha256"] = "1" * 64
    rows[0]["grading_input_fingerprint"]["step_index"] = True
    assert summarize(plan, rows, controls())["state"] == "failed"
    rows[0]["grading_input_fingerprint"]["step_index"] = 0
    rows[0]["grading_input_fingerprint"]["grading_input_sha256"] = "c" * 64
    result = summarize(plan, rows, controls())
    assert result["state"] == "failed"
    assert not result["cases"][0]["fixed_grading_input"]
    rows[0].pop("grading_input_fingerprint")
    assert summarize(plan, rows, controls())["state"] == "failed"


def test_captured_private_grade_requires_exact_shared_capture_and_full_denominator():
    plan = {**_summary_plan(), "grading_strategy": "capture_once", "binding_kind": "docker",
            "require_fixed_grading_input": True}
    rows = []
    for index, cell in enumerate(plan["cells"]):
        rows.append({
            **cell, "status": "graded", "reward": 1.0 if cell["case_id"] == "positive" else 0.0,
            "outcome_class": "graded", "transport": "captured_private_grade", "trial_sha256": None,
            "verifier_result_present": None, "candidate_environment": None, "exception": None,
            "private_verifier": {"sandbox_id": str(index)}, "grading_sha256": "a" * 64,
            "capture_manifest_sha256": ("b" if cell["case_id"] == "positive" else "c") * 64,
            "grading_input_fingerprint": {
                "schema_version": "capability-grading-input-fingerprint-v1",
                "submission_sha256": "d" * 64,
                "grading_input_sha256": ("e" if cell["case_id"] == "positive" else "f") * 64,
                "specification_sha256": "0" * 64, "protocol_sha256": "1" * 64,
                "transcript_sha256": "2" * 64, "payload_sha256": "3" * 64,
                "step_index": 0, "workspace_file_count": 1,
            },
        })
    assert summarize(plan, rows, controls())["state"] == "passed"
    rows[0]["capture_manifest_sha256"] = "4" * 64
    assert summarize(plan, rows, controls())["state"] == "failed"
    rows[0]["capture_manifest_sha256"] = "b" * 64
    rows[0]["transport"] = "harbor_trial"
    assert summarize(plan, rows, controls())["state"] == "failed"
    rows[0]["transport"] = "captured_private_grade"
    assert summarize(plan, rows[:-1], controls())["state"] == "failed"


def test_captured_isolation_checks_original_and_repeated_private_grades(tmp_path, monkeypatch):
    trials = tmp_path / "runtime-trials"
    capture_root = trials / "capture-positive"
    (capture_root / "verifier").mkdir(parents=True)
    (capture_root / "verifier/taskcompendium-result.json").write_text(
        json.dumps({"detail": {"verifier_sandbox_id": "original"}})
    )
    grade_root = trials / "regrade-positive-run-01"
    grade_root.mkdir()
    (grade_root / "private-grade.json").write_text(
        json.dumps({"detail": {"verifier_sandbox_id": "original"}})
    )

    def candidate(_artifact, _root, seen):
        seen.add("candidate")
        return {"sandbox_id": "candidate"}

    def private(result, seen, candidate_ids, **_kwargs):
        identifier = result["detail"]["verifier_sandbox_id"]
        if identifier in seen or identifier in candidate_ids:
            raise RuntimeError("sandbox was reused")
        seen.add(identifier)
        return {"sandbox_id": identifier}

    monkeypatch.setattr(regrade_module, "provider_isolation_record", candidate)
    monkeypatch.setattr(regrade_module, "verifier_isolation_record", private)
    captures = [{"trial_name": "capture-positive", "outcome_class": "captured"}]
    rows = [{"trial_name": "regrade-positive-run-01", "outcome_class": "graded"}]
    attest_trial_isolation(rows, trials, kind="docker", native=False,
                           runtime_image="image", supervisor_python="python3", captures=captures)
    assert captures[0]["private_verifier"] == {"sandbox_id": "original"}
    assert rows[0]["outcome_class"] == "invalid_verifier_evidence"


def test_execute_docker_capture_runs_one_harbor_trial_and_ten_direct_grades(tmp_path, monkeypatch):
    """Controller wiring must never regenerate the authored candidate per repeat."""
    binding = SimpleNamespace(environment=SimpleNamespace(kind="docker"), tools=[])
    class ContainerRuntime:
        image = "runtime-image"
        supervisor_python = "python3"

    class TaskTroveVerifier:
        mode = SimpleNamespace(value="script")

    task_verifier = TaskTroveVerifier()
    specification = SimpleNamespace(steps=[SimpleNamespace(verifier=task_verifier)])
    tc = types.ModuleType("taskcompendium")
    tc.__path__ = []
    execution = types.ModuleType("taskcompendium.execution")
    execution.HarborExecutionConfig = lambda *a: None
    execution.HarborLaunchConfig = lambda *a: None
    execution.HarborTaskBinding = lambda *a: binding
    execution.HarnessToolBinding = lambda *a: None
    execution.NoEnvironment = type("NoEnvironment", (), {})
    runner = types.ModuleType("taskcompendium.harbor.runner")
    lowering = types.ModuleType("taskcompendium.lowering")
    lowering.resolve_harbor_execution = lambda *a, **k: {"verifier": {"import_path": "original"}}
    models = types.ModuleType("taskcompendium.models")
    models.ContainerRuntime = ContainerRuntime
    models.TaskTroveVerifier = TaskTroveVerifier
    models.tasktrove_verifier = lambda *_: task_verifier
    models.verifier_runtime = lambda *_: ContainerRuntime()
    serialization = types.ModuleType("taskcompendium.serialization")
    serialization.from_json = lambda *_: specification
    serialization.renderings_from_json = lambda *_: object()
    verify = types.ModuleType("tasktrove_verify")
    verify.__path__ = []
    verify_spec = types.ModuleType("tasktrove_verify.spec")
    verify_spec.Mode = SimpleNamespace(SCRIPT=task_verifier.mode)
    msgspec = types.ModuleType("msgspec")
    msgspec.json = SimpleNamespace(decode=lambda *_args, **_kwargs: binding)
    msgspec.to_builtins = lambda value: {"kind": value.kind}
    for name, module in {
        "taskcompendium": tc, "taskcompendium.execution": execution,
        "taskcompendium.harbor.runner": runner, "taskcompendium.lowering": lowering,
        "taskcompendium.models": models, "taskcompendium.serialization": serialization,
        "tasktrove_verify": verify, "tasktrove_verify.spec": verify_spec,
        "msgspec": msgspec,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    bundle = tmp_path / "bundle"
    package = tmp_path / "package"
    tools = tmp_path / "tools"
    for path, data in {
        bundle / "binding.json": "{}", bundle / "specification.json": "{}",
        bundle / "renderings.json": "{}", package / "manifest.json": "{}",
        bundle / "controls.json": json.dumps({"cases": [
            {"id": "positive", "class": "positive", "response": "answer",
             "expect": {"status": "graded", "reward_min": 1, "reward_max": 1}}
        ]}),
    }.items():
        _write(path, data)
    tools.mkdir()
    plan = {"grading_strategy": "capture_once", "parallelism": 8,
            "identities": {"fixture": "fixed"}, "cells": [
                {"ordinal": i, "case_id": "positive", "repeat": i,
                 "trial_name": f"regrade-positive-run-{i:02d}"}
                for i in range(1, 11)
            ]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    args = SimpleNamespace(package=package, bundle=bundle, controls=bundle / "controls.json",
                           output=tmp_path / "out", plan=tmp_path / "plan.json",
                           plan_sha256="frozen", daytona_tools=tools,
                           candidate_resources=None)
    monkeypatch.setenv("CAPABILITY_REMOTE_REGRADE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-only")
    monkeypatch.setattr(regrade_module, "validate_package_manifest", lambda *a, **k: None)
    monkeypatch.setattr(regrade_module, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(regrade_module, "validate_persisted_plan", lambda *a: (tmp_path / "plan.json").read_bytes())
    monkeypatch.setattr(regrade_module, "environment_config", lambda *a: ({}, "docker"))
    monkeypatch.setattr(regrade_module, "authored_replay_agent_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(regrade_module, "authored_replay_inventory", lambda *a: {"fixture": True})
    monkeypatch.setattr(regrade_module, "summarize", lambda *a: {"state": "passed"})
    monkeypatch.setattr(regrade_module, "attest_trial_isolation", lambda *a, **k: None)
    fingerprint = {"grading_input_sha256": "b" * 64}
    manifest_sha = "c" * 64
    call_names = []

    async def run_trial(_package, execution_value, trials, name):
        call_names.append(name)
        assert execution_value["verifier"]["import_path"].endswith("CapturingDaytonaSemanticVerifier")
        root = trials / name
        (root / "verifier/fixed-grading-capture").mkdir(parents=True)
        (root / "result.json").write_text("{}")
        (root / "verifier/taskcompendium-result.json").write_text(json.dumps({
            "status": "graded", "reward": 1.0,
            "detail": {"grading_input_fingerprint": fingerprint},
        }))
        return SimpleNamespace(exception_info=None, verifier_result=object())

    runner.run_trial = run_trial
    capture_helper = types.ModuleType("capability_pipeline.fixed_grading_capture")
    capture_helper.load_capture = lambda *_a, **_k: {
        "manifest_sha256": manifest_sha,
        "manifest": {"source_specification_sha256": regrade_module.sha256(bundle / "specification.json"),
                     "source_renderings_sha256": regrade_module.sha256(bundle / "renderings.json"),
                     "fingerprint": fingerprint},
    }
    monkeypatch.setitem(sys.modules, "capability_pipeline.fixed_grading_capture", capture_helper)
    grade_calls = []

    def direct_grade(_root, *, expected_manifest_sha256):
        assert expected_manifest_sha256 == manifest_sha
        grade_calls.append(expected_manifest_sha256)
        return SimpleNamespace(status=SimpleNamespace(value="graded"), reward=1.0,
                               detail={"grading_input_fingerprint": fingerprint,
                                       "fixed_grading_capture_manifest_sha256": manifest_sha})

    daytona = types.ModuleType("capability_pipeline.daytona_verifier")
    daytona.grade_captured_in_daytona = direct_grade
    monkeypatch.setitem(sys.modules, "capability_pipeline.daytona_verifier", daytona)
    report = asyncio.run(regrade_module.execute(args))
    assert call_names == ["capture-positive"]
    assert len(grade_calls) == 10
    assert len(report["captures"]) == 1
    assert len(report["cells"]) == 10
    assert all(row["transport"] == "captured_private_grade" for row in report["cells"])
    assert all(row["trial_sha256"] is None for row in report["cells"])
    assert all((args.output / "runtime-trials" / row["trial_name"] / "private-grade.json").is_file()
               for row in report["cells"])


def _summary_plan():
    cells = [
        {"ordinal": ordinal, "case_id": case_id, "repeat": repeat,
         "trial_name": f"regrade-{case_id}-run-{repeat:02d}"}
        for ordinal, (case_id, repeat) in enumerate(
            ((case_id, repeat) for case_id in ("positive", "negative")
             for repeat in range(1, 11)), 1
        )
    ]
    return {"cell_count": len(cells), "repeats": 10, "cells": cells}


@pytest.mark.parametrize("corruption", ["duplicate", "invalid_evidence", "trial_exception"])
def test_summary_rejects_duplicate_or_invalid_cells(corruption):
    plan = _summary_plan()
    cells = [
        {**cell, "status": "graded", "reward": 1.0 if cell["case_id"] == "positive" else 0.0,
         "outcome_class": "graded", "private_verifier": {"sandbox_id": str(index)}}
        for index, cell in enumerate(plan["cells"])
    ]
    if corruption == "duplicate":
        cells[-1] = dict(cells[-2])
    elif corruption == "invalid_evidence":
        cells[-1]["outcome_class"] = "invalid_verifier_evidence"
    else:
        cells[-1]["outcome_class"] = "trial_exception"
    assert summarize(plan, cells, controls())["state"] == "failed"
