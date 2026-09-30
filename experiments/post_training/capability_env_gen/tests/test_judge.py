import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline.judge import (
    _assess_calibration,
    _exact_repeat_agreement,
    _machine_gate_path,
    _wilson_interval,
    calibrate,
    calibrate_task,
    judge,
    validate_calibration_fixture,
    validate_rubric,
    validate_task_calibration_fixture,
)
from capability_pipeline.validation import InvalidArtifact


def rubric():
    return {
        "id": "r",
        "task_context": "Explain an observation using evidence.",
        "disqualifiers": ["fabricated source"],
        "criteria": [
            {
                "id": "evidence",
                "description": "Uses supplied evidence",
                "weight": 1,
                "anchors": {str(i): f"quality anchor {i}" for i in range(5)},
            }
        ],
    }


class FakeStore:
    def __init__(self, response):
        self.response = response

    def generate(self, stage, identity, system, prompt, validator, **kwargs):
        validator(self.response)
        return self.response


def test_judge_cannot_fabricate_positive_evidence():
    store = FakeStore(
        {
            "status": "graded",
            "criteria": [
                {
                    "id": "evidence",
                    "score": 4,
                    "evidence": ["not present"],
                    "reason": "good",
                }
            ],
            "disqualifiers_triggered": [],
        }
    )
    result = judge(store, rubric(), "unrelated answer", "case")
    assert result["status"] == "infra_error"
    assert result["reward"] is None


def test_disqualifier_overrides_positive_criterion():
    store = FakeStore(
        {
            "status": "graded",
            "criteria": [
                {
                    "id": "evidence",
                    "score": 4,
                    "evidence": ["source"],
                    "reason": "quoted",
                }
            ],
            "disqualifiers_triggered": ["fabricated source"],
        }
    )
    result = judge(store, rubric(), "source", "case")
    assert result["status"] == "graded" and result["reward"] == 0


def test_invalid_task_remains_ungraded():
    result = judge(
        FakeStore({"status": "invalid_task", "reason": "Missing source"}),
        rubric(),
        "",
        "case",
    )
    assert result["status"] == "invalid_task" and result["reward"] is None


def test_rubric_rejects_nonfinite_weights_and_duplicate_disqualifiers():
    invalid = rubric()
    invalid["criteria"][0]["weight"] = float("inf")
    with pytest.raises(InvalidArtifact, match="finite positive"):
        validate_rubric(invalid)
    invalid = rubric()
    invalid["disqualifiers"] *= 2
    with pytest.raises(InvalidArtifact, match="duplicate disqualifier"):
        validate_rubric(invalid)


def _calibration_fixture(positives=40, negatives=40):
    cases = [
        {
            "id": f"positive-{index}",
            "kind": "oracle",
            "source_family": "one-fixed-task-source",
            "variant_group": f"positive-design-{index}",
            "design_label": f"acceptable-construction-{index}",
            "candidate": f"grounded positive response {index}",
            "expected_reward_range": [0.8, 1.0],
        }
        for index in range(positives)
    ]
    negative_kinds = ["plausible_wrong", "empty", "prompt_injection"]
    cases.extend(
        {
            "id": f"negative-{index}",
            "kind": negative_kinds[index % len(negative_kinds)],
            "source_family": "one-fixed-task-source",
            "variant_group": f"negative-design-{index}",
            "design_label": f"error-construction-{index}",
            "candidate": f"invalid negative response {index}",
            "expected_reward_range": [0.0, 0.2],
        }
        for index in range(negatives)
    )
    return {"rubric": rubric(), "cases": cases}


def _task_calibration_fixture(specification_sha256="a" * 64):
    fixture = _calibration_fixture()
    for case in fixture["cases"]:
        case["expected_judge_path"] = "model"
    return {
        "schema_version": "taskcompendium-judge-calibration-v1",
        "specification_sha256": specification_sha256,
        "cases": fixture["cases"],
    }


def test_task_calibration_rejects_nonobject_case_as_invalid_artifact():
    fixture = _task_calibration_fixture()
    fixture["cases"].append(None)
    with pytest.raises(InvalidArtifact, match="calibration cases must be objects"):
        validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15)


@pytest.mark.parametrize("field", ["id", "kind"])
@pytest.mark.parametrize("task_fixture", [False, True])
def test_calibration_rejects_unhashable_identity_fields(field, task_fixture):
    fixture = _task_calibration_fixture() if task_fixture else _calibration_fixture()
    fixture["cases"][0][field] = []
    with pytest.raises(InvalidArtifact):
        if task_fixture:
            validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15)
        else:
            validate_calibration_fixture(fixture, 3, 0.15)


def _composite_calibration_fixture(specification_sha256="a" * 64):
    fixture = _task_calibration_fixture(specification_sha256)
    for case in fixture["cases"]:
        case["expected_machine_gate"] = "pass"
    fixture["cases"].append(
        {
            "id": "deterministic-gate-negative",
            "kind": "deterministic_gate",
            "source_family": "one-fixed-task-source",
            "variant_group": "deterministic-gate-design",
            "design_label": "malformed-register-quote",
            "candidate": "a distinct response rejected by the quote gate",
            "expected_reward_range": [0.0, 0.0],
            "expected_judge_path": "skipped_machine_gate",
            "expected_machine_gate": "fail",
        }
    )
    return fixture


def _composite_result(reward, *, gate="pass"):
    detail = {
        "aggregation": "taskcompendium-composite-verifier-v1",
        "failed_machine_gates": [] if gate == "pass" else ["format"],
        "machine_results": [
            {
                "id": "format",
                "status": "graded",
                "reward": 1.0 if gate == "pass" else 0.0,
                "detail": {},
            }
        ],
    }
    if gate == "pass":
        detail["judge"] = {"judgments": [{"criterion": 0, "score": reward}]}
    else:
        detail["judge_path"] = "skipped_machine_gate"
    return {
        "status": "graded",
        "reward": reward,
        "detail": detail,
    }


def test_calibration_requires_documented_sample_size_and_finite_spread():
    with pytest.raises(InvalidArtifact, match="at least 40 positive"):
        validate_calibration_fixture(_calibration_fixture(1, 40), 3, 0.15)
    with pytest.raises(InvalidArtifact, match="at least 40 negative"):
        validate_calibration_fixture(_calibration_fixture(40, 3), 3, 0.15)
    with pytest.raises(InvalidArtifact, match="max spread"):
        validate_calibration_fixture(_calibration_fixture(), 3, float("nan"))
    validated_rubric, cases = validate_calibration_fixture(
        _calibration_fixture(), 3, 0.15
    )
    assert validated_rubric["id"] == "r"
    assert len(cases) == 80


def test_calibration_clusters_paraphrases_and_rejects_duplicate_candidates():
    fixture = _calibration_fixture()
    fixture["cases"][1]["variant_group"] = fixture["cases"][0]["variant_group"]
    fixture["cases"][1]["design_label"] = fixture["cases"][0]["design_label"]
    with pytest.raises(InvalidArtifact, match="40 positive variant groups"):
        validate_calibration_fixture(fixture, 3, 0.15)

    fixture = _calibration_fixture()
    fixture["cases"][1]["candidate"] = fixture["cases"][0]["candidate"].upper()
    with pytest.raises(InvalidArtifact, match="duplicate calibration candidate"):
        validate_calibration_fixture(fixture, 3, 0.15)


def test_calibration_repeat_metric_requires_identical_case_repeats():
    groups = [[True, True, True] for _ in range(34)] + [
        [True, True, False] for _ in range(6)
    ]
    successes, total = _exact_repeat_agreement(groups, 3)
    assert (successes, total) == (34, 40)
    assert successes / total == 0.85


def test_wilson_interval_is_bounded_and_conservative():
    low, high = _wilson_interval(40, 40)
    assert 0.9 < low < 1.0
    assert high == 1.0
    assert _wilson_interval(0, 0) == [0.0, 1.0]


def test_task_calibration_fixture_is_hash_bound_and_private_evidence_is_typed():
    fixture = _task_calibration_fixture()
    fixture["cases"][0]["workspace_files"] = {"/app/report.txt": "evidence"}
    fixture["cases"][0]["transcript"] = [{"role": "assistant", "content": "work"}]
    cases = validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15)
    assert len(cases) == 80

    with pytest.raises(InvalidArtifact, match="bind the TaskSpec"):
        validate_task_calibration_fixture(fixture, "b" * 64, 3, 0.15)
    fixture["specification_sha256"] = "b" * 64
    fixture["cases"][0]["workspace_files"] = {"relative.txt": "evidence"}
    with pytest.raises(InvalidArtifact, match="absolute declared paths"):
        validate_task_calibration_fixture(fixture, "b" * 64, 3, 0.15)


def test_composite_fixture_requires_model_cases_that_pass_machine_gates():
    fixture = _composite_calibration_fixture()
    cases = validate_task_calibration_fixture(
        fixture, "a" * 64, 3, 0.15, composite=True
    )
    assert len(cases) == 81

    fixture["cases"].pop()
    with pytest.raises(InvalidArtifact, match="deterministic-gate failure"):
        validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15, composite=True)

    fixture = _composite_calibration_fixture()
    fixture["cases"][0]["workspace_files"] = {"/app/report.txt": "bypass"}
    with pytest.raises(InvalidArtifact, match="Harbor replay commands"):
        validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15, composite=True)

    fixture = _composite_calibration_fixture()
    fixture["cases"][0]["transcript"] = [
        {"role": "assistant", "content": "fabricated transcript"}
    ]
    with pytest.raises(InvalidArtifact, match="actual Harbor replay transcript"):
        validate_task_calibration_fixture(fixture, "a" * 64, 3, 0.15, composite=True)


def test_composite_calibration_separates_native_model_and_machine_gate_paths():
    fixture = _composite_calibration_fixture()
    results = {}
    for case in fixture["cases"]:
        reward = 1.0 if case["kind"] == "oracle" else 0.0
        gate = case["expected_machine_gate"]
        for repeat in range(3):
            results[f"{case['id']}:{repeat}"] = _composite_result(reward, gate=gate)
    issues, metrics = _assess_calibration(
        fixture["cases"],
        results,
        {},
        3,
        0.15,
        require_model_judgment=True,
        require_composite_gate_pass=True,
    )
    assert not issues
    assert metrics["class_variant_group_counts"] == {
        "positive": 40,
        "negative": 40,
    }
    plausible = metrics["per_stratum"]["plausible_wrong"]
    assert plausible["judge_paths"]["model"] > 0
    deterministic = metrics["per_stratum"]["deterministic_gate"]
    assert deterministic["judge_paths"]["skipped_machine_gate"] == 3
    assert deterministic["machine_gate_paths"]["failed"] == 3
    assert _machine_gate_path(_composite_result(0.0, gate="fail")) == "failed"


def test_task_calibration_uses_pinned_adapter_and_conservative_gate(
    tmp_path, monkeypatch
):
    import hashlib

    from capability_pipeline import synthesis

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    specification = b'{"schema_version":"0.9"}\n'
    (bundle / "specification.json").write_bytes(specification)
    (bundle / "renderings.json").write_text("[]")
    specification_sha256 = hashlib.sha256(specification).hexdigest()
    fixture = _task_calibration_fixture(specification_sha256)
    (bundle / "judge-calibration.json").write_text(json.dumps(fixture))
    commands = []

    class FakeToolchain:
        def _command(self):
            return ["uv", "run", "python", "old-driver.py"]

    monkeypatch.setattr(
        synthesis.OfficialToolchain,
        "resolve",
        lambda *args: FakeToolchain(),
    )

    def fake_run(command, timeout):
        commands.append(command)
        output = command[command.index("--output") + 1]
        results = {}
        for case in fixture["cases"]:
            reward = 1.0 if case["kind"] == "oracle" else 0.0
            for repeat in range(3):
                results[f"{case['id']}:{repeat}"] = {
                    "status": "graded",
                    "reward": reward,
                    "detail": {"judgments": [{"score": reward, "model": "fixture"}]},
                }
        with open(output, "w") as stream:
            json.dump({"results": results, "failures": {}}, stream)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synthesis, "_run", fake_run)
    args = SimpleNamespace(
        bundle=bundle,
        out=tmp_path / "out",
        taskcompendium_source=None,
        api_key_env="GLM_API_TOKEN",
        concurrency=64,
        repeats=3,
        max_spread=0.15,
        timeout=60,
    )
    assert calibrate_task(args) == 0
    report = json.loads((tmp_path / "out/judge-calibration.json").read_text())
    assert report["state"] == "passed"
    assert report["mode"] == "taskcompendium-native-judge"
    assert commands[0][3].endswith("task_judge_calibrator.py")


def test_composite_calibration_uses_lowered_harbor_and_composed_reward(
    tmp_path, monkeypatch
):
    import hashlib

    from capability_pipeline import synthesis

    bundle = tmp_path / "bundle"
    package = tmp_path / "harbor"
    bundle.mkdir()
    package.mkdir()
    specification = b'{"schema_version":"0.9"}\n'
    (bundle / "specification.json").write_bytes(specification)
    (bundle / "renderings.json").write_text("[]")
    specification_sha256 = hashlib.sha256(specification).hexdigest()
    fixture = _composite_calibration_fixture(specification_sha256)
    (bundle / "judge-calibration.json").write_text(json.dumps(fixture))
    composite = (json.dumps({
        "schema_version": "taskcompendium-composite-verifier-v1",
        "steps": [{
            "step_index": 0,
            "machine_checks": [{"timeout": 60}],
            "judge": {"criterion_weights": [1.0] * 12,
                      "consensus": {"initial_samples": 2}},
        }],
    }) + "\n").encode()
    (bundle / "composite-verifier.json").write_bytes(composite)
    (package / "composite-verifier.json").write_bytes(composite)
    (package / "manifest.json").write_text('{"step_names":["step"]}\n')
    commands = []
    timeouts = []

    class FakeToolchain:
        def runtime_command(self):
            return ["uv", "run", "--extra", "harbor", "python", "runtime.py"]

    monkeypatch.setattr(
        synthesis.OfficialToolchain,
        "resolve",
        lambda *args: FakeToolchain(),
    )

    def fake_run(command, timeout):
        commands.append(command)
        timeouts.append(timeout)
        output = command[command.index("--output") + 1]
        results = {}
        for case in fixture["cases"]:
            reward = 1.0 if case["kind"] == "oracle" else 0.0
            for repeat in range(3):
                result = _composite_result(reward, gate=case["expected_machine_gate"])
                result["detail"].update(
                    {
                        "composite_adapter_sha256": hashlib.sha256(
                            Path(synthesis.__file__)
                            .with_name("composite_verifier.py")
                            .read_bytes()
                        ).hexdigest(),
                        "composite_policy_sha256": hashlib.sha256(
                            Path(synthesis.__file__)
                            .with_name("composite_policy.py")
                            .read_bytes()
                        ).hexdigest(),
                        "composite_config_sha256": hashlib.sha256(
                            composite
                        ).hexdigest(),
                    }
                )
                results[f"{case['id']}:{repeat}"] = result
        with open(output, "w") as stream:
            json.dump(
                {
                    "mode": "taskcompendium-composite-harbor",
                    "results": results,
                    "failures": {},
                },
                stream,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synthesis, "_run", fake_run)
    args = SimpleNamespace(
        bundle=bundle,
        package=package,
        out=tmp_path / "out",
        taskcompendium_source=None,
        api_key_env="GLM_API_TOKEN",
        concurrency=64,
        repeats=3,
        max_spread=0.15,
        timeout=60,
        shellsim_bridge=None,
    )
    assert calibrate_task(args) == 0
    assert commands[0][commands[0].index("--concurrency") + 1] == "16"
    assert timeouts[0] > args.timeout
    report = json.loads((tmp_path / "out/judge-calibration.json").read_text())
    assert report["state"] == "passed"
    assert report["mode"] == "taskcompendium-composite-harbor"
    assert report["composite_config_sha256"] == hashlib.sha256(composite).hexdigest()
    assert commands[0][commands[0].index("--package") + 1] == str(package.resolve())
    assert commands[0][commands[0].index("python") + 1].endswith(
        "task_judge_calibrator.py"
    )


def test_calibration_reuses_verified_runtime_overlay_without_reresolving_it(
    tmp_path, monkeypatch
):
    """A patched runner is a runtime overlay, never a candidate source checkout."""
    import hashlib

    from capability_pipeline import synthesis

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    specification = b'{"schema_version":"0.9"}\n'
    (bundle / "specification.json").write_bytes(specification)
    (bundle / "renderings.json").write_text("[]")
    fixture = _task_calibration_fixture(hashlib.sha256(specification).hexdigest())
    (bundle / "judge-calibration.json").write_text(json.dumps(fixture))
    source, overlay = tmp_path / "source", tmp_path / "runtime-overlay"
    source.mkdir()
    overlay.mkdir()
    toolchain = synthesis.OfficialToolchain(
        overlay, "uv", source_package_root=source
    )
    validated = []
    monkeypatch.setattr(
        synthesis.OfficialToolchain,
        "validate_runtime_overlay",
        lambda value: validated.append(value),
    )
    monkeypatch.setattr(
        synthesis.OfficialToolchain,
        "resolve",
        lambda *_args: pytest.fail("calibration must not resolve the patched overlay"),
    )

    def fake_run(command, timeout):
        output = Path(command[command.index("--output") + 1])
        results = {}
        for case in fixture["cases"]:
            reward = 1.0 if case["kind"] == "oracle" else 0.0
            for repeat in range(3):
                results[f"{case['id']}:{repeat}"] = {
                    "status": "graded",
                    "reward": reward,
                    "detail": {"judgments": [{"score": reward}]},
                }
        output.write_text(json.dumps({"results": results, "failures": {}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synthesis, "_run", fake_run)
    args = SimpleNamespace(
        bundle=bundle,
        out=tmp_path / "out",
        taskcompendium_source=source,
        toolchain=toolchain,
        api_key_env="GLM_API_TOKEN",
        concurrency=64,
        repeats=3,
        max_spread=0.15,
        timeout=60,
    )
    assert calibrate_task(args) == 0
    assert validated == [toolchain]


def test_calibration_confidence_uses_unique_cases_not_repeats():
    fixture = _calibration_fixture()
    fixture["cases"].append(
        {
            **fixture["cases"][0],
            "id": "positive-0-paraphrase",
            "candidate": "an alternate wording of grounded positive response zero",
        }
    )
    results = {}
    for case in fixture["cases"]:
        reward = 1.0 if case["kind"] == "oracle" else 0.0
        for repeat in range(3):
            results[f"{case['id']}:{repeat}"] = {
                "status": "graded",
                "reward": reward,
            }
    issues, metrics = _assess_calibration(fixture["cases"], results, {}, 3, 0.15)
    assert not issues
    assert metrics["class_variant_group_counts"] == {
        "positive": 40,
        "negative": 40,
    }
    assert metrics["false_accept_rate_interval"][1] == pytest.approx(
        1 - _wilson_interval(40, 40)[0]
    )
    assert metrics["repeat_level_descriptive"]["positive_repeats"] == 123
    assert metrics["repeat_level_descriptive"]["used_for_confidence_intervals"] is False

    results["positive-0:2"]["reward"] = 0.0
    issues, metrics = _assess_calibration(fixture["cases"], results, {}, 3, 0.15)
    assert issues
    assert metrics["balanced_accuracy"] == pytest.approx((39 / 40 + 1) / 2)


def test_native_calibration_excludes_exact_and_constraint_gates():
    fixture = _task_calibration_fixture()
    results = {}
    for case in fixture["cases"]:
        reward = 1.0 if case["kind"] == "oracle" else 0.0
        for repeat in range(3):
            results[f"{case['id']}:{repeat}"] = {
                "status": "graded",
                "reward": reward,
                "detail": {"gate": "exact" if reward else "constraints"},
            }
    issues, metrics = _assess_calibration(
        fixture["cases"],
        results,
        {},
        3,
        0.15,
        require_model_judgment=True,
    )
    assert "fewer than 40 positive variant groups reached the model judge" in issues
    assert "fewer than 40 negative variant groups reached the model judge" in issues
    assert metrics["class_variant_group_counts"] == {
        "positive": 0,
        "negative": 0,
    }
    assert metrics["repeat_agreement_variant_group_count"] == 0
    assert metrics["repeat_agreement_interval"] == [0.0, 1.0]
    assert "repeat agreement lower bound 0.000 below 0.90" in issues


def test_calibration_clears_stale_pass_before_fixture_validation(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "judge-calibration.json").write_text('{"state":"passed"}')
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(_calibration_fixture(1, 1)))
    args = SimpleNamespace(
        fixtures=fixture,
        out=output,
        repeats=3,
        max_spread=0.15,
        tier="interactive",
        concurrency=2,
    )

    with pytest.raises(InvalidArtifact):
        calibrate(args)

    assert (
        json.loads((output / "judge-calibration.json").read_text())["state"]
        == "running"
    )
