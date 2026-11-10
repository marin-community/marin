# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import runpy
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlencode, urlsplit

import click
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from iris.client.client import IrisClient
from iris.cluster.client.job_info import set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from scipy.stats import t

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_tests.fixtures import SELECTION_IDENTITY, write_receipt
from experiments.post_training.snowball_tests.fixtures import answer as _answer
from experiments.post_training.snowball_tests.fixtures import write_json as _json

pytestmark = pytest.mark.usefixtures("source_provenance")
VLLM = "vllm/median_generation_throughput"
KL = "policy/mismatch/staleness0/kl_k3"


def controls(write, name, **kwargs):
    pin = gate.control_pins(name)[0]
    return [
        write(
            name,
            presets.control_name(pin.preset, pin.commit, i),
            speed=100 + i,
            score=0.49 + i * 0.01,
            legacy=True,
            **kwargs,
        )
        for i in range(pin.runs)
    ]


def rescore(preset, *args, exit_code=None):
    result = CliRunner().invoke(gate.main, ["--preset", preset, *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), (result.output, result.exception)
    payload = json.loads(result.output.splitlines()[-1])
    assert result.exit_code == (exit_code if exit_code is not None else gate.EXIT_CODES[payload["verdict"]]), (
        result.output,
        result.exception,
    )
    return payload


def comparisons(report, role=None):
    return {row["metric"]: row for row in report["comparisons"] if role is None or row["role"] == role}


def edit_metrics(root, edit):
    for path in (root / "exports/metrics").glob("step-*.json"):
        values = json.loads(path.read_text())
        edit(values)
        _json(path, values)


def test_calibrate_pools_deviations_within_pins():
    pins = [gate.ControlPin("speed-medium", commit, presets.PRESET_REVISION, 3) for commit in ("a", "b")]
    noise = gate.calibrate(
        {pins[0]: [{"x": value} for value in (1, 2, 3)], pins[1]: [{"x": value} for value in (10, 12, 14)]}
    )["x"]
    assert (noise.mean, noise.sd, noise.df) == pytest.approx((7, math.sqrt(2.5), 4))


def test_rescore_leave_one_out_uses_median_and_two_guard_tolerance(recorded_runs, monkeypatch, capsys):
    roots = controls(recorded_runs, "speed-medium")
    path = roots[0] / "exports/metrics/step-000002.json"
    _json(path, json.loads(path.read_text()) | {"timing/step_wall/group_admission": 1100})
    before = {
        path: hashlib.sha256(path.read_bytes()).digest() for root in roots for path in root.rglob("*") if path.is_file()
    }
    report = rescore("speed-medium", "--leave-out", "r0")
    timing = comparisons(report, "objective")["speed/update_seconds"]
    assert [timing[key] for key in ("run", "control", "sd", "df", "n_run", "n_control", "delta")] == [
        100,
        102,
        1,
        2,
        1,
        3,
        -2,
    ]
    assert timing["tolerance"] == pytest.approx(t.ppf(1 - 0.025 / 2, 2) * math.sqrt(1 + 1 / 3))
    assert timing["delta_percent"] == pytest.approx(-200 / 102)
    assert timing["noise_percent"] == pytest.approx(100 * timing["tolerance"] / 102)
    assert (report["verdict"], report["label"]) == ("pass", "no change")
    assert comparisons(report)["timing/step_wall/group_admission"]["change"] == "changed"
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "constraint", "reported"}
    assert sum(row["metric"] == "speed/update_seconds" for row in report["comparisons"]) == 1
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    invariants = {row["name"]: row for row in report["invariants"]}
    assert all(row["comparator"] is not None for row in invariants.values())
    assert invariants["grader_error_fraction"]["comparator"] == "<="
    assert invariants["planned_updates"]["comparator"] == "=="
    assert invariants["finite_losses_and_gradients"]["comparator"] == "=="
    fixed = invariants["fixed_reply_tokens"]
    assert fixed["value"] is True and fixed["limit"] is True and fixed["comparator"] == "=="
    historical = json.loads(artifact.read_text())
    for row in historical["invariants"]:
        row.pop("comparator")
    decoded = gate.GateReport.model_validate_json(json.dumps(historical))
    assert all(row.comparator is None for row in decoded.invariants)
    monkeypatch.setattr(sys, "argv", ["snowball_rl_gate", "--preset", "speed-medium", "--leave-out", "r0"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as result:
        runpy.run_module("experiments.post_training.snowball_rl_gate", run_name="__main__")
    assert result.value.code == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"report": str(artifact), **report}
    assert before == {
        path: hashlib.sha256(path.read_bytes()).digest() for root in roots for path in root.rglob("*") if path.is_file()
    }

    source = recorded_runs("speed-medium", "users/test/inconsistent-phases", speed=101)

    def extra_phase(row):
        if row["step"] > 2:
            row["timing/step_wall/policy_training"] = 100

    edit_metrics(source, extra_phase)
    report = rescore("speed-medium", "--candidate", "users/test/inconsistent-phases")
    assert report["verdict"] == "invalid" and "speed/update_seconds" in report["reason"]


@pytest.mark.parametrize(
    "speed,telemetry,control_values,kl,verdict,label,change",
    [
        (80, 1, (100, 101, 102, 103), 0.08, "pass", "improves", "changed"),
        (130, 1000, (100, 101, 102, 103), 0.13, "fail", "regression", "changed"),
        (80, 1, (float("nan"), 101, 102, 103), 0.08, "pass", "improves", "unavailable"),
        (80, 1, (100, 101, 102, 103), 0.2, "fail", "trade-off", "changed"),
    ],
)
def test_verdict_labels_use_objectives_and_guards(
    recorded_runs, speed, telemetry, control_values, kl, verdict, label, change
):
    pin = gate.control_pins("speed-medium")[0]
    for i, value in enumerate(control_values):
        recorded_runs(
            "speed-medium", presets.control_name(pin.preset, pin.commit, i), speed=100 + i, metrics={VLLM: value}
        )
    recorded_runs("speed-medium", "users/test/telemetry", speed=speed, metrics={VLLM: telemetry, KL: kl})
    report = rescore("speed-medium", "--candidate", "users/test/telemetry")
    assert (report["verdict"], report["label"]) == (verdict, label)
    row = comparisons(report, "reported")[VLLM]
    assert row["change"] == change and row["higher_is_better"] is None
    if change == "unavailable":
        assert row["control"] is None and row["control_min"] is None and row["control_max"] is None


def test_zero_spread_is_unavailable_and_required_guards_are_invalid(recorded_runs):
    pin = gate.control_pins("learning-medium")[0]
    for i in range(pin.runs):
        value = (i + 1) / 4
        recorded_runs(
            "learning-medium", presets.control_name(pin.preset, pin.commit, i), score=value, initial_score=value
        )
    recorded_runs("learning-medium", "users/test/zero-objective", score=0.625, initial_score=0)
    report = rescore("learning-medium", "--candidate", "users/test/zero-objective")
    row = comparisons(report, "objective")["test_panel/family_mean/gain_per_h100_hour"]
    assert (row["sd"], row["noise_percent"], row["change"], report["label"]) == (0, None, "unavailable", "no change")
    for i in range(pin.runs):
        recorded_runs("learning-medium", presets.control_name(pin.preset, pin.commit, i), score=0.5)
    recorded_runs("learning-medium", "users/test/zero-guard")
    report = rescore("learning-medium", "--candidate", "users/test/zero-guard")
    assert report["verdict"] == "invalid" and "variation is zero" in report["reason"]


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    key = "policy/raw_grad_norm"
    pin = gate.control_pins("speed-medium")[0]
    for i in range(pin.runs):
        recorded_runs(
            "speed-medium", presets.control_name(pin.preset, pin.commit, i), speed=100 + i, metrics={key: 100 + i}
        )
    recorded_runs("speed-medium", "users/test/borderline", speed=108, metrics={key: 1})
    base = rescore("speed-medium", "--candidate", "users/test/borderline")
    added = rescore("speed-medium", "--candidate", "users/test/borderline", "--require", key + "<=control")
    duplicate = rescore(
        "speed-medium",
        "--candidate",
        "users/test/borderline",
        "--require",
        key + "<=control",
        "--require",
        key + "<=control",
    )
    repeated_guard = rescore(
        "speed-medium", "--candidate", "users/test/borderline", "--require", "speed/update_seconds<=control"
    )
    assert repeated_guard["comparisons"] == base["comparisons"]
    for report in (base, added, duplicate):
        assert (report["verdict"], report["label"]) == ("fail", "regression"), report
        assert {key: row for key, row in comparisons(report).items() if row["role"] in ("objective", "constraint")} == {
            key: row for key, row in comparisons(base).items() if row["role"] in ("objective", "constraint")
        }
    requirement = comparisons(added, "requirement")[key]
    assert requirement["passes"]
    assert requirement["tolerance"] == pytest.approx(t.ppf(0.975, 3) * math.sqrt(5 / 3) * math.sqrt(1.25))
    assert len([row for row in duplicate["comparisons"] if row["role"] == "requirement"]) == 1
    candidates = []
    for i, value in enumerate((0, 0.01)):
        name = f"users/test/absolute-{i}"
        path = recorded_runs("speed-medium", name, metrics={key: value})
        candidates.append(ArtifactStep.adopt(name, presets.PRESET_REVISION, str(path), kind=SkyRLRun))
    step = gate.gate_step(
        presets.PRESETS["speed-medium"],
        tuple(candidates),
        inputs=gate.GateInputs(preset_name="speed-medium", requirements=(key + "<=0", "speed/update_seconds>=0")),
    )
    report = run(step)[0]
    assert report.verdict == "fail"
    assert [(row.replicate, row.holds) for row in report.invariants if row.name == "require/" + key] == [
        (0, True),
        (1, False),
    ]
    assert [row.comparator for row in report.invariants if row.name == "require/" + key] == ["<=", "<="]
    assert [row.comparator for row in report.invariants if row.name == "require/speed/update_seconds"] == [">=", ">="]
    with pytest.raises(click.BadParameter, match="does not compute"):
        snowball.build_run(presets.PRESETS["speed-medium"], requirements=("typo>=0",))


@pytest.mark.parametrize(
    "preset,case,expected",
    [
        ("speed-medium", "weighted", "pass"),
        ("speed-medium", "empty", "invalid"),
        ("speed-medium", "nonfinite", "invalid"),
        ("speed-long", "pooled", "fail"),
        ("speed-long", "warmup", "pass"),
    ],
)
def test_mismatch_uses_selected_tokens_over_the_measured_window(recorded_runs, preset, case, expected):
    controls(recorded_runs, preset)
    root = recorded_runs(preset, "users/test/window", speed=101)
    first, second = list(presets.PRESETS[preset].measured)[:2]
    prefix = KL.rsplit("/", 1)[0]

    def edit(row):
        if case == "weighted":
            row.update(
                {
                    prefix + "/selected_tokens": 1 if row["step"] == first else 9 if row["step"] == second else 0,
                    KL: 0.5 if row["step"] == first else 0,
                }
            )
        elif case == "empty":
            row[prefix + "/selected_tokens"] = 0
        elif case == "nonfinite":
            row[prefix + "/finite_fraction"] = 0
        elif case == "pooled":
            row.update({KL: 0.2, "policy/mismatch/pooled/kl_k3": 0})
        elif row["step"] < first:
            row.update({KL: 1e9, prefix + "/finite_fraction": 0})

    edit_metrics(root, edit)
    report = rescore(preset, "--candidate", "users/test/window")
    assert report["verdict"] == expected, report
    if case == "weighted":
        assert comparisons(report)[KL]["run"] == pytest.approx(0.05)
    if preset == "speed-long" and report["verdict"] != "invalid":
        assert {row["metric"] for row in report["comparisons"] if row["passes"] is not None} == {
            "speed/update_seconds",
            KL,
        }


@pytest.mark.parametrize("fault", ["updates", "loss", "policy-loss", "union", "missing-counter"])
def test_failed_training_invariants_do_not_supply_a_directional_label(recorded_runs, fault):
    controls(recorded_runs, "speed-medium")
    root = recorded_runs("speed-medium", "users/test/invariant", speed=101)
    if fault == "updates":
        (root / "exports/metrics/step-000001.json").unlink()
    else:

        def edit(row):
            if fault == "loss":
                row["loss/avg_raw_advantages_abs"] = float("nan")
            elif fault == "policy-loss":
                row["policy/policy_loss"] = float("nan")
            elif fault == "union":
                row.update(
                    {
                        "generate/outcome/graded": 99,
                        "generate/outcome/unavailable": 1,
                        "generate/outcome/rollout_failure": 1,
                    }
                )
            else:
                row.pop("generate/outcome/graded", None)

        edit_metrics(root, edit)
    report = rescore("speed-medium", "--candidate", "users/test/invariant")
    assert (report["verdict"], report["label"]) == ("fail", "no change"), report
    assert any(not row["holds"] for row in report["invariants"])
    if fault == "updates":
        assert next(row for row in report["invariants"] if row["name"] == "planned_updates")["holds"] is False


@pytest.mark.parametrize("preset", ["speed-short", "data-short"])
def test_current_generation_counters_compare_with_stored_controls(recorded_runs, selection, preset):
    controls(recorded_runs, preset)
    inputs = None
    if preset == "data-short":
        selected = presets.selection_artifact("candidate-training@2026.11.10")
        shutil.copytree(selection.path(), selected.path())
        inputs = gate.GateInputs(preset_name=preset, tasks="candidate-training@2026.11.10")
    root = recorded_runs(preset, "users/test/current-generation", speed=101, inputs=inputs)
    if inputs is not None:
        path = Path(selected.path()) / "train.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            task = json.loads(row["task_spec"])
            task["id"] = "candidate-" + task["id"]
            row["task_spec"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(rows), path)
        path = root / "exports/dumped_evals/eval_only/panel.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["task_id"] = "candidate-" + row["task_id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    values = json.loads((root / "exports/metrics/step-000000.json").read_text())
    assert "generate/outcome/graded" not in values and "attempt_id" not in values
    report = rescore(preset, "--candidate", "users/test/current-generation")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    if preset == "speed-short":
        coverage = [row for row in report["invariants"] if row["name"].endswith("/panel_scored_fraction")]
        assert coverage and all(row["comparator"] == ">=" for row in coverage)
        values["eval/test_panel/all/outcome/graded"] = 1480
        values["eval/test_panel/all/outcome/unavailable"] = 20
        _json(root / "exports/metrics/step-000000.json", values)
        alias = root.parent.parent / "current-bad-counter" / root.name
        shutil.copytree(root, alias)
        report = rescore(preset, "--candidate", "users/test/current-bad-counter")
        assert report["verdict"] == "fail"


@pytest.mark.parametrize(
    "path,value,verdict",
    [
        (("skyrl", "generator", "enable_prefix_caching"), False, "pass"),
        (("runtime", "task_env", "VLLM_PROBE"), "1", "pass"),
        (("skyrl", "generator", "eval_sampling_params", "temperature"), 0.5, "invalid"),
        (("skyrl", "generator", "chat_template"), "different", "invalid"),
        (("runtime", "task_env", "TASK_PROBE"), "1", "invalid"),
    ],
)
def test_evaluation_protocol_compares_requests_and_accepts_training_or_engine_changes(
    recorded_runs, path, value, verdict
):
    controls(recorded_runs, "learning-medium")
    root = recorded_runs("learning-medium", "users/test/protocol")
    receipt = json.loads((root / "terminal.json").read_text())
    target = receipt["config"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    write_receipt(root, receipt)
    report = rescore("learning-medium", "--candidate", "users/test/protocol")
    assert report["verdict"] == verdict, report
    if verdict == "invalid":
        assert "evaluation protocol" in report["reason"]


@pytest.mark.parametrize(
    "case,expected", [("pass", 0), ("fail", 1), ("invalid", 2), ("missing", 3), ("native-failed", 2), ("child", 0)]
)
def test_submission_waits_for_the_coordinator_and_returns_the_verdict(recorded_runs, monkeypatch, case, expected):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    controls(recorded_runs, "speed-medium")
    preset = presets.PRESETS["speed-medium"]
    built = snowball.build_run(preset)
    source = recorded_runs(
        preset.name, built.runs[0].name, speed=150 if case == "fail" else 101, fingerprint=built.runs[0].fingerprint()
    )
    if case == "invalid":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["provenance"] = None
        _json(source / "terminal.json", receipt)
    if case == "native-failed":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["state"] = "failed"
        _json(source / "attempts/recorded-attempt.json", receipt)
        (source / "terminal.json").unlink()
    job = JobName.from_wire("/atqamar/coordinator-id-returned-by-iris")

    def finish(*args, **kwargs):
        if case not in ("missing", "native-failed"):
            run(gate.completed_gate(preset, built.runs, built.gate))
        return SimpleNamespace(state=JobState.FAILED if case == "native-failed" else JobState.SUCCEEDED)

    jobs = {job: SimpleNamespace(wait=finish)}
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = job
    client.job.side_effect = lambda name: jobs[name]
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    if case == "child":
        monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/coordinator/0")
        client.submit.side_effect = AssertionError("coordinator execution submitted another job")
    try:
        result = CliRunner().invoke(
            snowball.main, ["--preset", "speed-medium", "--version", presets.PRESET_REVISION, "--run", "--wait"]
        )
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    if case == "child":
        report = gate.completed_gate(preset, built.runs, built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"


def test_control_refuses_actual_untracked_source_before_execution(selection, tmp_path, monkeypatch):
    root = tmp_path / "git-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "experiments").mkdir()
    (root / "experiments/untracked.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(presets, "REPO_ROOT", root)
    with pytest.raises(click.BadParameter, match="stage source"):
        snowball.build_run(presets.PRESETS["speed-medium"], control=True)


@pytest.mark.parametrize("arguments", [("--wait",), ("--replicates", "0"), ("--set", "trainer.resume_mode=latest")])
def test_refusals_use_exit_four(selection, arguments):
    result = CliRunner().invoke(
        snowball.main, ["--preset", "speed-medium", "--version", presets.PRESET_REVISION, *arguments]
    )
    assert result.exit_code == 4, (result.output, result.exception)


def test_panel_scores_join_permuted_answers_and_aggregate_tasks(tmp_path):
    answers = [
        _answer("bounded", 3, low=1, high=5)
        | {"score": 0.8, "output_response": "first\u2028second\u2029third\u0085fourth"},
        _answer("mixed", 0),
        _answer("mixed", 1),
        _answer("mixed", None, status="error", error="TimeoutError"),
        _answer("stem", 0),
        _answer("mixed", None, status="error"),
        {key: value for key, value in _answer("mixed", 1).items() if key != "verification"},
    ]
    families = {
        "mixed": ("structural", "format"),
        "bounded": ("structural", "format"),
        "stem": ("stem", "scientific_reasoning"),
    }
    manifest = [
        {"task_id": task, "family": family, "aaii_category": category} for task, (family, category) in families.items()
    ]
    (tmp_path / "selection").mkdir()
    pq.write_table(pa.Table.from_pylist(manifest), tmp_path / "selection" / presets.TASK_MANIFEST_FILENAME)
    _json(
        tmp_path / "resolved-launch.yaml",
        {
            "config": {
                "artifacts": {"export_root": str(tmp_path / "configured-export")},
                "inputs": {
                    "validation_data": [{"uri": str(tmp_path / "selection")}],
                },
            }
        },
    )
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "configured-export/dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "panel.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in answers) + "\n")
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    scores = scoring.panel_scores(presets.EVALUATIONS["test_panel_100"], str(tmp_path))
    final = [row for row in scores.to_pylist() if row["step"] == 25]
    assert [(row["item"], row["attempt"], row["scored"], row["score"]) for row in final] == [
        ("bounded", 0, True, 0.5),
        ("mixed", 0, True, 0.0),
        ("mixed", 1, True, 1.0),
        ("mixed", 2, False, 0.0),
        ("stem", 0, True, 0.0),
        ("mixed", 3, False, 0.0),
        ("mixed", 4, False, 0.0),
    ]
    statistics = scoring.evaluation_statistics(pa.Table.from_pylist(final, schema=scores.schema))
    expected = {
        "mean": 0.375,
        "family_mean": 0.25,
        "mixed_fraction": 1 / 3,
        "all_right_fraction": 0,
        "all_wrong_fraction": 2 / 3,
    }
    expected |= {"scored_fraction": 4 / 7, "aaii_category/format/mean": 0.5}
    assert {key: statistics["test_panel_100/" + key] for key in expected} == pytest.approx(expected)


@pytest.mark.timeout(180)
def test_gate_compares_natively_composed_optimizer_and_initializer_changes(recorded_runs, native_json, tmp_path):
    preset = presets.PRESETS["learning-medium"]
    control_roots = controls(recorded_runs, preset.name)
    initializer = ArtifactStep.adopt(
        "tests/custom-init", "2026.11.07", str(tmp_path / "custom-init"), kind=LevanterCheckpoint
    )
    builds = [
        snowball.build_run(preset, control=True),
        snowball.build_run(preset, settings=(presets.parse_setting("trainer.policy.optimizer_config.lr=2.0e-6"),)),
        snowball.build_run(preset, init=initializer),
    ]
    paths = []
    for index, built in enumerate(builds):
        step = built.runs[0]
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        path = tmp_path / f"protocol-{index}.yaml"
        path.write_text(config.launch_config_yaml)
        paths.append(str(path))
    composed = native_json(
        """
import json, sys
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
print(json.dumps([OmegaConf.to_container(load_launch_config(path), resolve=True) for path in sys.argv[1:]]))
""",
        paths,
    )
    for index, config in enumerate(composed):
        roots = control_roots if index == 0 else [recorded_runs(preset.name, f"users/test/native-protocol-{index}")]
        for source in roots:
            document = json.loads((source / "terminal.json").read_text())
            document["config"].update(skyrl=config["skyrl"], runtime=config["runtime"])
            write_receipt(source, document)
    for index in [1, 2]:
        report = rescore(preset.name, "--candidate", f"users/test/native-protocol-{index}")
        assert (report["verdict"], report["label"]) == ("pass", "no change"), report


def test_task_set_identity_uses_contents_and_ignores_row_order_and_storage_location(recorded_runs):
    preset_name = "learning-medium"
    controls(recorded_runs, preset_name)
    source = recorded_runs(preset_name, "users/test/task-content")
    document = json.loads((source / "terminal.json").read_text())
    descriptor = document["config"]["inputs"]["validation_data"][0]
    copied = source / "panel-copy"
    shutil.copytree(descriptor["uri"], copied)
    descriptor["uri"] = str(copied)
    path = copied / descriptor["relative_path"]
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), path)
    write_receipt(source, document)
    assert rescore(preset_name, "--candidate", "users/test/task-content")["verdict"] == "pass"
    changed_source = recorded_runs(preset_name, "users/test/task-content-changed")
    changed_document = json.loads((changed_source / "terminal.json").read_text())
    changed_document["config"]["inputs"]["validation_data"][0]["uri"] = str(copied)
    write_receipt(changed_source, changed_document)
    task = json.loads(rows[0]["task_spec"])
    task["context"]["events"][0]["content"] = "Return dog."
    rows[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = rescore(preset_name, "--candidate", "users/test/task-content-changed")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report

    roots = controls(recorded_runs, preset_name)
    roots.append(recorded_runs(preset_name, "users/test/wrong-cohort"))
    panel_root = Path(presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path())
    manifest = pq.read_table(panel_root / presets.TASK_MANIFEST_FILENAME).to_pylist()
    manifest.append(manifest[0] | {"task_id": "task-outside-input"})
    pq.write_table(pa.Table.from_pylist(manifest), panel_root / presets.TASK_MANIFEST_FILENAME)
    for root in roots:
        for path in (root / "exports/dumped_evals").glob("*/*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if row["task_id"] == "task-0":
                    row["task_id"] = "task-outside-input"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = rescore(preset_name, "--candidate", "users/test/wrong-cohort")
    assert report["verdict"] == "invalid" and "failed invariants" in report["reason"]


@pytest.mark.parametrize("preset_name", ["learning-medium", "data-medium"])
def test_python_run_saves_typed_inputs_for_rescoring(recorded_runs, selection, tmp_path, monkeypatch, preset_name):
    controls(recorded_runs, preset_name)
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(presets, "MODEL_URI", str(model))
    preset = presets.PRESETS[preset_name]
    tasks = selection if preset_name == "data-medium" else None
    settings = (presets.parse_setting("trainer.policy.optimizer_config.lr=2.0e-6"),) if tasks is None else ()
    built = snowball.build_run(preset, settings=settings, tasks=tasks, requirements=("policy/raw_grad_norm<=10",))
    task_name = SELECTION_IDENTITY if tasks else None
    source = recorded_runs(preset.name, built.runs[0].name, fingerprint=built.runs[0].fingerprint())
    staged = source.with_name("native-staging")
    source.rename(staged)
    (staged / gate.GATE_INPUTS_FILENAME).unlink()

    def native(config):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(snowball, "run_skyrl", native)
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preset", "train-long", "--require", "unrelated"])
    run(*built.runs)
    recorded = gate.read_gate_inputs(str(source))
    assert recorded.preset_name == preset_name and recorded.requirements == ("policy/raw_grad_norm<=10",)
    decoded = dict(item.split("=", 1) for item in recorded.settings)
    assert {key: json.loads(value) for key, value in decoded.items()} == (
        {} if tasks else {"trainer.policy.optimizer_config.lr": 2e-6}
    )
    assert recorded.tasks == task_name
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    report = rescore(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["requirements"] == list(recorded.requirements)
    assert report["run"]["settings"] == list(recorded.settings)
    assert report["run"]["tasks"] == recorded.tasks
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert before == {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "case",
    [
        "pass",
        "fail-no-login",
        "invalid-no-login",
        "parent-early",
        "mixed-native-failed",
        "control-missing-receipt",
        "partial-retry",
        "child",
        "data-short",
    ],
)
def test_wait_publishes_stored_attempts_and_keeps_the_exit_code(recorded_runs, selection, monkeypatch, case):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    preset_name = "data-short" if case == "data-short" else "learning-medium"
    controls(recorded_runs, preset_name)
    preset = presets.PRESETS[preset_name]
    replicas = 2 if case in {"partial-retry", "mixed-native-failed", "control-missing-receipt"} else 1
    control = case == "control-missing-receipt"
    built = snowball.build_run(
        preset, control=control, replicates=replicas, tasks=selection if case == "data-short" else None
    )
    sources = [
        recorded_runs(
            preset.name,
            training.name,
            score=0.1 if case == "fail-no-login" else 0.5,
            fingerprint=training.fingerprint(),
            inputs=gate.GateInputs(preset_name=preset_name, tasks=SELECTION_IDENTITY) if case == "data-short" else None,
            attempt_id=f"native-{index}",
        )
        for index, training in enumerate(built.runs)
    ]
    if case == "invalid-no-login":
        receipt = json.loads((sources[0] / "terminal.json").read_text())
        receipt["result"]["provenance"] = None
        _json(sources[0] / "terminal.json", receipt)
    pending = {}
    for root in sources:
        for filename in ("terminal.json", "resolved-launch.yaml"):
            path = root / filename
            pending[path] = path.read_bytes()
            path.unlink()
    client = MagicMock(spec=IrisClient)
    job = JobName.from_wire("/atqamar/dashboard-coordinator")
    client.submit.return_value.job_id = job
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    phase = {"retry": False}

    def finish(*args, **kwargs):
        for path, data in pending.items():
            path.write_bytes(data)
        if case in {"parent-early", "mixed-native-failed"}:
            receipt = json.loads((sources[0] / "terminal.json").read_text())
            if not phase["retry"]:
                receipt["result"]["state"] = "failed"
                _json(sources[0] / "attempts/native-0.json", receipt)
                (sources[0] / "terminal.json").unlink()
                return SimpleNamespace(state=JobState.FAILED)
            receipt["config"]["run"]["attempt_id"] = "native-retry"
            _json(sources[0] / "terminal.json", receipt)
            _json(sources[0] / "attempts/native-retry.json", receipt)
            _json(sources[0] / "resolved-launch.yaml", {"config": receipt["config"]})
            for path in (sources[0] / "exports/metrics").glob("*.json"):
                row = json.loads(path.read_text())
                row["attempt_id"] = "native-retry"
                _json(path, row)
        if control and not phase["retry"]:
            (sources[0] / "terminal.json").unlink()
            (sources[0] / "attempts/native-0.json").unlink(missing_ok=True)
            return SimpleNamespace(state=JobState.FAILED)
        completed = gate.completed_gate(preset, built.runs, built.gate)
        if completed is not None:
            run(completed)
        return SimpleNamespace(state=JobState.SUCCEEDED)

    client.job.return_value.wait.side_effect = finish
    published = []
    unavailable = {"value": case in {"partial-retry", "fail-no-login", "invalid-no-login"}}

    def publish(record, service_url=None):
        if unavailable["value"] and (case != "partial-retry" or record["attempt_id"] == "native-1"):
            raise RuntimeError("Credential unavailable; run iris login")
        published.append(record)
        return "https://progress.test/#" + urlencode({"run": record["run_id"], "attempt": record["attempt_id"]})

    monkeypatch.setattr("experiments.post_training.publish_rl_progress.publish_result", publish)
    if case == "child":
        finish()
        monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/cached-coordinator/0")
        monkeypatch.setattr("marin.rl.skyrl.run_skyrl", lambda config: SkyRLRun.raw_load(str(sources[0])))
        monkeypatch.setattr(
            "experiments.post_training.publish_rl_progress.publish_result",
            lambda *args: pytest.fail("Coordinator contacted Marina"),
        )
    arguments = ["--preset", preset.name, "--version", presets.PRESET_REVISION, "--run", "--wait", "--issue", "73"]
    if control:
        arguments.append("--control")
    if case == "data-short":
        arguments.extend(["--tasks", SELECTION_IDENTITY])
    if replicas == 2:
        arguments.extend(["--replicates", "2"])
    expected = (
        1
        if case == "fail-no-login"
        else 2 if case in {"invalid-no-login", "parent-early", "mixed-native-failed"} else 3 if control else 0
    )
    try:
        result = CliRunner().invoke(snowball.main, arguments)
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    if case == "child":
        assert published == []
        return
    snapshot = yaml.safe_load("preset:" + result.output.rpartition("\npreset:")[2].split("\n# ", 1)[0])
    if case == "data-short":
        assert "publication" not in snapshot and published == []
        return
    if case == "pass":
        assert f"r0 planned_updates: {preset.updates} == {preset.updates}; within bound" in snapshot["constraints"]
    paths = [str(root / f"exports/attempts/native-{index}") for index, root in enumerate(sources)]
    if unavailable["value"]:
        assert snapshot["publication"] == {
            "served_records": paths[:1] if case == "partial-retry" else [],
            "unserved_records": paths[1:] if case == "partial-retry" else paths,
            "errors": (
                ["Credential unavailable; run iris login"] * (replicas - 1 if case == "partial-retry" else replicas)
            ),
            "retry": "Repeat the original command with --wait.",
        }
    if case == "mixed-native-failed":
        assert [row["attempt_id"] for row in published] == ["native-0"]
        assert snapshot["publication"] == {
            "served_records": paths[:1],
            "unserved_records": paths[1:],
            "errors": ["Gate report missing for this attempt; repeat the original command with --wait"],
            "retry": "Repeat the original command with --wait.",
        }
    if case in {"parent-early", "mixed-native-failed"}:
        assert published[0]["status"] == "failed" and published[0]["verdict"] is None
    if expected == 0 and not unavailable["value"]:
        for row in snapshot["runs"]:
            assert parse_qs(urlsplit(row["rl_progress_url"]).fragment)["attempt"] == [row["attempt_id"]]
    unavailable["value"] = False
    phase["retry"] = True
    arguments[arguments.index("--issue") + 1] = "74"
    repeated = CliRunner().invoke(snowball.main, arguments)
    assert repeated.exit_code == (
        0 if case in {"parent-early", "mixed-native-failed", "control-missing-receipt"} else expected
    ), (repeated.output, repeated.exception)
    recovered = yaml.safe_load("preset:" + repeated.output.rpartition("\npreset:")[2].split("\n# ", 1)[0])
    assert recovered["publication"]["unserved_records"] == [] and recovered["publication"]["errors"] == []
    assert all(row["issue"] == 74 for row in recovered["runs"])
    if case in {"parent-early", "mixed-native-failed"}:
        retry = next(row for row in published if row["attempt_id"] == "native-retry")
        assert retry["status"] == "succeeded" and retry["verdict"] == "pass" and retry["issue"] == 74
    else:
        assert any(row["attempt_id"] == "native-0" and row["issue"] == 74 for row in published)
