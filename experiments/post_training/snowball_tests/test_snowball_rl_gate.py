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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from iris.client.client import IrisClient
from iris.cluster.client.job_info import set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl import SkyRLRun
from marin.rl.skyrl_records import read_skyrl_metrics
from rigging.provenance import Provenance
from scipy.stats import t

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_tests.fixtures import answer as _answer
from experiments.post_training.snowball_tests.fixtures import write_json as _json
from experiments.post_training.snowball_tests.fixtures import write_metrics, write_receipt

pytestmark = pytest.mark.usefixtures("source_provenance")
VLLM = "vllm/median_generation_throughput"
KL = "policy/mismatch/staleness0/kl_k3"


def controls(write, name, **kwargs):
    pin = gate.control_pins(presets.PRESETS[name])[0]
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
    metrics = read_skyrl_metrics(str(root))
    for values in metrics.values():
        edit(values)
    write_metrics(root, metrics)


def test_calibrate_pools_deviations_within_pins():
    pins = [
        gate.ControlPin("speed-medium", commit, presets.PRESETS["speed-medium"].revision, 3) for commit in ("a", "b")
    ]
    noise = gate.calibrate(
        {pins[0]: [{"x": value} for value in (1, 2, 3)], pins[1]: [{"x": value} for value in (10, 12, 14)]}
    )["x"]
    assert (noise.mean, noise.sd, noise.df) == pytest.approx((7, math.sqrt(2.5), 4))


def test_rescore_leave_one_out_uses_median_and_two_guard_tolerance(recorded_runs, monkeypatch, capsys):
    roots = controls(recorded_runs, "speed-medium")
    values = read_skyrl_metrics(str(roots[0]))
    values[2]["timing/step_wall/group_admission"] = 1100
    write_metrics(roots[0], values)
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
    pin = gate.control_pins(presets.PRESETS["speed-medium"])[0]
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
    pin = gate.control_pins(presets.PRESETS["learning-medium"])[0]
    for i in range(pin.runs):
        value = (i + 1) / 4
        recorded_runs(
            "learning-medium", presets.control_name(pin.preset, pin.commit, i), score=value, initial_score=value
        )
    recorded_runs("learning-medium", "users/test/zero-objective", score=0.625, initial_score=0)
    report = rescore("learning-medium", "--candidate", "users/test/zero-objective")
    row = comparisons(report, "objective")["eval_lite/family_mean/gain_per_h100_hour"]
    assert (row["sd"], row["noise_percent"], row["change"], report["label"]) == (0, None, "unavailable", "no change")
    for i in range(pin.runs):
        recorded_runs("learning-medium", presets.control_name(pin.preset, pin.commit, i), score=0.5)
    recorded_runs("learning-medium", "users/test/zero-guard")
    report = rescore("learning-medium", "--candidate", "users/test/zero-guard")
    assert report["verdict"] == "invalid" and "variation is zero" in report["reason"]


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    key = "policy/raw_grad_norm"
    pin = gate.control_pins(presets.PRESETS["speed-medium"])[0]
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
        candidates.append(ArtifactStep.adopt(name, presets.PRESETS["speed-medium"].revision, str(path), kind=SkyRLRun))
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
    root = recorded_runs(
        "speed-medium", "users/test/invariant", speed=101, missing_steps=(1,) if fault == "updates" else ()
    )
    if fault != "updates":

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
        launch = json.loads((root / "terminal.json").read_text())["config"]
        descriptor = launch["inputs"]["validation_data"][0]
        path = Path(descriptor["uri"]) / descriptor["relative_path"]
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
    values = read_skyrl_metrics(str(root))[0]
    assert "generate/outcome/graded" not in values and values["attempt_id"] == "recorded-attempt"
    report = rescore(preset, "--candidate", "users/test/current-generation")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    if preset == "speed-short":
        coverage = [row for row in report["invariants"] if row["name"].endswith("/panel_scored_fraction")]
        assert coverage and all(row["comparator"] == ">=" for row in coverage)
        values["eval/eval_lite/all/outcome/graded"] = 1480
        values["eval/eval_lite/all/outcome/unavailable"] = 20
        write_metrics(root, {0: values})
        alias = root.parent.parent / "current-bad-counter" / root.name
        shutil.copytree(root, alias)
        report = rescore(preset, "--candidate", "users/test/current-bad-counter")
        assert report["verdict"] == "fail"


@pytest.mark.parametrize(
    "path,value,settings,verdict",
    [
        (("skyrl", "generator", "enable_prefix_caching"), False, ("generator.enable_prefix_caching=false",), "pass"),
        (("skyrl", "generator", "enable_prefix_caching"), False, (), "invalid"),
        (("runtime", "task_env", "VLLM_PROBE"), "1", ('+extra_env.VLLM_PROBE="1"',), "pass"),
        (("skyrl", "generator", "eval_sampling_params", "temperature"), 0.5, (), "invalid"),
        (("skyrl", "generator", "chat_template"), "different", (), "invalid"),
        (("runtime", "task_env", "TASK_PROBE"), "1", (), "invalid"),
        (("recorded-named",), None, (), "pass"),
    ],
)
def test_evaluation_protocol_compares_requests_and_recorded_training_or_engine_changes(
    recorded_runs, path, value, settings, verdict
):
    if path == ("recorded-named",):
        root = recorded_runs("speed-short", "users/test/recorded-named")
        launch = json.loads((root / "terminal.json").read_text())["config"]
        launch["skyrl"]["generator"]["eval_sampling_params"]["presence_penalty"] = 0.8
        launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 7
        protocol = scoring.evaluation_protocol(launch, "eval_lite")
        assert protocol["sampling_params"] == {
            "ignore_eos": False,
            "max_generate_length": 8192,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": -1,
            "logprobs": None,
            "stop": None,
        }
        assert protocol["n_samples_per_prompt"] == 1
        assert protocol["context_budget"]["max_new_tokens_per_turn"] == 8192
        return
    controls(recorded_runs, "learning-medium")
    root = recorded_runs(
        "learning-medium",
        "users/test/protocol",
        inputs=gate.GateInputs(preset_name="learning-medium", settings=settings),
    )
    receipt = json.loads((root / "terminal.json").read_text())
    target = receipt["config"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    write_receipt(root, receipt)
    report = rescore("learning-medium", "--candidate", "users/test/protocol")
    assert report["verdict"] == verdict, report
    if verdict == "invalid":
        assert "evaluation protocol" in report["reason"] or "config differs" in report["reason"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("pass", 0),
        ("fail", 1),
        ("invalid", 2),
        ("missing", 3),
        ("native-failed", 2),
        ("child", 0),
        ("changed-source", 0),
        ("stale-control", 4),
        ("stale-control-no-wait", 4),
    ],
)
def test_submission_waits_for_the_coordinator_and_returns_the_verdict(recorded_runs, monkeypatch, case, expected):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    roots = controls(recorded_runs, "speed-medium")
    preset = presets.PRESETS["speed-medium"]
    built = snowball.build_run(preset)
    source = recorded_runs(
        preset.name, built.runs[0].name, speed=150 if case == "fail" else 101, fingerprint=built.runs[0].fingerprint()
    )
    if case == "invalid":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["provenance"] = None
        write_receipt(source, receipt)
    if case == "native-failed":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["state"] = "failed"
        write_receipt(source, receipt)
        (source / "terminal.json").unlink()
    if case.startswith("stale-control"):
        receipt = json.loads((roots[-1] / "terminal.json").read_text())
        receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = 5e-6
        write_receipt(roots[-1], receipt)
    job = JobName.from_wire("/atqamar/coordinator-id-returned-by-iris")

    def finish(*args, **kwargs):
        if case not in ("missing", "native-failed"):
            run(gate.completed_gate(built.gate))
            if case == "changed-source":
                monkeypatch.setattr(
                    Provenance,
                    "capture",
                    lambda *args, **kwargs: Provenance("tree-edited-after-submit", "commit-one", True, "branch", "test"),
                )
        return SimpleNamespace(state=JobState.FAILED if case == "native-failed" else JobState.SUCCEEDED)

    jobs = {job: SimpleNamespace(wait=finish)}
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = job
    if case.startswith("stale-control"):
        client.submit.side_effect = AssertionError("stale controls reached the coordinator RPC")
    client.job.side_effect = lambda name: jobs[name]
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    if case == "child":
        monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/coordinator/0")
        client.submit.side_effect = AssertionError("coordinator execution submitted another job")
    try:
        result = CliRunner().invoke(
            snowball.main,
            [
                "--preset",
                "speed-medium",
                "--version",
                preset.revision,
                "--run",
                *(() if case == "stale-control-no-wait" else ("--wait",)),
            ],
        )
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    if case.startswith("stale-control"):
        assert "skyrl.trainer.policy.optimizer_config.lr" in result.output
    if case == "child":
        report = gate.completed_gate(built.gate)
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
        snowball.main, ["--preset", "speed-medium", "--version", presets.PRESETS["speed-medium"].revision, *arguments]
    )
    assert result.exit_code == 4, (result.output, result.exception)


def test_eval_lite_scores_join_permuted_answers_and_aggregate_tasks(tmp_path):
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
    scores = scoring.panel_scores(presets.EVALUATIONS["eval_lite_100"], str(tmp_path))
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
    assert {key: statistics["eval_lite_100/" + key] for key in expected} == pytest.approx(expected)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "case",
    [
        "optimizer",
        "initializer",
        "stale-default",
        "explicit-equal",
        "control-setting",
        "control-default-setting",
        "control-initializer",
        "control-tasks",
        "control-revision",
        "all-revisions",
        "untouched-sibling",
        "unbound-model",
        "unbound-data",
        "batch",
        "staged-profile",
        "wrong-profile-control",
        "home-alias",
        "alias-sibling",
        "unbound-gsm8k",
        "allocation",
        "source-identity",
        "relative-file",
        "authored-run-name",
        "cached-default",
        "cached-execution",
        "cached-source",
    ],
)
def test_gate_compares_natively_composed_optimizer_and_initializer_changes(recorded_runs, tmp_path, monkeypatch, case):
    short = {"staged-profile", "wrong-profile-control", "home-alias", "alias-sibling", "unbound-gsm8k"}
    preset = presets.PRESETS[
        "data-medium" if case == "control-tasks" else "speed-short" if case in short else "learning-medium"
    ]
    setting = "trainer.policy.optimizer_config.lr=2.0e-6"
    inputs = gate.GateInputs(preset_name=preset.name, settings=(setting,))
    if case in short:
        inputs = gate.GateInputs(preset_name=preset.name)
    if case == "wrong-profile-control":
        inputs = gate.GateInputs(preset_name=preset.name, settings=("generator.enable_prefix_caching=false",))
    if case == "batch":
        inputs = gate.GateInputs(
            preset_name=preset.name,
            settings=(
                "trainer.train_batch_size=64",
                "trainer.policy_mini_batch_size=64",
                "generator.n_samples_per_prompt=8",
                "trainer.micro_train_batch_size_per_gpu=2",
            ),
        )
    if case in ("cached-default", "cached-execution", "cached-source"):
        inputs = gate.GateInputs(preset_name=preset.name)
    if case == "control-default-setting":
        inputs = gate.GateInputs(preset_name=preset.name, settings=("trainer.policy.optimizer_config.lr=1.0e-6",))
    if case == "control-tasks":
        inputs = gate.GateInputs(preset_name=preset.name, tasks="fixture-selection@2026.11.10")
    if case in ("initializer", "unbound-model", "control-initializer"):
        name, version = "tests/custom-init", "2026.11.07"
        source = tmp_path / name / version
        initializer = SkyRLRun(
            path=str(source),
            hf_model_uri="s3://fixture-initializer/policy",
            global_step=25,
            tokenizer_uri=preset.model.tokenizer_uri,
            tokenizer_revision=preset.model.tokenizer_revision,
            checkpoint_root=str(source / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(source / "terminal.json"),
            iris_job_id="/tests/initializer",
        )
        _json(
            source / RECORD_FILENAME,
            ArtifactRecord(
                name=name,
                version=version,
                fingerprint="initializer",
                result_type=result_type_name(SkyRLRun),
                result=initializer.result_payload(),
            ).model_dump(mode="json"),
        )
        inputs = gate.GateInputs(preset_name=preset.name, init=f"{name}@{version}")
    if case in ("stale-default", "explicit-equal", "untouched-sibling"):
        control_roots = controls(recorded_runs, preset.name)
        promoted = replace(preset, settings=(*preset.settings, setting))
        monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: promoted})
        if case == "stale-default":
            inputs = gate.GateInputs(preset_name=preset.name)
    control_inputs = inputs if case in ("control-setting", "control-initializer", "control-tasks") else None
    if case not in ("stale-default", "explicit-equal", "untouched-sibling"):
        control_roots = controls(recorded_runs, preset.name, **({"inputs": control_inputs} if control_inputs else {}))
    name = f"users/test/native-{case}"
    candidate = recorded_runs(preset.name, name, inputs=inputs)
    fault_key = None
    if case in ("control-revision", "all-revisions"):
        for root in ([control_roots[0]] if case == "control-revision" else [*control_roots, candidate]):
            path = root / RECORD_FILENAME
            record = json.loads(path.read_text())
            record["version"] = "2026.11.10.1"
            path.write_text(json.dumps(record))
        fault_key = "preset revisions differ" if case == "control-revision" else "preset revision differs"
    elif case == "control-default-setting":
        document = json.loads((control_roots[0] / "terminal.json").read_text())
        document["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = 5e-6
        write_receipt(control_roots[0], document)
        fault_key = "skyrl.trainer.policy.optimizer_config.lr"
    elif case in ("control-setting", "control-initializer", "control-tasks"):
        fault_key = {
            "control-setting": "skyrl.trainer.policy.optimizer_config.lr",
            "control-initializer": "inputs.model.uri",
            "control-tasks": "inputs.train_data",
        }[case]
    elif case == "untouched-sibling":
        document = json.loads((control_roots[0] / "terminal.json").read_text())
        document["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["weight_decay"] = 0.1
        write_receipt(control_roots[0], document)
        fault_key = "weight_decay"
    elif case in ("unbound-model", "unbound-data"):
        document = json.loads((candidate / "terminal.json").read_text())
        if case == "unbound-model":
            document["config"]["skyrl"]["trainer"]["policy"]["model"]["path"] = "/unbound/model"
            fault_key = "skyrl.trainer.policy.model.path"
        else:
            document["config"]["skyrl"]["data"]["train_data"] = ["/unbound/train.parquet"]
            fault_key = "skyrl.data.train_data"
        write_receipt(candidate, document)
    elif case in ("staged-profile", "wrong-profile-control"):
        root = control_roots[0] if case == "wrong-profile-control" else candidate
        document = json.loads((root / "terminal.json").read_text())
        launch = document["config"]
        source = launch["inputs"]["evaluation_data"]["eval_lite"][0]
        for index, callback in enumerate(launch["skyrl"]["trainer"]["callbacks"]):
            if "eval_lite" in callback.get("additional_evaluations", {}):
                profile = callback["additional_evaluations"]["eval_lite"]
                profile["data_files"] = [str(Path(source["local_path"]) / source["relative_path"])]
                if case == "wrong-profile-control":
                    profile["data_files"] = ["/unbound/named-panel.parquet"]
                    fault_key = f"skyrl.trainer.callbacks.{index}.additional_evaluations.eval_lite.data_files"
        write_receipt(root, document)
    elif case in ("home-alias", "alias-sibling", "unbound-gsm8k"):
        document = json.loads((candidate / "terminal.json").read_text())
        launch = document["config"]
        launch["skyrl"]["data"]["train_data"] = ["/root/data/gsm8k/train.parquet"]
        if case == "unbound-gsm8k":
            launch["skyrl"]["data"]["train_data"] = ["/other/data/gsm8k/train.parquet"]
            fault_key = "skyrl.data.train_data"
        elif case == "alias-sibling":
            launch["skyrl"]["generator"]["max_num_seqs"] += 1
            fault_key = "skyrl.generator.max_num_seqs"
        write_receipt(candidate, document)
    elif case in ("allocation", "source-identity", "relative-file", "authored-run-name"):
        document = json.loads((candidate / "terminal.json").read_text())
        launch = document["config"]
        if case == "allocation":
            launch["iris"]["allocation"]["num_nodes"] += 1
            fault_key = "iris.allocation.num_nodes"
        elif case == "source-identity":
            source = launch["inputs"]["train_data"][0]
            source["identity"] = "other-source@2026.11.10"
            launch["skyrl"]["data"]["train_data"] = [dict(source)]
            fault_key = "inputs.train_data.0.identity"
        elif case == "relative-file":
            source = launch["inputs"]["validation_data"][0]
            shutil.copyfile(Path(source["uri"]) / source["relative_path"], Path(source["uri"]) / "other.parquet")
            source["relative_path"] = "other.parquet"
            launch["skyrl"]["data"]["val_data"] = [dict(source)]
            fault_key = "recorded panel file differs"
        else:
            launch["skyrl"]["trainer"]["run_name"] = "authored-run-name"
            fault_key = "skyrl.trainer.run_name"
        write_receipt(candidate, document)
    report = rescore(preset.name, "--candidate", name)
    if case in ("cached-default", "cached-execution", "cached-source"):
        assert report["verdict"] == "pass", report
        if case == "cached-default":
            promoted = replace(preset, settings=(*preset.settings, "trainer.policy.optimizer_config.weight_decay=0.1"))
            changed_key = "skyrl.trainer.policy.optimizer_config.weight_decay"
        elif case == "cached-execution":
            promoted = replace(
                preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_memory="123GB"))
            )
            changed_key = "iris.allocation.memory"
        else:
            document = json.loads((candidate / "terminal.json").read_text())
            document["config"]["skyrl"]["generator"]["eval_sampling_params"]["top_k"] = 1
            write_receipt(candidate, document)
            monkeypatch.setattr(
                Provenance,
                "capture",
                lambda *args, **kwargs: Provenance("tree-two", "commit-one", False, "branch", "test"),
            )
            changed_key = "evaluation protocol"
        if case != "cached-source":
            monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: promoted})
        updated = rescore(preset.name, "--candidate", name)
        assert updated["verdict"] == "invalid", updated
        assert changed_key in updated["reason"]
        assert updated["report"] != report["report"]
    elif fault_key is not None or case in ("stale-default", "explicit-equal"):
        assert report["verdict"] == "invalid", report
        assert (fault_key or "skyrl.trainer.policy.optimizer_config.lr") in report["reason"]
    else:
        assert (report["verdict"], report["label"]) == ("pass", "no change"), report
        if case in ("initializer", "optimizer", "batch"):
            expected_key = {
                "initializer": "inputs.model.uri",
                "batch": "skyrl.trainer.train_batch_size",
            }.get(case, "skyrl.trainer.policy.optimizer_config.lr")
            assert expected_key in report["run"]["config_changes"]


def test_task_set_content_identity_and_declared_panel_bindings(recorded_runs):
    preset_name = "learning-medium"
    controls(recorded_runs, preset_name)
    source = recorded_runs(preset_name, "users/test/task-content")
    document = json.loads((source / "terminal.json").read_text())
    descriptor = document["config"]["inputs"]["validation_data"][0]
    original = Path(descriptor["uri"]) / descriptor["relative_path"]
    rows = pq.read_table(original).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), original)
    report = rescore(preset_name, "--candidate", "users/test/task-content")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    source = recorded_runs(preset_name, "users/test/task-content-relocated")
    document = json.loads((source / "terminal.json").read_text())
    descriptor = document["config"]["inputs"]["validation_data"][0]
    copied = source / "panel-copy"
    shutil.copytree(descriptor["uri"], copied)
    descriptor["uri"] = str(copied)
    document["config"]["skyrl"]["data"]["val_data"] = [dict(descriptor)]
    path = copied / descriptor["relative_path"]
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), path)
    write_receipt(source, document)
    report = rescore(preset_name, "--candidate", "users/test/task-content-relocated")
    assert report["verdict"] == "invalid" and "config differs" in report["reason"], report
    changed_source = recorded_runs(preset_name, "users/test/task-content-changed")
    changed_document = json.loads((changed_source / "terminal.json").read_text())
    changed_document["config"]["inputs"]["validation_data"][0]["uri"] = str(copied)
    changed_document["config"]["skyrl"]["data"]["val_data"] = [
        dict(changed_document["config"]["inputs"]["validation_data"][0])
    ]
    write_receipt(changed_source, changed_document)
    task = json.loads(rows[0]["task_spec"])
    task["context"]["events"][0]["content"] = "Return dog."
    rows[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = rescore(preset_name, "--candidate", "users/test/task-content-changed")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report

    roots = controls(recorded_runs, preset_name)
    roots.append(recorded_runs(preset_name, "users/test/wrong-cohort"))
    panel_root = Path(presets.selection_artifact(presets.PRESETS[preset_name].eval_lite).path())
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
    preset = presets.PRESETS[preset_name]
    model = tmp_path / "model"
    model.mkdir()
    local = replace(
        preset.model,
        step=ArtifactStep.adopt(
            preset.model.step.name,
            preset.model.step.version,
            str(model),
            kind=preset.model.step.artifact_type,
        ),
    )
    preset = replace(preset, model=local)
    monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset_name: preset})
    controls(recorded_runs, preset_name)
    tasks = selection if preset_name == "data-medium" else None
    setting_text = ("trainer.policy.optimizer_config.lr=2.0e-6", '+extra_env.VLLM_PROBE="1"') if tasks is None else ()
    settings = tuple(map(presets.parse_setting, setting_text))
    built = snowball.build_run(preset, settings=settings, tasks=tasks, requirements=("policy/raw_grad_norm<=10",))
    task_name = "fixture-selection@2026.11.10" if tasks else None
    source = recorded_runs(
        preset.name,
        built.runs[0].name,
        fingerprint=built.runs[0].fingerprint(),
        inputs=gate.GateInputs(
            preset_name=preset.name, settings=setting_text, tasks=task_name, requirements=("policy/raw_grad_norm<=10",)
        ),
    )
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
        {} if tasks else {"trainer.policy.optimizer_config.lr": 2e-6, "+extra_env.VLLM_PROBE": "1"}
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
