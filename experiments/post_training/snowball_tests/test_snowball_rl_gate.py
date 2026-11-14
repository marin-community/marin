# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
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
from marin.rl.rl_progress import Change, Comparison, ComparisonRole, Invariant
from marin.rl.skyrl import SkyRLRun
from marin.rl.skyrl_records import RecordedLaunch, read_skyrl_launch_config, read_skyrl_metrics
from rigging.provenance import Provenance
from scipy.stats import t
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_rl_command import EXIT_CODES
from experiments.post_training.snowball_rl_scoring import GATE_INPUTS_FILENAME, GateInputs, read_gate_inputs
from experiments.post_training.snowball_tests.fixtures import answer as _answer
from experiments.post_training.snowball_tests.fixtures import write_json as _json
from experiments.post_training.snowball_tests.fixtures import write_metrics, write_receipt

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]
KL = "policy/mismatch/staleness0/kl_k3"


def controls(write, name, **kwargs):
    pin = gate.control_pin(presets.PRESETS[name])
    return [
        write(
            name,
            gate.control_name(pin.preset, pin.commit, i),
            speed=100 + i,
            score=0.49 + i * 0.01,
            **kwargs,
        )
        for i in range(pin.runs)
    ]


def rescore(preset, *args, exit_code=None):
    result = CliRunner().invoke(gate.main, ["--preset", preset, *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), (result.output, result.exception)
    payload = json.loads(result.output.splitlines()[-1])
    assert result.exit_code == (exit_code if exit_code is not None else EXIT_CODES[payload["verdict"]]), (
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


def test_control_calibration_and_guard_tolerance():
    noise = gate.calibrate([{"x": value} for value in (1, 2, 3, 4)])["x"]
    assert (noise.mean, noise.sd, noise.df) == pytest.approx((2.5, math.sqrt(5 / 3), 3))
    row = gate._comparison("x", ComparisonRole.OBJECTIVE, [{"x": 4}], noise, 4, 2, higher_is_better=True)
    assert (row.control, row.run, row.delta) == (2.5, 4, 1.5)
    assert row.tolerance == pytest.approx(t.ppf(1 - 0.025 / 2, 3) * math.sqrt(5 / 3) * math.sqrt(1.25))
    assert row.change == Change.SAME
    zero = gate.calibrate([{"x": 2}, {"x": 2}])["x"]
    unavailable = gate._comparison("x", ComparisonRole.OBJECTIVE, [{"x": 4}], zero, 2)
    assert (unavailable.tolerance, unavailable.change) == (None, Change.UNAVAILABLE)


@pytest.mark.parametrize(
    "objective,constraint,holds,verdict,label",
    [
        (Change.BETTER, Change.SAME, True, "pass", "improves"),
        (Change.WORSE, Change.SAME, True, "fail", "regression"),
        (Change.BETTER, Change.WORSE, True, "fail", "trade-off"),
        (Change.SAME, Change.SAME, False, "fail", "no change"),
    ],
)
def test_verdict_labels_use_objectives_and_guards(objective, constraint, holds, verdict, label):
    rows = [
        Comparison(
            metric="x",
            role=ComparisonRole.OBJECTIVE,
            n_run=1,
            n_control=4,
            change=objective,
            passes=objective != Change.WORSE,
        ),
        Comparison(
            metric="y",
            role=ComparisonRole.CONSTRAINT,
            n_run=1,
            n_control=4,
            change=constraint,
            passes=constraint != Change.WORSE,
        ),
        Comparison(metric="telemetry", role=ComparisonRole.REPORTED, n_run=1, n_control=4, change=Change.WORSE),
    ]
    facts = [Invariant(name="finite", value=holds, limit=True, holds=holds)]
    assert gate._verdict(facts, rows) == (verdict, label)


def test_rescore_leave_one_out_writes_the_printed_report(recorded_runs):
    controls(recorded_runs, "speed-medium")
    report = rescore("speed-medium", "--leave-out", "r0")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "constraint", "reported"}
    assert all(row["comparator"] is not None for row in report["invariants"])


def test_extra_requirements_cannot_widen_existing_guards(recorded_runs):
    preset = presets.PRESETS["speed-medium"]
    key = "policy/raw_grad_norm"
    pin = gate.control_pin(preset)
    for index in range(pin.runs):
        root = recorded_runs(
            preset.name, gate.control_name(pin.preset, pin.commit, index), speed=100 + index, metrics={key: 100 + index}
        )
    candidates = []
    for index, value in enumerate((0, 0.01)):
        name = f"users/test/requirements-{index}"
        root = recorded_runs(preset.name, name, speed=108, metrics={key: value})
        candidates.append(ArtifactStep.adopt(name, preset.revision, str(root), kind=SkyRLRun))
    report = run(
        gate.gate_step(
            preset,
            tuple(candidates),
            inputs=GateInputs(
                preset_name=preset.name,
                requirements=(
                    key + "<=control",
                    key + "<=control",
                    key + "<=0",
                    "speed/update_seconds<=control",
                ),
            ),
        )
    )[0]
    assert report.verdict == "fail" and report.label == "regression"
    assert next(row for row in report.comparisons if row.metric == "speed/update_seconds").passes is False
    requirement = report.rows(ComparisonRole.REQUIREMENT)
    assert len(requirement) == 1 and requirement[0].passes
    assert requirement[0].tolerance == pytest.approx(t.ppf(0.975, 3) * math.sqrt(5 / 3) * math.sqrt(0.5 + 0.25))
    assert [(row.replicate, row.holds) for row in report.invariants if row.name == "require/" + key] == [
        (0, True),
        (1, False),
    ]


@pytest.mark.parametrize("case,expected", [("weighted", 0.05), ("empty", None), ("nonfinite", None), ("pooled", 0.2)])
def test_mismatch_uses_selected_tokens_over_the_measured_window(case, expected):
    preset = replace(presets.PRESETS["speed-long"], measured=range(2, 4))
    prefix = KL.rsplit("/", 1)[0]
    metrics = {
        step: {
            "timing/step_wall/group_admission": value,
            "timing/step_wall/policy_training": 2,
            "timing/step_wall/checkpoint_work": 900,
            "timing/step_wall/evaluation": 700,
            KL: kl,
            prefix + "/selected_tokens": count,
            prefix + "/finite_fraction": 1,
            "policy/mismatch/pooled/kl_k3": 0,
        }
        for step, value, kl, count in [(1, 1e9, 1e9, 1000), (2, 5, 0.5, 1), (3, 9, 0, 9)]
    }
    for step in (2, 3):
        if case == "empty":
            metrics[step][prefix + "/selected_tokens"] = 0
        if case == "nonfinite":
            metrics[step][prefix + "/finite_fraction"] = 0
        if case == "pooled":
            metrics[step][KL] = 0.2
    values = scoring._window_values(preset, metrics)
    assert values["speed/update_seconds"] == 9
    assert (KL not in values) if expected is None else values[KL] == pytest.approx(expected)


@pytest.mark.parametrize("fault", ["updates", "loss", "policy-loss", "union", "missing-counter"])
def test_training_invariants_from_recorded_values(fault):
    preset = replace(presets.PRESETS["speed-medium"], updates=2, measured=range(1, 3))
    metrics = {
        step: {"policy/raw_grad_norm": 1, "policy/policy_loss": 1, "policy/final_loss": 1, "loss/total": 1}
        for step in (1, 2)
    }
    counters = {
        "generate/outcome/" + value: 99 if value == Outcome.GRADED else 0 for value in (*Outcome, "rollout_failure")
    }
    if fault == "updates":
        metrics.pop(2)
    elif fault == "loss":
        metrics[2]["loss/total"] = float("nan")
    elif fault == "policy-loss":
        metrics[2]["policy/policy_loss"] = float("nan")
    elif fault == "union":
        counters.update({"generate/outcome/unavailable": 1, "generate/outcome/rollout_failure": 1})
    else:
        counters.pop("generate/outcome/graded")
    facts = (
        *scoring.training_invariants(preset, metrics),
        scoring._grader_errors([counters], "generate/outcome/", "grader"),
    )
    assert any(not fact.holds for fact in facts)
    assert gate._verdict(facts, []) == ("fail", "no change")


@pytest.mark.parametrize("preset", ["speed-short", "data-short", "learning-medium"])
def test_current_generation_counters_compare_with_stored_controls(recorded_runs, selection, preset):
    controls(recorded_runs, preset)
    inputs = None
    if preset == "data-short":
        selected = presets.selection_artifact("candidate-training@2026.11.10")
        shutil.copytree(selection.path(), selected.path())
        inputs = GateInputs(preset_name=preset, tasks="candidate-training@2026.11.10")
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
        path = root / "exports/dumped_evals/eval_only/task_set.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["task_id"] = "candidate-" + row["task_id"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = read_skyrl_metrics(str(root))
    step = presets.PRESETS[preset].updates
    values = metrics[step]
    report = rescore(preset, "--candidate", "users/test/current-generation")
    assert (report["verdict"], report["label"]) == ("pass", "no change"), report
    if preset in ("speed-short", "learning-medium"):
        prefix = "eval/all/outcome/" if preset == "learning-medium" else "eval/eval_lite/all/outcome/"
        values[prefix + "graded"] = 19
        values[prefix + "unavailable"] = 1
        write_metrics(root, metrics)
        alias = root.parent.parent / "current-bad-counter" / root.name
        shutil.copytree(root, alias)
        report = rescore(preset, "--candidate", "users/test/current-bad-counter")
        assert report["verdict"] == "fail"
        if preset == "learning-medium":
            assert any(
                row["name"] == "eval_lite/grader_error_fraction" and not row["holds"] for row in report["invariants"]
            )
            return
        task_set = root / "exports/eval_lite/dumped_evals/eval_only/task_set.jsonl"
        answers = [json.loads(line) for line in task_set.read_text().splitlines()]
        for answer in answers[:1]:
            answer["verification"]["status"] = "error"
        task_set.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
        values["eval/eval_lite/all/outcome/graded"] = 20
        values["eval/eval_lite/all/outcome/unavailable"] = 0
        write_metrics(root, metrics)
        coverage_run = root.parent.parent / "current-low-coverage" / root.name
        shutil.copytree(root, coverage_run)
        report = rescore(preset, "--candidate", "users/test/current-low-coverage")
        coverage = next(row for row in report["invariants"] if row["name"].endswith("/scored_fraction"))
        assert coverage["value"] < 0.99 and coverage["holds"] is False and report["verdict"] == "fail"


@pytest.mark.parametrize(
    "path,value,settings,verdict",
    [
        (("skyrl", "generator", "eval_sampling_params", "temperature"), 0.5, (), "invalid"),
        (("skyrl", "trainer", "seed"), None, (), "invalid"),
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
        protocol = scoring.evaluation_protocol(
            RecordedLaunch.from_document(launch, attempt="recorded-attempt"), "eval_lite"
        )
        assert {
            key: protocol["sampling_params"][key]
            for key in ("temperature", "ignore_eos", "max_generate_length", "logprobs")
        } == {
            "temperature": 0,
            "ignore_eos": False,
            "max_generate_length": 8192,
            "logprobs": None,
        }
        assert protocol["n_samples_per_prompt"] == 1
        assert protocol["context_budget"]["max_new_tokens_per_turn"] == 8192
        return
    controls(recorded_runs, "learning-medium")
    root = recorded_runs(
        "learning-medium",
        "users/test/protocol",
        inputs=GateInputs(preset_name="learning-medium", settings=settings),
    )
    receipt = json.loads((root / "terminal.json").read_text())
    target = receipt["config"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    write_receipt(root, receipt)
    report = rescore("learning-medium", "--candidate", "users/test/protocol")
    assert report["verdict"] == verdict, report
    if path == ("skyrl", "trainer", "seed"):
        assert "malformed recorded configuration" in report["reason"] and "seed" in report["reason"]
    elif verdict == "invalid":
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
        ("candidate-compose", 4),
        ("train-candidate-compose", 4),
        ("train-pass", 0),
        ("train-failed", 3),
        ("malformed-control", 4),
        ("malformed-status", 4),
        ("compose-timeout", 4),
        ("unexpected", 3),
        ("missing-requirement", 4),
    ],
)
def test_submission_waits_for_the_coordinator_and_returns_the_verdict(recorded_runs, monkeypatch, case, expected):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    preset = presets.PRESETS["train-long" if case.startswith("train-") else "speed-medium"]
    roots = controls(recorded_runs, preset.name) if preset.gated else []
    built = snowball.build_run(preset)
    source = (
        None
        if "candidate-compose" in case
        else recorded_runs(
            preset.name,
            built.runs[0].name,
            speed=150 if case == "fail" else 101,
            fingerprint=built.runs[0].fingerprint(),
        )
    )
    if case == "train-failed":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["state"] = "failed"
        write_receipt(source, receipt)
    if case.startswith("malformed-"):
        root = roots[0] if case == "malformed-control" else source
        receipt = json.loads((root / "terminal.json").read_text())
        receipt["config"]["skyrl"]["trainer"]["seed"] = None
        write_receipt(root, receipt)
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
    if case == "compose-timeout":
        real_run = subprocess.run

        def timeout_native(command, *args, **kwargs):
            if any("launch_config.load_launch_config" in str(value) for value in command):
                raise subprocess.TimeoutExpired(command, 120)
            return real_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", timeout_native)
    if case == "unexpected":
        path = Path(presets.selection_artifact(preset.tasks).path()) / presets.SUMMARY_FILENAME
        summary = json.loads(path.read_text())
        summary.pop("grader_kind_counts")
        path.write_text(json.dumps(summary))
    job = JobName.from_wire("/test-user/coordinator-id-returned-by-iris")

    def finish(*args, **kwargs):
        if case not in ("missing", "native-failed") and built.gate is not None:
            run(gate.completed_gate(built.gate))
            if case == "changed-source":
                monkeypatch.setattr(
                    Provenance,
                    "capture",
                    lambda *args, **kwargs: Provenance("tree-edited-after-submit", "commit-one", True, "branch", "test"),
                )
        return SimpleNamespace(
            state=JobState.FAILED if case in ("native-failed", "train-failed") else JobState.SUCCEEDED
        )

    jobs = {job: SimpleNamespace(wait=finish)}
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = job
    if case.startswith("malformed-"):
        client.submit.side_effect = AssertionError("malformed stored configuration reached the coordinator RPC")
    if case.startswith("stale-control"):
        client.submit.side_effect = AssertionError("stale controls reached the coordinator RPC")
    if "candidate-compose" in case:
        client.submit.side_effect = AssertionError("a candidate that cannot compose reached the coordinator RPC")
    client.job.side_effect = lambda name: jobs[name]
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    if case == "child":
        monkeypatch.setenv("IRIS_TASK_ID", "/test-user/coordinator/0")
        client.submit.side_effect = AssertionError("coordinator execution submitted another job")
    try:
        result = CliRunner().invoke(
            snowball.main,
            [
                "--preset",
                preset.name,
                "--version",
                preset.revision,
                "--run",
                *(("--set", "trainer.algorithm.policy_loss_type=gspo") if "candidate-compose" in case else ()),
                *(("--require", "typo>=0") if case == "missing-requirement" else ()),
                *(() if case.endswith("no-wait") or case == "unexpected" else ("--wait",)),
            ],
        )
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    if case.startswith("malformed-"):
        assert "malformed recorded configuration" in result.output and "seed" in result.output
    if case.startswith("stale-control"):
        assert "skyrl.trainer.policy.optimizer_config.lr" in result.output
    if "candidate-compose" in case:
        assert "current preset recipe cannot be composed:" in result.output and "gspo" in result.output.lower()
    if case == "compose-timeout":
        assert "timed out after 120 s" in result.output
    if case == "unexpected":
        assert "KeyError" in result.output and "grader_kind_counts" in result.output
    if case == "missing-requirement":
        assert "does not compute" in result.output
    if case == "child":
        report = gate.completed_gate(built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"


@pytest.mark.parametrize("fault", ["untracked", "dirty"])
def test_control_refuses_actual_untracked_source_before_execution(selection, tmp_path, monkeypatch, fault):
    root = tmp_path / "git-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "experiments").mkdir()
    (root / "experiments/untracked.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(presets, "REPO_ROOT", root)
    if fault == "dirty":
        subprocess.run(["git", "-C", str(root), "add", "experiments/untracked.py"], check=True)
        monkeypatch.setattr(
            Provenance, "capture", lambda *args, **kwargs: Provenance("tree", "commit", True, "branch", "test")
        )
    with pytest.raises(click.BadParameter, match="stage source" if fault == "untracked" else "clean checkout"):
        snowball.build_run(presets.PRESETS["speed-medium"], control=True)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--wait",),
        ("--set", "trainer.dump_eval_results=false"),
        ("--set", 'trainer={"algorithm":{"off_policy_correction":"none"}}'),
        ("--set", "trainer.policy.optimizer_config.lr=wrong"),
    ],
)
def test_refusals_use_exit_four(selection, arguments):
    result = CliRunner().invoke(
        snowball.main, ["--preset", "speed-medium", "--version", presets.PRESETS["speed-medium"].revision, *arguments]
    )
    assert result.exit_code == 4, (result.output, result.exception)
    if "trainer.dump_eval_results=false" in arguments:
        assert "owned" in result.output


def test_eval_lite_scores_join_permuted_answers_and_aggregate_tasks(tmp_path, recorded_runs):
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
    root = recorded_runs("learning-medium", "users/test/score-order")
    launch = read_skyrl_launch_config(str(root))
    launch["artifacts"]["export_root"] = str(tmp_path / "configured-export")
    launch["inputs"]["validation_data"][0]["uri"] = str(tmp_path / "selection")
    _json(tmp_path / "resolved-launch.yaml", {"config": launch})
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "configured-export/dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "task_set.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in answers) + "\n"
        )
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    scores = scoring.evaluation_scores(presets.EVALUATIONS["eval_lite_100"], str(tmp_path))
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


@pytest.mark.parametrize(
    "case",
    [
        "initializer",
        "stale-default",
        "control-commit",
        "unbound-data",
        "batch",
        "staged-profile",
        "allocation",
        "cached-default",
        "zero-spread",
        "missing-guard",
    ],
)
def test_gate_compares_recorded_configuration_and_defaults(recorded_runs, tmp_path, monkeypatch, case):
    preset = presets.PRESETS[
        "speed-short" if case == "staged-profile" else "speed-medium" if case == "missing-guard" else "learning-medium"
    ]
    setting = "trainer.policy.optimizer_config.lr=2e-6"
    settings = () if case in ("stale-default", "cached-default", "zero-spread", "staged-profile") else (setting,)
    if case == "batch":
        settings = (
            "trainer.train_batch_size=64",
            "trainer.policy_mini_batch_size=64",
            "generator.n_samples_per_prompt=8",
            "trainer.micro_train_batch_size_per_gpu=2",
        )
    inputs = GateInputs(preset_name=preset.name, settings=settings)
    if case == "initializer":
        name, version = "tests/custom-init", "2026.11.07"
        root = tmp_path / name / version
        model = SkyRLRun(
            path=str(root),
            hf_model_uri="s3://fixture-initializer/policy",
            global_step=25,
            tokenizer_uri=preset.model.tokenizer_uri,
            tokenizer_revision=preset.model.tokenizer_revision,
            checkpoint_root=str(root / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(root / "terminal.json"),
            iris_job_id="/test-user/initializer",
        )
        _json(
            root / RECORD_FILENAME,
            ArtifactRecord(
                name=name,
                version=version,
                fingerprint="initializer",
                result_type=result_type_name(SkyRLRun),
                result=model.result_payload(),
            ).model_dump(mode="json"),
        )
        inputs = inputs.model_copy(update={"init": f"{name}@{version}"})
    control_roots = controls(recorded_runs, preset.name)
    if case == "zero-spread":
        pin = gate.control_pin(preset)
        control_roots = [
            recorded_runs(preset.name, gate.control_name(pin.preset, pin.commit, index), score=0.5)
            for index in range(pin.runs)
        ]
    if case == "stale-default":
        preset = replace(preset, recipe=replace(preset.recipe, config=preset.recipe.config.with_settings((setting,))))
        monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: preset})
    name = "users/test/config-" + case
    candidate = recorded_runs(preset.name, name, inputs=inputs)
    fault = {
        "stale-default": "control config differs",
        "control-commit": "does not match its pin",
        "unbound-data": "skyrl.data.train_data",
        "allocation": "iris.allocation.num_nodes",
        "zero-spread": "variation is zero",
        "missing-guard": "missing or non-finite guard metric",
    }.get(case)
    root = control_roots[0] if case == "control-commit" else candidate
    if case in ("control-commit", "unbound-data", "allocation", "staged-profile"):
        receipt = json.loads((root / "terminal.json").read_text())
        launch = receipt["config"]
        if case == "control-commit":
            receipt["result"]["provenance"]["base_commit"] = "wrong-commit"
        elif case == "unbound-data":
            launch["skyrl"]["data"]["train_data"] = ["/unbound/train.parquet"]
        elif case == "allocation":
            launch["iris"]["allocation"]["num_nodes"] += 1
        else:
            source = launch["inputs"]["evaluation_data"]["eval_lite"][0]
            for callback in launch["skyrl"]["trainer"]["callbacks"]:
                if "eval_lite" in callback.get("additional_evaluations", {}):
                    callback["additional_evaluations"]["eval_lite"]["data_files"] = [
                        str(Path(source["local_path"]) / source["relative_path"])
                    ]
        write_receipt(root, receipt)
    if case == "missing-guard":
        edit_metrics(candidate, lambda row: row.update({"policy/mismatch/staleness0/selected_tokens": 0}))
    report = rescore(preset.name, "--candidate", name)
    if fault:
        assert report["verdict"] == "invalid" and fault in report["reason"], report
    else:
        assert (report["verdict"], report["label"]) == ("pass", "no change"), report
        if case in ("initializer", "batch"):
            assert ("inputs.model.uri" if case == "initializer" else "skyrl.trainer.train_batch_size") in report["run"][
                "config_changes"
            ]
    if case == "cached-default":
        promoted = replace(
            preset,
            recipe=replace(
                preset.recipe,
                config=preset.recipe.config.with_settings(("trainer.policy.optimizer_config.weight_decay=0.1",)),
            ),
        )
        monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: promoted})
        updated = rescore(preset.name, "--candidate", name)
        assert updated["verdict"] == "invalid" and "weight_decay" in updated["reason"]
        assert updated["report"] != report["report"]


def test_task_set_content_hash_is_order_independent_and_checked(recorded_runs):
    preset = "learning-medium"
    controls(recorded_runs, preset)
    source = recorded_runs(preset, "users/test/task-content")
    receipt = json.loads((source / "terminal.json").read_text())
    descriptor = receipt["config"]["inputs"]["validation_data"][0]
    original = Path(descriptor["uri"]) / descriptor["relative_path"]
    rows = pq.read_table(original).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), original)
    report = rescore(preset, "--candidate", "users/test/task-content")
    assert report["verdict"] == "pass", report
    changed = recorded_runs(preset, "users/test/task-content-edited")
    receipt = json.loads((changed / "terminal.json").read_text())
    descriptor = receipt["config"]["inputs"]["validation_data"][0]
    copied = changed / "task-set-copy"
    shutil.copytree(descriptor["uri"], copied)
    descriptor["uri"] = str(copied)
    receipt["config"]["skyrl"]["data"]["val_data"] = [dict(descriptor)]
    path = copied / descriptor["relative_path"]
    rows = pq.read_table(path).to_pylist()
    task = json.loads(rows[0]["task_spec"])
    task["context"]["events"][0]["content"] = "Return dog."
    rows[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows), path)
    write_receipt(changed, receipt)
    report = rescore(preset, "--candidate", "users/test/task-content-edited")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


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
    setting_text = (
        ("trainer.policy.optimizer_config.lr=2.0e-6", 'extra_env.VLLM_PROBE="1"', "generator.max_num_seqs=1024")
        if tasks is None
        else ()
    )
    settings = setting_text
    built = snowball.build_run(preset, settings=settings, tasks=tasks, requirements=("policy/raw_grad_norm<=10",))
    task_name = "fixture-selection@2026.11.10" if tasks else None
    source = recorded_runs(
        preset.name,
        built.runs[0].name,
        fingerprint=built.runs[0].fingerprint(),
        inputs=GateInputs(
            preset_name=preset.name, settings=setting_text, tasks=task_name, requirements=("policy/raw_grad_norm<=10",)
        ),
    )
    staged = source.with_name("native-staging")
    source.rename(staged)
    (staged / GATE_INPUTS_FILENAME).unlink()

    def native(config):
        shutil.copytree(staged, source, dirs_exist_ok=True)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(snowball, "run_skyrl", native)
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preset", "train-long", "--require", "unrelated"])
    run(*built.runs)
    recorded = read_gate_inputs(str(source))
    assert recorded.preset_name == preset_name and recorded.requirements == ("policy/raw_grad_norm<=10",)
    decoded = dict(item.split("=", 1) for item in recorded.settings)
    assert {key: json.loads(value) for key, value in decoded.items()} == (
        {}
        if tasks
        else {
            "trainer": {"policy": {"optimizer_config": {"lr": 2e-6}}},
            "extra_env": {"VLLM_PROBE": "1"},
            "generator": {"max_num_seqs": 1024},
        }
    )
    assert recorded.tasks == task_name
    report = rescore(preset_name, "--candidate", built.runs[0].name)
    assert report["verdict"] == "pass", report
    assert report["run"]["requirements"] == list(recorded.requirements)
    assert report["run"]["settings"] == list(recorded.settings)
    assert report["run"]["tasks"] == recorded.tasks
    assert report["run"]["attempt_ids"] == ["recorded-attempt"]
    assert comparisons(report)["eval_lite/family_mean/gain_per_h100_hour"]["run"] == pytest.approx(0.005)
    if tasks is None:
        assert {"skyrl.trainer.policy.optimizer_config.lr", "skyrl.generator.max_num_seqs"} <= set(
            report["run"]["config_changes"]
        )
