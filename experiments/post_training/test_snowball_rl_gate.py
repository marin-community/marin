# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import runpy
import shutil
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

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
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.timing import Timestamp
from scipy.stats import t

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_rl_test_fixtures import answer as _answer
from experiments.post_training.snowball_rl_test_fixtures import write_json as _json

pytestmark = pytest.mark.usefixtures("source_provenance", "synthetic_current_controls")


def controls(write, preset, *, constant=False, fault=None, common_fault=None):
    pin = gate.control_pins(preset)[0]
    return [
        write(
            preset,
            presets.control_name(pin.preset, pin.main_commit, i),
            speed=100.0 + (0 if constant else i),
            score=0.49 + 0.01 * i,
            mixed=128 + i,
            **((common_fault or {}) | (fault if i == 0 and fault else {})),
        )
        for i in range(pin.runs)
    ]


def rescore(preset, *args, exit_code=None):
    result = CliRunner().invoke(gate.main, ["--preset", preset, *args])
    payload = json.loads(result.output.splitlines()[-1])
    expected = {"pass": 0, "fail": 1, "invalid": 2}[payload["verdict"]] if exit_code is None else exit_code
    assert result.exit_code == expected, (result.output, result.exception)
    return payload


def status(preset, argv):
    result = CliRunner().invoke(snowball.main, ["--preset", preset, "--version", presets.PRESET_REVISION, *argv])
    resume_warning, _, payload = result.stdout.partition("preset:")
    if "--resume-from" in argv:
        assert resume_warning.startswith("Cannot compare this checkout with resume source tree "), resume_warning
        assert "verify optimizer and checkpoint compatibility before continuing." in resume_warning
    else:
        assert not resume_warning, resume_warning
    snapshot = yaml.safe_load("preset:" + payload.split("StepSpec(", 1)[0])
    assert result.exit_code == {"pass": 0, "fail": 1, "invalid": 2, None: 0}[snapshot.get("verdict")], (
        result.output,
        result.exception,
    )
    return snapshot


def test_calibrate_pools_deviations_within_pins():
    first = gate.ControlPin("speed-short", "first", presets.PRESET_REVISION, 3)
    second = gate.ControlPin("speed-short", "second", presets.PRESET_REVISION, 2)
    noise = gate.calibrate({first: [{"x": x} for x in (1.0, 2.0, 3.0)], second: [{"x": x} for x in (10.0, 14.0)]})["x"]
    assert (noise.mean, noise.df) == (6, 3)
    assert noise.sd == pytest.approx(math.sqrt(10 / 3))


@pytest.mark.parametrize("outlier", [False, True])
def test_rescore_leave_one_out_uses_median_and_two_guard_tolerance(
    recorded_runs, outlier, tmp_path, monkeypatch, capsys
):
    roots = controls(recorded_runs, "speed-medium")
    (tmp_path / "test-panel-rates" / scoring.PASS_RATES_FILENAME).unlink()
    if outlier:
        path = roots[0] / "exports/metrics/step-000002.json"
        _json(path, json.loads(path.read_text()) | {"timing/step_wall/group_admission": 1100})
    before = {
        path: hashlib.sha256(path.read_bytes()).digest() for root in roots for path in root.rglob("*") if path.is_file()
    }
    report = rescore("speed-medium", "--leave-out", "r0")
    assert {f"timing/{phase}/{stat}" for phase in ("grading", "setup") for stat in ("mean", "p50", "p95", "max")} <= {
        row["metric"] for row in report["reported"]
    }
    timing, mismatch = report["guards"]
    assert [timing[key] for key in ("run", "control", "sd", "df", "n_run", "n_control", "delta")] == [
        100,
        102,
        1,
        2,
        1,
        3,
        2,
    ]
    assert timing["tolerance"] == pytest.approx(math.sqrt(2 * 0.975**2 / (1 - 0.975**2)) * math.sqrt(1 + 1 / 3))
    assert mismatch["delta"] == pytest.approx(0.002)
    (objective,) = report["objectives"]
    assert objective["metric"] == "speed/update_seconds"
    assert objective["delta_percent"] == pytest.approx(-200 / 102)
    assert objective["noise_percent"] == pytest.approx(100 * timing["tolerance"] / 102)
    assert [row["metric"] for row in report["constraints"]] == ["policy/mismatch/staleness0/kl_k3"]
    markdown = Path(report["report"]).with_suffix(".md").read_text()
    assert markdown.index("## Objectives") < markdown.index("## Constraints") < markdown.index("Result:")
    assert "delta %" in markdown and "noise +/- %" in markdown
    assert (report["verdict"], report["label"]) == ("pass", "regression" if outlier else "no change")
    rows = {row["metric"]: row for row in report["reported"]}
    assert rows["timing/step_wall/group_admission"]["run"] == pytest.approx(100 + (1000 / 9 if outlier else 0))
    assert [rows["startup/init_weight_sync_state"][key] for key in ("run", "control")] == pytest.approx([10, 10.2])
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert set(report) == {
        "preset",
        "verdict",
        "label",
        "reason",
        "run",
        "control",
        "invariants",
        "guards",
        "reported",
        "objectives",
        "constraints",
        "vs_baseline",
        "vs_baseline_reason",
        "gaps",
    }
    pin = gate.CONTROLS["speed-medium"][0]
    ordinary = rescore("speed-medium", "--candidate", presets.control_name(pin.preset, pin.main_commit, 0))
    assert ordinary["report"] != str(artifact) and [row["n_control"] for row in ordinary["guards"]] == [4, 4]
    arguments = ["snowball_rl_gate", "--preset", "speed-medium", "--leave-out", "r0"]
    monkeypatch.setattr(sys, "argv", arguments)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_module("experiments.post_training.snowball_rl_gate", run_name="__main__")
    output = capsys.readouterr()
    assert exit_status.value.code == 0
    assert [json.loads(line) for line in output.out.splitlines() if line.startswith('{"report":')] == [
        {"report": str(artifact), **report}
    ]
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in before}


def test_baseline_uses_control_spread_and_unique_finished_run_group(recorded_runs, tmp_path):
    controls(recorded_runs, "speed-medium")
    recorded_runs("speed-medium", "users/test/baseline-candidate", speed=101.5, argv=("--preset", "speed-medium"))
    first = recorded_runs("speed-medium", "users/test/baseline-a", speed=20, argv=("--preset", "speed-medium"))
    recorded_runs("speed-medium", "users/test/baseline-b", speed=24, argv=("--preset", "speed-medium"))
    alias = first.parents[1] / "baseline-alias" / presets.PRESET_REVISION
    record = json.loads((first / RECORD_FILENAME).read_text())
    _json(alias / RECORD_FILENAME, record | {"name": "users/test/baseline-alias", "source": str(first)})
    references = [
        f"users/test/{name}@{presets.PRESET_REVISION}"
        for name in ("baseline-b", "baseline-a", "baseline-a", "baseline-alias")
    ]
    original = rescore("speed-medium", "--candidate", "users/test/baseline-candidate")
    report = rescore(
        "speed-medium",
        "--candidate",
        "users/test/baseline-candidate",
        *(item for ref in references for item in ("--baseline", ref)),
    )
    assert (report["verdict"], report["label"]) == (original["verdict"], original["label"]) == ("pass", "no change")
    assert report["guards"] == original["guards"] and report["reported"] == original["reported"]
    assert report["report"] != original["report"]
    assert report["run"]["baselines"] == sorted(set(references))
    rows = {row["metric"]: row for row in report["vs_baseline"]}
    timing = rows["speed/update_seconds"]
    expected = {"run": 101.5, "control": 22, "n_run": 1, "n_control": 2, "df": 3, "delta": -79.5, "change": "worse"}
    assert {key: timing[key] for key in expected} == expected
    assert timing["sd"] == pytest.approx(math.sqrt(5 / 3))
    assert timing["tolerance"] == pytest.approx(t.ppf(1 - 0.025 / 2, 3) * math.sqrt(5 / 3) * math.sqrt(1.5))
    startup = rows["startup/init_weight_sync_state"]
    assert startup["control"] == pytest.approx(2.2)
    assert startup["tolerance"] == pytest.approx(t.ppf(0.975, 3) * math.sqrt(5 / 3) / 10 * math.sqrt(1.5))
    assert "## Vs baseline" in Path(report["report"]).with_suffix(".md").read_text()
    missing = recorded_runs("speed-medium", "users/test/missing-telemetry", argv=("--preset", "speed-medium"))
    for path in missing.glob("exports/metrics/*.json"):
        values = json.loads(path.read_text())
        values.pop("policy/mismatch/staleness0/kl_k3", None)
        _json(path, values)
    optional = rescore(
        "speed-medium",
        "--candidate",
        "users/test/baseline-candidate",
        "--baseline",
        f"users/test/missing-telemetry@{presets.PRESET_REVISION}",
    )
    assert optional["guards"] == original["guards"] and optional["reported"] == original["reported"]
    assert (optional["verdict"], optional["label"]) == (original["verdict"], original["label"])
    optional_rows = {row["metric"]: row for row in optional["vs_baseline"]}
    unavailable = optional_rows["policy/mismatch/staleness0/kl_k3"]
    assert unavailable["control"] is None and unavailable["change"] == "unavailable"
    wrong = recorded_runs("speed-medium", "users/test/wrong-baseline", argv=("--preset", "learning-medium"))
    old = recorded_runs(
        "speed-medium", "users/test/old-baseline", version="2026.10.26", argv=("--preset", "speed-medium")
    )
    unfinished = recorded_runs("speed-medium", "users/test/unfinished-baseline", missing_receipt=True)
    failed = recorded_runs("speed-medium", "users/test/failed-baseline", argv=("--preset", "speed-medium"))
    terminal = json.loads((failed / "terminal.json").read_text())
    terminal["result"]["state"] = "failed"
    _json(failed / "terminal.json", terminal)
    incomplete = recorded_runs(
        "speed-medium",
        "users/test/incomplete-baseline",
        metrics={"policy/raw_grad_norm": float("nan")},
        argv=("--preset", "speed-medium"),
    )
    controls(recorded_runs, "learning-medium")
    recorded_runs("learning-medium", "users/test/panel-candidate", argv=("--preset", "learning-medium"))
    other_panel = recorded_runs(
        "learning-medium",
        "users/test/other-panel",
        argv=("--preset", "learning-medium"),
        first_panel_task="other-heldout-task",
    )
    missing_record = first.parents[1] / "missing-record" / presets.PRESET_REVISION
    panel_original = rescore("learning-medium", "--candidate", "users/test/panel-candidate")
    cases = [
        ("speed-medium", root, reason)
        for root, reason in (
            (wrong, "not a recorded speed-medium"),
            (old, "revision differs"),
            (unfinished, "no launch receipt at"),
            (failed, "native job /tests/native-run is failed"),
            (incomplete, "failed invariants"),
            (missing_record, "record"),
        )
    ]
    different_protocol = recorded_runs(
        "learning-medium", "users/test/other-protocol", argv=("--preset", "learning-medium")
    )
    document = json.loads((different_protocol / "terminal.json").read_text())
    document["config"]["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.5
    _json(different_protocol / "terminal.json", document)
    _json(different_protocol / "resolved-launch.yaml", {"config": document["config"]})
    cases.append(("learning-medium", different_protocol, "evaluation protocol"))
    cases.append(("learning-medium", other_panel, "absent from its task manifest"))
    for preset_name, root, reason in cases:
        candidate_name = (
            "users/test/panel-candidate" if preset_name == "learning-medium" else "users/test/baseline-candidate"
        )
        expected = panel_original if preset_name == "learning-medium" else original
        ref = f"{root.relative_to(first.parents[3]).parent}@{root.name}"
        rejected = CliRunner().invoke(
            gate.main, ["--preset", preset_name, "--candidate", candidate_name, "--baseline", ref]
        )
        assert rejected.exit_code == 0, (ref, rejected.output, rejected.exception)
        payload = json.loads(rejected.output.strip().splitlines()[-1])
        assert (payload["verdict"], payload["label"]) == (expected["verdict"], expected["label"])
        assert payload["vs_baseline"] == [] and reason in payload["vs_baseline_reason"]
        refused = CliRunner().invoke(
            snowball.main, ["--preset", preset_name, "--version", presets.PRESET_REVISION, "--baseline", ref]
        )
        assert refused.exit_code == 4 and reason in refused.output


VLLM = "vllm/median_generation_throughput"
BOUNDARY = {
    "generate/outcome/graded": 9700,
    "generate/outcome/submission_failure": 200,
    "generate/outcome/infra_error": 100,
}
AMBIGUOUS = {
    "generate/outcome/graded": 9900,
    "generate/outcome/infra_error": 100,
    "generate/outcome/rollout_failure": 100,
}
BROKEN = {
    "metrics": {
        "policy/raw_grad_norm": float("nan"),
        "generate/outcome/graded": 502,
        "generate/outcome/infra_error": 10,
    },
    "unscored": 20,
    "initial_unscored": 20,
    "duplicate_task": True,
    "omit_step": 25,
    "argv": ("snowball", "--init", "users/test/missing@2026.10.27"),
}
BROKEN_FAILS = {"planned_updates", "finite_losses_and_gradients", "grader_error_fraction"} | {
    f"test_panel/{name}"
    for name in (
        "batch_finished",
        "panel_scored_fraction",
        "grader_error_fraction",
        "initial/batch_finished",
        "initial/panel_scored_fraction",
        "initial/grader_error_fraction",
    )
}


@pytest.mark.parametrize(
    "preset, fault, control_fault, verdict, label, failed, bounds",
    [
        pytest.param("learning-long", {"initial_score": 0.2}, {}, "pass", "no change", set(), None, id="long-complete"),
        pytest.param(
            "data-long",
            {"initial_score": 0.2},
            {"preset": "learning-long"},
            "pass",
            "no change",
            set(),
            None,
            id="data-long-complete",
        ),
        pytest.param(
            "learning-long",
            {"missing_final_panel": True, "initial_score": 0.2},
            {},
            "invalid",
            None,
            {"test_panel/batch_finished", "test_panel/panel_scored_fraction", "test_panel/grader_error_fraction"},
            None,
            id="long-missing-final",
        ),
        ("speed-medium", {"speed": 80}, {}, "pass", "improves", set(), None),
        ("speed-medium", {"speed": 80, "metrics": {VLLM: 1}}, {}, "pass", "trade-off", set(), None),
        ("speed-medium", {"speed": 130, "metrics": {VLLM: 1000}}, {}, "fail", "trade-off", set(), None),
        ("speed-long", {}, {}, "pass", "no change", set(), None),
        (
            "speed-long",
            {"metrics": {"async/performance/loss_tokens_per_configured_policy_gpu_second": 50}},
            {},
            "fail",
            "regression",
            set(),
            None,
        ),
        ("speed-long", {"metrics": {"policy/mismatch/pooled/kl_k3": 100}}, {}, "pass", "regression", set(), None),
        ("speed-long", {"metrics": {"policy/mismatch/staleness0/kl_k3": 100}}, {}, "fail", "regression", set(), None),
        pytest.param(
            "learning-medium",
            {"panel_identity": None},
            {},
            "invalid",
            None,
            set(),
            None,
            id="missing-recorded-panel-version",
        ),
        pytest.param(
            "learning-medium",
            {"panel_identity": "rl-data/task-selection/other-panel@2026.11.01"},
            {},
            "invalid",
            None,
            set(),
            None,
            id="recorded-panel-version-differs",
        ),
        ("learning-medium", BROKEN, {}, "fail", "regression", BROKEN_FAILS, None),
        ("speed-medium", {"fixed": False}, {}, "fail", "regression", {"fixed_reply_tokens"}, None),
        ("speed-medium", {}, {"fault": {"provenance": {"base_commit": "wrong"}}}, "invalid", None, set(), None),
        ("speed-medium", {}, {"fault": {"metrics": AMBIGUOUS}}, "invalid", None, set(), None),
        ("speed-medium", {"version": "2026.10.01"}, {}, "invalid", None, set(), None),
        pytest.param(
            "speed-medium",
            {},
            {"fault": {"record_version": "2026.10.01"}},
            "invalid",
            None,
            set(),
            None,
            id="candidate-and-control-evaluation-revisions-differ",
        ),
        (
            "learning-medium",
            {"drop_task": True},
            {},
            "invalid",
            None,
            {"test_panel/batch_finished", "test_panel/initial/batch_finished"},
            None,
        ),
        (
            "speed-medium",
            {"metrics": {"policy/mismatch/staleness0/selected_tokens": 0}},
            {},
            "invalid",
            None,
            set(),
            None,
        ),
        ("speed-medium", {}, {"constant": True}, "invalid", None, set(), None),
        pytest.param(
            "speed-medium",
            {},
            {"fault": {"metrics": {"vllm/latency_ttft_mean": float("nan")}}},
            "pass",
            "no change",
            set(),
            None,
            id="nonfinite_range",
        ),
        ("data-short", {}, {}, "pass", "no change", set(), None),
        ("speed-medium", {"metrics": BOUNDARY}, {}, "pass", "no change", set(), (0.01, 0.01)),
        ("speed-medium", {"metrics": AMBIGUOUS}, {}, "invalid", None, {"grader_error_fraction"}, (0.01, 0.02)),
        (
            "speed-medium",
            {"speed": 130, "metrics": AMBIGUOUS},
            {},
            "fail",
            "regression",
            {"grader_error_fraction"},
            (0.01, 0.02),
        ),
        ("speed-short", {}, {}, "pass", "no change", set(), None),
        pytest.param(
            "speed-short", {"quality_unscored": 15}, {}, "pass", "no change", set(), (0.01, 0.01), id="coverage-boundary"
        ),
        pytest.param(
            "speed-short",
            {"quality_unscored": 16},
            {},
            "fail",
            "regression",
            {"test_panel/panel_scored_fraction", "test_panel/grader_error_fraction"},
            (16 / 1500, 16 / 1500),
            id="quality-failure",
        ),
        pytest.param(
            "speed-short",
            {"quality_metrics": {"eval/test_panel/all/outcome/invalid_task": None}},
            {},
            "invalid",
            None,
            {"test_panel/grader_error_fraction"},
            (0, 1),
            id="quality-missing-counter",
        ),
        pytest.param(
            "speed-short",
            {
                "quality_unscored": 12,
                "quality_metrics": {
                    "eval/test_panel/all/outcome/graded": 3000,
                    "eval/test_panel/all/outcome/rollout_failure": 12,
                },
            },
            {},
            "invalid",
            None,
            {"test_panel/grader_error_fraction"},
            (0, 1),
            id="quality-inflated-denominator",
        ),
        pytest.param(
            "speed-short",
            {"first_answer_task": "unlisted"},
            {"common_fault": {"first_answer_task": "unlisted"}},
            "invalid",
            None,
            set(),
            None,
            id="quality-wrong-membership",
        ),
        pytest.param(
            "learning-medium",
            {"initial_unscored": 16},
            {},
            "fail",
            "regression",
            {"test_panel/initial/panel_scored_fraction", "test_panel/initial/grader_error_fraction"},
            None,
            id="initial-quality-failure",
        ),
        ("speed-medium", {"missing_receipt": True}, {}, "invalid", None, set(), None),
        ("speed-medium", {"unresolved_receipt": True}, {}, "invalid", None, set(), None),
        pytest.param(
            "speed-short", {"score": 0.2}, {}, "fail", "regression", set(), None, id="quality-score-regression"
        ),
        pytest.param(
            "learning-medium",
            {},
            {"fault": {"panel_identity": None}},
            "invalid",
            None,
            set(),
            None,
            id="control-panel-identity",
        ),
    ],
)
def test_rescore_reports_candidate_and_control_contracts(
    recorded_runs, monkeypatch, preset, fault, control_fault, verdict, label, failed, bounds
):
    if preset == "speed-long":
        monkeypatch.setitem(gate.CONTROLS, preset, (gate.ControlPin(preset, "09026fd100", presets.PRESET_REVISION, 4),))
    controls(
        recorded_runs,
        control_fault.get("preset", preset),
        **{key: value for key, value in control_fault.items() if key != "preset"},
    )
    recorded_fault = {key: value for key, value in fault.items() if key != "missing_final_panel"}
    source = recorded_runs(preset, f"users/test/{preset}", **({"score": 0.495} | recorded_fault))
    if fault.get("missing_final_panel"):
        shutil.rmtree(source / "exports/test_panel")
    report = rescore(
        preset,
        "--candidate",
        f"users/test/{preset}" + ("@" + fault["version"] if "version" in fault else ""),
        exit_code=4 if fault.get("unresolved_receipt") or fault.get("missing_receipt") else None,
    )
    if fault.get("unresolved_receipt") or fault.get("missing_receipt"):
        assert set(report) == {"preset", "verdict", "reason"}
        assert report["preset"] == preset and report["verdict"] == "invalid"
        if fault.get("missing_receipt"):
            assert report["reason"] == f"no launch receipt at {source}"
        else:
            assert report["reason"].startswith(f"cannot recover recorded command options for {source}:")
        return
    assert (report["verdict"], report["label"]) == (verdict, label), report
    if fault.get("first_answer_task") == "unlisted":
        assert "absent from its task manifest" in report["reason"]
    if "panel_identity" in control_fault.get("fault", {}):
        assert "recorded panel version is missing or malformed" in report["reason"]
    assert {row["name"] for row in report["invariants"] if not row["holds"]} == failed, report
    error_name = "test_panel/grader_error_fraction" if preset == "speed-short" else "grader_error_fraction"
    error = next((row for row in report["invariants"] if row["name"] == error_name), {})
    assert bounds is None or (error["lower_bound"], error["upper_bound"]) == bounds
    assert Path(report["report"]).with_suffix(".md").exists()
    if preset == "speed-short" and report["guards"]:
        throughput = next(row for row in report["guards"] if row["metric"] == "generate/tokens_per_second_per_gpu")
        assert throughput["passes"]
        fixed = next(row for row in report["invariants"] if row["name"] == "fixed_reply_tokens")
        assert fixed["holds"]
    if preset == "speed-long":
        assert {row["metric"] for row in report["guards"]} == {
            "speed/update_seconds",
            "async/performance/loss_tokens_per_configured_policy_gpu_second",
            "async/performance/response_tokens_per_configured_inference_gpu_second",
            "policy/mismatch/staleness0/kl_k3",
        }
        assert any(row["metric"] == "policy/mismatch/pooled/kl_k3" for row in report["reported"])
    if "version" in fault:
        assert fault["version"] in report["reason"] and presets.PRESET_REVISION in report["reason"]
    if "record_version" in control_fault.get("fault", {}):
        assert control_fault["fault"]["record_version"] in report["reason"]
        assert presets.PRESET_REVISION in report["reason"]
    if "panel_identity" in fault:
        if fault["panel_identity"] is None:
            assert (
                "recorded panel version is missing or malformed" in report["reason"] and "identity" in report["reason"]
            )
        else:
            assert presets.EVALUATIONS["test_panel"].source in report["reason"]
        if fault["panel_identity"]:
            assert fault["panel_identity"] in report["reason"]
    elif verdict != "invalid" and preset not in ("speed-medium", "speed-long"):
        expected_panel = (
            f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}"
            if preset == "data-short"
            else presets.EVALUATIONS["test_panel"].source
        )
        assert report["run"]["panel"] == report["control"]["panel"] == expected_panel


def test_detail_rows_use_table_counts_and_initializer_scores(recorded_runs, selection, tmp_path):
    panel = Path(presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path())
    summary = json.loads((panel / "summary.json").read_text())
    _json(panel / "summary.json", summary | {"full_validation_rows": 6})
    tasks = [
        {"task_id": f"task-{i}", "family": "ab"[i // 3], "aaii_category": "xy"[i // 3], "agentic": bool(i % 2)}
        for i in range(6)
    ]
    pq.write_table(pa.Table.from_pylist(tasks), panel / presets.TASK_MANIFEST_FILENAME)
    path = Path(summary["recipe"]["pass_rates"]["uri"])
    pq.write_table(pq.read_table(path).slice(0, 6), path)
    for i in range(4):
        pin = gate.CONTROLS["learning-medium"][0]
        recorded_runs(
            "learning-medium",
            presets.control_name(pin.preset, pin.main_commit, i),
            score=0.49 + 0.01 * i,
            metrics={"generate/std_num_tokens": 100 + i, "reward/domain/shared/avg_raw_reward": 0.1 * i},
        )
    answers = [_answer(f"task-{i}", value) for i, value in enumerate((0.2, 0.4, 0.8, 0.1, 0.5, 0.9))]
    recorded_runs(
        "learning-medium",
        "users/test/detail",
        answers=answers,
        metrics={
            "reward/domain/candidate-only/avg_raw_reward": 1,
            "reward/domain/shared/avg_raw_reward": 1,
            "async/rejected_rate": 0.5,
            "generate/std_num_tokens": 50,
        },
    )
    report = rescore("learning-medium", "--candidate", "users/test/detail")
    rows = {row["metric"]: row for row in report["reported"]}
    for dimension, category in (("aaii_category", "x"), ("family", "a")):
        for band, value, initial in (("hard", 0.2, 0.05), ("learnable", 0.4, 0.5), ("easy", 0.8, 0.95)):
            cell = rows[f"test_panel/{dimension}/{category}/difficulty/{band}/mean"]
            assert tuple(
                cell[key] for key in ("run", "initializer", "tasks", "control_min", "control_max")
            ) == pytest.approx((value, initial, 1, 0.49, 0.52))
            assert cell["tolerance"] == pytest.approx(
                t.ppf(1 - 0.025 / 6, 3) * math.sqrt(5 / 3) * 0.01 * math.sqrt(1.25)
            )
            assert cell["change"] == ("better" if band == "easy" else "worse")
    agentic = rows["test_panel/agentic/true/mean"]
    assert (agentic["run"], agentic["initializer"], agentic["tasks"]) == pytest.approx((1.4 / 3, 0.5, 3))
    assert rows["test_panel/aaii_category/x/mean"]["run"] == pytest.approx(1.4 / 3)
    assert agentic["tolerance"] == pytest.approx(t.ppf(1 - 0.025 / 5, 3) * math.sqrt(5 / 3) * 0.01 * math.sqrt(1.25))
    # A declared metric keeps k = 1 inside a breakdown table; constant controls leave a detail row unavailable.
    assert rows["test_panel/mean"]["tolerance"] == pytest.approx(
        t.ppf(0.975, 3) * math.sqrt(5 / 3) * 0.01 * math.sqrt(1.25)
    )
    assert rows["test_panel/scored_fraction"]["change"] == "unavailable"
    recorded_runs("learning-medium", "users/test/initial", answers=answers, initial_score=0.3)
    initial = {row["metric"]: row for row in rescore("learning-medium", "--candidate", "users/test/initial")["reported"]}
    for key in ("test_panel/agentic/true/mean", "test_panel/family/a/mean", "test_panel/difficulty/hard/mean"):
        assert initial[key]["tolerance"] == pytest.approx(rows[key]["tolerance"])
    assert initial["test_panel/mean/initial"]["change"] == "unavailable"
    assert (report["verdict"], report["label"]) == ("pass", "no change")
    assert rows["generate/std_num_tokens"]["change"] == "better"
    assert rows["reward/domain/shared/avg_raw_reward"]["change"] == "better"
    assert not {"reward/domain/candidate-only/avg_raw_reward", "async/rejected_rate"} & rows.keys()
    assert not {"test_panel/family_mean", "test_panel/mixed_fraction", "test_panel/all_wrong_fraction"} & rows.keys()
    partial = [
        _answer(f"task-{i}", value, error="Unavailable" if i == 0 else None)
        for i, value in enumerate((0.2, 0.4, 0.8, 0.1, 0.5, 0.9))
    ]
    first = recorded_runs("learning-medium", "users/test/partial", answers=partial)
    second = recorded_runs("learning-medium", "users/test/complete", answers=answers)
    candidates = tuple(
        ArtifactStep.adopt("users/test/" + path.parent.name, presets.PRESET_REVISION, str(path), kind=SkyRLRun)
        for path in (first, second)
    )
    multi = run(gate.gate_step(presets.PRESETS["learning-medium"], candidates, inputs=gate.GateInputs()))[0]
    category = next(row for row in multi.reported if row.metric == "test_panel/aaii_category/x/mean")
    assert category.tasks is None and category.initializer == pytest.approx(0.6125)
    assert category.run == pytest.approx(8 / 15)
    markdown = Path(report["report"]).with_suffix(".md").read_text()
    assert "## Breakdown" in markdown and "### test_panel: aaii_category x difficulty" in markdown
    wrong_rates = pq.read_table(path).to_pylist()
    wrong_rates[0]["task_id"] = "different-panel"
    pq.write_table(pa.Table.from_pylist(wrong_rates), path)
    recorded_runs("learning-medium", "users/test/wrong-rates", answers=answers)
    invalid_rates = rescore("learning-medium", "--candidate", "users/test/wrong-rates")
    assert invalid_rates["verdict"] == "invalid" and "pass-rate panel" in invalid_rates["reason"]


def test_requirement_preflight_accepts_only_keys_the_gate_computes(recorded_runs):
    for name in ("speed-short", "speed-medium", "learning-medium"):
        controls(recorded_runs, name)
    speed, learning = presets.PRESETS["speed-medium"], presets.PRESETS["learning-medium"]
    gate.validate_candidate(speed, ["speed/update_seconds<=120", VLLM + "<=control"], control=False)
    gate.validate_candidate(
        learning,
        ["test_panel/aaii_category/coding/mean>=control", "test_panel/family/large/difficulty/hard/mean>=0"],
        control=False,
    )
    for preset, requirement, kwargs in (
        (presets.PRESETS["speed-short"], "typo>=0", {}),
        (learning, "test_panel/aaii_category/codng/mean>=control", {}),
        # Every fixture task is non-agentic, so the panel has no agentic split.
        (learning, "test_panel/agentic/false/mean>=0", {}),
    ):
        with pytest.raises(click.BadParameter, match="does not compute"):
            snowball.build_run(preset, requirements=(requirement,), **kwargs)


def test_requirements_enforce_each_replicate_and_calibrate_added_guards(recorded_runs):
    key = "policy/mismatch/staleness0/log_ratio_abs_max"
    for i in range(4):
        pin = gate.CONTROLS["speed-medium"][0]
        recorded_runs(
            "speed-medium", presets.control_name(pin.preset, pin.main_commit, i), speed=100 + i, metrics={key: i * 0.001}
        )
    for value, expected in ((0, "pass"), (0.01, "fail")):
        recorded_runs(
            "speed-medium", f"users/test/required-{value}", metrics={key: value}, argv=["--require", key + "<=0"]
        )
        report = rescore("speed-medium", "--candidate", f"users/test/required-{value}")
        assert report["verdict"] == expected
        assert [(row["value"], row["holds"]) for row in report["invariants"] if row["name"] == "require/" + key] == [
            (value, value == 0)
        ]
    candidates = tuple(
        ArtifactStep.adopt(
            f"users/test/required-{value}",
            presets.PRESET_REVISION,
            str(Path(presets.artifact_prefix("east")) / f"users/test/required-{value}" / presets.PRESET_REVISION),
            kind=SkyRLRun,
        )
        for value in (0, 0.01)
    )
    multi = run(
        gate.gate_step(presets.PRESETS["speed-medium"], candidates, inputs=gate.GateInputs(requirements=(key + "<=0",)))
    )[0]
    assert multi.verdict == "fail"
    assert [(row.replicate, row.holds) for row in multi.invariants if row.name == "require/" + key] == [
        (0, True),
        (1, False),
    ]
    relative = rescore("speed-medium", "--candidate", "users/test/required-0.01", "--require", key + "<=control")
    row = relative["guards"][-1]
    assert row["tolerance"] == pytest.approx(t.ppf(1 - 0.025 / 3, 3) * math.sqrt(5 / 3) * 0.001 * math.sqrt(1.25))
    assert (row["metric"], row["higher_is_better"], row["passes"]) == (key, False, False)
    missing = rescore("speed-medium", "--candidate", "users/test/required-0.01", "--require", "absent>=1")
    assert missing["verdict"] == "invalid" and "absent" in missing["reason"]
    zero = rescore("speed-medium", "--candidate", "users/test/required-0.01", "--require", VLLM + ">=control")
    assert zero["verdict"] == "invalid" and "use a number bound" in zero["reason"]
    promoted = rescore(
        "speed-medium", "--candidate", "users/test/required-0.01", "--require", "startup/init_weight_sync_state<=control"
    )
    assert not {row["metric"] for row in promoted["guards"]} & {row["metric"] for row in promoted["reported"]}
    controls(recorded_runs, "learning-medium")
    recorded_runs("learning-medium", "users/test/repeated", score=0.45)
    duplicate = rescore(
        "learning-medium",
        "--candidate",
        "users/test/repeated",
        "--require",
        "test_panel/family_mean>=control",
        "--require",
        "test_panel/family_mean>=control",
    )
    assert duplicate["verdict"] == "fail" and len(duplicate["guards"]) == 1


def test_update_guard_alone_sets_improves(recorded_runs):
    pin = gate.CONTROLS["speed-medium"][0]
    for i, phase in enumerate((0, 200, 0, 200)):
        recorded_runs(
            "speed-medium",
            presets.control_name(pin.preset, pin.main_commit, i),
            speed=100 + i,
            metrics={
                "timing/step_wall/group_admission": phase,
                "timing/step_wall/weight_sync": 300 + i - phase,
                "async/performance/loss_tokens_per_configured_policy_gpu_second": 100 + i,
            },
        )
    recorded_runs(
        "speed-medium",
        "users/test/faster",
        metrics={
            "timing/step_wall/group_admission": 130,
            "timing/step_wall/weight_sync": 130,
            "async/performance/loss_tokens_per_configured_policy_gpu_second": 10,
        },
    )
    report = rescore("speed-medium", "--candidate", "users/test/faster")
    assert (report["verdict"], report["label"], report["guards"][0]["change"]) == ("pass", "improves", "better")
    assert all(
        row["change"] in ("same", "unavailable")
        for row in report["reported"]
        if not row["metric"].startswith("async/performance/")
    )
    assert next(row["change"] for row in report["reported"] if row["metric"].startswith("async/performance/")) == "worse"


REQUIREMENT = "policy/raw_grad_norm<=1"


@pytest.mark.timeout(240)
@pytest.mark.parametrize("preset", ["learning-medium", "data-medium"])
def test_built_gate_and_rescore_share_inputs_and_status(recorded_runs, tmp_path, preset, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "planned-team")
    controls(recorded_runs, preset)
    if preset == "learning-medium":
        init = ArtifactStep.adopt(
            "users/test/initializer", presets.PRESET_REVISION, str(tmp_path / "initializer"), kind=LevanterCheckpoint
        )
        _json(
            Path(init.path()) / RECORD_FILENAME,
            ArtifactRecord(result_type=result_type_name(LevanterCheckpoint)).model_dump(mode="json"),
        )
        run(init)
        settings = (
            "trainer.policy.optimizer_config.lr=2e-6",
            "trainer.algorithm={group_advantage_min_size: null, advantage_estimator: grpo}",
            "trainer.policy.optimizer_config.lr=1.0e-6",
        )
        built = snowball.build_run(
            presets.PRESETS[preset],
            init=init,
            settings=tuple(map(presets.parse_setting, settings)),
            requirements=(REQUIREMENT,),
        )
        argv = [
            "--require",
            REQUIREMENT,
            "--init",
            "users/test/unused@2026.10.27",
            "--init",
            f"{init.name}@{init.version}",
            *(item for value in settings for item in ("--set", value)),
        ]
        algorithm = {"trainer.algorithm.advantage_estimator": "grpo", "trainer.algorithm.group_advantage_min_size": None}
        expected = (
            algorithm | {"trainer.policy.optimizer_config.lr": 1e-6},
            None,
            f"{init.name}@{init.version}",
            [pytest.approx(0.295)],
        )
    else:
        tasks = f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}"
        built = snowball.build_run(
            presets.PRESETS[preset], tasks=presets.selection_artifact(tasks), requirements=(REQUIREMENT,)
        )
        argv = ["--require", REQUIREMENT, "--tasks", f"unused@{presets.DEFAULT_SELECTION_VERSION}", "--tasks", tasks]
        expected = ({}, tasks, None, [pytest.approx(0.195)])
    assert status(preset, argv)["runs"][0]["wandb_entity"] == "planned-team"
    initial = 0.2 if preset == "learning-medium" else None
    timers = {
        f"timing/{phase}/{stat}": float(index)
        for phase in ("grading", "setup")
        for index, stat in enumerate(("mean", "p50", "p95", "max"), 1)
    }
    source = recorded_runs(
        preset,
        built.runs[0].name,
        score=0.495,
        initial_score=initial,
        argv=["snowball_rl", *argv],
        fingerprint=built.runs[0].fingerprint(),
        metrics=timers,
    )
    built = replace(built, gate=gate.completed_gate(presets.PRESETS[preset], built.runs, built.gate))
    report = run(built.gate)[0].model_dump(mode="json", exclude={"path"})
    timer_rows = {row["metric"]: row for row in report["reported"] if row["metric"] in timers}
    assert {key: row["run"] for key, row in timer_rows.items()} == timers
    assert all(row["control"] is None and row["change"] == "unavailable" for row in timer_rows.values())
    markdown = (Path(built.gate.path()) / gate.GATE_REPORT_FILENAME).with_suffix(".md").read_text()
    assert all(key in markdown for key in timers)
    objective_rows = {row["metric"]: row for row in report["objectives"]}
    gain = objective_rows["test_panel/family_mean/gain_per_h100_hour"]
    assert gain["run"] == pytest.approx((0.495 - (0.2 if preset == "learning-medium" else 0.3)) / 40)
    assert gain["control"] == pytest.approx((0.505 - 0.3) / 40)
    printed_report = status(preset, argv)
    gain_line = (
        "0.005125 -> 0.007375; +43.9%; noise +/-22.4%; better"
        if preset == "learning-medium"
        else "0.005125 -> 0.004875; -4.9%; noise +/-22.4%; same"
    )
    assert printed_report["objectives"] == [
        "test_panel/family_mean: 0.505 -> 0.495; -2.0%; noise +/-9.1%; same",
        "test_panel/family_mean/gain_per_h100_hour: " + gain_line,
    ]
    assert "r0 planned_updates: 25 (required 25); within bound" in printed_report["constraints"]
    assert (
        list(printed_report).index("objectives")
        < list(printed_report).index("constraints")
        < list(printed_report).index("result")
    )
    assert rescore(preset, "--candidate", built.runs[0].name)["run"] == report["run"]
    settings = {key: json.loads(value) for key, _, value in (item.partition("=") for item in report["run"]["settings"])}
    gains = [row["run"] for row in report["reported"] if row["metric"] == "test_panel/mean/gain"]
    assert (settings, report["run"]["tasks"], report["run"]["init"], gains) == expected
    assert [(row["name"], row["holds"]) for row in report["invariants"] if row["name"].startswith("require/")] == [
        ("require/" + REQUIREMENT.split("<=")[0], True)
    ]
    assert "wandb_url" not in status(preset, argv)["runs"][0]
    assert status(preset, argv)["runs"][0]["wandb_entity"] == snowball.DEFAULT_WANDB_ENTITY
    _json(
        source / "exports/attempts/recorded-attempt/tracking.json",
        {
            "attempt_id": "recorded-attempt",
            "wandb_url": "https://wandb.example/team/project/runs/actual-id",
            "vllm_version": "fixture-vllm",
        },
    )
    _json(source / "exports/metrics/step-000026.json", {"step": 26, "startup/load_checkpoints": 3.0})
    _json(source / "exports/metrics/step-000028.json", {"step": 28, "trainer/global_step": 28})
    printed = status(preset, argv)
    assert printed["runs"][0]["wandb_url"] == "https://wandb.example/team/project/runs/actual-id"
    assert printed["runs"][0]["latest_completed_update"] == 25
    receipt = json.loads((source / "terminal.json").read_text())
    linked_receipt = receipt | {
        "config": {
            "run": {**receipt["config"]["run"], "id": "run / id?x=1&y=2"},
            "runtime": {"training_type": "sync"},
            "iris": {"wandb_entity": "receipt-team"},
        }
    }
    _json(source / "terminal.json", linked_receipt)
    started, finished = 1793840000000, 1793840060000
    linked_receipt["result"] = {
        **linked_receipt["result"],
        "started_at": Timestamp.from_ms(started).as_formatted_date(),
        "finished_at": Timestamp.from_ms(finished).as_formatted_date(),
    }
    _json(source / "terminal.json", linked_receipt)
    monkeypatch.setattr(
        "marin.rl.cli.iris_client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("status must use saved timing")),
    )
    linked = status(preset, argv)["runs"][0]
    assert linked["wandb_entity"] == "receipt-team"
    links = linked["grafana_links"]
    query = parse_qs(urlsplit(links["run"]).query)
    assert query["var-run"] == ["run / id?x=1&y=2"]
    assert query["from"] == [str(started)] and query["to"] == [str(finished)]
    linked_receipt["result"]["started_at"] = None
    _json(source / "terminal.json", linked_receipt)
    assert "grafana_links" not in status(preset, argv)["runs"][0]
    _json(source / "terminal.json", receipt)
    for step in (26, 28):
        (source / f"exports/metrics/step-{step:06d}.json").unlink()
    assert timers.keys() <= {row["key"] for row in printed["reported_metrics"]}
    assert (printed["verdict"], printed["label"]) == (report["verdict"], report["label"])
    assert printed["report"].endswith("gate_report.json") and "rl_path_changes_since_control" in printed
    recorded_runs(preset, "users/test/earlier-" + preset, score=0.49, argv=("--preset", preset))
    baseline = f"users/test/earlier-{preset}@{presets.PRESET_REVISION}"
    with_baseline = status(preset, [*argv, "--baseline", baseline])
    assert with_baseline["runs"][0]["name"] == printed["runs"][0]["name"]
    assert "report" not in with_baseline
    receipt = json.loads((source / "terminal.json").read_text())
    receipt["result"]["provenance"]["command_line"] += ["--baseline", baseline]
    _json(source / "terminal.json", receipt)
    recovered = rescore(preset, "--candidate", built.runs[0].name)
    assert recovered["run"]["baselines"] == [baseline] and recovered["vs_baseline"]
    assert (recovered["verdict"], recovered["label"]) == (report["verdict"], report["label"])
    Path(printed["report"]).unlink()
    terminal = json.loads((source / "terminal.json").read_text())
    _json(source / "attempts/failed.json", terminal | {"result": terminal["result"] | {"state": "failed"}})
    (source / "terminal.json").unlink()
    launch = json.loads((source / "resolved-launch.yaml").read_text())["config"]
    _json(
        source / "resolved-launch.yaml",
        {"config": launch | {"artifacts": {"attempts_root": str(source / "attempts")}, "run": {"attempt_id": "failed"}}},
    )
    printed = status(preset, argv)
    assert (printed["verdict"], printed["reason"], printed["iris_job_id"]) == (
        "invalid",
        "native job /tests/native-run is failed",
        "/tests/native-run",
    )


@pytest.mark.parametrize("preset", ["speed-long", "speed-medium", "data-long"])
def test_unpinned_candidates_refuse_gpu_submission(recorded_runs, monkeypatch, preset):
    if preset == "data-long":
        monkeypatch.delitem(gate.CONTROLS, "learning-long")
    if preset == "speed-medium":
        monkeypatch.setitem(
            gate.CONTROLS,
            preset,
            tuple(replace(pin, revision="2026.10.27") for pin in gate.CONTROLS[preset]),
        )
    client = MagicMock(spec=IrisClient)
    client.submit.side_effect = AssertionError("an unpinned candidate submitted a coordinator")
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    for command, arguments in (
        (
            snowball.main,
            [
                "--preset",
                preset,
                "--version",
                presets.PRESET_REVISION,
                "--run",
                "--wait",
                *(["--tasks", f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}"] if preset == "data-long" else []),
            ],
        ),
        (gate.main, ["--preset", preset, "--candidate", "tests/unpinned"]),
    ):
        refused = CliRunner().invoke(command, arguments)
        assert refused.exit_code == 4, (refused.output, refused.exception)
        payload = json.loads(refused.output)
        assert payload["preset"] == preset and payload["verdict"] == "invalid"
        control_preset = gate.SHARED_CONTROLS.get(preset, preset)
        assert payload["reason"].endswith(
            f"python -m experiments.post_training.snowball_rl --preset {control_preset} "
            f"--version {presets.PRESET_REVISION} --control --replicates 4 --run"
        )
        assert presets.PRESET_REVISION in payload["reason"]
        if preset == "speed-medium":
            assert "2026.10.27" in payload["reason"]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("pass", 0),
        ("fail", 1),
        ("invalid", 2),
        ("missing", 3),
        ("parent-early", 3),
        ("native-failed", 2),
        ("no-wait", 0),
        ("child", 0),
        ("control", 0),
        ("generate", 0),
    ],
)
def test_submission_waits_for_the_coordinator_and_returns_the_verdict(recorded_runs, monkeypatch, case, expected):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.setenv("WANDB_ENTITY", "submission-team")
    preset_name = "speed-short" if case == "generate" else "speed-medium"
    controls(recorded_runs, preset_name)
    preset = presets.PRESETS[preset_name]
    built = snowball.build_run(preset, control=case == "control")
    source = recorded_runs(
        preset,
        built.runs[0].name,
        speed=150 if case in {"fail", "no-wait"} else 101,
        fingerprint=built.runs[0].fingerprint(),
    )
    if case == "invalid":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["provenance"] = None
        _json(source / "terminal.json", receipt)
    built = replace(built, gate=gate.completed_gate(preset, built.runs, built.gate))
    if case not in {"missing", "control", "native-failed"}:
        run(built.gate)
    if case == "parent-early":
        shutil.rmtree(source)
        shutil.rmtree(built.gate.path())
    job = JobName.from_wire("/atqamar/coordinator-id-returned-by-iris")
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = job
    if case == "native-failed":
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["state"] = "failed"
        _json(source / "terminal.json", receipt)
    final = JobState.FAILED if case in {"parent-early", "native-failed"} else JobState.SUCCEEDED
    client.job.return_value.wait.return_value = SimpleNamespace(state=final)
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr("marin.rl.cli.iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    if case == "child":
        monkeypatch.setenv("IRIS_TASK_ID", "/atqamar/cached-coordinator/0")
        client.submit.side_effect = AssertionError("in-job execution submitted a coordinator")
    arguments = ["--preset", preset_name, "--version", presets.PRESET_REVISION, "--run"]
    if case != "no-wait":
        arguments.append("--wait")
    if case == "control":
        arguments.append("--control")
    try:
        result = CliRunner().invoke(snowball.main, arguments)
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    waited = [] if case in {"no-wait", "child"} else [(job,)]
    assert [call.args for call in client.job.call_args_list] == waited
    if case != "child":
        assert client.submit.call_args.kwargs["environment"].env_vars["WANDB_ENTITY"] == "submission-team"
    if case in {"pass", "fail", "invalid"}:
        assert "# speed-medium" in result.output
    if case == "native-failed":
        assert "result: invalid" in result.output and "native job /tests/native-run is failed" in result.output
        assert (
            rescore(preset_name, "--candidate", built.runs[0].name)["reason"] == "native job /tests/native-run is failed"
        )
    if case == "generate":
        snapshot = status(preset_name, [])
        assert snapshot["runs"][0]["latest_completed_update"] == 0
        assert {
            f"timing/{phase}/{stat}" for phase in ("grading", "setup") for stat in ("mean", "p50", "p95", "max")
        } <= {row["key"] for row in snapshot["reported_metrics"]}


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
                "inputs": {
                    "validation_data": [{"uri": str(tmp_path / "selection")}],
                }
            }
        },
    )
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "exports/dumped_evals" / session
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


def test_grafana_links_reference_dashboard_panels_and_variables():
    dashboard = json.loads((presets.REPO_ROOT / "infra/grafana/dashboards/rl_runs.json").read_text())
    variables = {value["name"] for value in dashboard["templating"]["list"]}
    panels, pending = set(), list(dashboard["panels"])
    while pending:
        panel = pending.pop()
        panels.add(panel["id"])
        pending.extend(panel.get("panels", []))
    for preset in presets.PRESETS.values():
        links = presets.grafana_links(
            preset,
            run_id="run / id?x=1&y=2",
            cluster="cw-us-east-02a",
            training_type="sync",
            started_ms=1793840000000,
            finished_ms=None,
        )
        for url in links.values():
            query = parse_qs(urlsplit(url).query)
            assert {key.removeprefix("var-") for key in query if key.startswith("var-")} <= variables
            assert all(int(value) in panels for value in query.get("viewPanel", ()))


@pytest.mark.parametrize(
    "change,verdict",
    [
        ("training", "pass"),
        ("attention", "pass"),
        ("prefix", "pass"),
        ("vllm-env", "pass"),
        ("vllm-version", "pass"),
        ("sampling", "invalid"),
        ("named-sampling", "invalid"),
        ("reply-limit", "invalid"),
        ("chat-template", "invalid"),
        ("task-env", "invalid"),
    ],
)
def test_evaluation_protocol_compares_training_and_engine_changes_and_binds_requests(recorded_runs, change, verdict):
    preset = "learning-long"
    controls(recorded_runs, preset)
    name = "users/test/protocol-" + change
    source = recorded_runs(preset, name)
    document = json.loads((source / "terminal.json").read_text())
    config = document["config"]["skyrl"]
    generator = config["generator"]
    if change == "training":
        config["trainer"]["policy"]["optimizer_config"]["lr"] = 2e-6
        config["trainer"]["algorithm"]["advantage_estimator"] = "grpo"
        generator["sampling_params"] = {"temperature": 0.7}
        generator["trajectory_reward_shaping"] = {"enabled": True}
        generator["weight_sync_backend"] = "nccl"
    elif change == "attention":
        generator["vllm_attention_backend"] = "another-backend"
    elif change == "prefix":
        generator["enable_prefix_caching"] = False
    elif change == "vllm-env":
        document["config"]["runtime"]["task_env"]["VLLM_PROBE"] = "1"
    elif change == "vllm-version":
        _json(
            source / "exports/attempts/recorded-attempt/tracking.json",
            {"attempt_id": "recorded-attempt", "engine": {"vllm_version": "another-wheel"}},
        )
    elif change == "sampling":
        generator["eval_sampling_params"]["temperature"] = 0.5
    elif change == "named-sampling":
        config["trainer"]["callbacks"][0]["additional_evaluations"]["test_panel"]["sampling_params"]["temperature"] = 0.5
    elif change == "reply-limit":
        generator["eval_sampling_params"]["max_generate_length"] = 4096
    elif change == "chat-template":
        generator["chat_template"] = "another-template"
    else:
        document["config"]["runtime"]["task_env"]["TASK_PROBE"] = "1"
    _json(source / "terminal.json", document)
    _json(source / "resolved-launch.yaml", {"config": document["config"]})
    report = rescore(preset, "--candidate", name)
    assert report["verdict"] == verdict, report
    if verdict == "invalid":
        assert "evaluation protocol" in report["reason"]


@pytest.mark.timeout(180)
def test_gate_compares_natively_composed_optimizer_and_initializer_changes(recorded_runs, native_json, tmp_path):
    preset = presets.PRESETS["learning-medium"]
    control_roots = controls(recorded_runs, preset.name)
    initializer = ArtifactStep.adopt(
        "tests/custom-init", "2026.11.07", str(tmp_path / "custom-init"), kind=LevanterCheckpoint
    )
    builds = [
        snowball.build_run(preset, control=True),
        snowball.build_run(preset, settings=(presets.parse_setting("trainer.policy.optimizer_config.lr=2e-6"),)),
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
        roots = control_roots if index == 0 else [recorded_runs(preset, f"users/test/native-protocol-{index}")]
        for source in roots:
            document = json.loads((source / "terminal.json").read_text())
            document["config"].update(skyrl=config["skyrl"], runtime=config["runtime"])
            _json(source / "terminal.json", document)
            _json(source / "resolved-launch.yaml", {"config": document["config"]})
    for index in [1, 2]:
        report = rescore(preset.name, "--candidate", f"users/test/native-protocol-{index}")
        assert (report["verdict"], report["label"]) == ("pass", "no change"), report


def test_task_set_identity_uses_contents_and_ignores_row_order_and_artifact_name(recorded_runs):
    preset = "learning-medium"
    controls(recorded_runs, preset)
    source = recorded_runs(preset, "users/test/task-content")
    document = json.loads((source / "terminal.json").read_text())
    descriptor = document["config"]["inputs"]["validation_data"][0]
    copied = source / "panel-copy"
    shutil.copytree(descriptor["uri"], copied)
    original = scoring.task_set_sha256(document["config"], "test_panel")
    descriptor["uri"] = str(copied)
    path = copied / descriptor["relative_path"]
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), path)
    assert scoring.task_set_sha256(document["config"], "test_panel") == original
    _json(source / "terminal.json", document)
    _json(source / "resolved-launch.yaml", {"config": document["config"]})
    assert rescore(preset, "--candidate", "users/test/task-content")["verdict"] == "pass"
    changed_source = recorded_runs(preset, "users/test/task-content-changed")
    changed_document = json.loads((changed_source / "terminal.json").read_text())
    changed_document["config"]["inputs"]["validation_data"][0]["uri"] = str(copied)
    _json(changed_source / "terminal.json", changed_document)
    _json(changed_source / "resolved-launch.yaml", {"config": changed_document["config"]})
    rows[0]["task_spec"] = json.dumps(json.loads(rows[0]["task_spec"]), indent=2)
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert scoring.task_set_sha256(document["config"], "test_panel") != original
    task = json.loads(rows[0]["task_spec"])
    task["context"]["events"][0]["content"] = "Return dog."
    rows[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert scoring.task_set_sha256(document["config"], "test_panel") != original
    report = rescore(preset, "--candidate", "users/test/task-content-changed")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


def test_issue_metadata_stays_in_launcher_status(recorded_runs):
    preset = presets.PRESETS["learning-medium"]
    controls(recorded_runs, preset.name)
    first = snowball.build_run(preset).runs[0]
    before = status(preset.name, ["--issue", "73"])["runs"][0]
    after = status(preset.name, ["--issue", "74"])["runs"][0]
    assert (before["name"], before["version"]) == (after["name"], after["version"]) == (first.name, first.version)
    assert (before["issue"], after["issue"]) == (73, 74)


@pytest.mark.parametrize("change,verdict", [("engine", "pass"), ("reply-limit", "invalid"), ("task-env", "invalid")])
def test_speed_short_compares_engine_changes_and_binds_the_evaluation_request(recorded_runs, change, verdict):
    preset = "speed-short"
    controls(recorded_runs, preset)
    source = recorded_runs(preset, "users/test/request-" + change)
    document = json.loads((source / "terminal.json").read_text())
    config = document["config"]
    if change == "engine":
        config["skyrl"]["generator"]["max_num_batched_tokens"] = 16384
        config["runtime"]["task_env"]["VLLM_LOGGING_LEVEL"] = "INFO"
        _json(
            source / "exports/attempts/recorded-attempt/tracking.json",
            {"attempt_id": "recorded-attempt", "engine": {"vllm_version": "another-wheel"}},
        )
    elif change == "reply-limit":
        config["skyrl"]["trainer"]["callbacks"][0]["additional_evaluations"]["test_panel"]["sampling_params"][
            "max_generate_length"
        ] = 4096
    else:
        config["runtime"]["task_env"]["NCCL_NET"] = "another-net"
    _json(source / "terminal.json", document)
    _json(source / "resolved-launch.yaml", {"config": config})
    report = rescore(preset, "--candidate", "users/test/request-" + change)
    assert report["verdict"] == verdict, report
    if verdict == "invalid":
        assert "evaluation protocol" in report["reason"]
    else:
        assert report["run"]["engines"] == [{"vllm_version": "another-wheel"}]


@pytest.mark.parametrize(
    "arguments", [["--unknown"], ["--preset", "speed-medium", "--set", "trainer.seed=3"], ["--issue", "0"]]
)
def test_refusals_exit_four_before_submission(recorded_runs, arguments):
    result = CliRunner().invoke(
        snowball.main, ["--preset", "speed-medium", "--version", presets.PRESET_REVISION, *arguments]
    )
    assert result.exit_code == 4, (result.output, result.exception)


def test_result_prints_the_verdict_before_a_tradeoff_label(recorded_runs):
    controls(recorded_runs, "speed-short")
    built = snowball.build_run(presets.PRESETS["speed-short"])
    recorded_runs("speed-short", built.runs[0].name, speed=120, score=0.1, fingerprint=built.runs[0].fingerprint())
    built = replace(built, gate=gate.completed_gate(presets.PRESETS["speed-short"], built.runs, built.gate))
    report = run(built.gate)[0]
    assert (report.verdict, report.label) == ("fail", "trade-off")
    assert status("speed-short", [])["result"] == "fail (trade-off)"
    markdown = (Path(report.path) / "gate_report.md").read_text()
    assert "Result: fail (trade-off)" in markdown
    assert "Verdict:" not in markdown and "Label:" not in markdown
