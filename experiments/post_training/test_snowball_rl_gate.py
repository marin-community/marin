# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from scipy.stats import t
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_pass_rates
from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring

pytest_plugins = ("experiments.post_training.snowball_rl_test_fixtures",)
pytestmark = pytest.mark.usefixtures("source_provenance")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def recorded_runs(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    panel = Path(presets.default_selection().path())
    tasks = [
        {
            "task_id": f"task-{i}",
            "family": "small" if i == 0 else "large",
            "aaii_category": "format" if i == 0 else "coding",
        }
        for i in range(910)
    ]
    pq.write_table(pa.Table.from_pylist(tasks), panel / presets.TASK_MANIFEST_FILENAME)
    rates = tmp_path / scoring.PASS_RATES_NAME / presets.PRESET_REVISION
    rates.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": row["task_id"],
                    "model_revision": presets.MODEL_REVISION,
                    "passed": (0, 4, 8)[i % 3],
                    "samples": 8,
                }
                for i, row in enumerate(tasks)
            ]
        ),
        rates / "pass_rates.parquet",
    )

    def write(
        preset_name,
        name,
        *,
        speed=100.0,
        score=0.5,
        metrics=None,
        unscored=0,
        duplicate_task=False,
        provenance=None,
        argv=(),
        initial_score=None,
        initial_missing=False,
        omit_step=None,
        version=presets.PRESET_REVISION,
        fixed=True,
        panel_shift=False,
        mixed=128,
        detail=False,
        fingerprint="",
        answers=None,
    ):
        preset = presets.PRESETS[preset_name] if isinstance(preset_name, str) else preset_name
        root = tmp_path / name / version
        result = SkyRLRun(
            path=str(root),
            hf_model_uri=None,
            global_step=preset.updates,
            tokenizer_uri="tests/tokenizer",
            tokenizer_revision="test-revision",
            checkpoint_root=str(root / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(root / "terminal.json"),
            iris_job_id="/tests/native-run",
        )
        record = ArtifactRecord(
            name=name,
            version=version,
            fingerprint=fingerprint,
            result_type=result_type_name(SkyRLRun),
            result=result.result_payload(),
        )
        _json(root / RECORD_FILENAME, record.model_dump(mode="json"))
        _json(
            root / "terminal.json",
            {
                "config": {},
                "result": {
                    "state": "succeeded",
                    "iris_job_id": "/tests/native-run",
                    "run_id": f"{name}-{version}",
                    "provenance": {
                        "tree_hash": "native-tree",
                        "base_commit": gate.CONTROLS.get(preset.name, gate.CONTROLS["learning-medium"])[0].main_commit,
                        "dirty": False,
                        "command_line": list(argv),
                        **(provenance or {}),
                    },
                },
            },
        )
        (root / ".executor_status").write_text("SUCCESS")
        for step in ((0,) if preset.recipe.layout.role_plan.generate_only else range(1, preset.updates + 1)):
            if step == omit_step:
                continue
            row = {f"generate/outcome/{outcome.value}": 0.0 for outcome in Outcome}
            row.update(
                {
                    "generate/outcome/graded": 512.0,
                    "generate/outcome/rollout_failure": 0.0,
                    "loss/policy_loss": 0.01,
                    "policy/raw_grad_norm": 1.0,
                    "policy/mismatch/staleness0/kl_k3": speed / 1000,
                    "policy/mismatch/staleness0/selected_tokens": 1000,
                    "generate/min_num_tokens": 4096 if fixed else 12,
                    "generate/max_num_tokens": 4096,
                    "generate/tokens_per_second_per_gpu": speed,
                    "generate/prompts_dropped_too_long": 0,
                    "vllm/median_generation_throughput": 100.0,
                    "startup/init_weight_sync_state": speed / 10,
                    "startup/load_checkpoints": 4.0,
                }
            )
            row.update({metric.key: 0.0 for metric in preset.reported if metric.key.startswith("timing/step_wall/")})
            row.update(
                {
                    "timing/step_wall/group_admission": speed,
                    "timing/step_wall/evaluation": 700,
                    "timing/step_wall/checkpoint_work": 900,
                    **(metrics or {}),
                }
            )
            _json(root / f"exports/metrics/step-{step:06d}.json", row)
        manifest = panel
        if panel_shift:
            manifest = root / "panel"
            manifest.mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [{**row, "task_id": "shifted" if i == 0 else row["task_id"]} for i, row in enumerate(tasks)]
                ),
                manifest / presets.TASK_MANIFEST_FILENAME,
            )
        _json(root / "resolved-launch.yaml", {"config": {"inputs": {"validation_data": [{"uri": str(manifest)}]}}})
        for evaluation in preset.evaluations:
            count = evaluation.sample or 910
            for step, average in [
                (-1 if preset.updates == 0 else preset.updates, score),
                *([(0, initial_score)] if initial_score is not None else []),
            ]:
                if step == 0 and initial_missing:
                    continue
                session = "eval_only" if step == -1 else f"global_step_{step}_evals"
                path = root / "exports/dumped_evals" / session / "panel.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as stream:
                    if answers is not None:
                        stream.writelines(json.dumps(row) + "\n" for row in answers)
                        continue
                    for i in range(count):
                        identifier = f"train-task-{i}" if evaluation.file == "train" else f"task-{i}"
                        if panel_shift and i == 0:
                            identifier = "shifted"
                        for attempt in range(evaluation.answers):
                            value = (
                                float(i < mixed and attempt == 0)
                                if evaluation.file == "train"
                                else 1.0 if detail and i == 0 else average
                            )
                            tokens = [1] * (
                                preset.recipe.fixed_reply_tokens
                                if fixed
                                and preset.recipe.layout.role_plan.generate_only
                                and preset.recipe.fixed_reply_tokens
                                else 1
                            )
                            line = (
                                json.dumps(
                                    {
                                        "task_id": identifier,
                                        "response_token_ids": tokens,
                                        "exception_type": "GraderUnavailable" if i < unscored else None,
                                        "verification": {
                                            "status": "verified",
                                            "score": value,
                                            "score_min": 0,
                                            "score_max": 1,
                                        },
                                    }
                                )
                                + "\n"
                            )
                            stream.write(line)
                            if duplicate_task and i == 0 and attempt == 0:
                                stream.write(line)
        return root

    return write


def controls(write, preset, *, speed=100.0, constant=False, fault=None):
    pin = gate.CONTROLS[preset][0]
    return [
        write(
            preset,
            presets.control_name(pin.preset, pin.main_commit, i),
            speed=speed + (0 if constant else i),
            score=0.49 + 0.01 * i,
            mixed=128 + i,
            **(fault if i == 0 and fault else {}),
        )
        for i in range(pin.runs)
    ]


def rescore(preset, name):
    result = CliRunner().invoke(gate.main, ["--preset", preset, "--candidate", name])
    assert result.exit_code == 0, (result.output, result.exception)
    return json.loads(result.output.splitlines()[-1])


def test_calibrate_pools_deviations_within_pins():
    first = gate.ControlPin("speed-short", "first", presets.PRESET_REVISION, 3)
    second = gate.ControlPin("speed-short", "second", presets.PRESET_REVISION, 2)
    noise = gate.calibrate(
        {first: [{"throughput": x} for x in (1.0, 2.0, 3.0)], second: [{"throughput": x} for x in (10.0, 14.0)]}
    )["throughput"]
    assert (noise.mean, noise.df) == (6, 3)
    assert noise.sd == pytest.approx(math.sqrt(10 / 3))


@pytest.mark.parametrize("outlier", [False, True])
def test_rescore_leave_one_out_uses_median_and_two_guard_tolerance(recorded_runs, outlier, tmp_path):
    roots = controls(recorded_runs, "speed-medium")
    next(tmp_path.glob("rl-data/snowball-pass-rates/*/*/pass_rates.parquet")).unlink()
    if outlier:
        path = roots[0] / "exports/metrics/step-000002.json"
        row = json.loads(path.read_text())
        row["timing/step_wall/group_admission"] += 1000
        _json(path, row)
    before = {
        path: hashlib.sha256(path.read_bytes()).digest() for root in roots for path in root.rglob("*") if path.is_file()
    }
    result = CliRunner().invoke(gate.main, ["--preset", "speed-medium", "--leave-out", "r0"])
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads(result.output.splitlines()[-1])
    timing, mismatch = report["guards"]
    quantile = math.sqrt(2 * 0.975**2 / (1 - 0.975**2))
    assert (
        timing["run"],
        timing["control"],
        timing["sd"],
        timing["df"],
        timing["n_run"],
        timing["n_control"],
        timing["delta"],
    ) == (100, 102, 1, 2, 1, 3, 2)
    assert timing["tolerance"] == pytest.approx(quantile * math.sqrt(1 + 1 / 3))
    assert mismatch["delta"] == pytest.approx(0.002)
    assert (report["verdict"], report["label"]) == ("pass", "regression" if outlier else "no change")
    mean = next(row for row in report["reported"] if row["metric"] == "timing/step_wall/group_admission")
    assert mean["run"] == pytest.approx(100 + (1000 / 9 if outlier else 0))
    startup = next(row for row in report["reported"] if row["metric"] == "startup/init_weight_sync_state")
    assert (startup["run"], startup["control"]) == pytest.approx((10, 10.2))
    artifact = Path(report["report"])
    assert json.loads(artifact.read_text()) == {k: v for k, v in report.items() if k != "report"}
    assert artifact.with_suffix(".md").exists()
    pin = gate.CONTROLS["speed-medium"][0]
    ordinary = rescore("speed-medium", presets.control_name(pin.preset, pin.main_commit, 0))
    assert ordinary["report"] != report["report"]
    assert [row["n_control"] for row in ordinary["guards"]] == [4, 4]
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in before}
    assert set(report) == {
        "report",
        "preset",
        "verdict",
        "label",
        "reason",
        "run",
        "control",
        "invariants",
        "guards",
        "reported",
        "gaps",
    }


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
BROKEN = {"policy/raw_grad_norm": float("nan"), "generate/outcome/graded": 502, "generate/outcome/infra_error": 10}


@pytest.mark.parametrize(
    "preset,fault,control_fault,verdict,label,failed",
    [
        ("speed-medium", {"speed": 80}, {}, "pass", "improves", set()),
        (
            "speed-medium",
            {"speed": 80, "metrics": {"vllm/median_generation_throughput": 1}},
            {},
            "pass",
            "trade-off",
            set(),
        ),
        (
            "speed-medium",
            {"speed": 130, "metrics": {"vllm/median_generation_throughput": 1000}},
            {},
            "fail",
            "trade-off",
            set(),
        ),
        (
            "learning-medium",
            {
                "metrics": BROKEN,
                "unscored": 20,
                "duplicate_task": True,
                "omit_step": 25,
                "argv": ("snowball", "--init", "users/test/missing@2026.10.27"),
                "initial_score": 0.2,
                "initial_missing": True,
            },
            {},
            "fail",
            "regression",
            {
                "planned_updates",
                "finite_losses_and_gradients",
                "grader_error_fraction",
                "panel-910/batch_finished",
                "panel-910/panel_scored_fraction",
                "panel-910/initial/batch_finished",
                "panel-910/initial/panel_scored_fraction",
            },
        ),
        ("speed-medium", {"fixed": False}, {}, "fail", "regression", {"fixed_reply_tokens"}),
        ("speed-medium", {}, {"fault": {"provenance": {"base_commit": "wrong"}}}, "invalid", None, set()),
        ("speed-medium", {}, {"fault": {"metrics": AMBIGUOUS}}, "invalid", None, set()),
        ("speed-medium", {"version": "2026.10.01"}, {}, "invalid", None, set()),
        ("learning-medium", {"panel_shift": True}, {}, "invalid", None, set()),
        ("speed-medium", {"metrics": {"policy/mismatch/staleness0/selected_tokens": 0}}, {}, "invalid", None, set()),
        ("speed-medium", {}, {"constant": True}, "invalid", None, set()),
        ("data-short", {}, {}, "pass", "no change", set()),
        ("learning-medium", {"detail": True}, {}, "pass", "no change", set()),
        ("speed-medium", {"metrics": BOUNDARY}, {}, "pass", "no change", set()),
        ("speed-medium", {"metrics": AMBIGUOUS}, {}, "invalid", None, {"grader_error_fraction"}),
        ("speed-medium", {"speed": 130, "metrics": AMBIGUOUS}, {}, "fail", "regression", {"grader_error_fraction"}),
        ("speed-short", {}, {}, "pass", "no change", set()),
    ],
)
def test_rescore_reports_candidate_and_control_contracts(
    recorded_runs, preset, fault, control_fault, verdict, label, failed
):
    controls(recorded_runs, preset, **control_fault)
    name = f"users/test/{preset}"
    recorded_runs(preset, name, score=0.495, **fault)
    report = rescore(preset, name + ("@" + fault["version"] if "version" in fault else ""))
    assert (report["verdict"], report["label"]) == (verdict, label), report
    assert {row["name"] for row in report["invariants"] if not row["holds"]} == failed, report
    if fault.get("metrics") in (BOUNDARY, AMBIGUOUS):
        error = next(row for row in report["invariants"] if row["name"] == "grader_error_fraction")
        assert (error["lower_bound"], error["upper_bound"]) == (0.01, 0.01 if fault["metrics"] == BOUNDARY else 0.02)
    if fault.get("detail"):
        detail = next(row for row in report["reported"] if row["metric"] == "panel-910/family/small/mean")
        assert detail["tolerance"] == pytest.approx(t.ppf(1 - 0.025 / 12, 3) * math.sqrt(5 / 3) * 0.01 * math.sqrt(1.25))
        assert any("/difficulty/" in row["metric"] for row in report["reported"])
        assert any(
            row.get("change") == "better" and row["metric"] not in {m.key for m in presets.PRESETS[preset].reported}
            for row in report["reported"]
        )
    assert Path(report["report"]).with_suffix(".md").exists()


@pytest.mark.parametrize("preset", ["learning-medium", "data-medium"])
def test_built_gate_and_rescore_share_inputs_and_status(recorded_runs, selection, tmp_path, preset):
    controls(recorded_runs, preset)
    if preset == "learning-medium":
        (summary,) = inspect.unwrap(snowball_pass_rates.main.callback)()
        producer = summary.deps[0]
        producer_preset = replace(
            presets.PRESETS["data-short"],
            focus=presets.Focus.LEARNING,
            evaluations=(scoring.PASS_RATE_EVALUATION,),
            guards=(),
        )
        answers = [
            {
                "task_id": f"task-{i}",
                "exception_type": "GraderUnavailable" if i == 1 and attempt == 7 else None,
                "verification": {
                    "status": "verified",
                    "score": float(attempt < (0, 4, 8)[i % 3]),
                    "score_min": 0,
                    "score_max": 1,
                },
            }
            for i in range(910)
            for attempt in range(8)
        ]
        recorded_runs(producer_preset, producer.name, fingerprint=producer.fingerprint(), answers=answers)
        run(summary)
        rates = pq.read_table(Path(summary.path()) / "pass_rates.parquet").to_pylist()
        assert {row["task_id"]: (row["passed"], row["samples"]) for row in rates} == {
            f"task-{i}": ((0, 4, 8)[i % 3], 7 if i == 1 else 8) for i in range(910)
        }
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
            "trainer.policy.optimizer_config.lr=1e-6",
        )
        built = snowball.build_run(
            presets.PRESETS[preset], init=init, settings=tuple(map(presets.parse_setting, settings))
        )
        argv = [
            "snowball_rl",
            "--init",
            "users/test/unused@2026.10.27",
            "--init",
            f"{init.name}@{init.version}",
            *(token for value in settings for token in ("--set", value)),
        ]
    else:
        tasks = presets.selection_artifact(f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}")
        built = snowball.build_run(presets.PRESETS[preset], tasks=tasks)
        argv = [
            "snowball_rl",
            "--tasks",
            f"unused@{presets.DEFAULT_SELECTION_VERSION}",
            "--tasks",
            f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}",
        ]
    source = recorded_runs(
        preset,
        built.runs[0].name,
        score=0.495,
        initial_score=0.2 if preset == "learning-medium" else None,
        argv=argv,
        fingerprint=built.runs[0].fingerprint(),
    )
    saved = run(built.gate)[0]
    report = saved.model_dump(mode="json", exclude={"path"})
    rescored = rescore(preset, built.runs[0].name)
    assert rescored["run"] == report["run"]
    metadata = report["run"]
    expected_settings = (
        {
            "trainer.algorithm.advantage_estimator": "grpo",
            "trainer.algorithm.group_advantage_min_size": None,
            "trainer.policy.optimizer_config.lr": "1e-6",
        }
        if preset == "learning-medium"
        else {}
    )
    assert {
        key: json.loads(value) for item in metadata["settings"] for key, _, value in (item.partition("="),)
    } == expected_settings
    assert (metadata["tasks"], metadata["init"]) == (
        (None, "users/test/initializer@2026.10.27")
        if preset == "learning-medium"
        else ("snowball-100k@2026.10.26", None)
    )
    if preset == "learning-medium":
        gain = next(row for row in report["reported"] if row["metric"] == "panel-910/mean/gain")
        assert gain["run"] == pytest.approx(0.295)
    result = CliRunner().invoke(snowball.main, ["--preset", preset, "--version", presets.PRESET_REVISION, *argv[1:]])
    assert result.exit_code == 0, (result.output, result.exception)
    status = yaml.safe_load(result.output.split("StepSpec(", 1)[0])
    assert (status["verdict"], status["label"]) == (report["verdict"], report["label"])
    assert status["report"].endswith("gate_report.json")
    assert "rl_path_changes_since_control" in status
    Path(status["report"]).unlink()
    terminal = json.loads((source / "terminal.json").read_text())
    terminal["result"]["state"] = "failed"
    (source / "terminal.json").unlink()
    (source / RECORD_FILENAME).unlink()
    (source / ".executor_status").write_text("FAILED")
    _json(source / "attempts/failed-fixture.json", terminal)
    launch = json.loads((source / "resolved-launch.yaml").read_text())
    launch["config"].update(
        {"artifacts": {"attempts_root": str(source / "attempts")}, "run": {"attempt_id": "failed-fixture"}}
    )
    _json(source / "resolved-launch.yaml", launch)
    result = CliRunner().invoke(snowball.main, ["--preset", preset, "--version", presets.PRESET_REVISION, *argv[1:]])
    assert result.exit_code == 0, (result.output, result.exception)
    status = yaml.safe_load(result.output.split("StepSpec(", 1)[0])
    assert (status["verdict"], status["reason"], status["iris_job_id"]) == ("fail", "run crashed", "/tests/native-run")


def test_panel_scores_join_permuted_answers_and_aggregate_tasks(tmp_path):
    def verified(score, low=0.0, high=1.0):
        return {"status": "verified", "score": score, "score_min": low, "score_max": high}

    answers = [
        {
            "task_id": "bounded",
            "score": 0.8,
            "verification": verified(3, 1, 5),
            "output_response": "first\u2028second\u2029third\u0085fourth",
        },
        {"task_id": "mixed", "verification": verified(0)},
        {"task_id": "mixed", "verification": verified(1)},
        {
            "task_id": "mixed",
            "verification": {"status": "error", "score": None, "score_min": 0, "score_max": 1},
            "exception_type": "TimeoutError",
        },
        {"task_id": "stem", "verification": verified(0)},
    ]
    selection = tmp_path / "selection"
    selection.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_id": "mixed", "family": "structural", "aaii_category": "format"},
                {"task_id": "bounded", "family": "structural", "aaii_category": "format"},
                {"task_id": "stem", "family": "stem", "aaii_category": "scientific_reasoning"},
            ]
        ),
        selection / presets.TASK_MANIFEST_FILENAME,
    )
    _json(tmp_path / "resolved-launch.yaml", {"config": {"inputs": {"validation_data": [{"uri": str(selection)}]}}})
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "exports/dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "panel.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in answers) + "\n")
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    scores = scoring.panel_scores(presets.EVALUATIONS["panel-100"], str(tmp_path))
    final = [row for row in scores.to_pylist() if row["step"] == 25]
    assert [(row["item"], row["attempt"], row["scored"], row["score"]) for row in final] == [
        ("bounded", 0, True, 0.5),
        ("mixed", 0, True, 0.0),
        ("mixed", 1, True, 1.0),
        ("mixed", 2, False, 0.0),
        ("stem", 0, True, 0.0),
    ]
    statistics = scoring.evaluation_statistics(pa.Table.from_pylist(final, schema=scores.schema))
    assert {
        key: statistics["panel-100/" + key]
        for key in (
            "mean",
            "family_mean",
            "mixed_fraction",
            "all_right_fraction",
            "all_wrong_fraction",
            "scored_fraction",
            "aaii_category/format/mean",
        )
    } == pytest.approx(
        {
            "mean": 0.375,
            "family_mean": 0.25,
            "mixed_fraction": 1 / 3,
            "all_right_fraction": 0,
            "all_wrong_fraction": 2 / 3,
            "scored_fraction": 0.8,
            "aaii_category/format/mean": 0.5,
        }
    )
