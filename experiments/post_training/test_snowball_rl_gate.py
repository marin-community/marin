# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import math
import runpy
import shutil
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import click
import marin.rl.skyrl as adapter
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient
from iris.cluster.client.job_info import set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from iris.rpc import controller_pb2
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.rl.skyrl import SkyRLRun
from marin.training.training import LevanterCheckpoint
from rigging.timing import Timestamp
from scipy.stats import t
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_pass_rates
from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring

pytest_plugins = ("experiments.post_training.snowball_rl_test_fixtures",)
pytestmark = pytest.mark.usefixtures("source_provenance", "synthetic_current_controls")


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _answer(task, score, *, low=0, high=1, status="verified", tokens=1, error=None):
    verification = {"status": status, "score": score, "score_min": low, "score_max": high}
    return {"task_id": task, "response_token_ids": [1] * tokens, "exception_type": error, "verification": verification}


@pytest.fixture
def recorded_runs(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    panel = Path(presets.default_selection().path())
    tasks = [
        {
            "task_id": f"task-{i}",
            "family": ("large", "small")[i == 0],
            "aaii_category": ("coding", "format")[i == 0],
            "agentic": False,
        }
        for i in range(910)
    ]
    pq.write_table(pa.Table.from_pylist(tasks), panel / presets.TASK_MANIFEST_FILENAME)
    test_panel = Path(presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path())
    full_tasks = [{**tasks[0 if i == 0 else 1], "task_id": f"task-{i}"} for i in range(1500)]
    pq.write_table(pa.Table.from_pylist(full_tasks), test_panel / presets.TASK_MANIFEST_FILENAME)
    rates = tmp_path / scoring.PASS_RATES_NAME / scoring.PASS_RATE_REVISION / "pass_rates.parquet"
    rates.parent.mkdir(parents=True)
    revision = presets.MODEL_REVISION
    rows = [
        {
            "task_id": f"task-{i}",
            "model_revision": revision,
            "passed": (0, 4, 8)[i % 3],
            "samples": 8,
            "mean_score": (0.05, 0.5, 0.95)[i % 3],
        }
        for i in range(910)
    ]
    pq.write_table(pa.Table.from_pylist(rows), rates)
    panel_rates = tmp_path / "test-panel-rates" / scoring.PASS_RATES_FILENAME
    panel_rates.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{**rows[i % len(rows)], "task_id": f"task-{i}"} for i in range(1500)]), panel_rates
    )

    def write(
        preset,
        name,
        *,
        speed=100.0,
        score=0.5,
        metrics=None,
        unscored=0,
        duplicate_task=False,
        drop_task=False,
        provenance=None,
        argv=(),
        initial_score=None,
        omit_step=None,
        version=presets.PRESET_REVISION,
        record_version=None,
        fixed=True,
        mixed=128,
        fingerprint="",
        answers=None,
        missing_receipt=False,
        unresolved_receipt=False,
        first_panel_task="task-0",
        quality_unscored=0,
        quality_metrics=None,
        initial_unscored=0,
        first_answer_task="task-0",
        panel_identity="",
    ):
        preset = presets.PRESETS[preset] if isinstance(preset, str) else preset
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
            version=record_version or version,
            fingerprint=fingerprint,
            result_type=result_type_name(SkyRLRun),
            result=result.result_payload(),
        )
        _json(root / RECORD_FILENAME, record.model_dump(mode="json"))
        commit = gate.CONTROLS.get(preset.name, gate.CONTROLS["learning-medium"])[0].main_commit
        source = {"tree_hash": "native-tree", "base_commit": commit, "dirty": False, "command_line": list(argv)}
        result = {"state": "succeeded", "iris_job_id": "/tests/native-run", "provenance": source | (provenance or {})}
        if not (missing_receipt or unresolved_receipt):
            _json(root / "terminal.json", {"config": {}, "result": result})
        (root / ".executor_status").write_text("SUCCESS")
        for step in (0,) if preset.recipe.layout.role_plan.generate_only else range(1, preset.updates + 1):
            row = {f"generate/outcome/{outcome}": 0.0 for outcome in (*Outcome, "rollout_failure")}
            row |= {"step": step, "trainer/global_step": step}
            if not preset.recipe.layout.role_plan.generate_only:
                row["timing/step_wall/unaccounted"] = 0.0
            row |= {metric.key: 0.0 for metric in preset.reported if metric.key.startswith("timing/step_wall/")}
            row |= {
                "attempt_id": "recorded-attempt",
                "generate/outcome/graded": 512.0,
                "loss/policy_loss": 0.01,
                "policy/raw_grad_norm": 1.0,
                "policy/mismatch/staleness0/kl_k3": speed / 1000,
                "policy/mismatch/pooled/kl_k3": speed / 1000,
                "policy/mismatch/staleness0/selected_tokens": 1000,
                "async/performance/loss_tokens_per_configured_policy_gpu_second": speed,
                "async/performance/response_tokens_per_configured_inference_gpu_second": 4 * speed,
                "generate/min_num_tokens": 4096 if fixed else 12,
                "generate/max_num_tokens": 4096,
                "generate/tokens_per_second_per_gpu": speed,
                "vllm/median_generation_throughput": 100.0,
                "startup/init_weight_sync_state": speed / 10,
                "timing/step_wall/group_admission": speed,
                "timing/step_wall/evaluation": 700,
                "timing/step_wall/checkpoint_work": 900,
            }
            if preset.recipe.layout.role_plan.generate_only:
                row["generate/wall_seconds"] = 1.0
            if step != omit_step:
                _json(root / f"exports/metrics/step-{step:06d}.json", row | (metrics or {}))
        launch = {"skyrl": {"trainer": {"seed": presets.SEED}}, "inputs": {"validation_data": [], "evaluation_data": {}}}
        if unresolved_receipt:
            launch |= {
                "artifacts": {"attempts_root": str(root / "attempts")},
                "run": {"attempt_id": "missing"},
                "iris": {"job_name": "lookup-missing-job"},
            }

            @contextmanager
            def endpoint(**kwargs):
                yield SimpleNamespace(url="http://iris.test", credentials=None)

            @contextmanager
            def rpc(*args):
                yield SimpleNamespace(list_jobs=lambda request: controller_pb2.Controller.ListJobsResponse())

            monkeypatch.setattr(adapter, "connect_controller", endpoint)
            monkeypatch.setattr(adapter, "rpc_client", rpc)
        if initial_score is None and any(evaluation.at_start for evaluation in preset.evaluations):
            initial_score = 0.3
        sessions = {-1 if preset.updates == 0 else preset.updates: score}
        sessions |= {0: initial_score} if initial_score is not None else {}
        for index, evaluation in enumerate(preset.evaluations):
            task_options = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--tasks"]
            default = presets.default_selection()
            selected = evaluation.tasks(
                presets.selection_artifact(task_options[-1]) if task_options else default, default
            )
            run_panel = Path(selected.path())
            if first_panel_task != "task-0":
                original = pq.read_table(run_panel / presets.TASK_MANIFEST_FILENAME).to_pylist()
                run_panel = root / f"heldout-{evaluation.name}"
                run_panel.mkdir()
                manifest = [
                    {**row, "task_id": first_panel_task} if row["task_id"] == "task-0" else row for row in original
                ]
                pq.write_table(pa.Table.from_pylist(manifest), run_panel / presets.TASK_MANIFEST_FILENAME)
            count = evaluation.sample or presets.selection_summary(selected)[f"{evaluation.file}_rows"]
            identifiers = [
                f"train-task-{i}" if evaluation.file == "train" else f"task-{i}" if i else first_panel_task
                for i in range(count)
            ]
            binding = selected
            if evaluation.sample is not None:
                binding = presets.sampled_tasks_step(
                    selected, evaluation.file, evaluation.sample, presets.SEED, local_only=evaluation.local_only
                )
            descriptor = {
                "uri": str(run_panel),
                "relative_path": f"{evaluation.file}.parquet",
                "kind": "directory",
                "identity": f"{binding.name}@{binding.version}",
                "local_path": str(run_panel),
            }
            if panel_identity is None:
                descriptor.pop("identity")
            elif panel_identity:
                descriptor["identity"] = panel_identity
            if index:
                launch["inputs"]["evaluation_data"][evaluation.name] = [descriptor]
            else:
                launch["inputs"]["validation_data"] = [descriptor]
            for step, average in sessions.items():
                failed_answers = initial_unscored if step == 0 else quality_unscored if index else unscored
                rows = answers or [
                    _answer(
                        first_answer_task if i == 0 and evaluation.file != "train" else identifier,
                        (float(i < mixed and attempt == 0) if evaluation.file == "train" else average),
                        tokens=(preset.recipe.fixed_reply_tokens or 1) if fixed and index == 0 else 1,
                        error="GraderUnavailable" if i < failed_answers else None,
                    )
                    for i, identifier in enumerate(identifiers[: len(identifiers) - drop_task])
                    for attempt in range(evaluation.answers)
                ]
                session = "eval_only" if step == -1 else f"global_step_{step}_evals"
                export = root / "exports" / evaluation.name if index else root / "exports"
                path = export / "dumped_evals" / session / "panel.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = rows[:1] * duplicate_task + rows
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                if evaluation.file != "train":
                    metric_step = 0 if step == -1 else step
                    metric_path = root / f"exports/metrics/step-{metric_step:06d}.json"
                    if metric_step != omit_step:
                        outcome_prefix = f"eval/{evaluation.name}/all/outcome/" if index else "eval/all/outcome/"
                        counts = {outcome_prefix + outcome: 0 for outcome in (*Outcome, "rollout_failure")}
                        failures = sum(bool(row.get("exception_type")) for row in rows)
                        counts.update(
                            {outcome_prefix + "graded": len(rows) - failures, outcome_prefix + "unavailable": failures}
                        )
                        previous = json.loads(metric_path.read_text()) if metric_path.exists() else {}
                        updated = previous | counts | (quality_metrics or {})
                        for key, value in (quality_metrics or {}).items():
                            if value is None:
                                del updated[key]
                        _json(metric_path, updated)
        if not missing_receipt:
            _json(root / "resolved-launch.yaml", {"config": launch})
        return root

    return write


def _native_launches(steps, tmp_path, native_json):
    paths, authored = [], []
    for index, step in enumerate(steps):
        context = StepContext.for_run(
            str(tmp_path / step.name / step.version), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        config = step.build_config(context)
        path = tmp_path / f"compose-{index}.yaml"
        path.write_text(config.launch_config_yaml)
        authored.append(config)
        paths.append(str(path))
    program = (
        "import json,sys\nfrom pathlib import Path\nfrom omegaconf import OmegaConf\n"
        "from cloud.iris import launch_config\n"
        "print(json.dumps([OmegaConf.to_container(launch_config.load_launch_config(Path(p)),resolve=True) "
        "for p in sys.argv[1:]]))\n"
    )
    return tuple(zip(authored, native_json(program, paths), strict=True))


def _record_launch(root, config, compiled):
    old = json.loads((root / "resolved-launch.yaml").read_text())["config"]["inputs"]
    for name, sources in {"primary": old["validation_data"], **old["evaluation_data"]}.items():
        actual = (
            compiled["inputs"]["validation_data"] if name == "primary" else compiled["inputs"]["evaluation_data"][name]
        )
        for source, destination in zip(sources, actual, strict=True):
            target = Path(urlsplit(destination["uri"]).path)
            if not target.exists():
                shutil.copytree(Path(urlsplit(source["uri"]).path), target)
    record = json.loads((root / RECORD_FILENAME).read_text())
    record["config"] = asdict(config)
    _json(root / RECORD_FILENAME, record)
    terminal = json.loads((root / "terminal.json").read_text())
    _json(root / "terminal.json", terminal | {"config": compiled})
    (root / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": compiled}))


def controls(write, preset, *, constant=False, fault=None, common_fault=None):
    pin = gate.CONTROLS[preset][0]
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


def rescore(preset, *args):
    result = CliRunner().invoke(gate.main, ["--preset", preset, *args])
    assert result.exit_code == 0, (result.output, result.exception)
    return json.loads(result.output.splitlines()[-1])


def status(preset, argv):
    result = CliRunner().invoke(snowball.main, ["--preset", preset, "--version", presets.PRESET_REVISION, *argv])
    assert result.exit_code == 0, (result.output, result.exception)
    return yaml.safe_load(result.output.split("StepSpec(", 1)[0])


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
    next(tmp_path.glob("rl-data/snowball-pass-rates/*/*/pass_rates.parquet")).unlink()
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
            (unfinished, "not a recorded speed-medium"),
            (failed, "run crashed"),
            (incomplete, "failed invariants"),
            (missing_record, "record"),
        )
    ]
    cases.append(("learning-medium", other_panel, "panel task IDs differ"))
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
        assert refused.exit_code == 2 and reason in refused.output


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
    "preset,fault,control_fault,verdict,label,failed,bounds,selection_change",
    [
        ("speed-medium", {"speed": 80}, {}, "pass", "improves", set(), None, False),
        ("speed-medium", {"speed": 80, "metrics": {VLLM: 1}}, {}, "pass", "trade-off", set(), None, False),
        ("speed-medium", {"speed": 130, "metrics": {VLLM: 1000}}, {}, "fail", "trade-off", set(), None, False),
        ("speed-long", {}, {}, "pass", "no change", set(), None, False),
        (
            "speed-long",
            {"metrics": {"async/performance/loss_tokens_per_configured_policy_gpu_second": 50}},
            {},
            "fail",
            "regression",
            set(),
            None,
            False,
        ),
        (
            "speed-long",
            {"metrics": {"async/performance/response_tokens_per_configured_inference_gpu_second": 50}},
            {},
            "fail",
            "regression",
            set(),
            None,
            False,
        ),
        ("speed-long", {"metrics": {"policy/mismatch/pooled/kl_k3": 100}}, {}, "pass", "regression", set(), None, False),
        (
            "speed-long",
            {"metrics": {"policy/mismatch/staleness0/kl_k3": 100}},
            {},
            "fail",
            "regression",
            set(),
            None,
            False,
        ),
        pytest.param(
            "learning-medium",
            {"panel_identity": None},
            {},
            "invalid",
            None,
            set(),
            None,
            False,
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
            False,
            id="recorded-panel-version-differs",
        ),
        ("learning-medium", BROKEN, {}, "fail", "regression", BROKEN_FAILS, None, False),
        ("speed-medium", {"fixed": False}, {}, "fail", "regression", {"fixed_reply_tokens"}, None, False),
        ("speed-medium", {}, {"fault": {"provenance": {"base_commit": "wrong"}}}, "invalid", None, set(), None, False),
        ("speed-medium", {}, {"fault": {"metrics": AMBIGUOUS}}, "invalid", None, set(), None, False),
        ("speed-medium", {"version": "2026.10.01"}, {}, "invalid", None, set(), None, False),
        pytest.param(
            "speed-medium",
            {},
            {"fault": {"record_version": "2026.10.01"}},
            "invalid",
            None,
            set(),
            None,
            False,
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
            False,
        ),
        (
            "speed-medium",
            {"metrics": {"policy/mismatch/staleness0/selected_tokens": 0}},
            {},
            "invalid",
            None,
            set(),
            None,
            False,
        ),
        ("speed-medium", {}, {"constant": True}, "invalid", None, set(), None, False),
        pytest.param(
            "speed-medium",
            {},
            {"fault": {"metrics": {"vllm/latency_ttft_mean": float("nan")}}},
            "pass",
            "no change",
            set(),
            None,
            False,
            id="nonfinite_range",
        ),
        ("data-short", {}, {}, "pass", "no change", set(), None, False),
        ("speed-medium", {"metrics": BOUNDARY}, {}, "pass", "no change", set(), (0.01, 0.01), False),
        ("speed-medium", {"metrics": AMBIGUOUS}, {}, "invalid", None, {"grader_error_fraction"}, (0.01, 0.02), False),
        (
            "speed-medium",
            {"speed": 130, "metrics": AMBIGUOUS},
            {},
            "fail",
            "regression",
            {"grader_error_fraction"},
            (0.01, 0.02),
            False,
        ),
        ("speed-short", {}, {}, "pass", "no change", set(), None, True),
        pytest.param(
            "speed-short",
            {"quality_unscored": 16},
            {},
            "fail",
            "regression",
            {"test_panel/panel_scored_fraction", "test_panel/grader_error_fraction"},
            (16 / 1500, 16 / 1500),
            False,
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
            False,
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
            False,
            id="quality-inflated-denominator",
        ),
        pytest.param(
            "speed-short",
            {"first_answer_task": "unlisted"},
            {},
            "invalid",
            None,
            set(),
            None,
            False,
            id="quality-wrong-membership",
        ),
        pytest.param(
            "learning-medium",
            {"initial_score": 0.3, "initial_unscored": 16},
            {},
            "fail",
            "regression",
            {"test_panel/initial/panel_scored_fraction", "test_panel/initial/grader_error_fraction"},
            None,
            False,
            id="initial-quality-failure",
        ),
        ("speed-medium", {"missing_receipt": True}, {}, "invalid", None, set(), None, False),
        ("speed-medium", {"unresolved_receipt": True}, {}, "invalid", None, set(), None, False),
        pytest.param(
            "speed-short", {"score": 0.2}, {}, "fail", "regression", set(), None, False, id="quality-score-regression"
        ),
    ],
)
def test_rescore_reports_candidate_and_control_contracts(
    recorded_runs, monkeypatch, preset, fault, control_fault, verdict, label, failed, bounds, selection_change
):
    if preset == "speed-long":
        monkeypatch.setitem(gate.CONTROLS, preset, (gate.ControlPin(preset, "09026fd100", presets.PRESET_REVISION, 4),))
    controls(recorded_runs, preset, **control_fault)
    source = recorded_runs(preset, f"users/test/{preset}", **({"score": 0.495} | fault))
    if selection_change:
        monkeypatch.setattr(presets, "DEFAULT_SELECTION_VERSION", "2099.01.01")
        _json(Path(presets.default_selection().path()) / "summary.json", {"full_validation_rows": 73})
    report = rescore(
        preset, "--candidate", f"users/test/{preset}" + ("@" + fault["version"] if "version" in fault else "")
    )
    assert (report["verdict"], report["label"]) == (verdict, label), report
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
        if selection_change:
            coverage = next(row for row in report["invariants"] if row["name"] == "test_panel/panel_scored_fraction")
            finished = next(row for row in report["invariants"] if row["name"] == "test_panel/batch_finished")
            assert coverage["value"] == 1.0
            assert finished["value"] == finished["limit"]
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
    elif verdict != "invalid":
        if preset in ("speed-medium", "speed-long"):
            pass
        else:
            expected_panel = (
                f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}"
                if preset == "data-short"
                else presets.EVALUATIONS["test_panel"].source
            )
            assert report["run"]["panel"] == report["control"]["panel"] == expected_panel
    assert not fault.get("missing_receipt") or report["reason"] == f"no launch receipt at {source}"
    assert not fault.get("unresolved_receipt") or report["reason"].startswith(
        "cannot identify the Iris job lookup-missing-job"
    )


def test_detail_rows_use_table_counts_and_initializer_scores(recorded_runs, selection, tmp_path):
    panel = Path(presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path())
    summary = json.loads((panel / "summary.json").read_text())
    summary["recipe"]["pass_rates"]["rows"] = 6
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
        (presets.PRESETS["train-long"], "test_panel/mean>=0", {}),
        (learning, "test_panel/mean>=0", {"control": True}),
        (learning, "invalid", {}),
    ):
        with pytest.raises(click.BadParameter):
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
@pytest.mark.parametrize("preset", ["learning-medium", "speed-short"])
def test_control_panel_rescore_preserves_recorded_protocol(recorded_runs, tmp_path, preset, monkeypatch, native_json):
    original_preset = presets.PRESETS[preset]
    pin = gate.CONTROLS[preset][0]
    originals = controls(recorded_runs, preset, common_fault={"metrics": {"timing/grading/mean": 2.0}})
    recorded_preset = replace(
        original_preset,
        recipe=replace(
            original_preset.recipe,
            engine_init_kwargs={**original_preset.recipe.engine_init_kwargs, "cpu_offload_gb": 1},
        ),
    )
    steps = snowball.build_run(recorded_preset, control=True, replicates=pin.runs).runs
    for root, (config, compiled) in zip(originals, _native_launches(steps, tmp_path, native_json), strict=True):
        record = json.loads((root / RECORD_FILENAME).read_text())
        record["result"].update(
            hf_model_uri=str(root / "retained-hf"),
            tokenizer_uri=compiled["inputs"]["model"]["tokenizer_uri"],
            tokenizer_revision=compiled["inputs"]["model"]["tokenizer_revision"],
        )
        _json(root / RECORD_FILENAME, record)
        _json(root / "retained-hf/config.json", {})
        _record_launch(root, config, compiled)
    original_bytes = {
        (root, filename): (root / filename).read_bytes()
        for root in originals
        for filename in (RECORD_FILENAME, "terminal.json", "resolved-launch.yaml")
    }
    panel_name, _ = (presets.DEFAULT_TEST_PANEL_NAME, presets.DEFAULT_TEST_PANEL_VERSION)
    new_panel = f"{panel_name}@2099.01.01"
    shutil.copytree(
        presets.selection_artifact(presets.EVALUATIONS["test_panel"].source).path(),
        presets.selection_artifact(new_panel).path(),
    )
    changed_presets = dict(presets.PRESETS)
    for name in (preset, "data-medium") if preset == "learning-medium" else (preset,):
        changed = replace(
            presets.PRESETS[name],
            evaluations=tuple(
                replace(item, source=new_panel) if item.name == "test_panel" else item
                for item in presets.PRESETS[name].evaluations
            ),
        )
        changed_presets[name] = changed
    monkeypatch.setattr(presets, "PRESETS", changed_presets)
    selected = presets.PRESETS[preset]
    candidate = f"users/test/new-panel-{preset}"
    recorded_runs(selected, candidate, score=0.495, initial_score=0.2, speed=101.5, argv=("--preset", preset))
    missing = rescore(preset, "--candidate", candidate, "--require", "test_panel/mean>=0.48")
    assert missing["verdict"] == "invalid" and new_panel in missing["reason"]
    rescored = snowball._panel_rescore_runs(selected, new_panel)
    assert [step.name for step in rescored.runs] == [
        presets.panel_rescore_name(presets.control_name(pin.preset, pin.main_commit, index), pin.revision, new_panel)
        for index in range(pin.runs)
    ]
    if preset == "learning-medium":
        for root, step in zip(originals, rescored.runs, strict=True):
            marker = root / ".executor_status"
            status_bytes = marker.read_bytes()
            marker.unlink()
            try:
                dependency = next(dep for dep in step.deps if dep.artifact_type is SkyRLRun)
                run(dependency)
                assert all(
                    (source / filename).read_bytes() == content for (source, filename), content in original_bytes.items()
                )
            finally:
                marker.write_bytes(status_bytes)
    evaluation_preset = presets.panel_rescore_preset(selected, new_panel)
    receipts = []
    for index, (step, (config, compiled)) in enumerate(
        zip(rescored.runs, _native_launches(rescored.runs, tmp_path, native_json), strict=True)
    ):
        assert compiled["runtime"]["entrypoint"] == "skyrl_train.entrypoints.main_generate"
        assert compiled["iris"]["allocation"]["num_nodes"] == 1
        assert compiled["iris"]["allocation"]["gpus_per_node"] == 8
        assert compiled["skyrl"]["generator"]["engine_init_kwargs"]["cpu_offload_gb"] == 1
        root = recorded_runs(
            evaluation_preset,
            step.name,
            score=0.49 + 0.01 * index,
            speed=9999,
            metrics={"timing/grading/mean": 9999.0},
        )
        _record_launch(root, config, compiled)
        receipts.append(root)
    recovered = rescore(preset, "--candidate", candidate, "--require", "test_panel/mean>=0.48")
    assert recovered["verdict"] == "pass", recovered
    assert recovered["report"] != missing["report"]
    assert recovered["control"]["panel"] == recovered["run"]["panel"] == new_panel
    assert not any(
        row["metric"].endswith(("/initial", "/gain")) and row["control"] is not None for row in recovered["reported"]
    )
    performance_key = "generate/tokens_per_second_per_gpu" if preset == "speed-short" else "timing/grading/mean"
    performance = next(row for row in (*recovered["guards"], *recovered["reported"]) if row["metric"] == performance_key)
    assert performance["control"] == pytest.approx(100 + (pin.runs - 1) / 2 if preset == "speed-short" else 2.0)
    assert rescore(preset, "--candidate", candidate, "--require", "test_panel/mean>=control")["verdict"] == "pass"
    baseline_run = snowball.build_run(
        selected, requirements=("test_panel/mean>=control",), baselines=(f"{candidate}@{presets.PRESET_REVISION}",)
    )
    assert baseline_run.gate is not None
    baseline_report = rescore(preset, "--candidate", candidate, "--baseline", f"{candidate}@{presets.PRESET_REVISION}")
    assert baseline_report["run"]["baselines"] == [f"{candidate}@{presets.PRESET_REVISION}"]
    assert baseline_report["vs_baseline"]
    assert rescore(preset, "--leave-out", "r0")["verdict"] == "pass"
    bounded = rescore(preset, "--leave-out", "r0", "--require", "test_panel/mean>=0.9")
    assert bounded["verdict"] == "fail", bounded
    assert any(row["name"] == "require/test_panel/mean" and not row["holds"] for row in bounded["invariants"])
    if preset == "learning-medium":
        data_candidate = "users/test/data-new-panel"
        recorded_runs(presets.PRESETS["data-medium"], data_candidate, score=0.495)
        assert rescore("data-medium", "--candidate", data_candidate)["verdict"] == "pass"
    first = receipts[0]
    terminal = json.loads((first / "terminal.json").read_text())
    faults = (
        (
            ("model", "config.inputs.model.identity", "other@2026.11.06", "model or tokenizer differs"),
            ("control", "config.run.id", "another-control", "names another control"),
            ("malformed", "config.inputs.model.identity", None, "invalid panel re-score"),
            (
                "protocol",
                "config.skyrl.generator.eval_sampling_params.temperature",
                1,
                "changes evaluation or engine settings",
            ),
            (
                "batch_invariance",
                "config.skyrl.trainer.algorithm.batch_invariant",
                False,
                "changes evaluation or engine settings",
            ),
            ("failed", "result.state", "failed", "no successful re-score"),
            ("incomplete", "", None, "control has"),
        )
        if preset == "learning-medium"
        else ()
    )
    for index, (fault, fields, value, reason) in enumerate(faults):
        broken = json.loads(json.dumps(terminal))
        broken["config"]["run"]["attempt_id"] = f"fault-{index}"
        if fields:
            fields = fields.split(".")
            target = broken
            for key in fields[:-1]:
                target = target[int(key) if key.isdecimal() else key]
            if value is ...:
                del target[fields[-1]]
            else:
                target[fields[-1]] = value
        dump = first / "exports/dumped_evals/eval_only/panel.jsonl"
        original_dump = dump.read_bytes()
        if fault == "incomplete":
            dump.write_text("\n".join(dump.read_text().splitlines()[:-1]) + "\n")
        _json(first / "terminal.json", broken)
        (first / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": broken["config"]}))
        invalid = rescore(preset, "--candidate", candidate)
        assert invalid["verdict"] == "invalid" and reason in invalid["reason"], (fault, invalid)
        dump.write_bytes(original_dump)
    _json(first / "terminal.json", terminal)
    (first / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": terminal["config"]}))
    if preset == "learning-medium":
        client = MagicMock(spec=IrisClient)
        client.submit.side_effect = AssertionError("a missing control export submitted GPU work")
        monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
        (originals[-1] / "retained-hf/config.json").unlink()
        refused = CliRunner().invoke(
            snowball.main,
            [
                "--preset",
                preset,
                "--version",
                presets.PRESET_REVISION,
                "--control",
                "--rescore-panel",
                new_panel,
                "--run",
            ],
        )
        assert refused.exit_code == 2, (refused.output, refused.exception)
        assert str(pin.runs - 1) in refused.output and "kept HF weights" in refused.output
        _json(originals[-1] / "retained-hf/config.json", {})
    assert all((root / filename).read_bytes() == content for (root, filename), content in original_bytes.items())


@pytest.mark.timeout(240)
@pytest.mark.parametrize("preset", ["learning-medium", "data-medium"])
def test_built_gate_and_rescore_share_inputs_and_status(recorded_runs, tmp_path, preset, monkeypatch, native_json):
    monkeypatch.setenv("WANDB_ENTITY", "planned-team")
    controls(recorded_runs, preset)
    if preset == "learning-medium":
        (summary,) = inspect.unwrap(snowball_pass_rates.main.callback)()
        answers = [
            _answer(
                f"task-{i}",
                0.5 if (i, attempt) == (0, 0) else float(attempt < (0, 4, 8)[i % 3]),
                error="GraderUnavailable" if (i, attempt) == (1, 7) else None,
            )
            for i in range(910)
            for attempt in range(8)
        ]
        recorded_runs(
            snowball_pass_rates.PRESET, summary.deps[0].name, fingerprint=summary.deps[0].fingerprint(), answers=answers
        )
        run(summary)
        rates = pq.read_table(Path(summary.path()) / "pass_rates.parquet").to_pylist()
        assert {row["task_id"]: (row["passed"], row["samples"]) for row in rates} == {
            f"task-{i}": ((0, 4, 8)[i % 3], 7 if i == 1 else 8) for i in range(910)
        }
        assert next(row["mean_score"] for row in rates if row["task_id"] == "task-0") == 0.0625
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
    report = run(built.gate)[0].model_dump(mode="json", exclude={"path"})
    timer_rows = {row["metric"]: row for row in report["reported"] if row["metric"] in timers}
    assert {key: row["run"] for key, row in timer_rows.items()} == timers
    assert all(row["control"] is None and row["change"] == "unavailable" for row in timer_rows.values())
    markdown = (Path(built.gate.path()) / gate.GATE_REPORT_FILENAME).with_suffix(".md").read_text()
    assert all(key in markdown for key in timers)
    assert rescore(preset, "--candidate", built.runs[0].name)["run"] == report["run"]
    settings = {key: json.loads(value) for key, _, value in (item.partition("=") for item in report["run"]["settings"])}
    gains = [row["run"] for row in report["reported"] if row["metric"] == "test_panel/mean/gain"]
    assert (settings, report["run"]["tasks"], report["run"]["init"], gains) == expected
    assert [(row["name"], row["holds"]) for row in report["invariants"] if row["name"].startswith("require/")] == [
        ("require/" + REQUIREMENT.split("<=")[0], True)
    ]
    assert "wandb_url" not in status(preset, argv)["runs"][0]
    assert status(preset, argv)["runs"][0]["wandb_entity"] == snowball.DEFAULT_WANDB_ENTITY
    _json(source / "exports/tracking.json", {"wandb_url": "https://wandb.example/team/project/runs/actual-id"})
    _json(source / "exports/metrics/step-000026.json", {"step": 26, "startup/load_checkpoints": 3.0})
    _json(source / "exports/metrics/step-000028.json", {"step": 28, "trainer/global_step": 28})
    printed = status(preset, argv)
    assert printed["runs"][0]["wandb_url"] == "https://wandb.example/team/project/runs/actual-id"
    assert printed["runs"][0]["latest_completed_update"] == 25
    receipt = json.loads((source / "terminal.json").read_text())
    linked_receipt = receipt | {
        "config": {
            "run": {"id": "run / id?x=1&y=2"},
            "runtime": {"training_type": "sync"},
            "iris": {"wandb_entity": "receipt-team"},
        }
    }
    _json(source / "terminal.json", linked_receipt)
    started, finished = 1793840000000, 1793840060000
    native_job = SimpleNamespace(
        state=JobState.SUCCEEDED, started_at=Timestamp.from_ms(started), finished_at=Timestamp.from_ms(finished)
    )
    client = MagicMock(spec=IrisClient)
    client.job_status.return_value = native_job
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    linked = status(preset, argv)["runs"][0]
    assert linked["wandb_entity"] == "receipt-team"
    links = linked["grafana_links"]
    query = parse_qs(urlsplit(links["run"]).query)
    assert query["var-run"] == ["run / id?x=1&y=2"]
    assert query["from"] == [str(started)] and query["to"] == [str(finished)]
    client.job_status.side_effect = ConnectError(Code.NOT_FOUND, "archived native job")
    assert "grafana_links" not in status(preset, argv)["runs"][0]
    client.job_status.side_effect = ConnectionError("controller unavailable")
    assert status(preset, argv)["verdict"] == report["verdict"]
    client.job_status.side_effect = None
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline")))
    assert "grafana_links" not in status(preset, argv)["runs"][0]
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    native_job.started_at = None
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
    if preset == "learning-medium":
        later = recorded_runs(preset, "users/test/later-replicate", score=0.495)
        receipt = json.loads((later / "terminal.json").read_text())
        _json(later / "terminal.json", receipt | {"result": receipt["result"] | {"provenance": None}})
        second = ArtifactStep.adopt("users/test/later-replicate", presets.PRESET_REVISION, str(later), kind=SkyRLRun)
        invalid = run(gate.gate_step(presets.PRESETS[preset], (built.runs[0], second), inputs=gate.GateInputs()))[0]
        assert (invalid.verdict, invalid.reason) == ("invalid", f"run has no recorded source provenance: {later}")
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
        "fail",
        "run crashed",
        "/tests/native-run",
    )


@pytest.mark.parametrize("preset", ["speed-long", "speed-medium"])
def test_unpinned_candidates_refuse_gpu_submission(recorded_runs, monkeypatch, preset):
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
        (snowball.main, ["--preset", preset, "--version", presets.PRESET_REVISION, "--run", "--wait"]),
        (gate.main, ["--preset", preset, "--candidate", "tests/unpinned"]),
    ):
        refused = CliRunner().invoke(command, arguments)
        assert refused.exit_code == 2, (refused.output, refused.exception)
        payload = json.loads(refused.output)
        assert payload["preset"] == preset and payload["verdict"] == "invalid"
        assert payload["reason"].endswith(
            f"python -m experiments.post_training.snowball_rl --preset {preset} "
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
    if case not in {"missing", "control"}:
        run(built.gate)
    if case == "parent-early":
        shutil.rmtree(source)
        shutil.rmtree(built.gate.path())
    job = JobName.from_wire("/atqamar/coordinator-id-returned-by-iris")
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = job
    final = JobState.FAILED if case == "parent-early" else JobState.SUCCEEDED
    client.job.return_value.wait.return_value = SimpleNamespace(state=final)
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
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
    assert all(call.kwargs["poll_interval"] == 60 for call in client.job.return_value.wait.call_args_list)
    if case != "child":
        assert client.submit.call_args.kwargs["environment"].env_vars["WANDB_ENTITY"] == "submission-team"
    if case in {"pass", "fail", "invalid"}:
        assert "# speed-medium" in result.output
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
    scores = scoring.panel_scores(presets.EVALUATIONS["panel-100"], str(tmp_path))
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
    assert {key: statistics["panel-100/" + key] for key in expected} == pytest.approx(expected)


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
