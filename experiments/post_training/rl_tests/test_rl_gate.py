# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
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
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from finestore.rl.run_records import (
    EvaluationAnswerRow,
    EvaluationTaskRow,
    read_attempt,
    write_attempt,
    write_evaluations,
)
from finestore.rl.strata import (
    MembershipState,
    SelectionDefinition,
    SelectionParent,
    SelectionTaskRow,
    read_selection_strata,
)
from finestore.schema import arrow_schema
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import RECORD_FILENAME, Artifact, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, materialized_config, run
from marin.rl import skyrl as skyrl_api
from marin.rl.rl_progress import Change, Comparison, ComparisonRole, Invariant
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from marin.rl.skyrl_records import RecordedLaunch, read_skyrl_metrics
from rigging.provenance import Provenance
from scipy.stats import t
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl as snowball
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_recipe as recipe
from experiments.post_training import rl_scoring as scoring
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_command import EXIT_CODES
from experiments.post_training.rl_scoring import GATE_INPUTS_FILENAME, GateInputs, read_gate_inputs
from experiments.post_training.rl_tests.fixtures import SELECTION_IDENTITY, append_metrics, write_metrics, write_receipt
from experiments.post_training.rl_tests.fixtures import answer as _answer
from experiments.post_training.rl_tests.fixtures import write_json as _json
from experiments.post_training.skyrl_test_support import native_json
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TASK_MANIFEST_FILENAME,
    EvalLiteRecipe,
    HeldoutSource,
    PanelFormat,
    eval_lite_step,
)

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]
KL = "policy/mismatch/staleness0/kl_k3"

INSPECT_NATIVE_LAUNCH = """
import json
import sys
from cloud.iris import launch

args = launch.create_parser().parse_args(sys.argv[1:])
config = launch.load_launch_config(args.config)
print(json.dumps({
    "forward_env": args.forward_env,
    "disk": config.iris.allocation.disk,
    "tokenizer": config.inputs.model.tokenizer_uri,
}))
"""


def controls(write, name, **kwargs):
    pin = gate.control_pin(snowball_model.SNOWBALL.by_name[name], table=snowball_model.SNOWBALL)
    return [
        write(
            name,
            gate.control_name(pin.preset, pin.commit, i, table=snowball_model.SNOWBALL),
            speed=100 + i,
            score=0.49 + i * 0.01,
            **kwargs,
        )
        for i in range(pin.runs)
    ]


def rescore(preset, *args, exit_code=None):
    result = CliRunner().invoke(gate.command(snowball_model.SNOWBALL), ["--preset", preset, *args])
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
    preset = snowball_model.SNOWBALL.by_name["speed-medium"]
    key = "policy/raw_grad_norm"
    pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
    for index in range(pin.runs):
        root = recorded_runs(
            preset.name,
            gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL),
            speed=100 + index,
            metrics={key: 100 + index},
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
            table=snowball_model.SNOWBALL,
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
    preset = replace(snowball_model.SNOWBALL.by_name["speed-long"], measured=range(2, 4))
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
    preset = replace(snowball_model.SNOWBALL.by_name["speed-medium"], updates=2, measured=range(1, 3))
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
        selected = recipe.selection_artifact("candidate-training@2026.11.10")
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
    step = snowball_model.SNOWBALL.by_name[preset].updates
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
        ("train-local-origin", 0),
        ("train-entry-override", 0),
        ("train-low-intermediate", 1),
        ("train-missing-intermediate", 1),
        ("train-best-checkpoint", 0),
        ("train-failed", 3),
        ("malformed-control", 4),
        ("malformed-status", 4),
        ("fail-no-login", 1),
        ("invalid-no-login", 2),
        ("mixed-native-failed", 2),
        ("control-missing-receipt", 3),
        ("partial-retry", 0),
        ("data-short", 0),
        ("compose-timeout", 4),
        ("unexpected", 3),
        ("missing-requirement", 4),
        ("missing-entry", 4),
        ("missing-env", 4),
    ],
)
def test_submission_waits_for_the_coordinator_and_returns_the_verdict(
    recorded_runs, selection, tmp_path, monkeypatch, case, expected
):
    monkeypatch.setenv("WANDB_API_KEY", "test-only-mocked-submission")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    publication_case = case in {
        "pass",
        "changed-source",
        "child",
        "fail-no-login",
        "invalid-no-login",
        "mixed-native-failed",
        "control-missing-receipt",
        "partial-retry",
        "data-short",
        "train-local-origin",
    }
    preset_name = (
        "train-long"
        if case.startswith("train-") or case == "malformed-status"
        else "data-short" if case == "data-short" else "learning-medium" if publication_case else "speed-medium"
    )
    preset = snowball_model.SNOWBALL.by_name[preset_name]
    if case == "child":
        model = "s3://marin-test/models/child-model"
        preset = replace(
            preset,
            model=replace(
                preset.model,
                step=ArtifactStep.adopt(
                    preset.model.step.name,
                    preset.model.step.version,
                    str(model),
                    kind=preset.model.step.artifact_type,
                ),
            ),
        )
    initializer = None
    if case in ("train-pass", "train-entry-override"):
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, host_disk="3TB")))
        initializer = ArtifactHfModel(
            ArtifactStep.adopt(
                "tests/python-init",
                "2026.11.17",
                "s3://marin-test/models/python-init",
                kind=preset.model.step.artifact_type,
            ),
            "tests/python-tokenizer",
            "python-tokenizer-revision",
            relative_path="",
        )
    monkeypatch.setattr(
        snowball_model,
        "SNOWBALL",
        replace(
            snowball_model.SNOWBALL,
            presets=tuple(preset if row.name == preset.name else row for row in snowball_model.SNOWBALL.presets),
        ),
    )
    if initializer is not None or case == "child":
        monkeypatch.setattr(
            "s3fs.S3FileSystem.exists",
            lambda self, path, **kwargs: str(path).removeprefix("s3://")
            in {"marin-test/models/python-init", "marin-test/models/child-model"},
        )
    roots = controls(recorded_runs, preset.name) if preset.gated else []
    replicas = 2 if case in {"partial-retry", "mixed-native-failed", "control-missing-receipt"} else 1
    control = case == "control-missing-receipt"
    settings = ("trainer.max_steps=2",) if case == "train-local-origin" else ()
    built = snowball.build_run(
        preset,
        table=snowball_model.SNOWBALL,
        init=initializer,
        control=control,
        replicates=replicas,
        tasks=selection if case == "data-short" else None,
        settings=settings,
    )
    sources = (
        []
        if "candidate-compose" in case
        else [
            recorded_runs(
                preset.name,
                training.name,
                speed=150 if case == "fail" else 101,
                score=0.1 if case == "fail-no-login" else 0.5,
                fingerprint=training.fingerprint(),
                inputs=built.inputs,
                attempt_id=f"native-{index}",
            )
            for index, training in enumerate(built.runs)
        ]
    )
    source = sources[0] if sources else None
    if case in ("train-low-intermediate", "train-missing-intermediate"):
        directory = source / "exports/dumped_evals/global_step_10_evals"
        if case == "train-missing-intermediate":
            shutil.rmtree(directory)
        else:
            answers = directory / "task_set.jsonl"
            rows = [json.loads(line) for line in answers.read_text().splitlines()]
            for row in rows[:2]:
                row["verification"]["status"] = "error"
            answers.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if case == "train-best-checkpoint":
        _json(source / "exports/best_checkpoint.json", {"step": 10, "mean": 0.5})
    if case in {"native-failed", "train-failed", "invalid", "invalid-no-login"} or case.startswith(
        ("malformed-", "stale-control")
    ):
        root = roots[0] if case == "malformed-control" else roots[-1] if case.startswith("stale-control") else source
        receipt = json.loads((root / "terminal.json").read_text())
        if case in {"native-failed", "train-failed"}:
            receipt["result"]["state"] = "failed"
        elif case in {"invalid", "invalid-no-login"}:
            receipt["result"]["provenance"] = None
        elif case.startswith("malformed-"):
            receipt["config"]["skyrl"]["trainer"]["seed"] = None
        else:
            receipt["config"]["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = 5e-6
        write_receipt(root, receipt)
        if case == "native-failed":
            (source / "terminal.json").unlink()
    pending = {}
    if publication_case:
        for root in sources:
            for filename in ("terminal.json", "resolved-launch.yaml"):
                path = root / filename
                pending[path] = path.read_bytes()
                path.unlink()
    phase = {"retry": False}
    submitted_provenance = Provenance.capture
    if case == "compose-timeout":
        real_run = subprocess.run

        def timeout_native(command, *args, **kwargs):
            if any("launch_config.load_launch_config" in str(value) for value in command):
                raise subprocess.TimeoutExpired(command, 120)
            return real_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", timeout_native)
    if case == "unexpected":
        path = Path(recipe.selection_artifact(preset.tasks).path()) / recipe.SUMMARY_FILENAME
        summary = json.loads(path.read_text())
        summary.pop("grader_kind_counts")
        path.write_text(json.dumps(summary))

    def assert_submission_link():
        output = click.get_binary_stream("stdout").getvalue().decode()
        urls = [
            line.split("run: ", 1)[1].strip() for line in output.splitlines() if "run: https://grafana.oa.dev/" in line
        ]
        query = next(parse_qs(urlparse(url).query) for url in urls if parse_qs(urlparse(url).query)["to"] == ["now"])
        assert query["var-run"] == [built.runs[0].name + "-" + preset.revision]
        assert query["var-cluster"] == [preset.recipe.layout.cluster]
        if preset.name == "train-long":
            assert query["var-training_type"] == ["async"]
        assert int(query["from"][0]) > 0

    def finish(*args, **kwargs):
        if case != "child":
            assert_submission_link()
        for path, data in pending.items():
            path.write_bytes(data)
        if case == "mixed-native-failed":
            receipt = json.loads((source / "terminal.json").read_text())
            if not phase["retry"]:
                receipt["result"]["state"] = "failed"
                write_receipt(source, receipt)
                (source / "terminal.json").unlink()
                return SimpleNamespace(state=JobState.FAILED)
            metrics = read_skyrl_metrics(str(source), attempt_id="native-0")
            receipt["config"]["run"]["attempt_id"] = "native-retry"
            write_receipt(source, receipt)
            for row in metrics.values():
                row["attempt_id"] = "native-retry"
            append_metrics(source, metrics)
        if control and not phase["retry"]:
            (source / "terminal.json").unlink()
            (source / "resolved-launch.yaml").unlink()
            return SimpleNamespace(state=JobState.FAILED)
        if case not in {"missing", "native-failed"} and built.gate is not None:
            run(gate.completed_gate(built.gate))
        if case == "changed-source":
            monkeypatch.setattr(
                Provenance,
                "capture",
                lambda *args, **kwargs: Provenance(
                    "tree-edited-before-publication",
                    "commit-one",
                    True,
                    "branch",
                    "test",
                ),
            )
        return SimpleNamespace(
            state=JobState.FAILED if case in {"native-failed", "train-failed"} else JobState.SUCCEEDED
        )

    client = MagicMock(spec=IrisClient)
    job = JobName.from_wire("/test-user/dashboard-coordinator")
    client.submit.return_value.job_id = job
    client.job.return_value.wait.side_effect = finish
    if case.startswith(("malformed-", "stale-control")) or "candidate-compose" in case:
        client.submit.side_effect = AssertionError("invalid stored inputs reached the coordinator RPC")
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(snowball, "open_iris_client", lambda **kwargs: nullcontext(client))
    published = []
    unavailable = {"value": case in {"partial-retry", "fail-no-login", "invalid-no-login"}}

    def publish(record, service_url=None):
        if unavailable["value"] and (case != "partial-retry" or record.attempt_id == "native-1"):
            raise RuntimeError("Credential unavailable; run iris login")
        published.append(record.model_dump(mode="json"))
        return (
            (service_url or "https://progress.test")
            + "/#"
            + urlencode({"run": record.run_id, "attempt": record.attempt_id})
        )

    monkeypatch.setattr("experiments.post_training.publish_rl_progress.publish_result", publish)
    if case == "child":
        finish()
        client.submit.side_effect = AssertionError("coordinator execution submitted another job")
        monkeypatch.setattr(snowball, "run_skyrl", lambda config, **kwargs: SkyRLRun.raw_load(str(source)))
        monkeypatch.setattr(
            "experiments.post_training.publish_rl_progress.publish_result",
            lambda *args: pytest.fail("Coordinator contacted Marina"),
        )
    arguments = ["--preset", preset.name, "--version", preset.revision, "--run", "--issue", "73"]
    if not case.endswith("no-wait") and case != "unexpected":
        arguments.append("--wait")
    if "candidate-compose" in case:
        arguments.extend(["--set", "trainer.algorithm.policy_loss_type=gspo"])
    if case == "missing-requirement":
        arguments.extend(["--require", "typo>=0"])
    if control:
        arguments.append("--control")
    if case == "data-short":
        arguments.extend(["--tasks", SELECTION_IDENTITY])
    if case == "train-local-origin":
        arguments.extend(["--set", settings[0], "--url", "http://127.0.0.1:55777"])
    if replicas == 2:
        arguments.extend(["--replicates", "2"])
    if case in ("train-pass", "train-entry-override"):
        monkeypatch.setenv("HF_TEST_KEY", "test-initializer-credential")
        monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
        module_name = "second_model_rl"
        entry_module = "override_model_rl" if case == "train-entry-override" else None
        coordinator_module = entry_module or module_name
        module = tmp_path / (module_name + ".py")
        module.write_text(
            "from dataclasses import replace\n"
            "from marin.execution.lazy import ArtifactStep\n"
            "from marin.rl.skyrl import ArtifactHfModel\n"
            "from experiments.post_training import rl, snowball_rl\n"
            "preset = snowball_rl.SNOWBALL.by_name['train-long']\n"
            "preset = replace(preset, recipe=replace(preset.recipe,\n"
            "    layout=replace(preset.recipe.layout, host_disk='3TB')))\n"
            "table = replace(snowball_rl.SNOWBALL, presets=tuple(preset if row.name == preset.name else row "
            "for row in snowball_rl.SNOWBALL.presets))\n"
            f"step = ArtifactStep.adopt('tests/python-init', '2026.11.17', {initializer.step.path()!r}, "
            "kind=preset.model.step.artifact_type)\n"
            "init = ArtifactHfModel(step, 'tests/python-tokenizer', 'python-tokenizer-revision', relative_path='')\n"
            f"raise SystemExit(rl.main(table, init=init, forward_env=('HF_TEST_KEY',), entry_module={entry_module!r}))\n"
        )
        if entry_module is not None:
            (tmp_path / (entry_module + ".py")).write_text(module.read_text())
        monkeypatch.syspath_prepend(str(tmp_path))
        arguments += ["--max-concurrent", "3"]
        staged = source.with_name("native-fixture-before-execution")
        source.rename(staged)

        def launch_native(command):
            native = native_json()(INSPECT_NATIVE_LAUNCH, command[command.index("marinskyrl") + 1 :])
            assert native == {"forward_env": ["HF_TEST_KEY"], "disk": "3TB", "tokenizer": "tests/python-tokenizer"}
            shutil.copytree(staged, source, dirs_exist_ok=True)
            response = json.loads((source / "terminal.json").read_text())["result"]
            response["model"] = dict(
                policy_export_uri=str(source / "policy"),
                global_step=preset.updates,
                tokenizer_uri="tests/python-tokenizer",
                tokenizer_revision="python-tokenizer-revision",
                checkpoint_root=str(source / "checkpoints"),
                terminal_manifest_uri=str(source / "terminal.json"),
            )
            return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

        monkeypatch.setattr(skyrl_api, "_run_launcher", launch_native)

        def submit(**kwargs):
            assert kwargs["entrypoint"].command == ["python", "-m", coordinator_module, *arguments]
            environment = kwargs["environment"].env_vars
            assert environment["HF_TEST_KEY"] == "test-initializer-credential"
            assert "UNRELATED_SECRET" not in environment
            with monkeypatch.context() as child:
                child.setenv("IRIS_TASK_ID", "/test-user/coordinator/0")
                child.setattr(sys, "argv", [module_name, *arguments])
                info = JobInfo(task_id=JobName.from_wire("/test-user/coordinator/0"))
                with iris_ctx_scope(IrisContext.from_job_info(info)):
                    with pytest.raises(SystemExit) as finished:
                        runpy.run_module(coordinator_module, run_name="__main__", alter_sys=True)
                assert finished.value.code == 0
            return SimpleNamespace(job_id=job)

        client.submit.side_effect = submit

        @click.command()
        def entry():
            with pytest.raises(SystemExit) as finished:
                if entry_module is not None:
                    runpy.run_path(str(module), run_name="__main__")
                else:
                    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            raise click.exceptions.Exit(finished.value.code)

        monkeypatch.setattr(sys, "argv", [module_name, *arguments])
        selected = entry
        invoked = []
    else:
        selected = snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            init=initializer,
            entry_module="experiments.post_training.snowball_rl",
        )
        invoked = arguments
    if case == "missing-entry":
        monkeypatch.setattr(sys.modules["__main__"], "__spec__", None)
    if case in ("missing-entry", "missing-env"):
        monkeypatch.delenv("RLCI_ABSENT_KEY", raising=False)
        selected = snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            forward_env=("RLCI_ABSENT_KEY",) if case == "missing-env" else (),
        )
        client.submit.side_effect = AssertionError("invalid caller reached coordinator RPC")
    info = JobInfo(task_id=JobName.from_wire("/test-user/coordinator/0")) if case == "child" else None
    set_job_info(info)
    try:
        with iris_ctx_scope(IrisContext.from_job_info(info)) if info is not None else nullcontext():
            result = CliRunner().invoke(selected, invoked)
    finally:
        set_job_info(None)
    assert result.exit_code == expected, (result.output, result.exception)
    if case in ("train-pass", "train-entry-override"):
        restored_config = yaml.safe_load(materialized_config(built.runs[0], str(source.parent)).launch_config_yaml)
        assert restored_config["inputs"]["model"]["tokenizer_uri"] == "tests/python-tokenizer"
        assert "test-initializer-credential" not in result.output
        assert "test-initializer-credential" not in json.dumps(restored_config)
    if case in ("train-low-intermediate", "train-missing-intermediate"):
        assert f"{preset.evaluations[0].name}/step-10/" in result.output and "holds: false" in result.output
        assert "state: succeeded" in result.output
    if case == "train-best-checkpoint":
        assert "best_checkpoint_step: 10" in result.output
        assert (
            f"{preset.evaluations[0].name}@100/scored" in result.output
            and f"{preset.evaluations[0].name}@100/attempted" in result.output
        )
    if case == "train-pass":
        assert "best_checkpoint_step: none eligible" in result.output
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
    if case == "missing-entry":
        assert "python -m MODULE" in result.output and "entry_module" in result.output
    if case == "missing-env":
        assert "RLCI_ABSENT_KEY" in result.output
    if case == "child":
        assert published == []
        report = gate.completed_gate(built.gate)
        assert Path(report.path(), gate.GATE_REPORT_FILENAME).exists()
        assert gate.GateReport.raw_load(report.path()).verdict == "pass"
        return
    if not publication_case:
        return
    snapshot = yaml.safe_load("preset:" + result.output.rpartition("\npreset:")[2].split("\n# ", 1)[0])
    if case == "data-short":
        assert "publication" not in snapshot and published == []
        return
    paths = [[f"{training.name}-{preset.revision}", f"native-{index}"] for index, training in enumerate(built.runs)]
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
    if case == "mixed-native-failed":
        assert published[0]["status"] == "failed" and published[0]["gate_result"] is None
    if expected == 0 and not unavailable["value"]:
        for row in snapshot["runs"]:
            assert parse_qs(urlsplit(row["rl_progress_url"]).fragment)["attempt"] == [row["attempt_id"]]
    if case == "train-local-origin":
        assert (snapshot["updates"], snapshot["measured_updates"]) == (2, [1, 2])
        assert "report" not in snapshot and "verdict" not in snapshot
        assert snapshot["runs"][0]["rl_progress_url"].startswith("http://127.0.0.1:55777/")
        assert (published[0]["planned_updates"], published[0]["measured_updates"], published[0]["optimizer_steps"]) == (
            2,
            [1, 2],
            2,
        )
    if case == "changed-source":
        monkeypatch.setattr(Provenance, "capture", submitted_provenance)
    unavailable["value"] = False
    phase["retry"] = True
    arguments[arguments.index("--issue") + 1] = "74"
    repeated = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL, argv=tuple(arguments), entry_module="experiments.post_training.snowball_rl"
        ),
        arguments,
    )
    assert repeated.exit_code == (0 if case in {"mixed-native-failed", "control-missing-receipt"} else expected), (
        repeated.output,
        repeated.exception,
    )
    recovered = yaml.safe_load("preset:" + repeated.output.rpartition("\npreset:")[2].split("\n# ", 1)[0])
    assert recovered["publication"]["unserved_records"] == [] and recovered["publication"]["errors"] == []
    assert all(row["issue"] == 74 for row in recovered["runs"])
    if case == "train-local-origin":
        assert recovered["runs"][0]["rl_progress_url"].startswith("http://127.0.0.1:55777/")
    if case == "mixed-native-failed":
        retry = next(row for row in published if row["attempt_id"] == "native-retry")
        assert retry["status"] == "succeeded" and retry["gate_result"]["verdict"] == "pass" and retry["issue"] == 74
    else:
        assert any(row["attempt_id"] == "native-0" and row["issue"] == 74 for row in published)


@pytest.mark.parametrize("fault", ["untracked", "dirty"])
def test_control_refuses_actual_untracked_source_before_execution(selection, tmp_path, monkeypatch, fault):
    root = tmp_path / "git-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "experiments").mkdir()
    (root / "experiments/untracked.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(recipe, "REPO_ROOT", root)
    if fault == "dirty":
        subprocess.run(["git", "-C", str(root), "add", "experiments/untracked.py"], check=True)
        monkeypatch.setattr(
            Provenance, "capture", lambda *args, **kwargs: Provenance("tree", "commit", True, "branch", "test")
        )
    with pytest.raises(click.BadParameter, match="stage source" if fault == "untracked" else "clean checkout"):
        snowball.build_run(snowball_model.SNOWBALL.by_name["speed-medium"], control=True, table=snowball_model.SNOWBALL)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--wait",),
        ("python-init-conflict",),
        ("python-init-rescore",),
        ("--set", "trainer.dump_eval_results=false"),
        ("--set", 'trainer={"algorithm":{"off_policy_correction":"none"}}'),
        ("--set", "trainer.policy.optimizer_config.lr=wrong"),
    ],
)
def test_refusals_use_exit_four(selection, arguments, capsys, tmp_path):
    if arguments == ("python-init-rescore",):
        assert (
            snowball.main(snowball_model.SNOWBALL, argv=["rescore"], init=snowball_model.SNOWBALL.presets[0].model) == 4
        )
        assert "Python init" in capsys.readouterr().err
        return
    initializer = snowball_model.SNOWBALL.presets[0].model if arguments == ("python-init-conflict",) else None
    preset_name = "speed-medium"
    if initializer is not None:
        preset_name = "train-long"
        name, version = "tests/valid-published-init", "2026.11.17"
        root = tmp_path / name / version
        _json(
            root / RECORD_FILENAME,
            ArtifactRecord(
                name=name, version=version, result_type=result_type_name(Artifact), source=str(root)
            ).model_dump(mode="json"),
        )
        arguments = ("--init", f"{name}@{version}")
    result = CliRunner().invoke(
        snowball.command(
            snowball_model.SNOWBALL,
            argv=tuple(arguments),
            init=initializer,
            entry_module="experiments.post_training.snowball_rl",
        ),
        ["--preset", preset_name, "--version", snowball_model.SNOWBALL.by_name[preset_name].revision, *arguments],
    )
    assert result.exit_code == 4, (result.output, result.exception)
    if initializer is not None:
        assert "Python init and --init" in result.output
    if "trainer.dump_eval_results=false" in arguments:
        assert "owned" in result.output


def test_eval_lite_scores_join_permuted_answers_and_aggregate_tasks(tmp_path, recorded_runs):
    answers = [
        _answer("bounded", 2, low=1, high=5)
        | {"score": 0.8, "output_response": "first\u2028second\u2029third\u0085fourth"},
        _answer("bounded", None, status="error", error="TimeoutError"),
        _answer("mixed", 0),
        _answer("mixed", 1),
        _answer("stem", 0),
        _answer("stem", None, status="error"),
        _answer("general", 1),
        {key: value for key, value in _answer("general", 1).items() if key != "verification"},
        _answer("coding", 0.8),
        _answer("coding", 0.4),
        _answer("agents", 1),
        _answer("agents", 1),
    ]
    normalized = [0.25, None, 0, 1, 0, None, 1, None, 0.8, 0.4, 1, 1]
    families = {
        "mixed": ("structural", "format", "hard"),
        "bounded": ("structural", "format", "learnable"),
        "stem": ("stem", "scientific_reasoning", "hard"),
        "general": ("general", "general", "easy"),
        "coding": ("competitive_coding", "coding", "learnable"),
        "agents": ("agents", "agents", "easy"),
    }
    root = recorded_runs("learning-medium", "users/test/score-order")
    native = json.loads((root / "terminal.json").read_text())
    launch = native["config"]
    launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] = 2
    source = launch["inputs"]["validation_data"][0]
    base = json.loads(pq.read_table(Path(source["uri"]) / source["relative_path"])["task_spec"][0].as_py())
    selection = tmp_path / "selection"
    selection.mkdir()
    specs = {task_id: {**base, "id": task_id} for task_id in families}
    manifest = [
        SelectionTaskRow(
            task_id=task_id,
            task_digest=canonical_sha256(specs[task_id]),
            family=family,
            aai_category=category,
            difficulty=difficulty,
            agentic=False,
            state=MembershipState.ACTIVE,
        )
        for task_id, (family, category, difficulty) in families.items()
    ]
    pq.write_table(
        pa.table({"task_spec": [json.dumps(value) for value in specs.values()]}), selection / "validation.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([row.model_dump(mode="json") for row in manifest], schema=arrow_schema(SelectionTaskRow)),
        selection / TASK_MANIFEST_FILENAME,
    )
    _json(
        selection / "summary.json",
        {
            "selection_strata": (
                SelectionDefinition(
                    name="permuted", version="2026.11.17.1", sampling="all", task_files=("validation.parquet",)
                ).model_dump(mode="json")
            )
        },
    )
    launch["artifacts"]["export_root"] = str(tmp_path / "configured-export")
    source["uri"] = str(selection)
    write_receipt(tmp_path, native)
    for session in ("global_step_25_evals", "eval_only"):
        directory = tmp_path / "configured-export/dumped_evals" / session
        directory.mkdir(parents=True)
        (directory / "task_set.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in answers) + "\n"
        )
        (directory / "aggregated_results.jsonl").write_text('{"avg_score": 0.5}\n')
    evaluation = snowball_model.EVALUATIONS["eval_lite_100"]
    historical = scoring.evaluation_scores(evaluation, str(tmp_path))
    final = historical.filter(pc.equal(historical["step"], 25)).to_pylist()
    assert [(row["item"], row["attempt"], row["scored"], row["score"]) for row in final] == [
        ("bounded", 0, True, 0.25),
        ("bounded", 1, False, 0),
        ("mixed", 0, True, 0),
        ("mixed", 1, True, 1),
        ("stem", 0, True, 0),
        ("stem", 1, False, 0),
        ("general", 0, True, 1),
        ("general", 1, False, 0),
        ("coding", 0, True, 0.8),
        ("coding", 1, True, 0.4),
        ("agents", 0, True, 1),
        ("agents", 1, True, 1),
    ]
    export, attempt_id = launch["artifacts"]["export_root"], launch["run"]["attempt_id"]
    record = read_attempt(export, attempt_id)
    assert record is not None
    write_attempt(export, record.model_copy(update={"evaluation_storage_version": 1}))
    annotations = {row.task_id: row for row in manifest}
    task_rows = []
    for step in (-1, 25):
        for task_id in families:
            task_answers = []
            for ordinal, (answer, score) in enumerate(zip(answers, normalized, strict=True)):
                if answer["task_id"] != task_id:
                    continue
                verification = answer.get("verification")
                task_answers.append(
                    EvaluationAnswerRow(
                        ordinal=ordinal,
                        native_uid=str(ordinal),
                        optimization_reward=answer.get("score", 0.0),
                        verifier_score=score,
                        raw_verifier_score=verification["score"] if verification else None,
                        score_min=verification["score_min"] if verification else None,
                        score_max=verification["score_max"] if verification else None,
                        verification_status=verification["status"] if verification else None,
                        passed=score == 1,
                        outcome="graded" if score is not None else "unavailable",
                        error_type=answer.get("exception_type"),
                        error_treatment=answer.get("error_treatment"),
                    )
                )
            task_rows.append(
                EvaluationTaskRow(
                    attempt_id=attempt_id,
                    step=step,
                    evaluation="eval/",
                    panel_name="permuted",
                    panel_version="2026.11.17.1",
                    task_id=task_id,
                    data_source="fixture",
                    strata=annotations[task_id].strata,
                    agentic=False,
                    score=None,
                    outcome="mixed",
                    error_type=None,
                    answers=task_answers,
                )
            )
    write_evaluations(export, task_rows)
    write_evaluations(
        export,
        [
            row.model_copy(
                update={
                    "attempt_id": "other-attempt",
                    "answers": [answer.model_copy(update={"verifier_score": 0.0}) for answer in row.answers],
                }
            )
            for row in task_rows
        ],
    )
    current = scoring.evaluation_scores(evaluation, str(tmp_path))
    assert current.equals(historical)
    with pytest.raises(scoring.InvalidComparison, match="dump is missing for attempt"):
        scoring.evaluation_answers(str(tmp_path))
    directory = Path(export) / "dumped_evals" / attempt_id / "global_step_25_evals"
    directory.mkdir(parents=True)
    current_answers = [dict(row, attempt_id=attempt_id, step=25, is_last_step=True) for row in answers]
    current_answers.insert(0, dict(current_answers[0], is_last_step=False))
    (directory / "task_set.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in current_answers) + "\n"
    )
    assert [row.answer["task_id"] for row in scoring.evaluation_answers(str(tmp_path))] == [
        row["task_id"] for row in answers
    ]
    statistics = scoring.evaluation_statistics(current.filter(pc.equal(current["step"], 25)))
    expected = {
        "mean": 109 / 180,
        "family_mean": 0.595,
        "mixed_fraction": 1 / 6,
        "all_right_fraction": 1 / 3,
        "all_wrong_fraction": 0.5,
        "scored_fraction": 0.75,
        "scored": 9,
        "attempted": 12,
        "aai_category/format/mean": 0.375,
        "difficulty/hard/mean": 0.25,
        "aai_weighted_mean": 0.72,
        "aai_category_weight_coverage": 1,
        "aai_category/scientific_reasoning/weight": 0.2,
        "aai_category/coding/weight": 0.2,
        "aai_category/general/weight": 0.3,
        "aai_category/agents/weight": 0.3,
    }
    assert {key: statistics["eval_lite_100/" + key] for key in expected} == pytest.approx(expected)
    unavailable = current.set_column(
        current.schema.get_field_index("scored"),
        "scored",
        pa.array([row["scored"] and row["item"] != "stem" for row in current.to_pylist()]),
    ).filter(pc.equal(current["step"], 25))
    assert "eval_lite_100/aai_weighted_mean" not in scoring.evaluation_statistics(unavailable)

    (selection / FULL_VALIDATION_FILENAME).write_bytes((selection / "validation.parquet").read_bytes())
    addition = {
        **base,
        "id": "addition",
        "tags": ["eval"],
        "context": {**base["context"], "events": [{**base["context"]["events"][0], "content": "Return a new answer."}]},
    }
    addition_file, excluded_file = tmp_path / "addition.parquet", tmp_path / "excluded.parquet"
    pq.write_table(pa.table({"task_spec": [json.dumps(addition)]}), addition_file)
    pq.write_table(pa.table({"task_spec": [json.dumps({**base, "id": "excluded"})]}), excluded_file)
    rates_file = tmp_path / "reference.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task_id,
                    "task_digest": canonical_sha256(document),
                    "model_revision": "reference-model",
                    "attempted": 8,
                    "samples": 8,
                    "passed": 0 if task_id == "bounded" else 4,
                    "mean_score": 0.0 if task_id == "bounded" else 0.5,
                }
                for task_id, document in specs.items()
            ]
        ),
        rates_file,
    )
    version = "2026.11.17.2"
    successor = EvalLiteRecipe(
        name="permuted",
        version=version,
        parent=SelectionParent(root=str(selection), name="permuted", version="2026.11.17.1"),
        seed=3,
        sources=(
            {
                "source_id": "addition",
                "rows": 1,
                "format": PanelFormat.TASK_SPEC,
                "family": "structural",
                "aaii_category": "format",
                "license": "fixture",
                "count": 1,
            },
        ),
        training=(),
        development=HeldoutSource(uri=str(excluded_file), version=version, rows=1, format=PanelFormat.TASK_SPEC),
        fast_counts={"structural": 2, "stem": 1, "general": 1, "competitive_coding": 1, "agents": 1},
        pass_rates={"uri": str(rates_file), "version": version, "rows": len(specs), "model_revision": "reference-model"},
    )
    successor_root = run(
        eval_lite_step(
            {"addition": ArtifactStep.adopt("fixture/version-addition", version, str(addition_file))}, successor
        )
    )[0].path
    child = read_selection_strata(successor_root)
    assert child is not None
    updated = [
        row.model_copy(
            update={"panel_name": "permuted", "panel_version": version, "strata": child.tasks[row.task_id].strata}
        )
        for row in task_rows
    ]
    addition_answers = [
        updated[-1].answers[0].model_copy(update={"ordinal": len(answers) + i, "native_uid": f"addition-{i}"})
        for i in range(2)
    ]
    updated.append(
        updated[-1].model_copy(
            update={"task_id": "addition", "strata": child.tasks["addition"].strata, "answers": addition_answers}
        )
    )
    write_evaluations(export, updated)
    source.update(uri=successor_root, relative_path=FULL_VALIDATION_FILENAME)
    write_receipt(tmp_path, native)
    previous = scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    assert previous.equals(current.filter(pc.equal(current["step"], 25)))
    target = next(row for row in updated if row.task_id == "bounded" and row.step == 25)
    for length in (0, 1):
        write_evaluations(export, [target.model_copy(update={"answers": target.answers[:length]})])
        with pytest.raises(scoring.InvalidComparison, match="incomplete answer coverage"):
            scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    write_evaluations(export, updated)
    measured_file = Path(successor_root) / FULL_VALIDATION_FILENAME
    original = measured_file.read_bytes()
    changed = pq.read_table(measured_file).to_pylist()
    changed_row = next(row for row in changed if json.loads(row["task_spec"])["id"] == "bounded")
    changed_spec = json.loads(changed_row["task_spec"])
    changed_spec["grader"]["parameters"] = {"expected": ["a different answer"]}
    changed_row["task_spec"] = json.dumps(changed_spec)
    pq.write_table(pa.Table.from_pylist(changed), measured_file)
    with pytest.raises(scoring.InvalidComparison, match="task content differs"):
        scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    measured_file.write_bytes(original)
    old_launch = RecordedLaunch.from_document(launch, attempt=attempt_id)
    old_receipt = read_attempt(export, attempt_id)
    active_attempt = "active-only"
    launch["run"]["attempt_id"] = native["result"]["attempt_id"] = active_attempt
    write_receipt(tmp_path, native)
    active_record = read_attempt(export, active_attempt)
    assert active_record is not None
    write_attempt(export, active_record.model_copy(update={"evaluation_storage_version": 1}))
    write_evaluations(
        export,
        [
            row.model_copy(update={"attempt_id": active_attempt})
            for row in updated
            if child.tasks[row.task_id].state == MembershipState.ACTIVE
        ],
    )
    with pytest.raises(scoring.InvalidComparison, match="recomputation is incomplete"):
        scoring.evaluation_scores(evaluation, str(tmp_path), panel_root=str(selection), step=25)
    with pytest.raises(scoring.InvalidComparison, match="differs from the selected receipt"):
        scoring.evaluation_scores(evaluation, str(tmp_path), launch=old_launch)
    with pytest.raises(scoring.InvalidComparison, match="differs from the selected receipt"):
        scoring.evaluation_answers(str(tmp_path), old_launch)
    assert scoring.evaluation_scores(
        evaluation, str(tmp_path), launch=old_launch, receipt=old_receipt, panel_root=str(selection), step=25
    ).equals(previous)


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
        "omitted-default",
        "omitted-current-nondefault",
        "omitted-candidate-default",
    ],
)
def test_gate_compares_recorded_configuration_and_defaults(recorded_runs, tmp_path, monkeypatch, case):
    preset = snowball_model.SNOWBALL.by_name[
        "speed-short" if case == "staged-profile" else "speed-medium" if case == "missing-guard" else "learning-medium"
    ]
    setting = "trainer.policy.optimizer_config.lr=2e-6"
    settings = (
        ()
        if case
        in (
            "stale-default",
            "cached-default",
            "zero-spread",
            "staged-profile",
            "omitted-default",
            "omitted-current-nondefault",
            "omitted-candidate-default",
        )
        else (setting,)
    )
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
        inputs = inputs.model_copy(
            update={
                "init": recipe.stored_initializer(
                    SkyRLPolicyModel(ArtifactStep.adopt(name, version, str(root), kind=SkyRLRun)), preset=preset
                )
            }
        )
    control_roots = controls(recorded_runs, preset.name)
    if case == "zero-spread":
        pin = gate.control_pin(preset, table=snowball_model.SNOWBALL)
        control_roots = [
            recorded_runs(
                preset.name, gate.control_name(pin.preset, pin.commit, index, table=snowball_model.SNOWBALL), score=0.5
            )
            for index in range(pin.runs)
        ]
    if case in ("stale-default", "omitted-current-nondefault"):
        changed = setting if case == "stale-default" else 'generator.chat_template_kwargs={"enable_thinking":false}'
        preset = replace(preset, recipe=replace(preset.recipe, config=preset.recipe.config.with_settings((changed,))))
        monkeypatch.setattr(
            snowball_model,
            "SNOWBALL",
            replace(
                snowball_model.SNOWBALL,
                presets=tuple(preset if row.name == preset.name else row for row in snowball_model.SNOWBALL.presets),
            ),
        )
    name = "users/test/config-" + case
    candidate = recorded_runs(preset.name, name, inputs=inputs)
    fault = {
        "stale-default": "control config differs",
        "control-commit": "does not match its pin",
        "unbound-data": "skyrl.data.train_data",
        "allocation": "iris.allocation.num_nodes",
        "zero-spread": "variation is zero",
        "missing-guard": "missing or non-finite guard metric",
        "omitted-current-nondefault": "control config differs",
        "omitted-candidate-default": "skyrl.generator.chat_template_kwargs",
    }.get(case)
    if case.startswith("omitted-"):
        omitted = [candidate] if case == "omitted-candidate-default" else control_roots
        for old in omitted:
            receipt = json.loads((old / "terminal.json").read_text())
            assert receipt["config"]["skyrl"]["generator"].pop("chat_template_kwargs") == {}
            if case == "omitted-default":
                assert receipt["config"]["inputs"].pop("starting_checkpoint") is None
                assert receipt["config"]["iris"].pop("role_topology") is None
            write_receipt(old, receipt)
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
        monkeypatch.setattr(
            snowball_model,
            "SNOWBALL",
            replace(
                snowball_model.SNOWBALL,
                presets=tuple(promoted if row.name == preset.name else row for row in snowball_model.SNOWBALL.presets),
            ),
        )
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
    inactive = json.loads(rows[0]["task_spec"])
    inactive["id"] = "retained-inactive"
    panel_root = original.parent
    pool_filename = "retained_pool.parquet"
    manifest_path = panel_root / TASK_MANIFEST_FILENAME
    annotations = pq.read_table(manifest_path).to_pylist()
    summary_path = panel_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    pool = pq.read_table(panel_root / summary["selection_strata"]["task_files"][0]).to_pylist()
    annotations.append(
        SelectionTaskRow(
            task_id=inactive["id"],
            task_digest=canonical_sha256(inactive),
            state=MembershipState.RETIRED,
        ).model_dump(mode="json")
    )
    pq.write_table(pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), manifest_path)
    pq.write_table(pa.Table.from_pylist([*pool, {"task_spec": json.dumps(inactive)}]), panel_root / pool_filename)
    summary["selection_strata"]["task_files"] = [pool_filename]
    summary_path.write_text(json.dumps(summary))
    task_set = scoring.read_recorded_task_set(
        RecordedLaunch.from_document(receipt["config"], attempt=receipt["config"]["run"]["attempt_id"]), "eval_lite_100"
    )
    assert set(task_set.task_ids) == {json.loads(row["task_spec"])["id"] for row in rows}
    assert inactive["id"] not in task_set.task_ids
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
    annotations = pq.read_table(copied / TASK_MANIFEST_FILENAME).to_pylist()
    for annotation in annotations:
        if annotation["task_id"] == task["id"]:
            annotation["task_digest"] = canonical_sha256(task)
    pq.write_table(
        pa.Table.from_pylist(annotations, schema=arrow_schema(SelectionTaskRow)), copied / TASK_MANIFEST_FILENAME
    )
    for filename in {pool_filename, *summary["selection_strata"]["active_files"]}:
        pool = pq.read_table(copied / filename).to_pylist()
        for row in pool:
            if json.loads(row["task_spec"])["id"] == task["id"]:
                row["task_spec"] = json.dumps(task)
        pq.write_table(pa.Table.from_pylist(pool), copied / filename)
    write_receipt(changed, receipt)
    report = rescore(preset, "--candidate", "users/test/task-content-edited")
    assert report["verdict"] == "invalid" and "TaskSpec content differs" in report["reason"], report


@pytest.mark.parametrize("preset_name", ["learning-medium", "data-medium"])
def test_python_run_saves_typed_inputs_for_rescoring(recorded_runs, selection, tmp_path, monkeypatch, preset_name):
    preset = snowball_model.SNOWBALL.by_name[preset_name]
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
    monkeypatch.setattr(
        snowball_model,
        "SNOWBALL",
        replace(
            snowball_model.SNOWBALL,
            presets=tuple(preset if row.name == preset_name else row for row in snowball_model.SNOWBALL.presets),
        ),
    )
    controls(recorded_runs, preset_name)
    tasks = selection if preset_name == "data-medium" else None
    setting_text = (
        ("trainer.policy.optimizer_config.lr=2.0e-6", 'extra_env.VLLM_PROBE="1"', "generator.max_num_seqs=1024")
        if tasks is None
        else ()
    )
    settings = setting_text
    export = model / "selected-hf"
    export.mkdir()
    for filename in ("config.json", "tokenizer_config.json"):
        (export / filename).write_text("{}")
    initializer = (
        ArtifactHfModel(
            ArtifactStep.adopt(
                "tests/noncanonical-initializer",
                "2026.11.17",
                str(model),
                kind=local.step.artifact_type,
                config={"tokenizer_uri": "tests/own-tokenizer"},
            ),
            "tests/own-tokenizer",
            "own-tokenizer-revision",
        )
        if preset.accepts_init
        else None
    )
    built = snowball.build_run(
        preset,
        settings=settings,
        tasks=tasks,
        init=initializer,
        requirements=("policy/raw_grad_norm<=10",),
        table=snowball_model.SNOWBALL,
    )
    task_name = SELECTION_IDENTITY if tasks else None
    source = recorded_runs(
        preset.name,
        built.runs[0].name,
        fingerprint=built.runs[0].fingerprint(),
        inputs=built.inputs,
    )
    staged = source.with_name("native-staging")
    source.rename(staged)
    (staged / GATE_INPUTS_FILENAME).unlink()

    def native(config, *, forward_env=()):
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
    if initializer is not None:
        assert (
            recorded.init.artifact_root,
            recorded.init.relative_path,
            recorded.init.tokenizer_uri,
            recorded.init.tokenizer_revision,
        ) == (str(model), None, "tests/own-tokenizer", "own-tokenizer-revision")
        assert json.loads(recorded.init.fingerprint_payload)["config"]["tokenizer_uri"] == "tests/own-tokenizer"
        newer = model / "newer-hf"
        newer.mkdir()
        for filename in ("config.json", "tokenizer_config.json"):
            (newer / filename).write_text("{}")
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
