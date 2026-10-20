"""SWE-bench outcomes survive grading, native reduction and archive persistence."""

import asyncio
import json
import sys
from argparse import Namespace
from dataclasses import replace
from types import ModuleType

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.chat_benchmarks.SWEbench.eval_instruct import SWEBenchBenchmark
from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.finestore_output import FineStoreOutput, read_finestore_output
from eval.contracts.task_outcome import FailureCategory, TaskRoute, TaskStatus
from eval.execution.driver import SourcePlan, run_evaluations, source_plans
from eval.execution.grading import Grader
from eval.execution.metrics import SourceMetrics
from eval.execution.run_spec import RunSpec
from eval.graders.swebench import swe_patch


class DockerClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def swebench_boundary(monkeypatch, tmp_path):
    client = DockerClient()
    docker = ModuleType("docker")
    docker.from_env = lambda: client
    constants = ModuleType("swebench.harness.constants")
    constants.KEY_INSTANCE_ID = "instance_id"
    constants.KEY_MODEL = "model_name_or_path"
    constants.KEY_PREDICTION = "model_patch"
    constants.APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
    constants.LOG_INSTANCE = "run_instance.log"
    constants.LOG_TEST_OUTPUT = "test_output.txt"
    constants.LOG_REPORT = "report.json"
    constants.RUN_EVALUATION_LOG_DIR = tmp_path
    test_spec = ModuleType("swebench.harness.test_spec.test_spec")
    test_spec.make_test_spec = lambda instance, **_kwargs: instance
    harness = ModuleType("swebench.harness.run_evaluation")
    for module in (docker, constants, test_spec, harness):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    harness.log_root = tmp_path
    return client, harness


@pytest.fixture
def swebench_spec():
    return EvalSpec(
        task="SWEbench", source=SourceRef("fixture"), source_id="repo-1", ordinal=0,
        requests=(), grader=GraderSpec("swe_patch"), doc={"instance_id": "repo-1"},
    )


@pytest.mark.parametrize("resolved", [False, True])
def test_swebench_grades_and_closes_client_without_context_manager(swebench_boundary, swebench_spec, resolved):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        assert not client.closed
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "report.json").write_text(json.dumps({prediction["instance_id"]: {"resolved": resolved}}))
        return {"completed": True, "resolved": resolved}

    harness.run_instance = run_instance

    grade = swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert grade.metrics == {"accuracy": float(resolved)}
    assert client.closed


def test_swebench_harness_failure_closes_client_and_propagates(swebench_boundary, swebench_spec):
    client, harness = swebench_boundary

    def run_instance(_instance, _prediction, **_kwargs):
        raise RuntimeError("Docker daemon unavailable")

    harness.run_instance = run_instance

    with pytest.raises(RuntimeError, match="Docker daemon unavailable"):
        swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert client.closed


def test_swebench_invalid_model_patch_is_scored_zero(swebench_boundary, swebench_spec):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "run_instance.log").write_text("INFO - >>>>> Patch Apply Failed:\nNo valid patches in input\n")
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance

    grade = swe_patch(swebench_spec, ["This response is not a patch."])

    assert grade.metrics == {"accuracy": 0.0}
    assert client.closed


@pytest.mark.parametrize("failure", ["Error while pulling image", "Test timed out after 1800 seconds"])
def test_swebench_incomplete_infrastructure_is_not_scored_zero(swebench_boundary, swebench_spec, failure):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "run_instance.log").write_text(failure)
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance

    item = replace(swebench_spec, requests=(ModelRequest(RequestKind.GENERATE, "question"),))
    plan = SourcePlan(
        "SWEbench", TaskRoute.CUSTOM, ([item],), ("SWEbench",),
        SourceMetrics(repeated=False, policy=SWEBenchBenchmark.REDUCER), {"swe_patch": Grader(swe_patch)},
    )
    result = asyncio.run(run_evaluations(
        SavedPatchModel({("question", 0): "diff --git a/file.py b/file.py"}), [plan], workers=1,
    ))

    assert result["results"] == {}
    assert result["task_outcomes"]["SWEbench"]["status"] is TaskStatus.FAILED
    assert result["task_outcomes"]["SWEbench"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE
    assert client.closed


def test_swebench_empty_model_patch_is_scored_zero(swebench_spec):
    grade = swe_patch(swebench_spec, [""])

    assert grade.metrics == {"accuracy": 0.0}


def test_swebench_candidate_test_timeout_is_scored_zero(swebench_boundary, swebench_spec):
    client, harness = swebench_boundary

    def run_instance(_instance, prediction, **kwargs):
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / prediction["instance_id"]
        log_dir.mkdir(parents=True)
        (log_dir / "run_instance.log").write_text("EvaluationError: Test timed out after 1800 seconds.")
        (log_dir / "test_output.txt").write_text("Tests started\n\nTimeout error: 1800 seconds exceeded.")
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance

    grade = swe_patch(swebench_spec, ["diff --git a/file.py b/file.py"])

    assert grade.metrics == {"accuracy": 0.0}
    assert grade.audit["test_timed_out"]
    assert client.closed


class SavedPatchModel:
    world_size = 1
    rank = 0

    def __init__(self, patches):
        self.patches = patches

    def tok_encode(self, text):
        return list(text.encode())

    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def generate_until(self, requests):
        return [self.patches[request.args[0], request.args[1].get("seed", 0)] for request in requests]


class ReversedReportDelivery:
    """Replay real item results in both orders before publishing their reduction."""

    def __init__(self, reducer, policy):
        self.reducer = reducer
        self.policy = policy
        self.results = []

    def add(self, result):
        self.results.append(result)

    def finish(self, task):
        forward = SourceMetrics(repeated=False, pass_at_k=(1, 2), policy=self.policy)
        for result in self.results:
            forward.add(result)
        for result in reversed(self.results):
            self.reducer.add(result)
        reduced = self.reducer.finish(task)
        assert reduced == forward.finish(task)
        return reduced


# Captured from pinned SWE-bench 4.1.0 make_run_report with the saved artifacts
# below. Native main supplies a Docker client, omitting cleanup-only metadata.
# Keep the reference data local so the base grader suite needs no SWE-bench extra.
_NATIVE_LIMITED_REPORT = {
    "total_instances": 7, "submitted_instances": 5, "completed_instances": 2,
    "resolved_instances": 1, "unresolved_instances": 1, "empty_patch_instances": 1, "error_instances": 2,
    "completed_ids": ["resolved", "unresolved"], "incomplete_ids": ["limited", "no-oracle"],
    "empty_patch_ids": ["empty"], "submitted_ids": ["empty", "rejected", "resolved", "timeout", "unresolved"],
    "resolved_ids": ["resolved"], "unresolved_ids": ["unresolved"], "error_ids": ["rejected", "timeout"],
    "schema_version": 2,
}
_NATIVE_DEBUG_REPORT = {
    "total_instances": 7, "submitted_instances": 2, "completed_instances": 0,
    "resolved_instances": 0, "unresolved_instances": 0, "empty_patch_instances": 1, "error_instances": 1,
    "completed_ids": [], "incomplete_ids": ["limited", "no-oracle", "resolved", "timeout", "unresolved"],
    "empty_patch_ids": ["empty"], "submitted_ids": ["empty", "rejected"],
    "resolved_ids": [], "unresolved_ids": [], "error_ids": ["rejected"], "schema_version": 2,
}


@pytest.mark.parametrize(("debug", "num_samples"), [(False, 1), (True, 1), (False, 2)])
def test_swebench_native_report_matches_limited_population_and_persisted_outcomes(
    swebench_boundary, tmp_path, monkeypatch, debug, num_samples,
):
    _client, harness = swebench_boundary
    ids = ["empty", "rejected", "timeout", "unresolved", "resolved", "limited", "no-oracle"]
    population = [{"instance_id": identity, "patch": "reference"} for identity in ids]
    oracle = [{"instance_id": identity, "text": identity} for identity in ["other-dataset", *ids[:-1]]]
    monkeypatch.setattr(
        "eval.chat_benchmarks.SWEbench.eval_instruct.load_dataset",
        lambda dataset, **_kwargs: oracle if dataset.endswith("_oracle") else population,
    )
    source = SWEBenchBenchmark(debug=debug)
    selected = ids[:2 if debug else 5]
    patches = {
        (identity, trial): "" if identity == "empty" else f"patch:{identity}:{trial}"
        for identity in selected for trial in range(num_samples)
    }
    native_payloads = {
        (identity, trial): {identity: {
            "patch_is_None": False, "patch_exists": True, "patch_successfully_applied": True,
            "resolved": (identity == "resolved") != bool(trial),
            "tests_status": {"FAIL_TO_PASS": {"success": ["test_fixed"], "failure": []}},
        }}
        for identity in ("unresolved", "resolved") for trial in range(num_samples)
    }

    def run_instance(_instance, prediction, **kwargs):
        identity = prediction["instance_id"]
        trial = int(prediction["model_patch"].rsplit(":", 1)[1])
        log_dir = harness.log_root / kwargs["run_id"] / prediction["model_name_or_path"] / identity
        log_dir.mkdir(parents=True, exist_ok=True)
        if identity == "rejected":
            (log_dir / "run_instance.log").write_text(">>>>> Patch Apply Failed: invalid patch")
        elif identity == "timeout":
            (log_dir / "run_instance.log").write_text("Test timed out after 1800 seconds.")
            (log_dir / "test_output.txt").write_text("Tests started\n\nTimeout error: 1800 seconds exceeded.")
        else:
            report = native_payloads[identity, trial]
            (log_dir / "report.json").write_text(json.dumps(report))
            return {"completed": True, "resolved": report[identity]["resolved"]}
        return {"completed": False, "resolved": False}

    harness.run_instance = run_instance
    expected_reports = {"0": _NATIVE_DEBUG_REPORT if debug else _NATIVE_LIMITED_REPORT}
    if num_samples > 1:
        expected_reports["1"] = {
            **_NATIVE_LIMITED_REPORT, "resolved_ids": ["unresolved"], "unresolved_ids": ["resolved"],
        }

    model = SavedPatchModel(patches)
    plans = source_plans(
        ["SWEbench"], {"SWEbench": TaskRoute.CUSTOM}, Namespace(get_benchmark=lambda _name: source), None,
        RunSpec(limit=5, num_samples=num_samples, pass_at_k=(1, 2)), model,
    )
    plans[0].graders = {"swe_patch": Grader(swe_patch)}
    if num_samples > 1:
        plans[0].reducer = ReversedReportDelivery(plans[0].reducer, source.REDUCER)
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "run", ["SWEbench"]) as output:
        result = asyncio.run(run_evaluations(model, plans, workers=2, sink=output, supports_request_seed=True))
        output.finish(result)

    assert result["task_outcomes"]["SWEbench"]["status"] is TaskStatus.SUCCEEDED
    restored = read_finestore_output(root, "run")
    assert restored == result
    metrics = restored["results"]["SWEbench"]
    assert metrics["total_instances"] == len(population)
    assert metrics["schema_version"] == 2
    if num_samples == 1:
        assert {key: metrics[key] for key in expected_reports["0"]} == expected_reports["0"]
        assert metrics["accuracy"] == (0.0 if debug else 0.2)
    else:
        assert metrics["pass@1"] == 0.2
        assert metrics["pass@2"] == 0.4
        for trial, expected in expected_reports.items():
            assert {"total_instances": metrics["total_instances"], "schema_version": metrics["schema_version"],
                    **metrics["reports_by_trial"][trial]} == expected

    rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).to_pylist(maps_as_pydicts="strict")
    assert len(rows) == len(selected) * num_samples
    statuses = {"empty": "empty_patch", "rejected": "patch_rejected", "timeout": "test_timeout"}
    for row in rows:
        sample = sample_from_archive_row(row)
        identity = json.loads(sample.doc)["instance_id"]
        trial = int(row["trial_id"] or 0)
        detail = json.loads(sample.grading.detail)
        expected = expected_reports[str(trial)]
        assert detail["report_outcome"] == {
            "instance_id": identity,
            **{name.removesuffix("_ids"): identity in values for name, values in expected.items() if name.endswith("_ids")},
        }
        if identity in statuses:
            assert detail["report_status"] == statuses[identity]
            assert "report" not in detail
        else:
            assert detail["report"] == native_payloads[identity, trial]
            assert detail["report_status"] == ("resolved" if sample.correct else "unresolved")
        assert "report_population" not in detail
        assert "report_metadata" not in detail
