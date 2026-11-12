# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin.evaluation.bfcl_parity import admit_native_run, compare_run_clusters

from experiments.evaluation import bfcl_parity as cli

TASKS = ("task-a", "task-b")


def native(run_id, rewards, *, target="model-a"):
    record = {
        "run_id": run_id,
        "status": "succeeded",
        "model": {
            "backend": "vllm",
            "config": {
                "name": target,
                "location": "s3://bucket/" + target,
                "revision": "weights-" + target,
                "identity": "producer-" + target,
                "tokenizer": "shared-tokenizer",
                "tokenizer_revision": "fixed-tokenizer",
                "generation": {"temperature": 1.0},
                "agent": {"version": "fixed-agent"},
                "serve": {"tool_call_parser": "hermes", "vllm_source": "fixed-runtime"},
            },
        },
        "eval": {
            "mechanism": "harbor",
            "source_digest": "fixed-yaml",
            "harbor": {"dataset": "bfcl_parity", "version": "fixed-tasks"},
            "tasks": [{"benchmark": {"n_benchmark": 2, "n_attempted": 2}}],
        },
        "provenance": {"eval_runtime": "fixed-harbor-runtime"},
    }
    samples = [
        {
            "doc_id": name,
            "task": "bfcl_parity",
            "filter": "",
            "metrics": [["reward", reward]],
            "grading": {
                "method": "harbor:verifier",
                "metric": "reward",
                "score": reward,
                "passed": None if reward is None else bool(reward),
                "detail": json.dumps(
                    {
                        "scored": reward is not None,
                        "rewards": {"reward": reward},
                        "error": None if reward is not None else {"type": "EnvironmentStartTimeoutError"},
                    }
                ),
            },
        }
        for name, reward in zip(TASKS, rewards, strict=True)
    ]
    return record, samples


def run(run_id, rewards, *, logical=None, target="model-a"):
    return admit_native_run(*native(run_id, rewards, target=target), logical_run=logical or run_id, task_ids=TASKS)


def test_native_scored_agent_zero_is_distinct_from_unscored_infrastructure():
    record, samples = native("physical", (0, None))
    detail = json.loads(samples[0]["grading"]["detail"])
    detail["error"] = {"type": "AgentTimeoutError", "category": "agent"}
    samples[0]["grading"]["detail"] = json.dumps(detail)
    samples[1]["metrics"] = [["reward", 0.0]]  # A transport default cannot turn scored:false into a zero.
    parsed = admit_native_run(record, samples, logical_run="slot", task_ids=TASKS)
    assert parsed.rewards == (0.0, None) and not parsed.complete
    assert parsed.errors["task-a"]["type"] == "AgentTimeoutError"
    assert parsed.errors["task-b"]["type"] == "EnvironmentStartTimeoutError"


def test_run_bootstrap_matches_analytic_binomial_cluster_interval():
    # Five all-or-none candidate runs give Binomial(5,.2)/5 bootstrap means:
    # exact population percentile endpoints are 0 and .6. Resampling tasks
    # independently across runs would use a different sampling unit.
    candidate = [run(f"c{i}", (int(i == 4),) * 2, target="candidate") for i in range(5)]
    reference = [run(f"r{i}", (0, 1), target="reference") for i in range(5)]
    result = compare_run_clusters({"candidate": candidate, "reference": reference}, reference="reference")
    assert result["arms"]["candidate"]["mean"] == 0.2
    assert result["arms"]["candidate"]["sample_sd"] == pytest.approx(math.sqrt(0.2))
    assert result["arms"]["candidate"]["ci95"] == [0, 0.6]
    assert result["contrasts"]["candidate"]["delta"] == pytest.approx(-0.3)
    assert result["contrasts"]["candidate"]["ci95"] == pytest.approx([-0.5, 0.1])


def test_one_complete_whole_replacement_retains_original_nulls_and_no_best_of():
    attempts = [run("original-1", (0, None), logical="1"), run("replacement-1", (1, 1), logical="1")]
    attempts += [run(f"original-{i}", (0, 1), logical=str(i)) for i in range(2, 6)]
    result = compare_run_clusters({"reference": attempts}, reference="reference", iterations=100)
    summary = result["arms"]["reference"]
    assert summary["n_complete"] == summary["n_logical"] == 5 and summary["n_physical"] == 6
    assert summary["logical_runs"]["1"]["attempts"][0]["rewards"] == (0, None)
    attempts.append(run("second-valid-for-1", (0, 0), logical="1"))
    with pytest.raises(ValueError, match="Multiple complete"):
        compare_run_clusters({"reference": attempts}, reference="reference", iterations=100)


def test_fewer_than_five_full_runs_has_no_primary_or_interval():
    runs = [run(str(i), (0, 1)) for i in range(4)] + [run("incomplete", (0, None))]
    result = compare_run_clusters({"reference": runs}, reference="reference", iterations=100)
    arm = result["arms"]["reference"]
    assert arm["n_complete"] == 4 and arm["n_physical"] == 5
    assert arm["mean"] is arm["sample_sd"] is arm["ci95"] is None
    assert arm["primary_available"] is False


@pytest.mark.parametrize("field", ["runtime", "tokenizer", "agent", "parser", "source", "hardware", "serving"])
def test_recorded_evaluator_method_drift_is_not_hidden_by_target_exclusions(field):
    a = run("a", (0, 1))
    record, samples = native("b", (0, 1), target="other-target")
    config = record["model"]["config"]
    if field == "runtime":
        record["provenance"]["eval_runtime"] = "different-runtime"
    elif field == "tokenizer":
        config["tokenizer_revision"] = "different-tokenizer"
    elif field == "agent":
        config["agent"]["version"] = "different-agent"
    elif field == "parser":
        config["serve"]["tool_call_parser"] = "different-parser"
    elif field in ("hardware", "serving"):
        record[field] = {"different-recorded-runtime": True}
    else:
        record["eval"]["source_digest"] = "different-yaml"
    b = admit_native_run(record, samples, logical_run="b", task_ids=TASKS)
    with pytest.raises(ValueError, match="method/task-set"):
        compare_run_clusters({"reference": [a], "candidate": [b]}, reference="reference", iterations=100)


def test_target_revision_exclusion_does_not_hide_default_tokenizer_revision():
    records = [native(str(i), (0, 1)) for i in range(2)]
    for record, _ in records:
        record["model"]["config"]["tokenizer"] = None
        record["model"]["config"]["tokenizer_revision"] = None
    records[1][0]["model"]["config"]["revision"] = "different-default-tokenizer-revision"
    arms = {
        label: [admit_native_run(*pair, logical_run=label, task_ids=TASKS)]
        for label, pair in zip(("reference", "candidate"), records, strict=True)
    }
    with pytest.raises(ValueError, match="method/task-set"):
        compare_run_clusters(arms, reference="reference", iterations=100)


def test_different_checkpoint_within_one_arm_is_rejected():
    with pytest.raises(ValueError, match="Different target"):
        compare_run_clusters(
            {"reference": [run("a", (0, 1)), run("b", (1, 1), target="other")]}, reference="reference", iterations=100
        )


@pytest.mark.parametrize(
    "corruption", ["duplicate", "wrong-task", "subset", "reward", "benchmark", "filter", "sample-filter", "passed"]
)
def test_native_task_and_grading_joins_reject_inconsistent_sources(corruption):
    record, samples = native("a", (0, 1))
    tasks = TASKS
    if corruption == "duplicate":
        samples.append(copy.deepcopy(samples[0]))
    elif corruption == "wrong-task":
        samples[0]["doc_id"] = "other"
    elif corruption == "subset":
        tasks = TASKS[:1]
        samples = samples[:1]
    elif corruption == "benchmark":
        samples[0]["task"] = "other-benchmark"
    elif corruption == "filter":
        samples[0]["grading"]["filter"] = "other-filter"
    elif corruption == "sample-filter":
        samples[0]["filter"] = "other-filter"
    elif corruption == "passed":
        samples[0]["grading"]["passed"] = True
    else:
        samples[0]["grading"]["score"] = 1
    with pytest.raises(ValueError):
        admit_native_run(record, samples, logical_run="a", task_ids=tasks)


def test_analysis_cli_verifies_native_byte_pins_and_writes_a_complete_report(tmp_path):
    arms = {}
    for label in ("reference", "candidate"):
        arms[label] = []
        for i in range(5):
            descriptor = {"logical_run": str(i)}
            for kind, value in zip(("record", "samples"), native(f"{label}{i}", (0, 1), target=label), strict=True):
                path = tmp_path / f"{label}{i}-{kind}.json"
                raw = json.dumps(value).encode()
                path.write_bytes(raw)
                descriptor[kind] = str(path)
                descriptor[kind + "_sha256"] = hashlib.sha256(raw).hexdigest()
            arms[label].append(descriptor)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task_ids": TASKS, "reference": "reference", "arms": arms}))
    output = tmp_path / "report.json"
    args = ["analyze", "--manifest", str(manifest), "--output", str(output), "--iterations", "100"]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["arms"]["candidate"]["n_complete"] == 5 and report["contrasts"]["candidate"]["ci95"] == [0, 0]
    Path(arms["reference"][0]["samples"]).write_text("changed")
    output.unlink()
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code != 0 and not output.exists()


def test_rollup_discrepancy_preserves_raw_scores_but_refuses_full_admission():
    record, samples = native("physical", (0, 1))
    record["coverage"] = {"bfcl_parity": {"n_scored": 1}}
    parsed = admit_native_run(record, samples, logical_run="slot", task_ids=TASKS)
    assert parsed.rewards == (0, 1) and parsed.coverage_mismatch and not parsed.complete
    assert parsed.native_coverage == record["coverage"]


def test_terminal_startup_failure_keeps_global_error_and_zero_observed_rows():
    record, _ = native("startup", (0, 1))
    record.update(status="infra_failed", error={"type": "RemoteInferenceStartupError"})
    parsed = admit_native_run(record, [], logical_run="slot", task_ids=TASKS)
    assert parsed.rewards == (None, None) and parsed.n_observed == 0 and not parsed.complete
    result = compare_run_clusters({"reference": [parsed]}, reference="reference", iterations=100)
    attempt = result["arms"]["reference"]["logical_runs"]["slot"]["attempts"][0]
    assert attempt["n_observed"] == attempt["n_scored"] == 0
    assert attempt["native_error"] == record["error"]
