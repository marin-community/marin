# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline recovery contracts; no model, grading account or cloud execution."""

import copy
import json
from pathlib import Path

import pytest
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from experiments.post_training.snowball_recovery import summarize_rows
from experiments.post_training.snowball_recovery_records import (
    LIBRARY_COMMIT,
    RUNTIME_COMMIT,
    encoded,
    join_receipts,
    pinned_bytes,
    prepare_records,
    require_disjoint,
    require_source,
    saved_answer,
    sha256,
)


def decode(ids, *, skip_special_tokens=False):
    return {
        (1,): "prompt",
        (2, 99): "<think>reason</think> answer<eos>",
        (2,): "<think>reason</think> answer",
        (2, 98): "<think>reason</think> answer<|eot_id|>",
        (3,): " unmarked answer ",
    }[tuple(ids)]


def extract(text):
    return text.split("</think>")[-1].strip().removesuffix("<|eot_id|>").strip()


@pytest.fixture(params=["sft", "drdoom"])
def recovery(request):
    arm = request.param
    failed_count = 2 if arm == "sft" else 1
    rows = [
        {
            "task_id": f"task-{i}",
            "evaluation_family": "stem" if i % 2 else "reasoning_gym",
            "contract_id": f"contract-{i}",
            "score": None if i < failed_count else 0.5,
            "stop_reason": "stop",
            "verification_status": "error" if i < failed_count else "verified",
            "failure_type": "VerifierRuntimeError" if i < failed_count else None,
        }
        for i in range(910)
    ]
    report = summarize_rows(rows)
    report.update(
        step=50,
        protocol={"max_new_tokens": 8192, "request_window_tokens": 24576, "temperature": 0.0},
        panel_sha256="a" * 64,
    )
    panel = [
        {"task_id": row["task_id"], "source_contract_id": row["contract_id"], "task_rollout_spec": "unused"}
        for row in rows
    ]
    probe = {
        "status": "passed",
        "validation_sha256": report["panel_sha256"],
        "validation_tasks": [{"task_id": row["task_id"], "dataset_uid": str(i)} for i, row in enumerate(rows)],
    }
    dumps = []
    for i in range(failed_count):
        spec = {
            "task": {
                "id": rows[i]["task_id"],
                "answer_type": "text",
                "answer_format": {"kind": "plain_text"},
                "grader": {"kind": "script"},
                "resources": {},
                "final_tools": [],
                "interaction_tools": [],
                "tool_bindings": {},
                "output_paths": [],
            },
            "task_machine": None,
            "verifier_machine": {"backend": "daytona", "startup_timeout": 600},
            "limits": {"max_turns": 1, "verifier_timeout": 900, "attempt_timeout": 3600},
        }
        panel[i]["task_rollout_spec"] = json.dumps(spec)
        row = {
            "uid": str(i),
            "verification": {"status": "error", "diagnostics": {} if arm == "drdoom" else {"operation": "grade"}},
            "server_error": None,
            "exception_type": "VerifierRuntimeError",
            "prompt_token_ids": [1],
            "response_token_ids": [2, 99],
            "input_prompt": decode([1]),
            "output_response": decode([2, 99]),
            "stop_reason": "stop",
        }
        raw = (json.dumps(row) + "\n").encode()
        dumps.append((row, sha256(raw)))
    reviewed = {rows[i]["task_id"]: dumps[i][1] for i in range(failed_count)}
    prepared = prepare_records(arm, report, probe, panel, dumps, reviewed, decode, 99, extract)
    source = {
        "runtime_commit": RUNTIME_COMMIT,
        "library_revisions": {
            name: LIBRARY_COMMIT for name in ("marin-taskcompendium", "marin-rolloutengine", "marin-verifyit")
        },
    }
    evidence, receipts = {}, []
    for row in prepared["requests"]:
        execution = {key: row[key] for key in ("task_id", "answer_sha256", "task_rollout_spec_sha256")}
        execution.update(
            source=source,
            score=1.0,
            grade={
                "status": "graded",
                "reward": 1.0,
                "score_min": 0.0,
                "score_max": 1.0,
                "diagnostics": {"stdout": "1.0", "stderr": "", "exit_code": 0},
            },
        )
        raw = encoded(execution)
        identity = sha256(raw)
        evidence[identity] = raw
        receipts.append(
            {
                **{key: value for key, value in execution.items() if key != "grade"},
                "attempt_count": 1,
                "execution_evidence_sha256": identity,
            }
        )
    return arm, report, probe, panel, dumps, reviewed, prepared, receipts, evidence


def test_saved_completions_recover_only_original_failures_and_preserve_every_other_row(recovery):
    arm, report, _, _, _, _, prepared, receipts, evidence = recovery
    before = encoded(report)
    derived = join_receipts(report, prepared, receipts, evidence, summarize_rows)
    assert derived["unchanged_scored_rows"] == (908 if arm == "sft" else 909)
    assert derived["summary"]["complete"] is True
    assert derived["summary"]["pairing_identity"] == report["pairing_identity"]
    assert derived["original_error_rows"] == [row for row in report["tasks"] if row["score"] is None]
    assert derived["regenerated_completions"] == 0 and derived["original_training_job_success"] is False
    assert encoded(report) == before
    original_good = [row for row in report["tasks"] if row["score"] is not None]
    assert original_good == [row for row in derived["summary"]["tasks"] if row["task_id"] not in reviewed_ids(prepared)]


def reviewed_ids(prepared):
    return {row["task_id"] for row in prepared["requests"]}


@pytest.mark.parametrize("change", ["answer", "machine", "missing", "duplicate", "evidence", "score"])
def test_receipt_cannot_change_answer_machine_or_leave_fixed_panel_incomplete(recovery, change):
    _, report, _, _, _, _, prepared, receipts, evidence = copy.deepcopy(recovery)
    if change == "answer":
        receipts[0]["answer_sha256"] = "b" * 64
    elif change == "machine":
        receipts[0]["task_rollout_spec_sha256"] = "b" * 64
    elif change == "missing":
        receipts.pop()
    elif change == "duplicate":
        receipts.append(receipts[0])
    elif change == "evidence":
        evidence[receipts[0]["execution_evidence_sha256"]] = b"changed"
    else:
        receipts[0]["score"] = 0.0
    with pytest.raises(ValueError):
        join_receipts(report, prepared, receipts, evidence, summarize_rows)


@pytest.mark.parametrize("change", ["saved_row", "tools", "pairing", "model_error", "panel_identity"])
def test_preparation_refuses_changed_original_evidence(recovery, change):
    arm, report, probe, panel, dumps, reviewed, _, _, _ = copy.deepcopy(recovery)
    if change == "saved_row":
        reviewed[next(iter(reviewed))] = "b" * 64
    elif change == "tools":
        spec = json.loads(panel[0]["task_rollout_spec"])
        spec["task"]["interaction_tools"] = [{"name": "shell"}]
        panel[0]["task_rollout_spec"] = json.dumps(spec)
    elif change == "pairing":
        report["pairing_identity"] = "b" * 64
    elif change == "model_error":
        dumps[0][0]["server_error"] = {"type": "failure"}
    else:
        panel[0]["source_contract_id"] = "another-original-contract"
    with pytest.raises(ValueError):
        prepare_records(arm, report, probe, panel, dumps, reviewed, decode, 99, extract)


def test_answer_reconstruction_uses_original_eos_and_reasoning_path_without_unmarked_guessing():
    assert saved_answer([2, 99], 99, decode, extract) == "answer"
    assert saved_answer([2], 99, decode, extract) == "answer"
    assert saved_answer([2, 98], 99, decode, extract) == "answer"
    with pytest.raises(ValueError):
        saved_answer([], 99, decode, extract)
    with pytest.raises(ValueError):
        saved_answer([3, 99], 99, decode, extract)


@pytest.mark.parametrize(
    "target", ["s3://bucket/original", "s3://bucket/original/recovery", "s3://bucket/", "s3://bucket/original/../new"]
)
def test_recovery_output_cannot_overlap_original_tree(target):
    with pytest.raises(ValueError):
        require_disjoint(target, ["s3://bucket/original"])


def test_pinned_input_and_source_identity_cannot_be_relabelled():
    raw = b"immutable original bytes"
    assert pinned_bytes(raw, sha256(raw)) == raw
    with pytest.raises(ValueError):
        pinned_bytes(raw + b"changed", sha256(raw))
    revisions = {name: LIBRARY_COMMIT for name in ("marin-taskcompendium", "marin-rolloutengine", "marin-verifyit")}
    require_source(RUNTIME_COMMIT, revisions)
    with pytest.raises(ValueError):
        require_source("new-runtime", revisions)
    revisions["marin-rolloutengine"] = "new-library"
    with pytest.raises(ValueError):
        require_source(RUNTIME_COMMIT, revisions)


def test_conditional_reservation_refuses_a_second_process_or_restart(tmp_path):
    marker = str(tmp_path / "attempt-started.json")
    conditional_object(marker).write(b'{"owner":"first"}', expected_version=None)
    with pytest.raises(ConditionalWriteError):
        conditional_object(marker).write(b'{"owner":"second"}', expected_version=None)
    assert conditional_object(marker).read().data == b'{"owner":"first"}'


@pytest.mark.manual
@pytest.mark.parametrize("root_kind", ["checkpoint_root", "resolved_config_uri", "ray"])
def test_regrade_rejects_original_tree_before_first_output_or_account_access(tmp_path, monkeypatch, capfd, root_kind):
    # Run explicitly with the clone's frozen external runtime available. These
    # are real CPU subprocess checks; no grader sandbox or policy is acquired.
    import subprocess  # noqa: PLC0415 - optional subprocess integration fixture

    import yaml  # noqa: PLC0415
    from marin.execution.lazy import StepContext  # noqa: PLC0415 - optional experiment dependencies

    from experiments.post_training.snowball_inventory_compare import Initializer, comparison_step  # noqa: PLC0415
    from experiments.post_training.snowball_recovery import RecoveryConfig, Stage, run_recovery  # noqa: PLC0415

    refs = {}
    roots = {}
    for arm in ("sft", "drdoom"):
        step = comparison_step(
            Initializer(arm),
            version="2026.10.20.999",
            model_uri=f"s3://marin-us-east-02a/test/{arm}/model",
            pool_uri="s3://marin-us-east-02a/test/pool",
            pool_version="2026.10.20.1",
            pool_sha256="a" * 64,
            validation_uri="s3://marin-us-east-02a/test/panel",
            validation_version="2026.10.20.1",
            validation_sha256="b" * 64,
        )
        config = step.build_config(
            StepContext.for_run(
                output_path=f"s3://marin-us-east-02a/test/{arm}",
                prefix="s3://marin-us-east-02a/test",
                runtime_args=step.runtime_args,
                deps=step.deps,
            )
        )
        document = yaml.safe_load(config.training.launch_config_yaml)
        base = tmp_path / "original" / arm
        for key in ("checkpoint_root", "export_root", "attempts_root"):
            document["artifacts"][key] = str(base / key)
        document["artifacts"]["resolved_config_uri"] = str(base / "controls" / "resolved.yaml")
        document["artifacts"]["terminal_manifest_uri"] = str(base / "controls" / "terminal.json")
        document["ray"]["rendezvous_dir"] = str(base / "ray" / "rendezvous")
        document["ray"]["log_dir"] = str(base / "ray" / "logs")
        path = tmp_path / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(document))
        refs[arm] = {
            "arm": arm,
            "training_config": {"uri": str(path), "sha256": sha256(path.read_bytes())},
            "checkpoint_root": document["artifacts"]["checkpoint_root"],
        }
        roots[arm] = document
    request = {**refs["sft"], "peer": refs["drdoom"]}
    manifest = tmp_path / "request.json"
    manifest.write_bytes(encoded(request))
    target = roots["sft"]["ray"]["rendezvous_dir"] if root_kind == "ray" else roots["sft"]["artifacts"][root_kind]
    if root_kind == "resolved_config_uri":
        target = str(Path(target).parent)
    monkeypatch.delenv("DAYTONA_EVAL_API_KEY", raising=False)
    with pytest.raises(subprocess.CalledProcessError):
        run_recovery(RecoveryConfig(str(manifest), sha256(manifest.read_bytes()), "sft", Stage.REGRADE, target))
    assert "Recovery output overlaps an original output or control tree" in capfd.readouterr().err
    assert not (tmp_path / "original").exists()
