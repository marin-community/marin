# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint
from rigging.runtime_bundle import RuntimeBundle
from taskcompendium.grading import exact_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi.bootstrap_loop import QualifiedTask, calibration_measurements
from experiments.post_training.russell_rsi.calibration_recovery import (
    CalibrationRecoveryConfig,
    GradeOnlyRecoveryConfig,
    PinnedFile,
    grade_only_recovery_step,
    recover_calibration,
    recover_grade_only_calibration,
    replay_grade,
)
from experiments.post_training.russell_rsi.repair_tasks import canonical_sha256


def pin(path, value):
    path.write_text(json.dumps(value) + "\n")
    return PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def recovery(tmp_path):
    tasks = [
        TaskSpec(
            id=f"task-{index}",
            context=ConversationInput(events=(TextMessage(role="user", content=f"Return {index}."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            verifier=exact_answer(str(index)),
            source=Source(dataset="source", revision="pin", row=str(index), importer_revision="1"),
        )
        for index in range(18)
    ]
    parquet = tmp_path / "train.parquet"
    write_tasks(str(parquet), tasks)
    bank = {
        "tasks": [
            {
                "task_id": task.id,
                "task_sha256": canonical_sha256(task.model_dump(mode="json")),
                "admission_sha256": f"admission-{task.id}",
                "source_id": f"source-{task.id}",
                "capability": "types",
                "contract_id": f"contract-{task.id}",
            }
            for task in tasks
        ]
    }
    runtime = RuntimeBundle("manifest", "manifest-hash", "archive", "archive-hash", "/opt")
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "original timestamp"}
    turn = {
        "message": {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "function": {"name": "shell", "arguments": '{"command": "ls -la"}'}}],
        },
        "response_token_ids": [3, 4],
    }
    records = [
        {
            "task_id": tasks[index // 8].id,
            "steps": [{"turn": deepcopy(turn), "transition": {"observations": [observation]}, "messages": []}],
            "messages": [observation],
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
            "loss_mask": [0, 0, 1, 1],
            "stop_reason": "stop",
            "grade": {"status": "graded", "reward": float(index % 2), "error": None, "failure": None},
            "interrupted_operation": None,
            "execution_error": None,
        }
        for index in range(144)
    ]
    for index in (103, 116, 141):
        records[index]["grade"] = {"status": "unavailable", "reward": None, "error": "No final grade"}
        records[index]["interrupted_operation"] = "start" if index == 103 else "grade"
    records[103].update(steps=[], messages=[], prompt_token_ids=[], response_token_ids=[], loss_mask=[])
    replacement_parquet = tmp_path / "replacement.parquet"
    write_tasks(str(replacement_parquet), [tasks[103 // 8]])
    lines = [(json.dumps(record) + "\n").encode() for record in records]
    traces = tmp_path / "traces.jsonl"
    traces.write_bytes(b"".join(lines))
    rewards = {
        task.id: [
            record["grade"]["reward"]
            for record in records
            if record["task_id"] == task.id and record["grade"]["status"] == "graded"
        ]
        for task in tasks
    }
    summary = {
        "model_identity": "parent",
        "tasks_identity": "bank",
        "count": 18,
        "samples_per_task": 8,
        "task_rewards": rewards,
    }
    replay_files = []
    reviews = []
    for index in (116, 141):
        recovered = deepcopy(records[index])
        recovered["grade"] = {"status": "graded", "reward": 0.0, "error": None, "failure": None}
        replay = {
            "original_line_index": index,
            "original_line_sha256": hashlib.sha256(lines[index]).hexdigest(),
            "task_id": records[index]["task_id"],
            "task_sha256": hashlib.sha256(tasks[index // 8].model_dump_json().encode()).hexdigest(),
            "source_commit": "original-runtime",
            "original_runtime_source": "original-runtime",
            "runtime_bundle": asdict(runtime),
            "model_requests": 0,
            "execution_error": None,
            "rollout": recovered,
        }
        evidence = pin(tmp_path / f"replay-{index}.json", replay)
        replay_files.append(evidence)
        reviews.append(
            {
                "original_line_index": index,
                "original_line_sha256": replay["original_line_sha256"],
                "replay_sha256": evidence.sha256,
                "accepted_reward": 0.0,
                "observation_exceptions": [],
            }
        )
    issuance = {
        "original_line_index": 103,
        "original_line_sha256": hashlib.sha256(lines[103]).hexdigest(),
        "original_traces_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
        "original_task_sha256": bank["tasks"][103 // 8]["task_sha256"],
        "tasks_sha256": hashlib.sha256(replacement_parquet.read_bytes()).hexdigest(),
        "runtime_module_hashes": {"engine": "engine-hash"},
        "evaluation": {
            "model_identity": "parent",
            "model_uri": "model",
            "tokenizer": "tokenizer",
            "tokenizer_revision": "revision",
            "runtime_bundle": asdict(runtime),
            "temperature": 1.0,
            "samples_per_task": 1,
            "limit": 1,
            "tasks_identity": "replacement",
        },
    }
    replacement = deepcopy(records[0])
    replacement["task_id"] = records[103]["task_id"]
    replacement_summary = {
        "model_identity": "parent",
        "tasks_identity": "replacement",
        "task_rewards": {replacement["task_id"]: [0.0]},
    }
    config = CalibrationRecoveryConfig(
        original_summary=pin(tmp_path / "summary.json", summary),
        original_traces=PinnedFile(str(traces), hashlib.sha256(traces.read_bytes()).hexdigest()),
        original_tasks=PinnedFile(str(parquet), hashlib.sha256(parquet.read_bytes()).hexdigest()),
        original_bank=pin(tmp_path / "bank.json", bank),
        review=pin(tmp_path / "review.json", {"replays": reviews, "startup_replacement": {"original_line_index": 103}}),
        replays=tuple(replay_files),
        replacement_issuance=pin(tmp_path / "issued.json", issuance),
        replacement_tasks=PinnedFile(str(replacement_parquet), issuance["tasks_sha256"]),
        replacement_summary=pin(tmp_path / "replacement-summary.json", replacement_summary),
        replacement_traces=pin(tmp_path / "replacement.jsonl", replacement),
        model_identity="parent",
        model_uri="model",
        tokenizer="tokenizer",
        tokenizer_revision="revision",
        bank_identity="bank",
        runtime_source="original-runtime",
        runtime_bundle=runtime,
        runtime_module_hashes={"engine": "engine-hash"},
        output_path=str(tmp_path / "result"),
    )
    return config, records, bank


def test_recovery_preserves_original_grades_and_links_three_completions(recovery, tmp_path):
    config, originals, bank = recovery
    recover_calibration(config)
    result = json.loads((tmp_path / "result" / "recovery.json").read_text())
    assert result["original_grade_count"] == 141
    assert len(result["attempts"]) == 144
    for index, attempt in enumerate(result["attempts"]):
        assert attempt["original_grade"] == originals[index]["grade"]
        if index in (103, 116, 141):
            assert attempt["completed_grade"]["reward"] == 0
            assert attempt["grade_evidence"]["uri"] != config.original_traces.uri
        else:
            assert attempt["completed_grade"] == originals[index]["grade"]
    summary = json.loads((tmp_path / "result" / "failure_summary.json").read_text())
    measured = calibration_measurements(summary, tuple(QualifiedTask(**row) for row in bank["tasks"]), "parent", "bank")
    assert len(measured) == 18
    assert all(len(item.rewards) == 8 for item in measured)
    sealed_bytes = (tmp_path / "result" / "recovery.json").read_bytes()
    recover_calibration(config)
    assert (tmp_path / "result" / "recovery.json").read_bytes() == sealed_bytes
    changed_review = config.review.read_json() | {"note": "different review input"}
    with pytest.raises(ValueError, match="Immutable record differs"):
        recover_calibration(replace(config, review=pin(tmp_path / "different-review.json", changed_review)))


@pytest.mark.parametrize("change", ["missing_grade", "model_identity", "tasks_identity"])
def test_selection_requires_complete_calibration_for_current_model_and_bank(recovery, tmp_path, change):
    config, _, bank = recovery
    recover_calibration(config)
    summary = json.loads((tmp_path / "result" / "failure_summary.json").read_text())
    if change == "missing_grade":
        summary["task_rewards"]["task-0"].pop()
    else:
        summary[change] = "different"
    with pytest.raises(ValueError):
        calibration_measurements(summary, tuple(QualifiedTask(**row) for row in bank["tasks"]), "parent", "bank")


@pytest.mark.parametrize(
    "change,message",
    [
        ("action", "saved model turn"),
        ("observation", "unreviewed observation"),
        ("tokens", "response_token_ids"),
        ("duplicate", "duplicates an attempt"),
        ("ungraded", "finite grade"),
        ("model", "checkpoint or task-bank identity"),
        ("bank", "checkpoint or task-bank identity"),
        ("replacement_task", "original attempt or sampling protocol"),
        ("temperature", "original attempt or sampling protocol"),
    ],
)
def test_recovery_rejects_changed_or_incomplete_evidence_before_output(recovery, tmp_path, change, message):
    config, _, _ = recovery
    if change in ("model", "bank"):
        config = replace(config, **{f"{change}_identity": "different"})
    elif change == "duplicate":
        config = replace(config, replays=(*config.replays, config.replays[0]))
    elif change == "ungraded":
        value = config.replacement_traces.read_json()
        value["grade"] = {"status": "unavailable", "reward": None}
        config = replace(config, replacement_traces=pin(tmp_path / "ungraded.jsonl", value))
    elif change == "replacement_task":
        config = replace(config, replacement_tasks=config.original_tasks)
        issuance = config.replacement_issuance.read_json()
        issuance["tasks_sha256"] = config.original_tasks.sha256
        config = replace(config, replacement_issuance=pin(tmp_path / "changed-issuance.json", issuance))
    elif change == "temperature":
        issuance = config.replacement_issuance.read_json()
        issuance["evaluation"]["temperature"] = 0.0
        config = replace(config, replacement_issuance=pin(tmp_path / "changed-issuance.json", issuance))
    else:
        value = config.replays[0].read_json()
        value["all_observations_match"] = True
        value["all_model_requests_match"] = True
        if change == "action":
            value["rollout"]["steps"][0]["turn"]["message"]["tool_calls"][0]["function"][
                "arguments"
            ] = '{"command": "echo changed"}'
        elif change == "observation":
            value["rollout"]["steps"][0]["transition"]["observations"][0]["content"] = "different output"
        else:
            value["rollout"]["response_token_ids"] = [9]
        replay = pin(tmp_path / "changed-replay.json", value)
        review = config.review.read_json()
        review["replays"][0]["replay_sha256"] = replay.sha256
        config = replace(
            config, replays=(replay, config.replays[1]), review=pin(tmp_path / "changed-review.json", review)
        )
    with pytest.raises(ValueError, match=message):
        recover_calibration(config)
    assert not (tmp_path / "result").exists()


def test_replay_accepts_only_exact_reviewed_observation_change(recovery):
    config, originals, _ = recovery
    replay = config.replays[0].read_json()
    decision = config.review.read_json()["replays"][0]
    after = {"role": "tool", "tool_call_id": "call-1", "content": "reviewed timestamp"}
    replay["rollout"]["steps"][0]["transition"]["observations"] = [after]
    replay["rollout"]["messages"] = [after]
    decision.update(
        exact_command="ls -la",
        tool_call_id="call-1",
        observation_exceptions=[{"step_index": 0, "original": originals[116]["messages"], "replay": [deepcopy(after)]}],
    )
    assert replay_grade(originals[116], replay, decision)["reward"] == 0
    after["content"] = "unreviewed timestamp"
    with pytest.raises(ValueError, match="exact reviewed"):
        replay_grade(originals[116], replay, decision)


@pytest.fixture
def grade_only(recovery, tmp_path):
    old, records, bank = recovery
    tasks = list(read_tasks(old.original_tasks.uri))
    for index in range(18, 26):
        task = tasks[-1].model_copy(update={"id": f"task-{index}"})
        tasks.append(task)
        bank["tasks"].append(
            {
                **bank["tasks"][-1],
                "task_id": task.id,
                "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            }
        )
        for _ in range(8):
            records.append({**deepcopy(records[-1]), "task_id": task.id})
    task_path = tmp_path / "grade-only-tasks.parquet"
    write_tasks(str(task_path), tasks)
    old = replace(
        old,
        original_tasks=PinnedFile(str(task_path), hashlib.sha256(task_path.read_bytes()).hexdigest()),
        original_bank=pin(tmp_path / "grade-only-bank.json", bank),
    )
    for index in (103, 141):
        records[index]["grade"] = {"status": "graded", "reward": float(index % 2), "error": None, "failure": None}
    for index, record in enumerate(records):
        record["sample_index"] = 7 - index % 8
    lines = [(json.dumps(record) + "\n").encode() for record in records]
    path = tmp_path / "grade-only-originals.jsonl"
    path.write_bytes(b"".join(lines))
    summary = old.original_summary.read_json()
    summary["count"] = 26
    summary["task_rewards"] = {
        row["task_id"]: [
            record["grade"]["reward"]
            for record in records
            if record["task_id"] == row["task_id"] and record["grade"]["status"] == "graded"
        ]
        for row in bank["tasks"]
    }
    source = tmp_path / "source.py"
    source.write_bytes(b"print('saved submission')\n")
    patch = tmp_path / "model.patch"
    patch.write_bytes(
        b"diff --git a/source.py b/source.py\nnew file mode 100644\nindex 0000000..1234567\n"
        b"--- /dev/null\n+++ b/source.py\n@@ -0,0 +1 @@\n+print('saved submission')\n"
    )
    fields = {
        "original_summary": pin(tmp_path / "grade-only-summary.json", summary),
        "original_traces": PinnedFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest()),
        "original_tasks": old.original_tasks,
        "original_bank": old.original_bank,
    }
    slot = 116
    bindings = {
        "original_line_index": slot,
        "original_line_sha256": hashlib.sha256(lines[slot]).hexdigest(),
        "sample_index": records[slot]["sample_index"],
        "task_id": records[slot]["task_id"],
        "task_sha256": bank["tasks"][slot // 8]["task_sha256"],
        "source_commit": old.runtime_source,
        "runtime_bundle": asdict(old.runtime_bundle),
        "source_module_hashes": old.runtime_module_hashes,
    }
    manifest = {
        **{name: asdict(value) for name, value in fields.items()},
        **bindings,
        "protocol": "russell-rsi-grade-only-recovery-v1",
        "model_identity": old.model_identity,
        "bank_identity": old.bank_identity,
        "model_requests": 0,
        "maximum_vm_start_attempts": 3,
        "source_file": {
            "uri": str(source),
            "path": "source.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
        },
        "expected_changed_file_modes": {"source.py": "100644"},
    }
    manifest_pin = pin(tmp_path / "manifest.json", manifest)
    result = {
        **bindings,
        "manifest_sha256": manifest_pin.sha256,
        "model_requests": 0,
        "execution_error": None,
        "source_file_sha256": manifest["source_file"]["sha256"],
        "patch_uri": str(patch),
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "changed_files": ["source.py"],
        "changed_file_modes": {"source.py": "100644"},
        "startup_attempts": {"reconstruction": 1, "private_grader": 2},
        "startup_failures": [{"phase": "private_grader", "attempt": 1, "returned_machine": False}],
        "grade": {"status": "graded", "reward": 0.0, "error": None, "failure": None},
    }
    config = GradeOnlyRecoveryConfig(
        **fields,
        reconstruction_manifest=manifest_pin,
        grade_result=pin(tmp_path / "result.json", result),
        model_identity=old.model_identity,
        bank_identity=old.bank_identity,
        runtime_source=old.runtime_source,
        runtime_bundle=old.runtime_bundle,
        runtime_module_hashes=old.runtime_module_hashes,
        output_path=str(tmp_path / "grade-only-output"),
    )
    return config, records, bank


def test_grade_only_preserves_all_model_evidence_and_original_grades(grade_only):
    config, originals, bank = grade_only
    recover_grade_only_calibration(config)
    output = Path(config.output_path)
    completed = json.loads((output / "completed_traces.json").read_text())["attempts"]
    assert completed == [
        {**record, "grade": config.grade_result.read_json()["grade"]} if index == 116 else record
        for index, record in enumerate(originals)
    ]
    lineage = json.loads((output / "recovery.json").read_text())
    assert lineage["original_grade_count"] == 207
    assert len(lineage["attempts"]) == 208
    assert [item["sample_index"] for item in lineage["attempts"]] == [record["sample_index"] for record in originals]
    summary = json.loads((output / "failure_summary.json").read_text())
    measured = calibration_measurements(summary, tuple(QualifiedTask(**row) for row in bank["tasks"]), "parent", "bank")
    assert all(len(item.rewards) == 8 for item in measured)
    sealed = (output / "recovery.json").read_bytes()
    recover_grade_only_calibration(config)
    assert (output / "recovery.json").read_bytes() == sealed


@pytest.mark.parametrize(
    "change", ["slot", "model", "bank", "model_request", "patch", "ungraded", "runtime", "source", "returned_machine"]
)
def test_grade_only_rejects_changed_submission_or_identity(grade_only, tmp_path, change):
    config, _, _ = grade_only
    result = config.grade_result.read_json()
    if change == "slot":
        result["sample_index"] += 1
    elif change == "model":
        config = replace(config, model_identity="another-checkpoint")
    elif change == "bank":
        config = replace(config, bank_identity="another-bank")
    elif change == "model_request":
        result["model_requests"] = 1
    elif change == "patch":
        patch = tmp_path / "changed.patch"
        patch.write_bytes(Path(result["patch_uri"]).read_bytes().replace(b"saved submission", b"another candidate"))
        result.update(patch_uri=str(patch), patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest())
    elif change == "runtime":
        result["source_module_hashes"] = {"changed": "runtime"}
    elif change == "source":
        result["source_file_sha256"] = "0" * 64
    elif change == "returned_machine":
        result["startup_failures"][0]["returned_machine"] = True
    else:
        result["grade"] = {"status": "unavailable", "reward": None}
    config = replace(config, grade_result=pin(tmp_path / "changed-result.json", result))
    with pytest.raises(ValueError):
        recover_grade_only_calibration(config)
    assert not Path(config.output_path).exists()


def test_grade_only_rejects_duplicate_model_sample_slot(grade_only, tmp_path):
    config, records, _ = grade_only
    records[0]["sample_index"] = records[1]["sample_index"]
    trace = tmp_path / "duplicate.jsonl"
    trace.write_bytes(b"".join((json.dumps(record) + "\n").encode() for record in records))
    trace_pin = PinnedFile(str(trace), hashlib.sha256(trace.read_bytes()).hexdigest())
    manifest = config.reconstruction_manifest.read_json()
    manifest["original_traces"] = asdict(trace_pin)
    manifest_pin = pin(tmp_path / "duplicate-manifest.json", manifest)
    result = config.grade_result.read_json()
    result["manifest_sha256"] = manifest_pin.sha256
    config = replace(
        config,
        original_traces=trace_pin,
        reconstruction_manifest=manifest_pin,
        grade_result=pin(tmp_path / "duplicate-result.json", result),
    )
    with pytest.raises(ValueError, match="model sample slot"):
        recover_grade_only_calibration(config)
    assert not Path(config.output_path).exists()


def test_grade_only_uses_declared_bank_bytes_instead_of_external_task_pin(grade_only, tmp_path):
    config, _, _ = grade_only
    changed_bank = tmp_path / "changed-bank"
    changed_bank.mkdir()
    (changed_bank / "train.parquet").write_bytes(b"another task bank")
    (changed_bank / "bank.json").write_bytes(config.original_bank.read_bytes())
    bank = ArtifactStep.adopt("documents/bank", "2026.10.05", str(changed_bank), kind=Artifact)
    model = ArtifactStep.adopt("checkpoints/model", "2026.10.05", "/tmp/model", kind=LevanterCheckpoint)
    value = asdict(config)
    step = grade_only_recovery_step(value, "2026.10.05.2", bank, model, config.runtime_bundle)
    ctx = StepContext.for_run(str(tmp_path / "bound-output"), str(tmp_path), deps=step.deps)
    bound = step.build_config(ctx)
    bound = replace(bound, model_identity=config.model_identity, bank_identity=config.bank_identity)
    with pytest.raises(ValueError, match="digest mismatch"):
        recover_grade_only_calibration(bound)
    assert not Path(bound.output_path).exists()
