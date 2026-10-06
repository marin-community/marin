# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace

import jax.random as jrandom
import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from marin.execution.lazy import StepContext, artifact_identity
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import RolloutContractError
from taskcompendium.models import TaskSpec

from experiments.post_training.russell_rsi import teacher_chat_study as study
from experiments.post_training.russell_rsi import test_teacher_collection, test_teacher_four_pass
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_study import collect_remaining_rows, qualified_row
from experiments.post_training.russell_rsi.teacher_collection import TeacherTask
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, preflight_task

student_tokenizer = test_teacher_collection.student_tokenizer
continuation_inputs = test_teacher_four_pass.continuation_inputs
study_inputs = test_teacher_four_pass.study_inputs


def inputs(tokenizer):
    selected, tasks = [], {}
    for index in range(10):
        task = TaskSpec.model_validate_json(
            preflight_task(index + 20, PREFLIGHT_INSTRUCTION, 90000 + index).model_dump_json()
        )
        tasks[task.id] = task
        selected.append(
            asdict(TeacherTask(f"family-{index}", "boundaries", task.id, digest(task.model_dump(mode="json"))))
        )
    original: dict = {
        "selection": {
            "selected": selected,
            "teacher_model": {"max_tokens": 16384, "temperature": 0.0, "reasoning_effort": "medium"},
        }
    }
    retained = []
    for index, slot in [(0, "00-1"), (2, "02-0")]:
        record = rollout(f"seed-{index}")
        row = qualified_row(record, tasks[selected[index]["task_id"]], tokenizer)
        assert row is not None
        retained.append(
            {
                "task": selected[index],
                "attempt": int(slot[-1]),
                "slot": slot,
                "row": row["example"],
                "witness": row,
                "row_sha256": compact_json_sha256(row["example"]),
            }
        )
    return original, tasks, retained


def rollout(value, reward=1):
    return {
        "execution_error": None,
        "grade": {"status": "graded", "reward": reward},
        "stop_reason": "stop",
        "messages": [
            {"role": "user", "content": "Fix the public workspace."},
            {"role": "assistant", "content": value, "reasoning": "private teacher thought"},
        ],
    }


@pytest.mark.parametrize("interrupted", [False, True])
def test_frozen_remaining_order_retains_seeds_and_never_reissues_slots(tmp_path, student_tokenizer, interrupted):
    original, tasks, retained = inputs(student_tokenizer)
    root = StoragePath(str(tmp_path / "collection"))
    if interrupted:
        write_once(
            root / "trajectories/05-1/trajectory.json",
            {"task": original["selection"]["selected"][5], "attempt": 1, "slot": "05-1"},
        )
    calls = []

    async def run(task, slot, model):
        calls.append(slot.name)
        return rollout("answer-" + slot.name, reward=0 if slot.name == "05-1" else 1)

    first = asyncio.run(collect_remaining_rows(original, tasks, retained, student_tokenizer, root, run))
    assert calls == (["06-0", "07-0"] if interrupted else ["05-1", "06-0", "07-0"])
    assert first["status"] == "passed"
    assert [row["slot"] for row in first["accepted"]] == ["00-1", "02-0", "06-0", "07-0"]
    assert first["accepted"][:2] == retained
    assert first["new_trajectories"] == 3 and first["cumulative_trajectories"] == 13
    assert not (root / "trajectories/05-0").exists() and not (root / "trajectories/06-1").exists()
    second = asyncio.run(collect_remaining_rows(original, tasks, retained, student_tokenizer, root, run))
    assert second == first
    assert len(calls) == (2 if interrupted else 3)
    assert all("reasoning" not in message for row in first["accepted"] for message in row["row"]["messages"])


def test_exhausted_budget_does_not_train_or_replace_seed_rows(tmp_path, student_tokenizer):
    original, tasks, retained = inputs(student_tokenizer)
    calls = []

    async def run(task, slot, model):
        calls.append(slot.name)
        return rollout("failed", reward=0)

    result = asyncio.run(
        collect_remaining_rows(original, tasks, retained, student_tokenizer, StoragePath(str(tmp_path)), run)
    )
    assert calls == ["05-1", "06-0", "06-1", "07-0", "07-1", "08-0", "08-1", "09-0", "09-1"]
    assert result["status"] == "insufficient_rows" and result["accepted"] == retained
    assert result["new_trajectories"] == 9 and result["cumulative_trajectories"] == 19


def test_changed_task_and_fatal_resume_fail_before_provider(tmp_path, student_tokenizer):
    original, tasks, retained = inputs(student_tokenizer)
    calls = []

    async def run(task, slot, model):
        calls.append(slot.name)
        raise RolloutContractError("missing actual token IDs")

    broken = json.loads(json.dumps(original))
    broken["selection"]["selected"][9]["task_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen admission"):
        asyncio.run(collect_remaining_rows(broken, tasks, retained, student_tokenizer, StoragePath(str(tmp_path)), run))
    assert calls == []
    for _ in range(2):
        with pytest.raises(RolloutContractError):
            asyncio.run(
                collect_remaining_rows(original, tasks, retained, student_tokenizer, StoragePath(str(tmp_path)), run)
            )
    assert calls == ["05-1"]
    marker = json.loads((tmp_path / "contract-failure.json").read_text())
    assert marker["slot"] == "05-1" and marker["exception_type"] == "RolloutContractError"


def test_admitted_serialized_task_survives_new_schema_defaults(tmp_path, student_tokenizer):
    original, tasks, retained = inputs(student_tokenizer)
    for entry in original["selection"]["selected"]:
        serialized = tasks[entry["task_id"]].model_dump(mode="json")
        del serialized["interaction_tools"]
        del serialized["output_paths"]
        entry["task_sha256"] = digest(serialized)
        tasks[entry["task_id"]] = TaskSpec.model_validate_json(json.dumps(serialized))
    calls = []

    async def run(task, slot, model):
        calls.append(slot.name)
        return rollout("not accepted", reward=0)

    result = asyncio.run(
        collect_remaining_rows(original, tasks, retained, student_tokenizer, StoragePath(str(tmp_path / "valid")), run)
    )
    assert result["new_trajectories"] == 9 and calls[0] == "05-1"
    calls.clear()
    changed = tasks[original["selection"]["selected"][0]["task_id"]]
    changed_payload = changed.model_dump(mode="json", exclude_unset=True)
    changed_payload["context"]["events"][0]["content"] = "Changed admitted public task"
    tasks[changed.id] = TaskSpec.model_validate_json(json.dumps(changed_payload))
    with pytest.raises(ValueError, match="frozen admission"):
        asyncio.run(
            collect_remaining_rows(
                original, tasks, retained, student_tokenizer, StoragePath(str(tmp_path / "changed")), run
            )
        )
    assert calls == [] and not (tmp_path / "changed" / "plan.json").exists()


def test_actual_mixture_repeats_four_rows_eight_times_in_built_four_batches(study_inputs, tmp_path, monkeypatch):
    original, _ = study_inputs
    original["student_context_amendment"] = {"context_tokens": 16384}
    original["runtime_commit"] = MARIN_SKYRL.commit
    dummy = {"uri": str(tmp_path / "not-read"), "sha256": "0" * 64}
    config = {
        "protocol": study.PROTOCOL,
        "version": original["version"],
        "runtime_commit": MARIN_SKYRL.commit,
        "original_study": dummy,
        "prospective_decision": dummy,
        "canonical_proof": dummy,
        "predecessors": [],
        "retained_rows": [],
    }
    monkeypatch.setattr(study, "source_study", lambda parsed: original)
    outputs = study.chat_study_workflow(config)
    trained = outputs["train"]
    pod = trained.build_config(
        StepContext.for_run(
            str(tmp_path / "trained"), str(tmp_path / "artifacts"), deps=trained.deps, runtime_args=trained.runtime_args
        )
    )
    built = pod.train_config
    reload = outputs["reload"].build_config(
        StepContext.for_run(
            str(tmp_path / "reload"),
            str(tmp_path / "artifacts"),
            deps=outputs["reload"].deps,
            runtime_args=outputs["reload"].runtime_args,
        )
    )
    assert built.trainer.train_batch_size == 8 and built.trainer.num_train_steps == 4 and built.train_seq_len == 16384
    assert reload.model.identity == artifact_identity(trained) and reload.model.location.endswith("/hf/step-3")
    mixture = MixtureDataset(
        {"teacher": ListAsyncDataset(["a", "b", "c", "d"])},
        {"teacher": 1.0},
        stop_strategy=built.data.stop_strategy,
        key=jrandom.PRNGKey(0),
        block_size=built.data.mixture_block_size,
    )
    count = built.trainer.train_batch_size * built.trainer.num_train_steps
    rows = asyncio.run(mixture.get_batch(list(range(count))))
    assert Counter(rows) == {"a": 8, "b": 8, "c": 8, "d": 8}
    assert all(
        Counter(rows[start : start + built.trainer.train_batch_size]) == {"a": 2, "b": 2, "c": 2, "d": 2}
        for start in range(0, count, built.trainer.train_batch_size)
    )


@pytest.mark.parametrize("defect", ["reservation", "second_identity", "decision_budget"])
def test_predecessor_binding_rejects_changed_consumed_task_before_collection(
    tmp_path, student_tokenizer, monkeypatch, defect
):
    original, _, retained = inputs(student_tokenizer)
    monkeypatch.setattr(study, "require_four_pass_condition", lambda value: {})

    def pin(name, value):
        raw = value.encode() if isinstance(value, str) else json.dumps(value).encode()
        path = tmp_path / name
        path.write_bytes(raw)
        return PinnedFile(str(path), hashlib.sha256(raw).hexdigest())

    plan = {"selected": original["selection"]["selected"], "capabilities": {"skills": [{"label": "boundaries"}]}}
    original["selection"]["capabilities"] = plan["capabilities"]
    predecessors = []
    for version, slots in [("2026.10.05.11", ("00-0",)), ("2026.10.05.12", study.V12_CONSUMED_SLOTS)]:
        uri = "s3://fixture/" + version
        reservations = []
        for slot in slots:
            family, attempt = map(int, slot.split("-"))
            reservations.append(
                study.ReservedSlot(slot, pin(version + slot, {"task": plan["selected"][family], "attempt": attempt}))
            )
        predecessors.append(
            study.Predecessor(
                "documents/fixture@" + version + ":12345678",
                uri,
                pin(
                    version + "info",
                    {
                        "name": "documents/fixture",
                        "output_path": uri,
                        "config": {"version": version, "fingerprint": "12345678"},
                    },
                ),
                pin(version + "status", "FAILED\n"),
                pin(version + "plan", plan),
                pin(version + "fatal", {"slot": slots[-1], "plan_sha256": compact_json_sha256(plan)}),
                tuple(reservations),
            )
        )
    first, second = predecessors
    original["collection_recovery"] = {
        "predecessor_identity": first.identity,
        "fatal_marker_sha256": first.contract_failure.sha256,
        "executor_info_sha256": first.executor_info.sha256,
        "executor_status_sha256": first.executor_status.sha256,
        "plan_sha256": first.plan.sha256,
        "slot_reservation_sha256": first.reservations[0].reservation.sha256,
    }
    original_pin = pin("original", original)
    proof = pin("canonical", {"rows": retained})
    decision = {
        "protocol": study.PROTOCOL,
        "source_study_config": {"sha256": original_pin.sha256},
        "source_rows": {"canonical_proof_sha256": proof.sha256, "slots": list(study.RETAINED_SLOTS)},
        "selection": {
            "permitted_slots": list(study.PERMITTED_SLOTS),
            "consumed_slots": 10,
            "maximum_new_trajectories": 9,
            "original_total_ceiling": 20,
        },
        "sft": {
            "unique_jsonl_rows": 4,
            "batch_size": 8,
            "optimizer_updates": 4,
            "passes": 8,
            "context_tokens": 16384,
            "example_exposures": 32,
            "assistant_only_loss": True,
            "teacher_reasoning_in_student": False,
            "truncate": False,
        },
        "collection": original["selection"]["teacher_model"],
        "source_collection": {
            "identity": second.identity,
            "uri": second.uri,
            "failure_sha256": second.contract_failure.sha256,
        },
    }
    seed_files = tuple(study.RetainedRow(slot, proof, proof, proof) for slot in study.RETAINED_SLOTS)
    bound = study.ChatStudy(original_pin, pin("decision", decision), proof, tuple(predecessors), seed_files)
    assert study.source_study(bound) == original
    if defect == "reservation":
        corrupted = json.loads(first.reservations[0].reservation.read_bytes())
        corrupted["task"]["task_sha256"] = "f" * 64
        bad_first = replace(first, reservations=(study.ReservedSlot("00-0", pin("wrong-reservation", corrupted)),))
        bad = replace(bound, predecessors=(bad_first, second))
    elif defect == "second_identity":
        bad = replace(bound, predecessors=(first, replace(second, identity="documents/other@2026.10.05.12:12345678")))
    else:
        decision["selection"]["maximum_new_trajectories"] = 10
        bad = replace(bound, prospective_decision=pin("wrong-decision", decision))
    with pytest.raises(ValueError):
        study.source_study(bad)


@pytest.mark.parametrize("defect", [None, "grade", "canonical_mask", "source_hash"])
def test_exact_retained_row_grade_and_canonical_proof_binding(tmp_path, student_tokenizer, defect):
    original, tasks, seeds = inputs(student_tokenizer)

    def pin(name, value):
        raw = json.dumps(value).encode()
        path = tmp_path / name
        path.write_bytes(raw)
        return PinnedFile(str(path), hashlib.sha256(raw).hexdigest())

    original_pin = pin("original.json", original)
    retained = []
    attested: list[dict] = []
    for seed in seeds:
        record = rollout("seed-" + seed["task"]["family"].rsplit("-", 1)[-1])
        legacy = seed["witness"]
        record_pin, row_pin = pin(seed["slot"] + "rollout", record), pin(seed["slot"] + "student", legacy)
        grade = {
            "task": seed["task"],
            "attempt": seed["attempt"],
            "status": "accepted",
            "rollout_sha256": compact_json_sha256(record),
        }
        retained.append(study.RetainedRow(seed["slot"], record_pin, row_pin, pin(seed["slot"] + "grade", grade)))
        attested.append(
            {
                "slot": seed["slot"],
                "canonical": json.loads(json.dumps(legacy)),
                "source_file_hashes": {"student-row.json": row_pin.sha256, "rollout.json": record_pin.sha256},
            }
        )
    identity = "documents/predecessor@2026.10.05.12:12345678"
    proof = {"collection_identity": identity, "collection_config_sha256": original_pin.sha256, "rows": attested}
    if defect == "grade":
        bad = retained[0].qualification.read_json()
        bad["status"] = "failed"
        retained[0] = replace(retained[0], qualification=pin("wrong-grade", bad))
    elif defect == "canonical_mask":
        proof["rows"][0]["canonical"]["assistant_mask"][0] = 1 - proof["rows"][0]["canonical"]["assistant_mask"][0]
    elif defect == "source_hash":
        proof["rows"][0]["source_file_hashes"]["rollout.json"] = "0" * 64
    proof_pin = pin("proof.json", proof)
    predecessor = study.Predecessor(identity, "s3://fixture", proof_pin, proof_pin, proof_pin, proof_pin, ())
    bound = study.ChatStudy(original_pin, proof_pin, proof_pin, (predecessor, predecessor), tuple(retained))
    if defect is None:
        actual = study.retained_rows(bound, original, tasks, student_tokenizer)
        assert [row["row"] for row in actual] == [row["row"] for row in seeds]
    else:
        with pytest.raises(ValueError, match="Retained successful row"):
            study.retained_rows(bound, original, tasks, student_tokenizer)
