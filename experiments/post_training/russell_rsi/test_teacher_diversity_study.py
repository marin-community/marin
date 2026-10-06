# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import jax.random as jrandom
import numpy as np
import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle
from taskcompendium.models import TaskSpec
from taskcompendium.parquet import write_tasks

from experiments.post_training.russell_rsi import (
    test_evaluation_journal,
    test_teacher_collection,
    test_teacher_four_pass,
)
from experiments.post_training.russell_rsi.bootstrap_loop import QualifiedTask
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.launch_post_teacher_sft import seal_study_calibration
from experiments.post_training.russell_rsi.launch_teacher_diversity_post_sft import (
    AMENDMENT_PROTOCOL,
    LAUNCH_PROTOCOL,
    PERMITTED_CHANGES,
    durable_diversity_post_workflow,
)
from experiments.post_training.russell_rsi.launch_teacher_diversity_sft import SFT_VERSION, durable_sft_stages
from experiments.post_training.russell_rsi.launch_teacher_sft import StudentTrainingTemplate, TeacherCollectionConfig
from experiments.post_training.russell_rsi.rollout_eval import calibration_evaluation_journal
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_study import PROTOCOL as RETAINED_PROTOCOL
from experiments.post_training.russell_rsi.teacher_chat_study import (
    collect_chat_dataset,
    collect_chat_rows,
    qualified_row,
)
from experiments.post_training.russell_rsi.teacher_collection import TeacherTask
from experiments.post_training.russell_rsi.teacher_diversity_study import (
    PASSES,
    PERMITTED_SLOTS,
    PROTOCOL,
    RETAINED_SLOTS,
    ROWS,
    diversity_plan,
    diversity_post_workflow,
    diversity_workflow,
    require_diversity_condition,
    validated_diversity_post_workflow,
)
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, preflight_task

student_tokenizer = test_teacher_collection.student_tokenizer
continuation_inputs = test_teacher_four_pass.continuation_inputs
study_inputs = test_teacher_four_pass.study_inputs
frozen_comparison = test_evaluation_journal.frozen_comparison
calibration_config = test_evaluation_journal.calibration_config


def pinned(tmp_path, name, value):
    path = tmp_path / f"{name}-{compact_json_sha256(value)}.json"
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return {"uri": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def set_pin(config, name, pin):
    config[f"{name}_uri"], config[f"{name}_sha256"] = pin["uri"], pin["sha256"]


def decision(config):
    return {
        "protocol": PROTOCOL,
        "original_study": config["original_study"],
        **{
            f"{key}_sha256": config[f"{key}_sha256"]
            for key in ("current_selection", "current_coding", "capability_release", "capability_review", "bank_record")
        },
        "train_sha256": config["selection"]["train_sha256"],
        **{
            key: config[key]
            for key in (
                "retained_collection",
                "retained_train",
                "retained_canonical_proof",
                "historical_attempts",
                "retained_rows",
            )
        },
        "selected": config["selection"]["selected"],
        "maximum_new_trajectories": 12,
        "first_qualified_new_rows": 4,
        "sft": {
            "rows": 8,
            "passes": 4,
            "batch_size": 8,
            "updates": 4,
            "context_tokens": 16384,
            "learning_rate": 1e-6,
            "assistant_only_loss": True,
            "truncate": False,
        },
    }


@pytest.fixture
def diversity_inputs(study_inputs, tmp_path):
    original, _ = study_inputs
    original = deepcopy(original)
    original["student_context_amendment"] = {}
    original["runtime_commit"] = MARIN_SKYRL.commit
    lineage = json.loads(Path(original["continuation_selection_uri"]).read_bytes())
    config = {
        **original,
        "protocol": PROTOCOL,
        "original_study": pinned(tmp_path, "original", original),
        "version": "2026.10.06.20",
        "collection_version": "2026.10.06.20",
    }
    candidate = {"checkpoint_identity": "new-sft@v1:current", "development": [27 / 32, 25 / 32], "retention": 1 / 3}
    current = {
        "incumbent": lineage["incumbent"],
        "sft": candidate,
        "sft_rl": None,
        "selected": candidate,
        "promoted": lineage["incumbent"],
        "original_parent": lineage["original_parent"],
    }
    set_pin(config, "current_selection", pinned(tmp_path, "current-selection", current))
    source = json.loads(Path(original["continuation_config_uri"]).read_bytes())
    panel = json.loads(Path(source["panel_uri"]).read_bytes())
    rows = [
        {**item, "pass_rate": int(index < passes)}
        for suite, passes in (("humanevalplus", 27), ("mbppplus", 25))
        for index, item in enumerate(row for row in panel["items"] if row["suite"] == suite)
    ]
    coding = {
        "model_identity": candidate["checkpoint_identity"],
        "panel_sha256": original["coding_panel_sha256"],
        "scores": {"humanevalplus": 27 / 32, "mbppplus": 25 / 32},
        "rows": rows,
        "records_sha256": ["a" * 64, "b" * 64],
    }
    set_pin(config, "current_coding", pinned(tmp_path, "current-coding", coding))
    config["current_coding_identity"] = "coding-completed-current"
    set_pin(
        config,
        "capability_review",
        pinned(
            tmp_path,
            "new-review",
            {
                "decision": "approve",
                "binding": {
                    "candidate_identity": candidate["checkpoint_identity"],
                    "coding_evidence_identity": config["current_coding_identity"],
                    "coding_evidence_sha256": config["current_coding_sha256"],
                    "coding_panel_sha256": original["coding_panel_sha256"],
                    "capability_release_sha256": config["capability_release_sha256"],
                    "source": "coding-development",
                },
            },
        ),
    )
    bank = json.loads(Path(config["bank_record_uri"]).read_bytes())
    config["selection"] = {
        **original["selection"],
        "selected": [
            asdict(
                TeacherTask(bank["family_by_task"][str(index)], "types", str(index), bank["tasks"][index]["task_sha256"])
            )
            for index in range(4, 10)
        ],
    }
    set_pin(
        config,
        "bank_expansion",
        pinned(
            tmp_path,
            "new-bank-review",
            {
                "decision": "approve",
                "binding": {
                    "source_bank_sha256": original["bank_record_sha256"],
                    "bank_sha256": config["bank_record_sha256"],
                    "train_sha256": config["selection"]["train_sha256"],
                    "family_map_sha256": compact_json_sha256(bank["family_by_task"]),
                    "capability_release_sha256": config["capability_release_sha256"],
                },
                "additions": [],
            },
        ),
    )
    set_pin(
        config,
        "exclusion_review",
        pinned(
            tmp_path,
            "excluded",
            {
                "decision": "approve",
                "bank_sha256": config["bank_record_sha256"],
                "family_map_sha256": compact_json_sha256(bank["family_by_task"]),
                "excluded_families": ["acceptance-family", "final-family"],
            },
        ),
    )
    for key in ("retained_collection", "retained_train", "retained_canonical_proof", "historical_attempts"):
        config[key] = pinned(tmp_path, key, {})
    config["retained_rows"] = [{"slot": f"retained-{index}"} for index in range(4)]
    config["prospective_decision"] = pinned(tmp_path, "decision", decision(config))
    return config


def test_current_coding_is_independent_of_historical_lineage_and_compiles_sft(diversity_inputs, tmp_path):
    config = diversity_inputs
    original_bytes = Path(config["original_study"]["uri"]).read_bytes()
    original = require_diversity_condition(config).original_study
    assert config["current_coding_sha256"] != original["candidate_coding_sha256"]
    outputs = diversity_workflow(config)
    trained = outputs["train"]
    bound = trained.build_config(
        StepContext.for_fingerprint(deps=trained.deps, runtime_arg_keys=trained.runtime_args.keys())
    )
    train = bound.train_config
    assert train.trainer.train_batch_size == 8 and train.trainer.num_train_steps == 4
    assert train.train_seq_len == 16384 and train.data.mixture_block_size == 8
    assert train.trainer.metrics_start_step == 0 and train.trainer.tracker == (JsonLoggerConfig(),)
    assert bound.env_vars["WANDB_MODE"] == "disabled"
    mixture = MixtureDataset(
        {"teacher": ListAsyncDataset(list(range(8)))},
        {"teacher": 1.0},
        stop_strategy=train.data.stop_strategy,
        key=jrandom.PRNGKey(0),
        block_size=train.data.mixture_block_size,
    )
    exposed = asyncio.run(mixture.get_batch(list(range(32))))
    assert Counter(exposed) == {index: 4 for index in range(8)}
    assert all(Counter(exposed[start : start + 8]) == {index: 1 for index in range(8)} for start in range(0, 32, 8))
    assert Path(config["original_study"]["uri"]).read_bytes() == original_bytes
    changed = deepcopy(config)
    stale = json.loads(Path(original["capability_review_uri"]).read_bytes())
    set_pin(changed, "capability_review", pinned(tmp_path, "stale-label-review", stale))
    with pytest.raises(ValueError, match="current coding provenance"):
        diversity_workflow(changed)


@pytest.mark.parametrize("field", ["pass_rate", "benchmark_id", "prompt_sha256"])
def test_scored_coding_rows_cannot_be_replaced_by_unchanged_aggregates(diversity_inputs, tmp_path, field):
    config = deepcopy(diversity_inputs)
    coding = json.loads(Path(config["current_coding_uri"]).read_bytes())
    coding["rows"][-1][field] = 1 if field == "pass_rate" else "different"
    set_pin(config, "current_coding", pinned(tmp_path, "changed-row", coding))
    with pytest.raises(ValueError, match=r"saved outcomes|invalid development outcomes"):
        require_diversity_condition(config)


def successful(value):
    return {
        "execution_error": None,
        "grade": {"status": "graded", "reward": 1},
        "stop_reason": "stop",
        "messages": [
            {"role": "user", "content": "Public request"},
            {"role": "assistant", "content": value, "reasoning": "private reasoning"},
        ],
    }


def retained_inputs(tmp_path, tokenizer):
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 70000 + index) for index in range(101, 133)]
    records = {task.id: task.model_dump_json() for task in tasks}
    entries = [
        asdict(TeacherTask(f"family-{index}", "boundaries", task.id, digest(json.loads(records[task.id]))))
        for index, task in enumerate(tasks)
    ]
    accepted, pins, canonical = [], [], []
    for index, (task, entry) in enumerate(zip(tasks[:4], entries[:4], strict=True)):
        slot, rollout = RETAINED_SLOTS[index], successful(f"retained-answer-{index}")
        attempt = int(slot.split("-")[1])
        row = qualified_row(rollout, task, tokenizer)
        assert row is not None
        accepted.append(
            {
                "task": entry,
                "attempt": attempt,
                "slot": slot,
                "row_sha256": compact_json_sha256(row["example"]),
                "row": row["example"],
                "witness": row,
            }
        )
        pins.append(
            {
                "slot": slot,
                "rollout": pinned(tmp_path, slot + "-rollout", rollout),
                "student_row": pinned(tmp_path, slot + "-row", row),
                "qualification": pinned(
                    tmp_path,
                    slot + "-grade",
                    {
                        "status": "accepted",
                        "task": entry,
                        "attempt": attempt,
                        "rollout_sha256": compact_json_sha256(rollout),
                    },
                ),
            }
        )
        canonical.append(
            {"slot": slot, "tokens": len(row["input_ids"]), "assistant_targets": sum(row["assistant_mask"])}
        )
    train = tmp_path / "retained.jsonl"
    train.write_text("".join(json.dumps(row["row"], sort_keys=True) + "\n" for row in accepted))
    train_pin = {"uri": str(train), "sha256": hashlib.sha256(train.read_bytes()).hexdigest()}
    bank = {
        "tasks": [
            asdict(
                QualifiedTask(
                    task.id, entry["task_sha256"], f"admitted-{index}", f"source-{index}", "boundaries", entry["family"]
                )
            )
            for index, (task, entry) in enumerate(zip(tasks, entries, strict=True))
        ],
        "family_by_task": {task.id: entry["family"] for task, entry in zip(tasks, entries, strict=True)},
    }
    config = {
        "retained_rows": pins,
        "retained_collection": pinned(
            tmp_path,
            "retained-collection",
            {"protocol": RETAINED_PROTOCOL, "status": "passed", "accepted": accepted, "cumulative_trajectories": 16},
        ),
        "retained_train": train_pin,
        "retained_canonical_proof": pinned(
            tmp_path,
            "canonical",
            {"status": "canonical_runtime_rows_passed", "train_sha256": train_pin["sha256"], "rows": canonical},
        ),
        "historical_attempts": pinned(
            tmp_path,
            "history",
            {
                "consumed_trajectories": 16,
                "records": [
                    {"producer": "previous-collection", "slot": str(index), "family": f"historical-{index}"}
                    for index in range(16)
                ],
            },
        ),
        "selection": {
            "selected": entries[4:10],
            "teacher_model": {"max_tokens": 512, "temperature": 0, "reasoning_effort": "medium"},
        },
    }
    set_pin(config, "bank_record", pinned(tmp_path, "bank", bank))
    return config, records


def test_retained_full_rows_survive_new_collection_and_completed_resume_has_no_calls(tmp_path, student_tokenizer):
    config, records = retained_inputs(tmp_path, student_tokenizer)
    plan = diversity_plan(config, records, student_tokenizer)
    directory = StoragePath(str(tmp_path / "new-collection"))
    calls = []

    async def model(task, slot, settings):
        calls.append(slot.name)
        return successful("new-answer-" + slot.name)

    result = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, model, required_rows=ROWS))
    assert calls == ["00-0", "01-0", "02-0", "03-0"]
    assert result["accepted"][:4] == plan["retained"] and len(result["accepted"]) == 8
    assert result["new_trajectories"] == 4 and result["cumulative_trajectories"] == 20
    assert len(result["accepted"]) * PASSES == 32 and not (directory / "trajectories/03-1").exists()
    second = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, model, required_rows=ROWS))
    assert second == result and len(calls) == 4
    config["selection"]["selected"][0] = plan["retained"][0]["task"]
    with pytest.raises(ValueError, match="overlap retained"):
        diversity_plan(config, records, student_tokenizer)
    assert len(PERMITTED_SLOTS) == 12


def test_completed_shared_setup_writes_eight_rows_without_runtime_or_provider_credentials(
    tmp_path, student_tokenizer, monkeypatch
):
    config, records = retained_inputs(tmp_path, student_tokenizer)
    plan = diversity_plan(config, records, student_tokenizer)
    directory = StoragePath(str(tmp_path / "collection"))

    async def scripted(task, slot, settings):
        return successful("new-answer-" + slot.name)

    result = asyncio.run(collect_chat_rows(plan, records, student_tokenizer, directory, scripted, required_rows=8))
    bank = tmp_path / "input-bank"
    bank.mkdir()
    write_tasks(str(bank / "train.parquet"), [TaskSpec.model_validate_json(raw) for raw in records.values()])
    template = tmp_path / "template.jinja"
    template.write_text(MARIN_CHAT_TEMPLATE)
    parent = tmp_path / "tokenizer"
    base = TeacherCollectionConfig(
        config["selection"],
        "unused",
        "unused",
        str(bank),
        str(parent),
        "parent-update8",
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in parent.iterdir() if path.is_file()},
        RuntimeBundle(
            archive_uri="/absent-runtime",
            archive_sha256="a" * 64,
            manifest_uri="/absent-manifest",
            manifest_sha256="b" * 64,
            installation_parent=str(tmp_path / "absent-installation"),
        ),
        "never-called",
        str(directory),
    )
    monkeypatch.delenv(GLM_TOKEN_ENV, raising=False)
    collect_chat_dataset(
        base,
        train_sha256=hashlib.sha256((bank / "train.parquet").read_bytes()).hexdigest(),
        template=StudentTrainingTemplate(str(template), hashlib.sha256(template.read_bytes()).hexdigest()),
        plan=lambda actual_records, tokenizer: diversity_plan(config, actual_records, tokenizer),
        required_rows=8,
        passes=4,
        batch_size=8,
        updates=4,
    )
    dataset = json.loads((directory / "dataset.json").read_text())
    examples = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
    assert examples == [row["row"] for row in result["accepted"]]
    assert dataset["rows"] == 8 and dataset["passes"] == 4 and dataset["example_exposures"] == 32
    assert dataset["sha256"] == hashlib.sha256((directory / "train.jsonl").read_bytes()).hexdigest()
    assert not (tmp_path / "absent-installation").exists()


def test_expanded_calibration_binds_all_288_slots_and_refuses_changed_bank(calibration_config):
    config = calibration_config
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 23) for index in range(101, 137)]
    write_tasks(config.tasks_path, tasks)
    config = replace(config, limit=36)
    journal = calibration_evaluation_journal(config)
    binding = journal.binding
    assert len(binding["attempts"]["task"]) == 288 and len(binding["attempts"]["preflight"]) == 2
    assert (
        journal.attempt("task", f"{tasks[-1].id}/7", digest(tasks[-1].model_dump(mode="json"))).binding["key"]
        == f"{tasks[-1].id}/7"
    )
    write_tasks(config.tasks_path, tasks[:-1])
    with pytest.raises(ValueError, match="complete eight-sample"):
        calibration_evaluation_journal(config)


@pytest.mark.parametrize("signal", [False, True])
def test_full_36_task_calibration_preserves_conditional_rl_barriers(diversity_inputs, tmp_path, signal):
    study = deepcopy(diversity_inputs)
    original = json.loads(Path(study["original_study"]["uri"]).read_bytes())
    bank = json.loads(Path(study["bank_record_uri"]).read_bytes())
    additions = [
        QualifiedTask(
            str(index),
            f"hash-{index}",
            f"admission-{index}",
            f"independent-source-{index}",
            "api_contracts,types",
            f"independent-family-{index}",
            relation="new_contract",
        )
        for index in range(32, 36)
    ]
    bank["tasks"].extend(asdict(task) for task in additions)
    bank["family_by_task"].update({task.task_id: task.contract_id for task in additions})
    set_pin(study, "bank_record", pinned(tmp_path, "36-bank", bank))
    study["bank"] = {**study["bank"], "identity_config": {"bank_sha256": study["bank_record_sha256"]}}
    study["selection"]["bank_sha256"] = study["bank_record_sha256"]
    evidence = []
    for task in additions:
        admitted = pinned(tmp_path, "admitted-" + task.task_id, {"task_id": task.task_id})
        excluded = pinned(tmp_path, "excluded-" + task.task_id, {"decision": "approve"})
        evidence.append(
            {
                "task": {
                    key: getattr(task, key)
                    for key in ("task_id", "task_sha256", "source_id", "contract_id", "admission_sha256")
                },
                "admission_evidence_uri": admitted["uri"],
                "admission_evidence_sha256": admitted["sha256"],
                "exclusion_review_uri": excluded["uri"],
                "exclusion_review_sha256": excluded["sha256"],
            }
        )
    set_pin(
        study,
        "bank_expansion",
        pinned(
            tmp_path,
            "36-review",
            {
                "decision": "approve",
                "binding": {
                    "source_bank_sha256": original["bank_record_sha256"],
                    "bank_sha256": study["bank_record_sha256"],
                    "train_sha256": study["selection"]["train_sha256"],
                    "family_map_sha256": compact_json_sha256(bank["family_by_task"]),
                    "capability_release_sha256": study["capability_release_sha256"],
                },
                "additions": evidence,
            },
        ),
    )
    set_pin(
        study,
        "exclusion_review",
        pinned(
            tmp_path,
            "36-exclusion",
            {
                "decision": "approve",
                "bank_sha256": study["bank_record_sha256"],
                "family_map_sha256": compact_json_sha256(bank["family_by_task"]),
                "excluded_families": ["acceptance-family", "final-family"],
            },
        ),
    )
    study["prospective_decision"] = pinned(tmp_path, "36-decision", decision(study))
    producer = diversity_workflow(study)["train"]
    post = {**study, "sft_uri": producer.path(str(tmp_path / "artifacts"))}
    set_pin(post, "sft_config", pinned(tmp_path, "sft-config", study))
    qualification = test_teacher_four_pass.four_update_qualification(artifact_identity(producer), post["sft_uri"])
    qualification["source_config_sha256"] = post["sft_config_sha256"]
    set_pin(post, "qualification", pinned(tmp_path, "qualified", qualification))
    outputs = diversity_post_workflow(post, "calibrate")
    bound = outputs["decision"].build_config(
        StepContext.for_run(str(tmp_path / "decision"), str(tmp_path / "artifacts"), deps=outputs["decision"].deps)
    )
    assert len(bound.record.plan.task_bank) == 36
    Path(bound.record.summary_path).mkdir(parents=True)
    summary = {
        "model_identity": bound.record.plan.current_checkpoint,
        "tasks_identity": bound.record.plan.bank_identity,
        "count": 36,
        "samples_per_task": 8,
        "task_rewards": {task.task_id: [0, 1] * 4 if signal else [0] * 8 for task in bound.record.plan.task_bank},
    }
    (Path(bound.record.summary_path) / "failure_summary.json").write_text(json.dumps(summary))
    seal_study_calibration(bound)
    set_pin(
        post,
        "calibration_decision",
        {
            "uri": str(tmp_path / "decision/calibration-decision.json"),
            "sha256": hashlib.sha256((tmp_path / "decision/calibration-decision.json").read_bytes()).hexdigest(),
        },
    )
    post["calibration_summary_uri"] = str(Path(bound.record.summary_path) / "failure_summary.json")
    evaluated = diversity_post_workflow(post, "evaluate")
    if signal:
        for key in ("coding-sft", "coding-sft-rl", "retention-sft", "retention-sft-rl"):
            dependencies = graph_handles([evaluated[key]])
            assert evaluated["rl"] in dependencies and evaluated["reload"] in dependencies
    else:
        assert "rl" not in evaluated and "coding-sft-rl" not in evaluated
    qualification["optimizer_steps"] = qualification["optimizer_steps"][1:]
    set_pin(post, "qualification", pinned(tmp_path, "missing-step0", qualification))
    with pytest.raises(ValueError, match="complete finite optimizer"):
        diversity_post_workflow(post, "calibrate")


def test_durable_sft_preserves_collection_and_science(diversity_inputs, tmp_path):
    original = diversity_workflow(diversity_inputs)
    updated = durable_sft_stages(original)
    assert artifact_identity(updated["collect"]) == artifact_identity(original["collect"])
    assert updated["train"].deps == original["train"].deps
    before = original["train"].build_config(
        StepContext.for_fingerprint(deps=original["train"].deps, runtime_arg_keys=original["train"].runtime_args)
    )
    after = updated["train"].build_config(
        StepContext.for_fingerprint(deps=updated["train"].deps, runtime_arg_keys=updated["train"].runtime_args)
    )
    assert updated["train"].version == updated["reload"].version == SFT_VERSION
    assert updated["reload"].deps == (updated["train"],)
    reload_config = updated["reload"].build_config(
        StepContext.for_fingerprint(deps=updated["reload"].deps, runtime_arg_keys=updated["reload"].runtime_args)
    )
    assert reload_config.model.identity.endswith(updated["train"].fingerprint())
    assert reload_config.model.location.endswith("hf/step-3")
    assert updated["train"].fingerprint() != original["train"].fingerprint()
    trainer = after.train_config.trainer
    assert trainer.tracker == (JsonLoggerConfig(metric_destination="<output_path>/optimizer-telemetry"),)
    restored = replace(trainer, id=before.train_config.trainer.id, tracker=before.train_config.trainer.tracker)
    assert replace(after, train_config=replace(after.train_config, trainer=restored)) == before


def test_post_workflow_binds_durable_training_and_keeps_qualification_gate(diversity_inputs, tmp_path):
    stages = durable_sft_stages(diversity_workflow(diversity_inputs))
    trained = stages["train"]
    post = {**diversity_inputs, "sft_uri": trained.path(str(tmp_path / "artifacts"))}
    set_pin(post, "sft_config", pinned(tmp_path, "durable-science", diversity_inputs))
    qualification = test_teacher_four_pass.four_update_qualification(artifact_identity(trained), post["sft_uri"])
    qualification["source_config_sha256"] = post["sft_config_sha256"]
    set_pin(post, "qualification", pinned(tmp_path, "durable-qualified", qualification))
    outputs = validated_diversity_post_workflow(post, "calibrate", trained=trained)
    identities = {artifact_identity(handle) for handle in graph_handles([outputs["terminal"]])}
    assert artifact_identity(trained) not in identities
    assert artifact_identity(stages["reload"]) not in identities
    qualified_model = outputs["calibration"].deps[1]
    assert qualified_model.adopt_config is not None
    assert qualified_model.adopt_config["sft"] == artifact_identity(trained)
    assert qualified_model.adopt_source == qualification["hf_export_uri"]
    qualification["sft_identity"] = artifact_identity(diversity_workflow(diversity_inputs)["train"])
    set_pin(post, "qualification", pinned(tmp_path, "wrong-producer", qualification))
    with pytest.raises(ValueError, match="pinned 4-update export"):
        validated_diversity_post_workflow(post, "calibrate", trained=trained)


@pytest.fixture
def durable_post_inputs(diversity_inputs, tmp_path):
    study = deepcopy(diversity_inputs)
    study["version"] = study["collection_version"] = "2026.10.06.15"
    study["prospective_decision"] = pinned(tmp_path, "v15-decision", decision(study))
    stages = durable_sft_stages(diversity_workflow(study))
    artifact_prefix = str(tmp_path / "artifacts")
    post: dict = {**study, "version": "2026.10.06.18", "sft_uri": stages["train"].path(artifact_prefix)}
    set_pin(post, "sft_config", pinned(tmp_path, "v15-study", study))
    config_pin = {"uri": post["sft_config_uri"], "sha256": post["sft_config_sha256"]}
    source = {"head": "f" * 40, "runtime_commit": MARIN_SKYRL.commit, "files": {"worker.py": "a" * 64}}
    post["sft_source_review"] = pinned(
        tmp_path,
        "source-review",
        {
            "status": "approved",
            "source_head": source["head"],
            "runtime_commit": MARIN_SKYRL.commit,
        },
    )
    amendment: dict = {
        "protocol": AMENDMENT_PROTOCOL,
        "config": config_pin,
        "source": source,
        "collection_identity": artifact_identity(stages["collect"]),
        "changes": PERMITTED_CHANGES,
        "source_only_changes": ["exclude_stale_parent_weight_manifest"],
        "science": {
            "rows": 8,
            "passes": 4,
            "batch_size": 8,
            "updates": 4,
            "context_tokens": 16384,
            "learning_rate": 1e-6,
        },
    }
    records = {}
    launches = {}
    for role, handle in (("sft", stages["train"]), ("reload", stages["reload"])):
        root = Path(handle.path(artifact_prefix))
        root.mkdir(parents=True)
        (root / ".executor_status").write_text("SUCCESS")
        ctx = StepContext.for_run(str(root), artifact_prefix, deps=handle.deps, runtime_args=handle.runtime_args)
        bound = json.loads(canonical_json(handle.build_config(ctx)))
        amendment[role] = {"version": SFT_VERSION, "identity": artifact_identity(handle), "output_path": str(root)}
        request = pinned(
            tmp_path,
            f"{role}-request",
            {
                "stage": role,
                "version": SFT_VERSION,
                "source_head": source["head"],
                "runtime_commit": MARIN_SKYRL.commit,
                "config_uri": config_pin["uri"],
                "config_sha256": config_pin["sha256"],
            },
        )
        preflight = pinned(
            tmp_path,
            f"{role}-preflight",
            {
                "exit_code": 0,
                "identity": {"source_head": source["head"], "request_sha256": request["sha256"]},
            },
        )
        launch = {
            "protocol": LAUNCH_PROTOCOL,
            "source_head": source["head"],
            "runtime_commit": MARIN_SKYRL.commit,
            "source_files": source["files"],
            "config": config_pin,
            "source_review": post["sft_source_review"],
            "producer_identity": artifact_identity(handle),
            "output_path": str(root),
            "request": request,
            "preflight": preflight,
            "bound_config": bound,
            "rl_authorized": False,
            "signal_gate_passed": None,
        }
        record = {
            "name": handle.name,
            "version": handle.version,
            "fingerprint": handle.fingerprint(),
            "output_path": str(root),
            "config": bound,
            "provenance": {"base_commit": source["head"], "dirty": False},
        }
        if role == "reload":
            reload_run_id = "reload-smoke"
            (root / reload_run_id).mkdir()
            canonical_record = {
                "run_id": reload_run_id,
                "status": "succeeded",
                "error": None,
                "metrics": {"mmlu_abstract_algebra_0shot": {"sample_len": 1.0, "acc,none": 0.0}},
                "model": {
                    "location": bound["model"]["location"],
                    "config": {"identity": bound["model"]["identity"]},
                },
                "eval": {"name": "mmlu-smoke", "evalchemy": {"max_eval_instances": 1}},
            }
            result_path = root / reload_run_id / "record.json"
            raw = json.dumps(canonical_record, sort_keys=True).encode()
            result_path.write_bytes(raw)
            result_pin = {"uri": str(result_path), "sha256": hashlib.sha256(raw).hexdigest()}
            record["result"] = {
                "records_prefix": str(root),
                "run_ids": [reload_run_id],
                "results_paths": [str(root / reload_run_id / "results")],
            }
        path = root / ".artifact.json"
        raw = json.dumps(record, sort_keys=True).encode()
        path.write_bytes(raw)
        post[f"{role}_producer"] = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        post[f"{role}_launch_proof"] = pinned(tmp_path, f"{role}-launch", launch)
        records[role], launches[role] = record, launch
    trainer = records["sft"]["config"]["train_config"]["trainer"]
    amendment["sft"].update(run_id=trainer["id"], metric_destination=trainer["tracker"][0]["metric_destination"])
    post["sft_telemetry_amendment"] = pinned(tmp_path, "telemetry-amendment", amendment)
    qualification: dict = test_teacher_four_pass.four_update_qualification(
        artifact_identity(stages["train"]), post["sft_uri"]
    )
    qualification["source_config_sha256"] = post["sft_config_sha256"]
    qualification["serving_reload"].update(evidence_uri=result_pin["uri"], evidence_sha256=result_pin["sha256"])
    destination = Path(amendment["sft"]["metric_destination"])
    destination.mkdir()
    event_pins = []
    for step in qualification["optimizer_steps"]:
        event = {
            "tracker": "json_logger",
            "event": "log",
            "run_id": trainer["id"],
            "step": step["step"],
            "metrics": {
                "train/loss": step["loss"],
                "optim/learning_rate": float(np.float32(step["learning_rate"])),
                "grad/norm/total": step["gradient_norm"],
                "updates/norm/total": step["update_norm"],
            },
        }
        raw = json.dumps(event, sort_keys=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = destination / f"step-{step['step']}-{digest}.json"
        path.write_bytes(raw)
        event_pins.append({"uri": str(path), "sha256": digest})
    qualification["optimizer_telemetry"] = {
        "files": event_pins,
        "skip_bad_steps": False,
        "crash_on_nan": True,
        "crash_on_inf": True,
        "learning_rate_dtype": "float32",
    }
    set_pin(post, "qualification", pinned(tmp_path, "qualified-v17", qualification))
    return post, stages, qualification, event_pins, result_pin, destination, records, launches


def test_public_durable_post_factory_requires_raw_metrics_and_exact_reload(durable_post_inputs, tmp_path):
    post, stages, qualification, event_pins, result_pin, destination, records, launches = durable_post_inputs
    outputs = durable_diversity_post_workflow(post, "calibrate")
    identities = {artifact_identity(handle) for handle in graph_handles([outputs["terminal"]])}
    assert artifact_identity(stages["train"]) not in identities
    assert artifact_identity(stages["reload"]) not in identities

    qualification["optimizer_telemetry"]["files"] = event_pins[:-1]
    set_pin(post, "qualification", pinned(tmp_path, "missing-metric-step", qualification))
    with pytest.raises(ValueError, match="cover all qualified"):
        durable_diversity_post_workflow(post, "calibrate")
    qualification["optimizer_telemetry"]["files"] = event_pins
    canonical_raw = Path(result_pin["uri"]).read_bytes()
    canonical_record = json.loads(canonical_raw)
    for failed_record in (
        {**canonical_record, "status": "failed", "error": {"message": "worker failed"}},
        {**canonical_record, "metrics": {}},
        {**canonical_record, "model": {"location": "another-export", "config": {"identity": "another-model"}}},
    ):
        raw = json.dumps(failed_record, sort_keys=True).encode()
        Path(result_pin["uri"]).write_bytes(raw)
        qualification["serving_reload"]["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
        set_pin(post, "qualification", pinned(tmp_path, "failed-reload", qualification))
        with pytest.raises(ValueError, match="reload record did not succeed"):
            durable_diversity_post_workflow(post, "calibrate")
    Path(result_pin["uri"]).write_bytes(canonical_raw)
    qualification["serving_reload"]["evidence_sha256"] = result_pin["sha256"]
    conflict = json.loads(Path(event_pins[0]["uri"]).read_bytes())
    conflict["metrics"]["train/loss"] += 1
    raw = json.dumps(conflict, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    conflict_path = destination / f"step-0-{digest}.json"
    conflict_path.write_bytes(raw)
    qualification["optimizer_telemetry"]["files"] = [*event_pins, {"uri": str(conflict_path), "sha256": digest}]
    set_pin(post, "qualification", pinned(tmp_path, "conflicting-metrics", qualification))
    with pytest.raises(ValueError, match="conflicting values"):
        durable_diversity_post_workflow(post, "calibrate")
    qualification["optimizer_telemetry"]["files"] = event_pins
    qualification["serving_reload"]["evidence_uri"] = str(tmp_path / "another-reload")
    set_pin(post, "qualification", pinned(tmp_path, "wrong-reload-evidence", qualification))
    with pytest.raises(ValueError, match="different serving reload"):
        durable_diversity_post_workflow(post, "calibrate")
    qualification["serving_reload"]["evidence_uri"] = result_pin["uri"]
    set_pin(post, "qualification", pinned(tmp_path, "qualified-again", qualification))
    records["reload"]["config"]["model"]["location"] = str(tmp_path / "different-export")
    raw = json.dumps(records["reload"], sort_keys=True).encode()
    path = Path(post["reload_producer"]["uri"])
    path.write_bytes(raw)
    post["reload_producer"]["sha256"] = hashlib.sha256(raw).hexdigest()
    post["reload_launch_proof"] = pinned(tmp_path, "wrong-export-launch", launches["reload"])
    with pytest.raises(ValueError, match="different telemetry or reload"):
        durable_diversity_post_workflow(post, "calibrate")
