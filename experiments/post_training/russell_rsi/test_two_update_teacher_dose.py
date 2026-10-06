# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import importlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import jax.random as jrandom
import numpy as np
import pytest
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi import test_teacher_diversity_study as fixture_source
from experiments.post_training.russell_rsi.launch import adopted
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_diversity_study import diversity_workflow
from experiments.post_training.russell_rsi.test_teacher_four_pass import four_update_qualification
from experiments.post_training.russell_rsi.two_update_teacher_dose import (
    CONSUMER_MODULES,
    DECISION_PROTOCOL,
    PROTOCOL,
    VERSION,
    qualified_two_update_sft,
    two_update_dose_evaluation,
    two_update_dose_stages,
)

pinned = fixture_source.pinned
study_inputs = fixture_source.study_inputs
continuation_inputs = fixture_source.continuation_inputs
diversity_inputs = fixture_source.diversity_inputs
durable_post_inputs = fixture_source.durable_post_inputs


@pytest.fixture
def dose_inputs(durable_post_inputs, tmp_path):
    original = durable_post_inputs[0]
    study = json.loads(Path(original["sft_config_uri"]).read_bytes())
    expected = diversity_workflow(study)["collect"]
    root = tmp_path / "completed-training-collection"
    root.mkdir()
    rows = [
        {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "qualified"}]}
        for i in range(8)
    ]
    accepted = [{"row": row, "task": {"family": f"family-{i}"}} for i, row in enumerate(rows)]
    collection = {"status": "passed", "accepted": accepted}

    result = pinned(tmp_path, "collection", collection)
    Path(result["uri"]).rename(root / "collection.json")
    result["uri"] = str(root / "collection.json")
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (root / "train.jsonl").write_bytes(content)
    train = {"uri": str(root / "train.jsonl"), "sha256": hashlib.sha256(content).hexdigest()}
    dataset = pinned(
        tmp_path,
        "dataset",
        {
            "rows": 8,
            "sha256": train["sha256"],
            "collection_sha256": compact_json_sha256(collection),
        },
    )
    Path(dataset["uri"]).rename(root / "dataset.json")
    dataset["uri"] = str(root / "dataset.json")
    producer = pinned(
        tmp_path,
        "producer",
        {
            "name": expected.name,
            "version": expected.version,
            "fingerprint": expected.fingerprint(),
            "output_path": str(root),
        },
    )
    Path(producer["uri"]).rename(root / ".artifact.json")
    producer["uri"] = str(root / ".artifact.json")
    (root / ".executor_status").write_text("SUCCESS")
    status = {"uri": str(root / ".executor_status"), "sha256": hashlib.sha256(b"SUCCESS").hexdigest()}
    consumer = pinned(
        tmp_path,
        "consumer",
        {
            "status": "canonical_runtime_rows_passed",
            "collection_identity": artifact_identity(expected),
            "runtime_commit": MARIN_SKYRL.commit,
            "train_sha256": train["sha256"],
            "full_rows_no_overflow": True,
            "rows": [{"row_sha256": compact_json_sha256(row), "tokens": 10, "assistant_targets": 1} for row in rows],
            "consumer_imports": [
                {
                    "module": name,
                    "sha256": hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest(),
                }
                for name in sorted(CONSUMER_MODULES)
            ],
        },
    )
    pins = {
        "producer": producer,
        "status": status,
        "result": result,
        "dataset": dataset,
        "train": train,
        "canonical_runtime_rows": consumer,
    }
    original_pin = pinned(tmp_path, "original-post", original)

    parent = adopted(study["parent"], LevanterCheckpoint)
    decision = {
        "protocol": DECISION_PROTOCOL,
        "original_post_config": original_pin,
        "collection": pins,
        "parent_identity": artifact_identity(parent),
        "rows": 8,
        "passes": 2,
        "example_exposures": 16,
        "batch_size": 8,
        "optimizer_updates": 2,
        "learning_rate": 1e-6,
        "context_tokens": 16384,
        "fresh_optimizer": True,
        "conditions": ["sft"],
        "new_collection": False,
        "calibration": False,
        "rl_authorized": False,
    }
    return {
        "protocol": PROTOCOL,
        "version": VERSION,
        "original_post_config": original_pin,
        "collection": pins,
        "prospective_decision": pinned(tmp_path, "dose-decision", decision),
    }


def test_two_update_dose_consumes_each_existing_row_twice_with_fresh_optimizer(dose_inputs, tmp_path):
    stages = two_update_dose_stages(dose_inputs)
    train = stages["train"]
    prefix = str(tmp_path / "artifacts")
    pod = train.build_config(
        StepContext.for_run(train.path(prefix), prefix, deps=train.deps, runtime_args=train.runtime_args)
    )
    config = pod.train_config
    assert config.trainer.num_train_steps == 2 and config.trainer.train_batch_size == 8
    assert config.trainer.load_checkpoint is False and config.trainer.initialize_from is None
    assert config.train_seq_len == 16384 and config.data.mixture_block_size == 8
    key = jrandom.PRNGKey(config.trainer.seed)
    shuffled = ListAsyncDataset(list(range(config.trainer.train_batch_size))).shuffle(key)
    mixture = MixtureDataset(
        {"teacher": shuffled},
        {"teacher": 1.0},
        stop_strategy=config.data.stop_strategy,
        key=key,
        block_size=config.data.mixture_block_size,
    )
    observed = asyncio.run(
        mixture.get_batch(list(range(config.trainer.num_train_steps * config.trainer.train_batch_size)))
    )
    assert Counter(observed) == {i: 2 for i in range(8)}
    assert all(Counter(observed[a : a + 8]) == {i: 1 for i in range(8)} for a in (0, 8))
    handles = list(graph_handles([stages["reload"]]))
    assert not any(
        "conversations" in step.name or "calibration" in step.name or "sft-rl" in step.name for step in handles
    )
    reload = stages["reload"]
    bound = reload.build_config(
        StepContext.for_run(reload.path(prefix), prefix, deps=reload.deps, runtime_args=reload.runtime_args)
    )
    assert bound.model.identity == artifact_identity(train)
    assert bound.model.location == prefix_join(train.path(prefix), "hf/step-1")
    assert config.hf_save_steps == 2


def test_four_update_export_cannot_be_relabelled_as_two_update_dose(tmp_path):
    qualification = four_update_qualification("four-update-model", str(tmp_path))
    qualification["protocol"] = "teacher-sft-two-update-qualification-v1"
    with pytest.raises(ValueError, match="pinned 2-update"):
        qualified_two_update_sft(qualification, identity="four-update-model", root=str(tmp_path))


def test_completed_two_update_evaluation_binds_actual_producer_and_raw_telemetry(
    dose_inputs, durable_post_inputs, tmp_path
):
    original_post, _, old_qualification, _, _, _, old_records, old_launches = durable_post_inputs
    config_pin = pinned(tmp_path, "prospective-dose", dose_inputs)
    stages = two_update_dose_stages(dose_inputs)
    review = json.loads(Path(original_post["sft_source_review"]["uri"]).read_bytes())
    review["source_files"] = old_launches["sft"]["source_files"]
    post = {"dose_config": config_pin, "sft_source_review": pinned(tmp_path, "dose-source", review)}
    records, launches = {}, {}
    prefix = str(tmp_path / "dose-artifacts")
    for role, handle in (("sft", stages["train"]), ("reload", stages["reload"])):
        root = Path(handle.path(prefix))
        root.mkdir(parents=True)
        (root / ".executor_status").write_text("SUCCESS")
        ctx = StepContext.for_run(str(root), prefix, deps=handle.deps, runtime_args=handle.runtime_args)
        bound = json.loads(canonical_json(handle.build_config(ctx)))
        launch = deepcopy(old_launches[role])
        launch.update(
            config=config_pin,
            source_review=post["sft_source_review"],
            producer_identity=artifact_identity(handle),
            output_path=str(root),
            bound_config=bound,
        )
        launch["request"] = pinned(
            tmp_path,
            f"dose-{role}-request",
            {
                "stage": role,
                "version": handle.version,
                "source_head": review["source_head"],
                "runtime_commit": MARIN_SKYRL.commit,
                "config_uri": config_pin["uri"],
                "config_sha256": config_pin["sha256"],
            },
        )
        launch["preflight"] = pinned(
            tmp_path,
            f"dose-{role}-preflight",
            {
                "exit_code": 0,
                "identity": {"source_head": review["source_head"], "request_sha256": launch["request"]["sha256"]},
            },
        )
        record = deepcopy(old_records[role])
        record.update(
            name=handle.name,
            version=handle.version,
            fingerprint=handle.fingerprint(),
            output_path=str(root),
            config=bound,
        )
        if role == "reload":
            reload_record = {
                "run_id": "dose-reload",
                "status": "succeeded",
                "error": None,
                "metrics": {"mmlu_abstract_algebra_0shot": {"sample_len": 1.0, "acc,none": 0.0}},
                "model": {"location": bound["model"]["location"], "config": {"identity": bound["model"]["identity"]}},
                "eval": {"name": "mmlu-smoke", "evalchemy": {"max_eval_instances": 1}},
            }
            (root / "dose-reload").mkdir()
            reload_pin = pinned(root / "dose-reload", "record", reload_record)
            Path(reload_pin["uri"]).rename(root / "dose-reload" / "record.json")
            reload_pin["uri"] = str(root / "dose-reload" / "record.json")
            record["result"] = {
                "records_prefix": str(root),
                "run_ids": ["dose-reload"],
                "results_paths": [str(root / "dose-reload/results")],
            }
        post[f"{role}_producer"] = pinned(root, ".artifact", record)
        Path(post[f"{role}_producer"]["uri"]).rename(root / ".artifact.json")
        post[f"{role}_producer"]["uri"] = str(root / ".artifact.json")
        post[f"{role}_launch_proof"] = pinned(tmp_path, f"dose-{role}-launch", launch)
        records[role], launches[role] = record, launch
    qualification = deepcopy(old_qualification)
    qualification.update(
        protocol="teacher-sft-two-update-qualification-v1",
        source_config_sha256=config_pin["sha256"],
        sft_identity=artifact_identity(stages["train"]),
        sft_root=records["sft"]["output_path"],
        hf_export_uri=records["reload"]["config"]["model"]["location"],
        optimizer_updates=2,
    )
    qualification["optimizer_steps"] = qualification["optimizer_steps"][:2]
    qualification["serving_reload"].update(
        model_identity=qualification["sft_identity"],
        model_uri=qualification["hf_export_uri"],
        evidence_uri=reload_pin["uri"],
        evidence_sha256=reload_pin["sha256"],
    )
    destination = Path(records["sft"]["config"]["train_config"]["trainer"]["tracker"][0]["metric_destination"])
    destination.mkdir()
    event_pins = []
    for step in qualification["optimizer_steps"]:
        event = {
            "tracker": "json_logger",
            "event": "log",
            "run_id": records["sft"]["config"]["train_config"]["trainer"]["id"],
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
    qualification["optimizer_telemetry"]["files"] = event_pins
    post["qualification"] = pinned(tmp_path, "two-qualified", qualification)
    outputs = two_update_dose_evaluation(post)
    assert not any(
        "calibration" in handle.name or "sft-rl" in handle.name for handle in graph_handles([outputs["terminal"]])
    )
    records["sft"]["config"]["train_config"]["trainer"]["num_train_steps"] = 4
    launches["sft"]["bound_config"] = records["sft"]["config"]
    post["sft_producer"] = pinned(Path(records["sft"]["output_path"]), ".artifact", records["sft"])
    canonical = Path(records["sft"]["output_path"]) / ".artifact.json"
    Path(post["sft_producer"]["uri"]).replace(canonical)
    post["sft_producer"]["uri"] = str(canonical)
    post["sft_launch_proof"] = pinned(tmp_path, "wrong-dose-launch", launches["sft"])
    with pytest.raises(ValueError, match="executed different training"):
        two_update_dose_evaluation(post)
