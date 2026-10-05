# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import haliax as hax
import numpy as np
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from levanter.data.text.preference import PreferencePairDataset
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.bfcl_rl.collect import DATA_URI, MODELS, NATIVE_AGENT_PROFILES, ModelSource
from experiments.post_training.bfcl_rl.data import DATASET_COMMIT, BFCLPartition, TaskIdentity
from experiments.post_training.bfcl_rl.offline_collect import TEACHER_MODEL, TEACHER_REVISION
from experiments.post_training.bfcl_rl.offline_curate import (
    NativeCollectionInput,
    OfflineCollectionInput,
    collection_teacher_traces,
)
from experiments.post_training.bfcl_rl.offline_data import (
    NativeModelTrace,
    build_verified_sft_store,
    native_chat_document,
    native_model_trace,
    verifier_selected_chat,
)
from experiments.post_training.bfcl_rl.offline_preferences import NativePreferenceConfig, build_native_preference_cache
from experiments.post_training.bfcl_rl.preferences import PairDisposition, select_pair
from experiments.post_training.bfcl_rl.recovery_data import (
    RecoveryPreferenceCache,
    collection_receipt,
    recovery_cache_value,
    recovery_preference_rows,
    write_recovery_cache,
)
from experiments.post_training.bfcl_rl.retained_preferences import (
    CollectionIdentity,
    TokenStep,
    causal_token_sequence,
    pretokenized_preference,
    read_retained_archives,
    retained_rollout,
)

TASK = TaskIdentity("bfcl-simple-python-13", "simple_python_13", "audited-task-digest")
HOLDOUT = TaskIdentity("bfcl-simple-python-12", "simple_python_12", "holdout-task-digest")
PARTITION = BFCLPartition(DATASET_COMMIT, (TASK,), (HOLDOUT,))


def _record(model: str, score: float, *, task: TaskIdentity = TASK) -> dict:
    completion = 10 if model == "teacher" else 30
    return {
        "schema_version": 6,
        "record_id": f"{model}-{task.name}",
        "run_id": f"{model}-collection",
        "phase": "eval",
        "global_step": 0,
        "trajectory": {
            "instance_id": task.name,
            "repetition_id": 0,
            "environment_extras": {"data_source": f"/staged/bfcl_complement/{task.name}"},
        },
        "provenance": {"model_source_identity": f"{model}@pinned"},
        "verification_result": {"status": "verified", "score": score, "passed": None, "score_min": 0, "score_max": 1},
        "reward": {"outcome": score, "shaped": -0.25},
        "disposition": {"server_error": None, "exception_type": None, "error_treatment": None},
        "prompt": {"token_ids": [1, 2]},
        "response": {
            "token_ids": [completion, completion + 1, 20, 21],
            "loss_mask": [1, 1, 0, 1],
            "step_boundaries": [
                {"prompt_token_ids": [1, 2], "token_start": 0, "token_end": 2},
                {"prompt_token_ids": [1, 2, completion, completion + 1, 99], "token_start": 2, "token_end": 4},
            ],
        },
    }


def _identity(model: str) -> CollectionIdentity:
    return CollectionIdentity(
        f"{model}-collection",
        f"{model}@pinned",
        f"{model}-revision",
        "pi@0.87.0",
        DATASET_COMMIT,
        "/staged/bfcl_complement",
    )


def test_verified_teacher_traces_reuse_harmony_store_with_student_masks(tmp_path: Path):
    messages = [
        {"role": "system", "content": "SYSTEM_CONTEXT"},
        {"role": "user", "content": "USER_CONTEXT"},
        {
            "role": "assistant",
            "content": "ASSISTANT_REASONING\n</think>\n",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": {"key": "TOOL_ARGUMENT"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "TOOL_OBSERVATION"},
        {"role": "assistant", "content": "VERIFIER_CORRECT"},
    ]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    # The teacher's IDs intentionally lie outside this student's small vocabulary.
    teacher_record = _record("teacher", 1.0)
    teacher_record["response"] = {
        "token_ids": [9, 248000, 248001, 99, 248002, 248003, 42],
        "loss_mask": [0, 1, 1, 0, 0, 1, 0],
        "step_boundaries": [{"prompt_token_ids": [1, 2], "token_start": 0, "token_end": 7}],
    }
    identity = replace(_identity("teacher"), harness="opencode@1.18.2")
    trial = {
        "task_name": TASK.name,
        "exception_info": None,
        "verifier_result": {"rewards": {"reward": 1.0}},
        "config": {"agent": {"name": "opencode"}},
        "agent_info": {"name": "opencode", "version": "1.18.2"},
        "agent_result": {"metadata": {"rollout_correlation_id": "native-trial"}},
    }
    entries = [
        {
            "trial_id": "native-trial",
            "timestamp": index,
            "status_code": 200,
            "request": {"messages": messages[:2] if index == 0 else messages[:-1], "tools": tools},
            "literal": {
                "prompt_token_ids": [1, 2, 9] if index == 0 else [1, 2, 9, 248000, 248001, 99, 248002],
                "completion_token_ids": tokens,
                "assistant_message": assistant,
            },
        }
        for index, tokens, assistant in ((0, [248000, 248001], messages[2]), (1, [248003], messages[-1]))
    ]
    foreign = {**entries[0], "trial_id": "another-trial"}
    auxiliary = {
        **entries[0],
        "request": {"messages": [{"role": "user", "content": "Auxiliary request"}], "tools": tools},
        "timestamp": -1,
        "literal": {**entries[0]["literal"], "prompt_token_ids": [999], "completion_token_ids": [248999]},
    }
    trace = native_model_trace(
        identity=identity,
        seed=7,
        retained_record=teacher_record,
        retained_uri="retained",
        native_trace_uri="literal",
        trial_result=trial,
        literal_entries=[foreign, auxiliary, *reversed(entries)],
        partition=PARTITION,
        assistant_prefill="<think>\n",
        model_tokenizer="unused-model",
    )
    assert trace.messages == messages
    assert trace.initial_messages == messages[:2]
    continuation_prompt = [1, 2, 9, 248000, 248001, 99]
    continuation_record = {
        **teacher_record,
        "prompt": {"token_ids": continuation_prompt},
        "response": {
            "token_ids": [248002, 248003, 42],
            "loss_mask": [0, 1, 0],
            "step_boundaries": [{"prompt_token_ids": continuation_prompt, "token_start": 0, "token_end": 3}],
        },
    }
    continuation = native_model_trace(
        identity=identity,
        seed=7,
        retained_record=continuation_record,
        retained_uri="continuation",
        native_trace_uri="literal",
        trial_result=trial,
        literal_entries=entries,
        partition=PARTITION,
        assistant_prefill="<think>\n",
        model_tokenizer="unused-model",
    )
    assert continuation.messages == messages
    assert continuation.initial_prompt_sha256 == trace.initial_prompt_sha256
    changed_root = {
        **entries[0],
        "request": {"messages": [{"role": "user", "content": "ANOTHER_TASK"}], "tools": tools},
    }
    with pytest.raises(ValueError, match="Every native assistant turn"):
        native_model_trace(
            identity=identity,
            seed=7,
            retained_record=continuation_record,
            retained_uri="continuation",
            native_trace_uri="literal",
            trial_result=trial,
            literal_entries=[changed_root, entries[1]],
            partition=PARTITION,
            assistant_prefill="<think>\n",
            model_tokenizer="unused-model",
        )
    with pytest.raises(ValueError, match="differ from retained trainable"):
        native_model_trace(
            identity=identity,
            seed=7,
            retained_record=teacher_record,
            retained_uri="retained",
            native_trace_uri="literal",
            trial_result=trial,
            literal_entries=[entries[-1]],
            partition=PARTITION,
            assistant_prefill="<think>\n",
            model_tokenizer="unused-model",
        )
    duplicate = replace(
        trace,
        seed=8,
        identity=replace(trace.identity, run_id="teacher-collection-seed8"),
        retained_record={**teacher_record, "run_id": "teacher-collection-seed8"},
    )
    incorrect_record = _record("teacher", 0.0)
    incorrect_record["record_id"] = "incorrect-teacher"
    incorrect = replace(trace, retained_record=incorrect_record)
    tokenizer_path = tmp_path / "student-tokenizer"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [json.dumps(messages), json.dumps(tools)],
        trainer=trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
    )
    hf_tokenizer.chat_template = MARIN_CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(tokenizer_path)
    trace = replace(
        trace,
        model_tokenizer=str(tokenizer_path),
        assistant_completion_token_ids=tuple(
            tuple(hf_tokenizer.encode(text, add_special_tokens=False))
            for text in (
                'ASSISTANT_REASONING\n</think>\n<tool_call>\n{"name":"lookup","arguments":{"key":"TOOL_ARGUMENT"}}\n</tool_call>',
                "VERIFIER_CORRECT",
            )
        ),
    )
    duplicate = replace(trace, seed=8, identity=duplicate.identity, retained_record=duplicate.retained_record)
    incorrect = replace(trace, retained_record=incorrect_record)
    raw_loop = "<tool_call>\n" + "RAW_LOOP " * 20
    captured = tuple([*hf_tokenizer.encode(raw_loop, add_special_tokens=False), hf_tokenizer.eos_token_id])
    unparsed = replace(
        trace,
        messages=[*trace.messages[:-1], {"role": "assistant", "content": ""}],
        assistant_completion_token_ids=(*trace.assistant_completion_token_ids[:-1], captured),
        model_tokenizer=str(tokenizer_path),
    )
    literal_document = native_chat_document(unparsed)
    assert literal_document["assistant_literals"][-1] == raw_loop
    with set_current_client(LocalClient()):
        store = build_verified_sft_store(
            [trace, duplicate, incorrect],
            partition=PARTITION,
            output_path=str(tmp_path / "curation"),
            student_tokenizer=str(tokenizer_path),
            max_length=4096,
            seed=42,
            num_shards=1,
            max_workers=1,
        )
    cache = TreeCache.load(
        store.cache_path, {"input_ids": np.zeros(0, np.int32), "assistant_masks": np.zeros(0, np.int32)}
    )
    assert len(cache) == 1
    row = cache[0]
    student = load_tokenizer(str(tokenizer_path))
    masked = student.decode(np.asarray(row["input_ids"])[np.asarray(row["assistant_masks"], dtype=bool)].tolist())
    assert "ASSISTANT_REASONING" in masked
    assert "TOOL_ARGUMENT" in masked
    assert "VERIFIER_CORRECT" in masked
    assert "USER_CONTEXT" not in masked
    assert "SYSTEM_CONTEXT" not in masked
    assert "TOOL_OBSERVATION" not in masked
    selection = json.loads((tmp_path / "curation/selection.json").read_text())
    assert {item["seed"] for item in selection["selected"]} == {7, 8}
    assert selection["dispositions"] == {"verifier_correct": 2, "incorrect_or_unscored": 1}
    assert all(item["task"]["digest"] == TASK.digest for item in selection["selected"])
    assert all(item["assistant_prefill"] == "<think>\n" for item in selection["selected"])
    assert store.sources["bfcl-complement"].conversations == 1


def test_teacher_harmony_curation_rejects_parity_before_adapting_messages():
    trace = NativeModelTrace(
        _identity("teacher"),
        7,
        _record("teacher", 1.0, task=HOLDOUT),
        "retained",
        "native",
        [{"role": "user", "content": "Holdout"}, {"role": "assistant", "content": "Answer"}],
        [],
        "",
        [],
        [],
        "initial-prompt",
        "unused-model",
        (),
    )
    with pytest.raises(ValueError, match="outside the BFCL training complement"):
        verifier_selected_chat(trace, PARTITION)


def test_retained_archives_produce_exact_preferences_with_tool_context_masked(tmp_path: Path):
    records = []
    for model, score in (("teacher", 1.0), ("student", 0.0)):
        path = tmp_path / f"{model}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("records/record.json.gz", gzip.compress(json.dumps(_record(model, score)).encode()))
        records.append(read_retained_archives([str(path)], identity=_identity(model), partition=PARTITION)[0])
    teacher, student = records
    pair = select_pair(teacher.rollout, student.rollout).pair
    assert pair is not None and pair.chosen.model_revision == "teacher-revision"
    row = pretokenized_preference(teacher, student, max_length=16)
    assert row == {
        "chosen_input_ids": [1, 2, 10, 11, 99, 20, 21],
        "chosen_assistant_masks": [0, 0, 1, 1, 0, 0, 1],
        "rejected_input_ids": [1, 2, 30, 31, 99, 20, 21],
        "rejected_assistant_masks": [0, 0, 1, 1, 0, 0, 1],
    }
    assert pair.chosen.task_digest == TASK.digest


def test_retained_verdict_overrides_shaping_and_discards_infrastructure_failures():
    teacher_record = _record("teacher", 1.0)
    student_record = _record("student", 0.0)
    teacher = retained_rollout(
        teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    student_record["disposition"]["server_error"] = {"status_code": 503}
    student = retained_rollout(
        student_record, identity=_identity("student"), partition=PARTITION, trajectory_uri="student"
    )
    assert select_pair(teacher.rollout, student.rollout).disposition == PairDisposition.UNSCORED
    teacher_record["reward"]["outcome"] = 0.0
    with pytest.raises(ValueError, match="differs from the BFCL verifier"):
        retained_rollout(teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher")


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_zero_treated_agent_exit_is_excluded_from_verified_preferences(score):
    teacher_record = _record("teacher", score)
    teacher_record["reward"]["outcome"] = 0.0
    teacher_record["disposition"].update(error_treatment="zero", exception_type="NonZeroAgentExitCodeError")
    teacher = retained_rollout(
        teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    student = retained_rollout(
        _record("student", 1.0 - score), identity=_identity("student"), partition=PARTITION, trajectory_uri="student"
    )
    selection = select_pair(teacher.rollout, student.rollout)
    assert selection.disposition == PairDisposition.UNSCORED
    assert selection.pair is None


def test_retained_holdout_and_changed_model_cannot_form_training_preferences():
    with pytest.raises(ValueError, match="outside the BFCL training complement"):
        retained_rollout(
            _record("teacher", 1.0, task=HOLDOUT),
            identity=_identity("teacher"),
            partition=PARTITION,
            trajectory_uri="holdout",
        )
    with pytest.raises(ValueError, match="different model source"):
        retained_rollout(
            _record("teacher", 1.0),
            identity=replace(_identity("teacher"), model_source_identity="unrelated@model"),
            partition=PARTITION,
            trajectory_uri="teacher",
        )


def test_setup_failure_without_model_tokens_is_unscored_and_cannot_form_a_preference():
    record = _record("student", 0.0)
    record["verification_result"] = {"status": "unavailable", "reason": "NonZeroAgentExitCodeError"}
    record["disposition"]["exception_type"] = "NonZeroAgentExitCodeError"
    record["prompt"]["token_ids"] = []
    record["response"] = {"token_ids": [], "loss_mask": [], "step_boundaries": []}
    student = retained_rollout(record, identity=_identity("student"), partition=PARTITION, trajectory_uri="setup-failed")
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    selection = select_pair(teacher.rollout, student.rollout)
    assert selection.disposition == PairDisposition.UNSCORED
    assert selection.pair is None
    record["verification_result"] = _record("student", 0.0)["verification_result"]
    with pytest.raises(ValueError, match="verified rollout requires exact model-token evidence"):
        retained_rollout(record, identity=_identity("student"), partition=PARTITION, trajectory_uri="invalid-verified")


def test_context_forks_and_overlong_preferences_cannot_be_silently_rewritten():
    steps = (TokenStep((1, 2), (10, 11), (1, 1)), TokenStep((99, 98), (12,), (1,)))
    with pytest.raises(ValueError, match="context fork"):
        causal_token_sequence(steps, max_length=16)
    with pytest.raises(ValueError, match="truncation would change the rollout"):
        causal_token_sequence(steps[:1], max_length=3)
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=_identity("teacher"), partition=PARTITION, trajectory_uri="t"
    )
    student = retained_rollout(
        _record("student", 0.0), identity=_identity("student"), partition=PARTITION, trajectory_uri="s"
    )
    changed_prompt = replace(student, steps=(TokenStep((3, 4), (20,), (1,)),))
    with pytest.raises(ValueError, match="exact initial prompt"):
        pretokenized_preference(teacher, changed_prompt, max_length=16)


def _receipts(model: str) -> tuple[dict, dict]:
    expected = MODELS[model]
    source = {
        "uri": DATA_URI,
        "identity": "complement@pinned",
        "relative_path": f"bfcl_complement/{TASK.name}",
        "local_path": "/staged",
    }
    terminal = {
        "result": {
            "state": "succeeded",
            "run_id": f"logical-{model}",
            "attempt_id": "attempt",
            "iris_job_id": f"iris-{model}",
        },
        "config": {
            "run": {"id": f"logical-{model}", "attempt_id": "attempt"},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.terminal_bench_generate"},
            "ingress": {"record_literal": True},
            "inputs": {
                "model": {
                    "uri": expected.uri,
                    "identity": f"{model}@pinned",
                    "local_path": f"/staged/{model}-alias",
                    "tokenizer_uri": expected.model,
                    "tokenizer_revision": expected.revision,
                },
                "train_data": [source],
                "validation_data": [],
            },
        },
    }
    resolved = {
        "train_data_sources": [dict(source)],
        "val_data_sources": [],
        "config": {
            "trainer": {
                "policy": {"model": {"source_uri": expected.uri, "source_identity": f"{model}@pinned"}},
                "algorithm": {"tito_full": True},
            },
            "generator": {
                "n_samples_per_prompt": 1,
                "max_input_length": 32768,
                "sampling_params": {"temperature": 1.0},
                "trajectory_retention": {
                    "enabled": True,
                    "required": True,
                    "sample_fraction": 1.0,
                    "phases": ["eval"],
                    "run_id": f"{model}-collection",
                    "output_path": f"/{model}/trajectories",
                },
            },
            "terminal_bench_config": {
                "model_info": {"max_input_tokens": 32768, "max_output_tokens": 8192},
                "harbor": {
                    "name": "pi",
                    "version": "0.87.0",
                    "thinking_format": "chat-template",
                    "container_profile": "gvisor",
                    "import_path": "marinskyrl.iris_harbor_environment:IrisEnvironment",
                },
            },
        },
    }
    resolved["config"] = {"skyrl": resolved["config"]}
    return terminal, resolved


def test_completed_native_collection_joins_archives_and_literal_messages(tmp_path: Path):
    terminal, resolved = _receipts("teacher")
    config = terminal["config"]
    skyrl = resolved["config"]["skyrl"]
    teacher_source = "region-local-qwen-snapshot"
    config["inputs"]["model"].update(
        uri=teacher_source, tokenizer_uri=TEACHER_MODEL, tokenizer_revision=TEACHER_REVISION
    )
    config["inputs"]["train_data"][0]["relative_path"] = "bfcl_complement"
    resolved["train_data_sources"][0]["relative_path"] = "bfcl_complement"
    skyrl["trainer"]["policy"]["model"]["source_uri"] = teacher_source
    skyrl["trainer"]["seed"] = 7
    skyrl["terminal_bench_config"]["harbor"]["agent_profiles"] = list(NATIVE_AGENT_PROFILES)
    skyrl["generator"]["trajectory_retention"]["output_path"] = str(tmp_path / "trajectories")
    config["artifacts"] = {
        "attempts_root": str(tmp_path / "attempts"),
        "resolved_config_uri": str(tmp_path / "resolved.json"),
    }
    config["runtime"].update(experiments_dir=str(tmp_path / "literal"), launcher_commit="pinned-runtime")
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    (tmp_path / "resolved.json").write_text(json.dumps(resolved))
    messages = [
        {"role": "user", "content": "BFCL instruction"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "Correct"},
    ]
    trial = {
        "task_name": TASK.name,
        "exception_info": None,
        "verifier_result": {"rewards": {"reward": 1.0}},
        "config": {"agent": {"name": "opencode"}},
        "agent_info": {"name": "opencode", "version": "1.18.2"},
        "agent_result": {"metadata": {"rollout_correlation_id": "correct-trial"}},
    }
    trial_path = tmp_path / "attempts/trace_jobs/eval_sessions/native/task/result.json"
    trial_path.parent.mkdir(parents=True)
    trial_path.write_text(json.dumps(trial))
    record = _record("teacher", 1.0)
    archive_path = tmp_path / "trajectories/schema_v6/archives/phase=eval/step=00000000/records.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("records/correct.json.gz", gzip.compress(json.dumps(record).encode()))
    literal_path = tmp_path / "literal/logs/native_literal.jsonl"
    literal_path.parent.mkdir(parents=True)
    entries = [
        {
            "trial_id": "correct-trial",
            "timestamp": index,
            "status_code": 200,
            "request": {"messages": messages[:1] if index == 0 else messages[:-1]},
            "literal": {
                "prompt_token_ids": boundary["prompt_token_ids"] + ([20] if index else []),
                "completion_token_ids": completion,
                "assistant_message": messages[1] if index == 0 else messages[-1],
            },
        }
        for index, (boundary, completion) in enumerate(
            zip(record["response"]["step_boundaries"], ([10, 11], [21]), strict=True)
        )
    ]
    literal_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    source = OfflineCollectionInput(str(tmp_path / "terminal.json"), teacher_source, 7)
    traces = list(collection_teacher_traces(source, PARTITION, str(tmp_path / "audit.json")))
    assert len(traces) == 1
    assert traces[0].messages == messages
    assert traces[0].identity.model_revision == TEACHER_REVISION
    assert traces[0].identity.harness == "opencode@1.18.2"
    assert traces[0].retained_uri == f"{archive_path}#records/correct.json.gz"
    assert traces[0].native_trace_uri == str(trial_path)
    assert json.loads((tmp_path / "audit.json").read_text())["retained_tasks"] == 1
    literal_path.write_text(
        json.dumps({**entries[0], "literal": {**entries[0]["literal"], "completion_token_ids": [999]}}) + "\n"
    )
    with pytest.raises(ValueError, match="differ from retained"):
        list(collection_teacher_traces(source, PARTITION, str(tmp_path / "invalid-audit.json")))
    assert not (tmp_path / "invalid-audit.json").exists()


def _native_pair_collection(
    root: Path,
    model: str,
    outcomes: tuple[float | None, ...],
    partition: BFCLPartition,
    fault: str,
    seed: int = 7,
    date: str = "",
) -> NativeCollectionInput:
    root.mkdir()
    terminal, resolved = _receipts(model)
    config = terminal["config"]
    skyrl = resolved["config"]["skyrl"]
    locator = ModelSource(
        TEACHER_MODEL if model == "teacher" else MODELS["student"].model,
        TEACHER_REVISION if model == "teacher" else MODELS["student"].revision,
        f"pinned-{model}-snapshot",
        "pinned",
    )
    locator = replace(locator, model=str(root / "collection-tokenizer"))
    tokenizer_path = Path(f"{locator.model}@{locator.revision}")
    _native_pair_tokenizer(tokenizer_path)
    collection_tokenizer = load_tokenizer(str(tokenizer_path))
    config["inputs"]["model"].update(uri=locator.uri, tokenizer_uri=locator.model, tokenizer_revision=locator.revision)
    config["inputs"]["train_data"][0]["relative_path"] = "bfcl_complement"
    resolved["train_data_sources"][0]["relative_path"] = "bfcl_complement"
    skyrl["trainer"]["policy"]["model"]["source_uri"] = locator.uri
    skyrl["trainer"]["seed"] = seed
    skyrl["generator"]["trajectory_retention"]["run_id"] = f"{root.name}-collection"
    skyrl["terminal_bench_config"]["harbor"].update(
        name="opencode", version="1.18.2", agent_profiles=list(NATIVE_AGENT_PROFILES)
    )
    skyrl["generator"]["trajectory_retention"]["output_path"] = str(root / "trajectories")
    config["artifacts"] = {"attempts_root": str(root / "attempts"), "resolved_config_uri": str(root / "resolved.json")}
    config["runtime"].update(experiments_dir=str(root / "literal"), launcher_commit="pinned-runtime")
    (root / "terminal.json").write_text(json.dumps(terminal))
    (root / "resolved.json").write_text(json.dumps(resolved))
    archive_path = root / "trajectories/schema_v6/archives/phase=eval/step=00000000/records.zip"
    archive_path.parent.mkdir(parents=True)
    entries = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index, (task, score) in enumerate(zip(partition.complement, outcomes, strict=True)):
            record = _record(model, score or 0.0, task=task)
            record["run_id"] = f"{root.name}-collection"
            profile = NATIVE_AGENT_PROFILES[index % len(NATIVE_AGENT_PROFILES)]
            trial_id = f"{model}-{index}"
            trial = {
                "task_name": task.name,
                "exception_info": None,
                "verifier_result": {"rewards": {"reward": score or 0.0}},
                "config": {"agent": {"name": profile["name"]}},
                "agent_info": {"name": profile["name"], "version": profile["version"]},
                "agent_result": {"metadata": {"rollout_correlation_id": trial_id}},
            }
            if model == "student" and index == 0 and fault == "agent_error":
                trial["exception_info"] = {"exception_type": "AgentTimeoutError"}
                record["disposition"]["exception_type"] = "AgentTimeoutError"
            if score is None:
                record["verification_result"] = {"status": "unavailable", "reason": "setup_failed"}
                record["prompt"]["token_ids"] = []
                record["response"] = {"token_ids": [], "loss_mask": [], "step_boundaries": []}
                trial["exception_info"] = {"exception_type": "EnvironmentStartTimeoutError"}
            else:
                first_prompt = [1000, 1001] if model == "teacher" else [3, 4]
                record["prompt"]["token_ids"] = first_prompt
                tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
                user = {"role": "user", "content": f"BFCL_USER_CONTEXT_{task.name}"}
                if model == "student" and index == 0 and fault == "context":
                    user["content"] = "DIFFERENT_INITIAL_CONTEXT"
                assistant = {
                    "role": "assistant",
                    "content": f"{model.upper()}_REASONING\n</think>\n",
                    "tool_calls": [
                        {
                            "id": "lookup-call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"key": f"{model.upper()}_ARGUMENT"}},
                        }
                    ],
                }
                observation = {
                    "role": "tool",
                    "tool_call_id": "lookup-call",
                    "content": f"{model.upper()}_TOOL_OBSERVATION",
                }
                final = {"role": "assistant", "content": f"{model.upper()}_FINAL_{index}"}
                if model == "student" and index == 0 and fault == "malformed_calls":
                    assistant["tool_calls"][0]["function"] = {
                        "name": "hallucinated_tool",
                        "arguments": '{"key": "STUDENT_ARGUMENT',
                    }
                messages = [user, assistant, observation, final]
                if profile["name"] == "opencode":
                    identity_line = (
                        f"You are powered by the model named {model}-alias. "
                        f"The exact model ID is hosted_vllm/{model}-alias"
                    )
                    messages.insert(0, {"role": "system", "content": f"SYSTEM_INSTRUCTIONS\n{identity_line}\n{date}"})
                    # A model identification quoted in task data must remain literal.
                    user["content"] += (
                        "\nYou are powered by the model named teacher-alias. "
                        "The exact model ID is hosted_vllm/teacher-alias"
                    )
                initial_count = 2 if profile["name"] == "opencode" else 1
                raw_call = (
                    f"{model.upper()}_REASONING\n</think>\n<tool_call>\n"
                    + json.dumps(assistant["tool_calls"][0]["function"])
                    + "\n</tool_call>"
                )
                if model == "student" and index == 0 and fault == "malformed_calls":
                    raw_call += "RAW_LOOP " * 20
                completions = [
                    collection_tokenizer.encode(raw_call),
                    collection_tokenizer.encode(final["content"]),
                ]
                first_end = len(completions[0])
                record["response"] = {
                    "token_ids": [*completions[0], 99, *completions[1]],
                    "loss_mask": [*[1] * first_end, 0, *[1] * len(completions[1])],
                    "step_boundaries": [
                        {"prompt_token_ids": first_prompt, "token_start": 0, "token_end": first_end},
                        {
                            "prompt_token_ids": [999, 998],
                            "token_start": first_end,
                            "token_end": first_end + 1 + len(completions[1]),
                        },
                    ],
                }
                boundaries = record["response"]["step_boundaries"]
                for step, boundary in enumerate(boundaries):
                    response = record["response"]["token_ids"][boundary["token_start"] : boundary["token_end"]]
                    masks = record["response"]["loss_mask"][boundary["token_start"] : boundary["token_end"]]
                    completion = [token for token, mask in zip(response, masks, strict=True) if mask]
                    if model == "student" and index == 0 and step == 1 and fault == "negative_tokens":
                        completion = [888]
                    entries.append(
                        {
                            "trial_id": trial_id,
                            "timestamp": index * 10 + step,
                            "status_code": 200,
                            "request": {
                                "messages": messages[:initial_count] if step == 0 else messages[:-1],
                                "tools": tools,
                            },
                            "literal": {
                                "prompt_token_ids": (
                                    boundary["prompt_token_ids"] + ([response[0]] if masks[0] == 0 else [])
                                ),
                                "completion_token_ids": completion,
                                "assistant_message": assistant if step == 0 else final,
                            },
                        }
                    )
            path = root / f"attempts/trace_jobs/eval_sessions/native/{task.name}/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(trial))
            archive.writestr(f"records/{index}.json.gz", gzip.compress(json.dumps(record).encode()))
    literal = root / "literal/logs/native_literal.jsonl"
    literal.parent.mkdir(parents=True)
    literal.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return NativeCollectionInput(str(root / "terminal.json"), locator, seed)


def _native_pair_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["USER_CONTEXT REASONING ARGUMENT TOOL_OBSERVATION FINAL"],
        trainer=trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    hf.chat_template = MARIN_CHAT_TEMPLATE
    hf.save_pretrained(path)


@pytest.mark.parametrize("fault", ["none", "context", "negative_tokens", "agent_error", "malformed_calls"])
def test_native_dpo_cache_retokenizes_both_models_and_preserves_pair_and_loss_semantics(tmp_path: Path, fault: str):
    tasks = tuple(TaskIdentity(f"bfcl-simple-python-{i}", f"simple_python_{i}", f"digest-{i}") for i in range(13, 18))
    partition = replace(PARTITION, complement=tasks)
    teacher = _native_pair_collection(tmp_path / "teacher", "teacher", (1.0, 0.0, 0.0, 1.0, 1.0), partition, fault)
    student = _native_pair_collection(tmp_path / "student", "student", (0.0, 1.0, 0.0, 1.0, None), partition, fault)
    tokenizer_path = tmp_path / "student-tokenizer"
    _native_pair_tokenizer(tokenizer_path)
    config = NativePreferenceConfig(
        (teacher,), student, "unused", str(tokenizer_path), 4096, str(tmp_path / "cache"), 1, "student-alias"
    )
    if fault == "negative_tokens":
        with pytest.raises(ValueError, match="differ from retained trainable"):
            build_native_preference_cache(config, partition)
        assert not (tmp_path / "cache/train").exists()
        return
    with set_current_client(LocalClient()):
        value = build_native_preference_cache(config, partition)
    assert value.num_preferences == (1 if fault in ("context", "agent_error") else 2)
    report = json.loads((tmp_path / "cache/selection.json").read_text())
    assert report["dispositions"] == {
        "preference": 1 if fault == "agent_error" else 2,
        "both_incorrect": 1,
        "both_correct": 1,
        "unscored": 2 if fault == "agent_error" else 1,
    }
    assert [pair["chosen"]["model_revision"] for pair in report["preferences"]] == (
        [MODELS["student"].revision]
        if fault in ("context", "agent_error")
        else [TEACHER_REVISION, MODELS["student"].revision]
    )
    assert [item["reason"] for item in report["excluded_preferences"]] == (
        ["initial_context_mismatch"] if fault == "context" else []
    )
    adaptations = report["model_identity_adaptations"]
    teacher_initial = next(item for item in adaptations if item["source_id"].endswith(tasks[0].name))
    assert teacher_initial["original_initial_prompt_sha256"] != teacher_initial["student_initial_prompt_sha256"]
    if fault != "agent_error":
        student_initial = next(
            item
            for item in adaptations
            if item["source_alias"] == "student-alias" and item["source_id"].endswith(tasks[0].name)
        )
        assert (
            teacher_initial["student_initial_prompt_sha256"] == student_initial["student_initial_prompt_sha256"]
        ) == (fault != "context")
    else:
        assert report["collections"][1]["dispositions"]["opencode@1.18.2/unscored"] == 2
    cache = TreeCache.load(
        str(tmp_path / "cache/train"),
        {
            key: np.zeros(0, np.int32)
            for key in ("chosen_input_ids", "chosen_assistant_masks", "rejected_input_ids", "rejected_assistant_masks")
        },
    )
    tok = load_tokenizer(str(tokenizer_path))
    example = PreferencePairDataset(cache, hax.Axis("position", 4096)).as_sync_dataset()[0]
    for role, branch in (("chosen", example.chosen), ("rejected", example.rejected)):
        row = cache[0]
        ids = np.asarray(row[f"{role}_input_ids"])
        masks = np.asarray(row[f"{role}_assistant_masks"], dtype=bool)
        assert ids.max() < tok.vocab_size
        masked = tok.decode(ids[masks].tolist())
        assert "REASONING" in masked and "ARGUMENT" in masked and "FINAL" in masked
        assert "USER_CONTEXT" not in masked and "TOOL_OBSERVATION" not in masked
        if fault == "malformed_calls" and role == "rejected":
            assert masked.count("RAW_LOOP") == 20
        targets = np.roll(np.asarray(branch.tokens.array), -1)[np.asarray(branch.loss_weight.array) > 0]
        np.testing.assert_array_equal(targets, ids[masks])
        assert "TOOL_OBSERVATION" in tok.decode(ids.tolist())
        if fault not in ("context", "agent_error"):
            text = tok.decode(ids.tolist())
            assert "SYSTEM_INSTRUCTIONS\nYou are powered by the model named student-alias." in text
            assert "The exact model ID is hosted_vllm/teacher-alias" in text
            assert "SYSTEM_INSTRUCTIONS\nYou are powered by the model named teacher-alias." not in text


def test_native_teacher_pool_matches_context_without_reweighting_student_trajectories(tmp_path: Path):
    tasks = tuple(TaskIdentity(f"bfcl-simple-python-{i}", f"simple_python_{i}", f"digest-{i}") for i in range(13, 18))
    partition = replace(PARTITION, complement=tasks)
    earlier = _native_pair_collection(
        tmp_path / "earlier", "teacher", (1.0, 0.0, 0.0, 1.0, 1.0), partition, "none", date="Today's date: Oct 4\n"
    )
    fresh = _native_pair_collection(
        tmp_path / "fresh",
        "teacher",
        (1.0, 0.0, 1.0, 1.0, 1.0),
        partition,
        "none",
        seed=11,
        date="Today's date: Oct 5\n",
    )
    student = _native_pair_collection(
        tmp_path / "student", "student", (0.0, 1.0, 0.0, 1.0, None), partition, "none", date="Today's date: Oct 5\n"
    )
    tokenizer_path = tmp_path / "student-tokenizer"
    _native_pair_tokenizer(tokenizer_path)
    config = NativePreferenceConfig(
        (earlier, fresh), student, "unused", str(tokenizer_path), 4096, str(tmp_path / "cache"), 1, "student-alias"
    )
    with set_current_client(LocalClient()):
        value = build_native_preference_cache(config, partition)
    assert value.num_preferences == 3
    report = json.loads((tmp_path / "cache/selection.json").read_text())
    assert [collection["seed"] for collection in report["collections"]] == [7, 11, 7]
    assert [item["reason"] for item in report["excluded_preferences"]] == [
        "initial_context_mismatch",
        "duplicate_student_counterpart",
    ]
    pairs = {pair["chosen"]["task_source_id"]: pair for pair in report["preferences"]}
    assert set(pairs) == {task.source_id for task in tasks[:3]}
    assert "/fresh/" in pairs[tasks[0].source_id]["chosen"]["trajectory_uri"]
    assert "/earlier/" in pairs[tasks[1].source_id]["rejected"]["trajectory_uri"]
    assert pairs[tasks[1].source_id]["chosen"]["model_revision"] == MODELS["student"].revision
    documents = [json.loads(line) for line in (tmp_path / "cache/native-chat/branches.jsonl").read_text().splitlines()]
    dated = {
        document["source_id"]: document["messages"][0]["content"][0]["text"]
        for document in documents
        if document["messages"][0]["role"] == "system"
    }
    assert "Today's date: Oct 4" in dated[f"earlier-collection/teacher-{tasks[0].name}"]
    assert "Today's date: Oct 5" in dated[f"fresh-collection/teacher-{tasks[0].name}"]


def test_recovery_cache_roundtrip_preserves_causal_scoring_with_tool_context(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, score), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, score, receipt in zip(("teacher", "student"), (1.0, 0.0), receipts, strict=True)
    ]
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    write_recovery_cache(rows, report, str(tmp_path / "cache"))
    value = recovery_cache_value(str(tmp_path / "cache"))
    write_record(
        ArtifactRecord(
            output_path=value.path,
            result_type=result_type_name(RecoveryPreferenceCache),
            result=value.result_payload(),
        )
    )
    reloaded = RecoveryPreferenceCache.raw_load(value.path)
    assert reloaded.num_preferences == 1
    assert reloaded.tokenizer_revision == MODELS["student"].revision
    assert reloaded.max_length == 16
    exemplar = {key: np.zeros((0,), dtype=np.int32) for key in rows[0]}
    cache = TreeCache.load(str(tmp_path / "cache" / "train"), exemplar)
    example = PreferencePairDataset(cache, hax.Axis("position", 16)).as_sync_dataset()[0]
    chosen_targets = np.roll(np.asarray(example.chosen.tokens.array), -1)[
        np.asarray(example.chosen.loss_weight.array) > 0
    ]
    rejected_targets = np.roll(np.asarray(example.rejected.tokens.array), -1)[
        np.asarray(example.rejected.loss_weight.array) > 0
    ]
    np.testing.assert_array_equal(chosen_targets, [10, 11, 21])
    np.testing.assert_array_equal(rejected_targets, [30, 31, 21])
    assert report["dispositions"] == {"preference": 1}
    assert receipts[0].identity.run_id == "teacher-collection"
    assert (
        json.loads((tmp_path / "cache" / "selection.json").read_text())["preferences"][0]["chosen"]["model_revision"]
        == MODELS["teacher"].revision
    )


def test_incomplete_collections_cannot_publish_a_preference_cache(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=receipts[0].identity, partition=PARTITION, trajectory_uri="teacher"
    )
    student = retained_rollout(
        _record("student", 0.0), identity=receipts[1].identity, partition=PARTITION, trajectory_uri="student"
    )
    unseen = TaskIdentity("bfcl-simple-python-14", "simple_python_14", "unseen-digest")
    partition = replace(PARTITION, complement=(TASK, unseen))
    expanded = [replace(receipt, task_names=frozenset({TASK.name, unseen.name})) for receipt in receipts]
    with pytest.raises(ValueError, match="complete collection selection"):
        recovery_preference_rows(
            [teacher],
            [student],
            teacher_receipt=expanded[0],
            student_receipt=expanded[1],
            partition=partition,
            max_length=16,
        )
    assert not (tmp_path / "cache").exists()


def test_mismatched_initial_context_is_excluded_with_pair_provenance():
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, score), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, score, receipt in zip(("teacher", "student"), (1.0, 0.0), receipts, strict=True)
    ]
    student = replace(student, steps=(TokenStep((3, 4), (30,), (1,)),))
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    assert rows == []
    assert report["preferences"] == []
    excluded = report["excluded_preferences"]
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "initial_prompt_mismatch"
    assert excluded[0]["pair"]["chosen"]["trajectory_uri"] == "teacher"
    assert excluded[0]["pair"]["rejected"]["trajectory_uri"] == "student"


def test_two_wrong_rollouts_produce_no_optimizer_data(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, 0.0), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, receipt in zip(("teacher", "student"), receipts, strict=True)
    ]
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    assert rows == [] and report["dispositions"] == {"both_incorrect": 1}
    with pytest.raises(ValueError, match="must perform no update"):
        write_recovery_cache(rows, report, str(tmp_path / "cache"))
    assert not (tmp_path / "cache").exists()


def test_collection_receipts_reject_holdout_sources_and_sampling_mismatches():
    terminal, resolved = _receipts("teacher")
    terminal["config"]["inputs"]["train_data"][0]["relative_path"] = f"bfcl_complement/{HOLDOUT.name}"
    with pytest.raises(ValueError, match="outside the BFCL complement"):
        collection_receipt(terminal, resolved, model="teacher", partition=PARTITION)
    teacher_receipt = collection_receipt(*_receipts("teacher"), model="teacher", partition=PARTITION)
    terminal, resolved = _receipts("student")
    resolved["config"]["skyrl"]["generator"]["sampling_params"]["temperature"] = 0.5
    student_receipt = collection_receipt(terminal, resolved, model="student", partition=PARTITION)
    with pytest.raises(ValueError, match="different harness or sampling conditions"):
        recovery_preference_rows(
            [], [], teacher_receipt=teacher_receipt, student_receipt=student_receipt, partition=PARTITION, max_length=16
        )
