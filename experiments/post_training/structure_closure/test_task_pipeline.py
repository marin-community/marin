# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import sys
from dataclasses import asdict, replace

import httpx
import pytest
from openai import AsyncOpenAI
from rolloutengine.chat import ChatModel
from rolloutengine.models import ModelTurn, RolloutFailure, TraceMode
from rolloutengine.test_support import rollout_limits
from taskcompendium.models import Source, TaskSpec, TextMessage, verifyit_spec
from verifyit.spec import StructureClosureSpec
from verifyit.tool_call_markup import THINK_CLOSE

from experiments.post_training.structure_closure.benchmark import (
    CLOSE_REQUIRED,
    ClosureCase,
    Family,
    build_benchmark,
    task_cases,
)
from experiments.post_training.structure_closure.evaluate import completion_model, evaluate_cohort, evaluate_task
from experiments.post_training.structure_closure.rollout_audit import (
    EVALUATION_RECORD,
    audit_rollouts,
    compare_evaluations,
    rollout_summary,
)
from experiments.post_training.structure_closure.rollout_audit import main as audit_main
from experiments.post_training.structure_closure.task_data import (
    case_task,
    load_validation,
    paired_task_probes,
    prepare_tasks,
)
from experiments.post_training.structure_closure.token_data import ToolPrompt, tokenized_case_record
from experiments.post_training.structure_closure.tool_prompt_probe import paired_tool_probes
from experiments.post_training.structure_closure.training_prefixes import prepare_training_prefixes

SOURCE = Source(dataset="closure-fixture", revision="source-revision", row="0", importer_revision="1")
SAMPLING = {"temperature": 0, "max_tokens": 256}


def object_task():
    return case_task(
        ClosureCase(
            "object",
            "parent",
            "validation",
            Family.OBJECT,
            CLOSE_REQUIRED,
            "Finish the object.",
            '{"task":"x"',
            '{"task":"x"}',
            "source",
        ),
        SOURCE,
    )


def model_answer(text, *, prompt_tokens=(10, 11)):
    async def complete(request):
        assert request.prompt_token_ids is None
        return ModelTurn({"role": "assistant", "content": text}, prompt_tokens, (20,), (-0.5,), "stop", text=text)

    return complete


def write_cases(directory, *, overlap=False):
    directory.mkdir()
    hashes = {}
    for split in ("train", "validation"):
        cases = task_cases("parent" if overlap else split + "_parent", [{"search": {"query": "value"}}], split, "source")
        if overlap:
            cases = [
                replace(case, case_id=hashlib.sha256(f"{split}/{case.case_id}".encode()).hexdigest()) for case in cases
            ]
        data = "".join(json.dumps(asdict(case)) + "\n" for case in cases).encode()
        (directory / f"{split}.jsonl").write_bytes(data)
        hashes[split] = hashlib.sha256(data).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({"hashes": hashes}))


@pytest.mark.parametrize("tool_prompt", list(ToolPrompt))
def test_generated_cohort_runs_each_family_through_tasks_harness_and_grader(tmp_path, tokenizer, tool_prompt):
    cases = tmp_path / "cases"
    write_cases(cases)
    near = tmp_path / "near"
    prepare_training_prefixes(cases, near)
    initializer = tmp_path / "tokenizer"
    tokenizer.save_pretrained(initializer)
    output = tmp_path / "tasks"
    prepare_tasks(near, output, tool_prompt=tool_prompt, initializer=initializer)
    tasks = load_validation(output)
    answers = {}
    for task in tasks:
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, StructureClosureSpec)
        answers[tuple((event.role, event.content) for event in task.context.events)] = spec.expected[len(spec.prefix) :]

    async def complete(request):
        # The standard submission instruction is inserted before the assistant prefill.
        key = (
            (request.messages[0]["role"], request.messages[0]["content"]),
            (request.messages[-1]["role"], request.messages[-1]["content"]),
        )
        return await model_answer(answers[key])(request)

    records = asyncio.run(
        evaluate_cohort(tasks, complete, limits=rollout_limits(max_turns=1), concurrency=3, sampling=SAMPLING)
    )
    restored = [EVALUATION_RECORD.validate_json(EVALUATION_RECORD.dump_json(record)) for record in records]
    audit = audit_rollouts(tasks, restored, sampling=SAMPLING)
    assert audit.complete
    assert {result.family for result in audit.results} == set(Family)
    assert all(result.success for result in audit.results)


@pytest.mark.parametrize(
    "marker", ["<tool_call>", "</tool_call>", '<tool_call>{"name":"nested","arguments":{}}</tool_call>']
)
def test_paired_tool_probes_preserve_quoted_protocol_markers(tokenizer, marker):
    case = next(
        case
        for case in task_cases("parent", [{"search": {"query": marker}}], "validation", "source")
        if case.family == Family.TOOL and case.control == CLOSE_REQUIRED
    )
    record = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
    historical = paired_tool_probes(record, tokenizer)
    tasks = paired_task_probes(case, SOURCE, tokenizer)
    assert len(historical) == len(tasks) == 4
    for old, task in zip(historical, tasks, strict=True):
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, StructureClosureSpec)
        assert spec.prefix == old["prefix"]
        assert json.loads(spec.expected.removeprefix("<tool_call>").removesuffix("</tool_call>"))["arguments"][
            "result"
        ] == [{"search": {"query": marker}}]
        actual = asyncio.run(
            evaluate_task(
                task,
                model_answer("</tool_call>"),
                limits=rollout_limits(max_turns=1),
                trace=TraceMode.MESSAGES,
                sampling=SAMPLING,
            )
        )
        assert actual.result.grade.reward == 1.0
        assert bool(actual.request.options.get("tools")) == old["probe_variant"].startswith("declared-")


@pytest.mark.parametrize("tokens", [(10, 11), (72, 84, 96)])
def test_text_evaluation_accepts_actual_model_tokenization_and_semantic_output(tokens):
    task = object_task()
    record = asyncio.run(
        evaluate_task(
            task,
            model_answer(" } extra output", prompt_tokens=tokens),
            limits=rollout_limits(max_turns=1),
            trace=TraceMode.EXACT_TOKENS,
            sampling=SAMPLING,
        )
    )
    audit = audit_rollouts([task], [record], sampling=SAMPLING)
    assert audit.complete
    assert audit.results[0].success
    assert audit.results[0].trailing_content
    assert not audit.results[0].whole_response_success
    assert record.result.rollout.conversations[0].prompt_token_ids == tokens


@pytest.mark.parametrize("failure", ["missing", "duplicate", "conditioning", "sampling", "reward", "interrupted"])
def test_audit_rejects_changed_or_incomplete_evidence_without_shrinking_denominators(failure):
    first = object_task()
    second = first.model_copy(update={"id": "structure-closure:second"})
    tasks = [first, second]
    records = [
        asyncio.run(
            evaluate_task(
                task, model_answer("}"), limits=rollout_limits(max_turns=1), trace=TraceMode.MESSAGES, sampling=SAMPLING
            )
        )
        for task in tasks
    ]
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records.append(records[1])
    else:
        record = records[1]
        assert record.request is not None
        if failure == "conditioning":
            request = replace(record.request, messages=({"role": "user", "content": "different task"},))
            records[1] = replace(record, request=request)
        elif failure == "sampling":
            request = replace(record.request, options={**record.request.options, "temperature": 1})
            records[1] = replace(record, request=request)
        elif failure == "reward":
            records[1] = replace(record, result=replace(record.result, grade=replace(record.result.grade, reward=0)))
        else:
            rollout = replace(record.result.rollout, failure=RolloutFailure("TimeoutError"))
            records[1] = replace(record, result=replace(record.result, rollout=rollout))
    audit = audit_rollouts(tasks, records, sampling=SAMPLING)
    report = rollout_summary(tasks, audit, sampling=SAMPLING)
    assert not audit.complete
    assert report["groups"]["json_object"]["expected"] == 2
    if failure != "duplicate":
        assert report["groups"]["json_object"]["unscored"] == 1
        assert report["groups"]["json_object"]["success_rate"] == 0.5
    with pytest.raises(ValueError, match="Incomplete"):
        compare_evaluations(report, report)


def test_audit_compares_real_model_failures_and_improvement_on_the_same_cases():
    task = object_task()
    reports = []
    for response in ("", "}"):
        record = asyncio.run(
            evaluate_task(
                task,
                model_answer(response),
                limits=rollout_limits(max_turns=1),
                trace=TraceMode.MESSAGES,
                sampling=SAMPLING,
            )
        )
        audit = audit_rollouts([task], [record], sampling=SAMPLING)
        assert audit.complete
        reports.append(rollout_summary([task], audit, sampling=SAMPLING))
    assert compare_evaluations(*reports)["json_object"] == {"baseline": 0, "candidate": 1, "change": 1}
    changed = {**reports[1], "sampling": {**SAMPLING, "temperature": 1}}
    with pytest.raises(ValueError, match="sampling"):
        compare_evaluations(reports[0], changed)


def test_task_export_rejects_cross_split_parent_leakage_without_partial_splits(tmp_path):
    cases = tmp_path / "cases"
    write_cases(cases, overlap=True)
    output = tmp_path / "tasks"
    with pytest.raises(ValueError, match="task conversion failures"):
        prepare_tasks(cases, output, tool_prompt=ToolPrompt.COMPACT, initializer=None)
    assert not (output / "train.jsonl").exists()
    assert not (output / "validation.jsonl").exists()
    assert not (output / "manifest.json").exists()
    assert "same parent" in json.loads((output / "conversion-failures.json").read_text())[0]["error"]


def test_benchmark_builder_uses_disjoint_parent_tasks_and_persists_both_splits(tmp_path):
    source = tmp_path / "bfcl"
    source.mkdir()
    (source / "bfcl_complement-manifest.json").write_text(
        json.dumps({"complement": [{"source_id": name, "name": name} for name in ("train_parent", "validation_parent")]})
    )
    for name in ("train_parent", "validation_parent"):
        path = source / "bfcl_complement" / name / "solution" / "solve.sh"
        path.parent.mkdir(parents=True)
        path.write_text("cat << 'ORACLE_EOF'\n" + json.dumps([{"search": {"query": name}}]) + "\nORACLE_EOF\n")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "train_task_ids": ["train_parent"],
                "diagnostic_reserved_task_ids": ["validation_parent"],
                "parity_task_ids": [],
            }
        )
    )
    built = tmp_path / "built"
    build_benchmark(source, split, built)
    output = tmp_path / "tasks"
    prepare_tasks(built, output, tool_prompt=ToolPrompt.COMPACT, initializer=None)
    training = [TaskSpec.model_validate_json(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert {task.id for task in training}.isdisjoint(task.id for task in load_validation(output))


async def test_text_prefill_and_experiment_settings_reach_chat_endpoint():
    captured = []

    async def serve(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "}"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        record = await evaluate_task(
            object_task(),
            ChatModel(client, "test-model", max_tokens=256),
            limits=rollout_limits(max_turns=1),
            trace=TraceMode.MESSAGES,
            sampling=SAMPLING,
        )
    assert captured[0]["messages"][-1] == {"role": "assistant", "content": '{"task":"x"'}
    assert captured[0]["continue_final_message"] is True
    assert captured[0]["add_generation_prompt"] is False
    assert captured[0]["temperature"] == 0
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert audit_rollouts([object_task()], [record], sampling=SAMPLING).complete


def test_cohort_keeps_interrupted_attempts_and_later_successful_cases():
    first = object_task()
    second = first.model_copy(
        update={
            "id": "structure-closure:second",
            "context": first.context.model_copy(
                update={"events": (TextMessage(role="user", content="Second task."), first.context.events[-1])}
            ),
        }
    )

    async def complete(request):
        if request.messages[0]["content"] == "Finish the object.":
            raise RuntimeError("Inference transport failed")
        return await model_answer("}")(request)

    tasks = [first, second]
    records = asyncio.run(
        evaluate_cohort(tasks, complete, limits=rollout_limits(max_turns=1), concurrency=2, sampling=SAMPLING)
    )
    assert len(records) == 2
    assert records[0].result.rollout.failure is not None
    audit = audit_rollouts(tasks, records, sampling=SAMPLING)
    assert not audit.complete
    assert [(result.uid, result.success) for result in audit.results] == [(second.id, True)]
    report = rollout_summary(tasks, audit, sampling=SAMPLING)
    assert report["groups"]["json_object"]["expected"] == 2
    assert report["groups"]["json_object"]["unscored"] == 1


def test_valid_but_wrong_tool_action_is_a_scored_model_failure():
    async def complete(request):
        return ModelTurn(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "write_result", "arguments": "{}"}}
                ],
            },
            (1,),
            (2,),
            (-1,),
            "stop",
        )

    task = object_task()
    record = asyncio.run(
        evaluate_task(task, complete, limits=rollout_limits(max_turns=1), trace=TraceMode.MESSAGES, sampling=SAMPLING)
    )
    audit = audit_rollouts([task], [record], sampling=SAMPLING)
    assert audit.complete
    assert not audit.results[0].success
    assert record.result.grade.reward == 0


def test_audit_cli_retains_malformed_rows_and_writes_incomplete_comparison(tmp_path, monkeypatch):
    cases = tmp_path / "cases"
    write_cases(cases)
    tasks_dir = tmp_path / "tasks"
    prepare_tasks(cases, tasks_dir, tool_prompt=ToolPrompt.COMPACT, initializer=None)
    tasks = load_validation(tasks_dir)
    task = tasks[0]
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StructureClosureSpec)
    record = asyncio.run(
        evaluate_task(
            task,
            model_answer(spec.expected[len(spec.prefix) :]),
            limits=rollout_limits(max_turns=1),
            trace=TraceMode.MESSAGES,
            sampling=SAMPLING,
        )
    )
    dump = tmp_path / "rollouts.jsonl"
    dump.write_bytes(EVALUATION_RECORD.dump_json(record) + b'\n{"broken":\n')
    sampling = tmp_path / "sampling.json"
    sampling.write_text(json.dumps(SAMPLING))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"complete": False}))
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--tasks",
            str(tasks_dir),
            "--rollouts",
            str(dump),
            "--sampling",
            str(sampling),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        audit_main()
    assert caught.value.code == 1
    report = json.loads(output.read_text())
    assert not report["complete"]
    assert report["expected"] == len(tasks)
    assert len(report["missing_uids"]) == len(tasks) - 1
    assert report["failures"][0]["uid"] == "line:2"
    assert len(report["results"]) == 1
    assert "comparison_error" in report


@pytest.mark.parametrize("split", ["train", "validation"])
def test_near_boundary_variant_rejects_modified_source_splits(tmp_path, split):
    cases = tmp_path / "cases"
    write_cases(cases)
    path = cases / f"{split}.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    output = tmp_path / "variant"
    with pytest.raises(ValueError):
        prepare_training_prefixes(cases, output)
    assert not output.exists()


async def test_raw_completion_transport_preserves_reasoning_closing_delimiter(tokenizer):
    case = next(
        case
        for case in task_cases("parent", [], "validation", "source")
        if case.family == Family.REASONING and case.control == CLOSE_REQUIRED
    )
    task = case_task(case, SOURCE)
    captured = []

    async def serve(request):
        assert request.url.path == "/v1/completions"
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "text_completion",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "text": THINK_CLOSE, "logprobs": None}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        record = await evaluate_task(
            task,
            completion_model(client, model="test-model", tokenizer=tokenizer),
            limits=rollout_limits(max_turns=1),
            trace=TraceMode.MESSAGES,
            sampling=SAMPLING,
        )
    assert captured[0]["prompt"].endswith(case.prefix)
    assert record.result.rollout.conversations[0].steps[0].turn.text == THINK_CLOSE
    assert record.result.grade.reward == 1
    audit = audit_rollouts([task], [record], sampling=SAMPLING)
    assert audit.complete
    assert audit.results[0].success


def test_parsed_reasoning_is_unscored_instead_of_a_false_model_failure():
    case = next(
        case
        for case in task_cases("parent", [], "validation", "source")
        if case.family == Family.REASONING and case.control == CLOSE_REQUIRED
    )
    task = case_task(case, SOURCE)

    async def parsed(request):
        return ModelTurn(
            {"role": "assistant", "content": "", "reasoning_content": "parser removed delimiters"},
            (1,),
            (2,),
            (-1,),
            "stop",
        )

    record = asyncio.run(
        evaluate_task(task, parsed, limits=rollout_limits(max_turns=1), trace=TraceMode.MESSAGES, sampling=SAMPLING)
    )
    audit = audit_rollouts([task], [record], sampling=SAMPLING)
    assert not audit.complete
    assert not audit.results
    assert len(audit.failures) == 1
