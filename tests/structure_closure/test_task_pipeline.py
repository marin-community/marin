# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace

import pytest
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.test_support import ReplayModel, rollout_spec
from taskcompendium.models import Source, TaskSpec, verifyit_spec
from verifyit.spec import StructureClosureSpec
from verifyit.tool_call_markup import THINK_CLOSE

from experiments.post_training.structure_closure.benchmark import (
    CLOSE_REQUIRED,
    Family,
    build_benchmark,
    task_cases,
)
from experiments.post_training.structure_closure.task_data import case_task, paired_task_probes, prepare_tasks
from experiments.post_training.structure_closure.token_data import ToolPrompt, tokenized_case_record
from experiments.post_training.structure_closure.tool_prompt_probe import paired_tool_probes
from experiments.post_training.structure_closure.training_prefixes import prepare_training_prefixes

SOURCE = Source(dataset="closure-fixture", revision="source-revision", row="0", importer_revision="1")


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
@pytest.mark.asyncio
async def test_exported_tasks_grade_each_family_through_prefill_harness(tmp_path, tokenizer, tool_prompt):
    cases = tmp_path / "cases"
    write_cases(cases)
    near = tmp_path / "near"
    prepare_training_prefixes(cases, near)
    initializer = tmp_path / "tokenizer"
    tokenizer.save_pretrained(initializer)
    output = tmp_path / "tasks"
    prepare_tasks(near, output, tool_prompt=tool_prompt, initializer=initializer)
    tasks = [TaskSpec.model_validate_json(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    originals = {row["case_id"]: row for row in map(json.loads, (cases / "validation.jsonl").read_text().splitlines())}
    assert {task.id for task in tasks} == {"structure-closure:" + identity for identity in originals}
    assert {original["family"] for original in originals.values()} == set(Family)
    engine = RolloutEngine({}, {"prefill": PrefillHarness()})
    for task in tasks:
        original = originals[task.id.removeprefix("structure-closure:")]
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, StructureClosureSpec)
        if original["family"] in (Family.TOOL, Family.REPEATED_OPENER):
            assert json.loads(spec.expected.removeprefix("<tool_call>").removesuffix("</tool_call>")) == json.loads(
                original["expected"].removeprefix("<tool_call>").removesuffix("</tool_call>")
            )
        else:
            assert spec.expected == original["expected"]
        model = ReplayModel([{"role": "assistant", "content": spec.expected[len(spec.prefix) :]}])
        result = await engine.run_task(rollout_spec(task, harness="prefill", max_turns=1), model.complete)
        assert result.grade.reward == 1
        assert model.requests[0].messages[-1] == {"role": "assistant", "content": spec.prefix}


@pytest.mark.parametrize(
    "marker", ["<tool_call>", "</tool_call>", '<tool_call>{"name":"nested","arguments":{}}</tool_call>']
)
@pytest.mark.asyncio
async def test_paired_tool_probes_preserve_quoted_protocol_markers(tokenizer, marker):
    case = next(
        case
        for case in task_cases("parent", [{"search": {"query": marker}}], "validation", "source")
        if case.family == Family.TOOL and case.control == CLOSE_REQUIRED
    )
    record = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
    historical = paired_tool_probes(record, tokenizer)
    tasks = paired_task_probes(case, SOURCE, tokenizer)
    assert len(historical) == len(tasks) == 4
    engine = RolloutEngine({}, {"prefill": PrefillHarness()})
    for old, task in zip(historical, tasks, strict=True):
        spec = verifyit_spec(task.grader)
        assert isinstance(spec, StructureClosureSpec)
        assert spec.prefix == old["prefix"]
        assert json.loads(spec.expected.removeprefix("<tool_call>").removesuffix("</tool_call>"))["arguments"][
            "result"
        ] == [{"search": {"query": marker}}]
        model = ReplayModel([{"role": "assistant", "content": "</tool_call>"}])
        result = await engine.run_task(rollout_spec(task, harness="prefill", max_turns=1), model.complete)
        assert result.grade.reward == 1
        assert bool(model.requests[0].options.get("tools")) == old["probe_variant"].startswith("declared-")


def test_task_export_rejects_cross_split_parent_leakage_without_partial_splits(tmp_path):
    cases = tmp_path / "cases"
    write_cases(cases, overlap=True)
    output = tmp_path / "tasks"
    with pytest.raises(ValueError):
        prepare_tasks(cases, output, tool_prompt=ToolPrompt.COMPACT, initializer=None)
    assert not (output / "train.jsonl").exists()
    assert not (output / "validation.jsonl").exists()
    assert not (output / "manifest.json").exists()
    assert len(json.loads((output / "conversion-failures.json").read_text())) == 1


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
    validation = [TaskSpec.model_validate_json(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert training and validation
    assert {row["task_id"] for row in map(json.loads, (built / "train.jsonl").read_text().splitlines())} == {
        "train_parent"
    }
    assert {row["task_id"] for row in map(json.loads, (built / "validation.jsonl").read_text().splitlines())} == {
        "validation_parent"
    }
    assert {task.id for task in training}.isdisjoint(task.id for task in validation)


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


@pytest.mark.parametrize("continuation, reward", [(THINK_CLOSE, 1), ("unfinished reasoning", 0)])
@pytest.mark.asyncio
async def test_reasoning_task_grades_raw_closing_delimiter(continuation, reward):
    case = next(
        case
        for case in task_cases("parent", [], "validation", "source")
        if case.family == Family.REASONING and case.control == CLOSE_REQUIRED
    )
    task = case_task(case, SOURCE)
    model = ReplayModel([{"role": "assistant", "content": continuation}])
    result = await RolloutEngine({}, {"prefill": PrefillHarness()}).run_task(
        rollout_spec(task, harness="prefill", max_turns=1), model.complete
    )
    assert result.grade.reward == reward
    assert result.rollout.conversations[0].steps[0].turn.message["content"] == continuation
