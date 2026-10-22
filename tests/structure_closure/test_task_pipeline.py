# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace

import pytest
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.test_support import ReplayModel, rollout_spec
from taskcompendium.models import Source, TaskSpec, verifyit_spec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.spec import StructureClosureSpec
from verifyit.tool_call_markup import THINK_CLOSE
from zephyr.readers import load_parquet

from experiments.post_training.structure_closure.token_data import ToolPrompt, tokenized_case_record
from experiments.post_training.structure_closure.tool_prompt_probe import paired_tool_probes
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    CLOSE_REQUIRED,
    Family,
    task_cases,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.prefixes import prepare_training_prefixes
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.source import TrainingPrefix, sources
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.tasks import (
    case_task,
    paired_task_probes,
    prepare_tasks,
)
from experiments.post_training.task_curation.local import run_local_steps
from experiments.post_training.task_curation.pipeline import HfSource, process_rows

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
    training = [TaskSpec.model_validate_json(line) for line in (output / "train.jsonl").read_text().splitlines()]
    training_tool = next(
        task for task in training if "family:tool_delimiter" in task.tags and "control:continue_required" in task.tags
    )
    training_spec = verifyit_spec(training_tool.grader)
    assert isinstance(training_spec, StructureClosureSpec)
    assert 'train_parent"' in training_spec.prefix
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


def test_curation_reader_generates_disjoint_task_cohorts(tmp_path):
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
    pinned = HfSource("fixture/bfcl", "a" * 40, ("bfcl_complement-manifest.json",), SourceFormat.JSON)
    declared = sources(
        pinned,
        HfSource("fixture/bfcl", "a" * 40, ("bfcl_complement/*/solution/solve.sh",), SourceFormat.GENERATED),
        HfSource("fixture/split", "b" * 40, ("split.json",), SourceFormat.JSON),
        split_filename="split.json",
        training_prefix=TrainingPrefix.NEAR_BOUNDARY,
        tool_prompt=ToolPrompt.COMPACT,
    )
    context = ConversionContext(
        inputs={"oracles": StoragePath(str(source)), "split": StoragePath(str(tmp_path))},
        grader_environment=None,
    )
    cohorts = {}
    for declared_source in declared:
        recipe = declared_source.config
        assert recipe is not None
        assert declared_source.pipeline is process_rows
        assert recipe.source.read is not None
        cases = list(recipe.source.read(StoragePath(str(source / pinned.files[0])), context))
        cohort = [recipe.convert(RawRow(case["case_id"], SOURCE, case), context) for case in cases]
        assert all(isinstance(task, TaskSpec) for task in cohort)
        cohorts[declared_source.name] = cohort
    training = cohorts["bfcl-structure-closure-train"]
    validation = cohorts["bfcl-structure-closure-validation"]
    assert training and validation
    assert all("train_parent" in verifyit_spec(task.grader).expected for task in training)
    assert all("validation_parent" in verifyit_spec(task.grader).expected for task in validation)
    assert {task.id for task in training}.isdisjoint(task.id for task in validation)
    tool = next(
        task for task in training if "family:tool_delimiter" in task.tags and "control:continue_required" in task.tags
    )
    assert 'train_parent"' in verifyit_spec(tool.grader).prefix

    runtime = CampaignRuntime()
    options = PipelineOptions(
        SourceProcessingMode.QUICK,
        runtime,
        inputs=InputOverrides(str(source), auxiliary={"oracles": str(source), "split": str(tmp_path)}),
    )
    steps = {item.name: item.pipeline(item, options) for item in declared}
    output = tmp_path / "curated"
    run_local_steps(
        steps,
        output,
        runtime=runtime,
        max_workers=1,
        worker_resources=ResourceConfig(cpu=1, ram="4g"),
        cache_dir=tmp_path / "cache",
        source_ids={item.name: item.info.id for item in declared},
    )
    for name, expected in cohorts.items():
        records = [row for shard in (output / name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))]
        curated = [TaskSpec.model_validate_json(row["task_json"]) for row in records]
        assert len(curated) == len(expected)
        assert {verifyit_spec(task.grader).expected for task in curated} == {
            verifyit_spec(task.grader).expected for task in expected
        }


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
