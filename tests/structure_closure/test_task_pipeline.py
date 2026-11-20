# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, AssistantPrefixHarness
from rolloutengine.test_support import ReplayModel, rollout_spec
from taskcompendium.models import TaskSpec, verifyit_spec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.closure_scoring import closure_state
from verifyit.spec import StructureClosureSpec
from verifyit.tool_call_markup import THINK_CLOSE, TOOL_CLOSE, TOOL_OPEN
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import Split
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.inputs import (
    COMPLEMENT_SHA256,
    INVENTORY_FILE,
    SPLIT_FILE,
    TOKENIZER_FILES,
)
from experiments.post_training.task_curation.datasets.bfcl_structure_closure.source import CaseReader, sources
from experiments.post_training.task_curation.local import run_local_steps

QUOTED_MARKERS = '<tool_call>{"name":"nested","arguments":{}}</tool_call>'
ORACLE_ANSWER = [{"search": {"query": QUOTED_MARKERS}}]


@pytest.fixture
def bfcl_manifests(tmp_path):
    inventory = tmp_path / INVENTORY_FILE
    inventory.parent.mkdir(parents=True)
    oracle_dir = inventory.parent / "oracles"
    oracle_dir.mkdir()
    split_digest = "a" * 64
    oracle = ("cat << 'ORACLE_EOF'\n" + json.dumps(ORACLE_ANSWER) + "\nORACLE_EOF\n").encode()
    parents = []
    for cohort in Split:
        parent = f"{cohort}_parent"
        source_name = f"bfcl-{parent}"
        (oracle_dir / f"{source_name}.sh").write_bytes(oracle)
        parents.append(
            {
                "task": parent,
                "source_name": source_name,
                "cohort": cohort,
                "solution_sha256": hashlib.sha256(oracle).hexdigest(),
            }
        )
    inventory.write_text(
        json.dumps(
            {
                "source_partition_sha256": COMPLEMENT_SHA256,
                "split_manifest_sha256": split_digest,
                "records": parents,
                "failures": [],
            }
        )
    )
    split = tmp_path / SPLIT_FILE
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "source_split_sha256": split_digest,
                "task_ids": {
                    "train": ["train_parent"],
                    "validation": ["validation_parent"],
                    "parity": ["parity_parent"],
                },
            }
        )
    )
    return tmp_path, inventory, split


@pytest.mark.asyncio
async def test_process_rows_curates_and_grades_native_tool_and_reasoning_tasks(tmp_path, tokenizer, bfcl_manifests):
    input_root, _, _ = bfcl_manifests
    tokenizer.save_pretrained(input_root / TOKENIZER_FILES[0].rsplit("/", 1)[0])
    declared = sources()
    runtime = CampaignRuntime()
    download_options = PipelineOptions(SourceProcessingMode.QUICK, runtime)
    train_graph, validation_graph = (source.pipeline(source, download_options) for source in declared)
    assert [(dep.name, dep.fingerprint()) for dep in train_graph.deps] == [
        (dep.name, dep.fingerprint()) for dep in validation_graph.deps
    ]

    options = PipelineOptions(
        SourceProcessingMode.QUICK,
        runtime,
        inputs=InputOverrides(
            str(input_root),
            auxiliary={"split": str(input_root), "tokenizer": str(input_root)},
        ),
    )
    output = tmp_path / "curated"
    run_local_steps(
        {source.name: source.pipeline(source, options) for source in declared},
        output,
        runtime=runtime,
        max_workers=1,
        worker_resources=ResourceConfig(cpu=1, ram="4g"),
        cache_dir=tmp_path / "cache",
        source_ids={source.name: source.info.id for source in declared},
    )

    cohorts = {}
    engine = RolloutEngine({}, {ASSISTANT_PREFIX_HARNESS: AssistantPrefixHarness()})
    for cohort in Split:
        shards = (output / f"bfcl-structure-closure-{cohort}" / "normalize").glob("*.parquet")
        tasks = [TaskSpec.model_validate_json(row["task_json"]) for shard in shards for row in load_parquet(str(shard))]
        cohorts[cohort] = {task.id for task in tasks}
        assert {tag for task in tasks for tag in task.tags if tag.startswith("family:")} == {
            "family:json_object",
            "family:json_array",
            "family:json_nested",
            "family:json_string_escape",
            "family:tool_delimiter",
            "family:reasoning_delimiter",
            "family:code_fence",
            "family:repeated_opener_retention",
        }
        tool_task = next(
            task for task in tasks if "family:tool_delimiter" in task.tags and "control:continue_required" in task.tags
        )
        tool_spec = verifyit_spec(tool_task.grader)
        assert isinstance(tool_spec, StructureClosureSpec)
        assert tool_spec.prefix.startswith(TOOL_OPEN + "\n")
        assert json.loads(tool_spec.expected.removeprefix(TOOL_OPEN).removesuffix(TOOL_CLOSE)) == {
            "name": "write_result",
            "arguments": {"task": f"{cohort}_parent", "result": ORACLE_ANSWER},
        }
        if cohort == Split.TRAIN:
            early_close = closure_state(tool_spec.prefix, tool_spec.format).closing_suffix
            assert early_close
            model = ReplayModel([{"role": "assistant", "content": early_close}])
            rollout = rollout_spec(tool_task, harness=ASSISTANT_PREFIX_HARNESS, max_turns=1)
            failed = await engine.run_task(rollout, model.complete)
            assert failed.grade.reward == 0
        else:
            assert tool_spec.prefix.endswith('"validati')

        model = ReplayModel([{"role": "assistant", "content": tool_spec.expected[len(tool_spec.prefix) :]}])
        completed = await engine.run_task(
            rollout_spec(tool_task, harness=ASSISTANT_PREFIX_HARNESS, max_turns=1), model.complete
        )
        assert completed.grade.reward == 1
        assert model.requests[0].messages[-1] == {"role": "assistant", "content": tool_spec.prefix}
        assert model.requests[0].options["tools"][0]["function"]["name"] == "write_result"

        reasoning_task = next(
            task for task in tasks if "family:reasoning_delimiter" in task.tags and "control:close_required" in task.tags
        )
        model = ReplayModel([{"role": "assistant", "content": THINK_CLOSE}])
        result = await engine.run_task(
            rollout_spec(reasoning_task, harness=ASSISTANT_PREFIX_HARNESS, max_turns=1), model.complete
        )
        assert result.grade.reward == 1
        assert result.rollout.conversations[0].steps[0].turn.message["content"] == THINK_CLOSE

    assert cohorts[Split.TRAIN].isdisjoint(cohorts[Split.VALIDATION])


@pytest.mark.parametrize(
    "changed, error",
    [
        ({"train": ["train_parent", "validation_parent"]}, "disjoint"),
        ({"validation": ["outside_complement"]}, "audited complement"),
        ({"parity": ["train_parent"]}, "disjoint"),
        ({"train": "train_parent"}, "valid array"),
    ],
)
def test_reader_rejects_invalid_split_before_opening_oracles(bfcl_manifests, changed, error):
    root, inventory, split = bfcl_manifests
    record = json.loads(split.read_text())
    record["task_ids"].update(changed)
    split.write_text(json.dumps(record))
    context = ConversionContext(inputs={"split": StoragePath(str(root))}, grader_environment=None)
    with pytest.raises(ValueError, match=error):
        list(CaseReader(Split.TRAIN)(StoragePath(str(inventory)), context))


def test_reader_rejects_split_from_a_different_manifest(bfcl_manifests):
    root, inventory, split = bfcl_manifests
    split.write_text(json.dumps(json.loads(split.read_text()) | {"source_split_sha256": "b" * 64}))
    context = ConversionContext(inputs={"split": StoragePath(str(root))}, grader_environment=None)
    with pytest.raises(ValueError, match="different split manifests"):
        list(CaseReader(Split.TRAIN)(StoragePath(str(inventory)), context))
