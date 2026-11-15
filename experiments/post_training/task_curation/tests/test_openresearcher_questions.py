# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
from rolloutengine.harness.browsecomp import OpenResearcherProvider
from rolloutengine.harness.shell import ShellHarness, task_prompt
from rolloutengine.test_support import rollout_limits
from taskcompendium.harbor.adapter import TASK_SPEC_FILE, write_harbor_task
from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import resolve_shell_tool

from experiments.post_training.task_curation.datasets.openresearcher import questions
from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS_REVISION,
    DATASET,
    DATASET_REVISION,
    RegionalInputs,
)
from experiments.post_training.task_curation.datasets.openresearcher.tasks import convert_question


def test_seed_repetitions_collapse_but_conflicting_gold_answers_remain_visible(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"qid": 1, "question": "Synthetic question?", "answer": "one"},
                {"qid": 1, "question": "Synthetic question?", "answer": "one"},
                {"qid": 1, "question": "Synthetic question?", "answer": "two"},
            ]
        ),
        raw / "seed-fixture.parquet",
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    manifest = json.dumps(
        [{"dataset_id": DATASET, "revision": DATASET_REVISION, "files": [{"path": "seed-fixture.parquet"}]}]
    )
    (stage / "source-manifest.json").write_text(manifest)
    receipt = RegionalInputs(
        path=str(stage),
        source_roots={DATASET: str(raw)},
        source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        license_receipts_sha256="0" * 64,
        verified_files=1,
        verified_bytes=0,
        iris_task_id="synthetic-fixture",
    ).model_dump_json()
    receipt_file = tmp_path / "receipt.json"
    receipt_file.write_text(receipt)
    monkeypatch.setenv("IRIS_TASK_ID", "synthetic-fixture")
    monkeypatch.setattr(questions, "EXPECTED_TRAJECTORIES", 3)
    output = tmp_path / "output"
    result = questions.extract_questions(
        questions.QuestionConfig(str(receipt_file), hashlib.sha256(receipt.encode()).hexdigest(), str(output))
    )
    rows = pq.read_table(output / "questions.parquet").to_pylist()
    assert (
        result.raw_trajectories,
        result.unique_contracts,
        result.unique_questions,
        result.conflicting_answer_questions,
    ) == (3, 2, 1, 1)
    assert sorted(len(json.loads(row["source_refs"])) for row in rows) == [1, 2]
    assert {row["answer"] for row in rows} == {"one", "two"}


def test_research_native_roundtrip_preserves_both_tools_and_keeps_gold_out_of_actor_contract(tmp_path):
    state = OpenResearcherState(
        implementation_revision="a" * 40,
        corpus_revision=CORPUS_REVISION,
        index_manifest_sha256="b" * 64,
        tokenizer_revision="c" * 40,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    images = {
        "actor_image": "ghcr.io/marin-community/rl-data-openresearcher@sha256:" + "d" * 64,
        "grader_image": "ghcr.io/marin-community/rl-data-openresearcher@sha256:" + "e" * 64,
    }
    row = {
        "question": "Who founded Cedar Press in 1974?",
        "answer": "PRIVATE_REFERENCE_ALPHA",
        "source_refs": json.dumps([{"file": "seed-fixture.parquet", "row": 7, "qid": "fixture"}]),
    }
    task = convert_question(row, state=state, **images)
    changed_gold = convert_question({**row, "answer": "PRIVATE_REFERENCE_BETA"}, state=state, **images)
    assert (task.id, task.source, task.context) == (changed_gold.id, changed_gold.source, changed_gold.context)
    directory = tmp_path / "native"
    write_harbor_task(
        task,
        directory,
        ExecutionRequirements(cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60),
    )
    restored = TaskSpec.model_validate_json((directory / TASK_SPEC_FILE).read_text())
    harness = ShellHarness(
        command_timeout=None, tools={OPENRESEARCHER_ACTION_INTERFACE: OpenResearcherProvider(command_timeout=None)}
    )
    harness.check(restored, rollout_limits())
    prompt = task_prompt(restored, resolve_shell_tool(restored, harness.shell_tool), ())
    assert {tool["function"]["name"] for tool in prompt.options["tools"]} == {"search", "get_document"}
    assert row["question"] in prompt.messages[0]["content"]
    assert "PRIVATE_REFERENCE" not in json.dumps(prompt.messages)
    private = next(resource for resource in restored.resources.verifier if resource.path == "config.json")
    assert json.loads(resource_bytes(private)) == {name: row[name] for name in ("question", "answer")}
