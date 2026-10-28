# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from zephyr.context import ZephyrContext
from zephyr.stage_io import ZephyrWorkerError

from taskcompendium.convert.answers import exact_answer_task, source_defect
from taskcompendium.evaluation import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    Sampling,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.pipeline.evaluation import (
    MANIFEST_FILENAME,
    chat_item,
    evaluation_cases,
    evaluation_items,
    publish_evaluation_corpus,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, NormalizationChange, NormalizedTask, RawRow, SourceRecipe


def convert_evaluation(row: RawRow, _context: ConversionContext):
    if row.data.get("reject"):
        return source_defect("missing_reference", "The source row has no reference")
    task = exact_answer_task(row, prompt=row.data["prompt"], answers=("red",), ignore_case=False)
    task = task.model_copy(update={"tags": ("eval", "source:fixture")})
    if row.data.get("rewrite"):
        return NormalizedTask(
            task,
            (
                NormalizationChange(
                    field="context",
                    reason="Changed source wording",
                    original="old",
                    replacement="new",
                ),
            ),
        )
    return task


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [{"prompt": "Name the color"}, {"prompt": "Name the color"}, {"prompt": "Choose a color"}]
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    recipe = SourceRecipe(
        name="colors",
        version="1",
        source=SourceFiles("colors", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert_evaluation,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    protocol = EvaluationProtocol(
        benchmark="colors",
        version="1",
        operation="chat_generation",
        sampling=Sampling(temperature=0, top_p=1, top_k=None, max_tokens=20, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        reducer="mean_v1",
        expected_population=3,
    )
    return source, tmp_path / "output", recipe, protocol


def publish(corpus, tmp_path, item_builder=chat_item):
    source, output, recipe, protocol = corpus
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            return publish_evaluation_corpus(recipe, protocol, item_builder, context, str(source), str(output))
    finally:
        client.shutdown()


def test_evaluation_publication_retains_duplicate_source_positions_and_detects_changed_files(corpus, tmp_path):
    manifest = publish(corpus, tmp_path)
    output = corpus[1]
    cases = list(evaluation_cases(str(output)))
    assert manifest.rows == 3
    assert [case.item.source_row for case in cases] == ["rows.jsonl:0", "rows.jsonl:1", "rows.jsonl:2"]
    assert cases[0].task.context == cases[1].task.context
    assert cases[0].task.id != cases[1].task.id
    assert all("eval" in case.task.tags for case in cases)
    (output / manifest.shards[0].items.path).write_bytes(b"changed after publication")
    with pytest.raises(ValueError, match="changed after publication"):
        list(evaluation_cases(str(output)))


@pytest.mark.parametrize("change", ["rewrite", "reject", "missing"])
def test_evaluation_publication_withholds_release_for_changed_population_or_prompts(corpus, tmp_path, change):
    source, output, _, _ = corpus
    rows = [{"prompt": "Name the color", change: True}, {"prompt": "Name the color"}, {"prompt": "Choose a color"}]
    if change == "missing":
        rows.pop()
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, ZephyrWorkerError)):
        publish(corpus, tmp_path)
    assert not (output / MANIFEST_FILENAME).exists()


def test_evaluation_publication_requires_context_metadata_for_a_selection_policy(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    selected = protocol.model_copy(update={"context_policy": ContextPolicy(kind="mrcr", max_context=8192)})
    with pytest.raises((ValueError, ZephyrWorkerError), match="context metadata"):
        publish((source, output, recipe, selected), tmp_path)
    assert not (output / MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("corruption", ["missing_shards", "changed_bounds", "changed_provenance"])
def test_evaluation_reader_rejects_inconsistent_committed_manifest(corpus, tmp_path, corruption):
    publish(corpus, tmp_path)
    output = corpus[1]
    marker = output / MANIFEST_FILENAME
    manifest = json.loads(marker.read_text())
    if corruption == "missing_shards":
        manifest["shards"] = []
    elif corruption == "changed_bounds":
        manifest["shards"][0]["first_source_row"] = "rows.jsonl:1"
    else:
        manifest["source_revision"] = "unrelated-revision"
    marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        list(evaluation_items(str(output)))


def test_evaluation_metadata_scan_and_selected_rows_avoid_unselected_task_decoding(corpus, tmp_path):
    manifest = publish(corpus, tmp_path)
    output = corpus[1]
    shard = manifest.shards[0]
    file = output / shard.tasks.path
    table = pq.read_table(file)
    rows = table.to_pylist()
    rows[0]["task_json"] = "Task JSON must not be decoded during a sidecar scan or an unrelated row selection"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    marker = output / MANIFEST_FILENAME
    committed = json.loads(marker.read_text())
    committed["shards"][0]["tasks"]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    marker.write_text(json.dumps(committed))
    assert [item.source_row for item in evaluation_items(str(output))] == [
        "rows.jsonl:0",
        "rows.jsonl:1",
        "rows.jsonl:2",
    ]
    selected = list(evaluation_cases(str(output), source_rows={"rows.jsonl:2"}))
    assert [case.task.source.row for case in selected] == ["rows.jsonl:2"]
    with pytest.raises(ValueError):
        list(evaluation_cases(str(output)))


def raw_item(task):
    return chat_item(task).model_copy(update={"operation": TextGeneration(context=task.context.events[0].content)})


def mismatched_raw_item(task):
    return raw_item(task).model_copy(update={"operation": TextGeneration(context="Different prompt")})


def test_raw_publication_withholds_release_when_sidecar_prompt_differs_from_canonical_task(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    raw_protocol = protocol.model_copy(update={"operation": "text_generation"})
    with pytest.raises((ValueError, ZephyrWorkerError), match="Raw evaluation prompt differs"):
        publish((source, output, recipe, raw_protocol), tmp_path, mismatched_raw_item)
    assert not (output / MANIFEST_FILENAME).exists()


def test_raw_reader_rejects_committed_prompt_mismatch_for_selected_task(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    raw_protocol = protocol.model_copy(update={"operation": "text_generation"})
    manifest = publish((source, output, recipe, raw_protocol), tmp_path, raw_item)
    assert len(list(evaluation_cases(str(output)))) == 3
    shard = manifest.shards[0]
    file = output / shard.items.path
    table = pq.read_table(file)
    rows = table.to_pylist()
    item = json.loads(rows[0]["evaluation_json"])
    item["operation"]["context"] = "Different prompt"
    rows[0]["evaluation_json"] = json.dumps(item)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    marker = output / MANIFEST_FILENAME
    committed = json.loads(marker.read_text())
    committed["shards"][0]["items"]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    marker.write_text(json.dumps(committed))
    assert len(list(evaluation_cases(str(output), source_rows={"rows.jsonl:2"}))) == 1
    with pytest.raises(ValueError, match="Raw evaluation prompt differs"):
        list(evaluation_cases(str(output), source_rows={"rows.jsonl:0"}))
