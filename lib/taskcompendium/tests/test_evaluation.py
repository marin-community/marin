# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from zephyr.context import ZephyrContext
from zephyr.stage_io import ZephyrWorkerError

from taskcompendium.convert.answers import exact_answer_task, source_defect
from taskcompendium.evaluation.corpus import MANIFEST_FILENAME, chat_item, evaluation_cases, evaluation_items, task_rows
from taskcompendium.evaluation.selection import select_cases
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    ChatPromptVariant,
    CohortOrder,
    CohortPolicy,
    ContextPolicy,
    EvaluationProtocol,
    LimitStage,
    Metric,
    MetricKind,
    OperationKind,
    OutputBudget,
    Sampling,
    SummaryKind,
    TaskGroup,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, NormalizationChange, NormalizedTask, RawRow, SourceRecipe


def convert_evaluation(row: RawRow, _context: ConversionContext):
    if row.data.get("reject"):
        return source_defect("missing_reference", "The source row has no reference")
    task = exact_answer_task(row, prompt=row.data["prompt"], answers=("red",), ignore_case=False)
    task = task.model_copy(
        update={"tags": ("eval", "source:fixture"), "labels": {"color": row.data.get("color", "red")}}
    )
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
        summary=SummaryKind.MEAN,
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
    original = (source / "rows.jsonl").read_bytes()
    rows = [{"prompt": "Name the color", change: True}, {"prompt": "Name the color"}, {"prompt": "Choose a color"}]
    if change == "missing":
        rows.pop()
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, ZephyrWorkerError)):
        publish(corpus, tmp_path)
    assert not (output / MANIFEST_FILENAME).exists()
    (source / "rows.jsonl").write_bytes(original)
    assert publish(corpus, tmp_path).rows == 3
    assert len(list(evaluation_cases(str(output)))) == 3


def test_evaluation_publication_requires_reference_text_for_reference_budget(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    selected = protocol.model_copy(
        update={"context_policy": ContextPolicy(output_budget=OutputBudget(minimum=4, reserve=2, multiplier=2))}
    )
    with pytest.raises((ValueError, ZephyrWorkerError), match="requires reference text"):
        publish((source, output, recipe, selected), tmp_path)
    assert not (output / MANIFEST_FILENAME).exists()
    assert publish(corpus, tmp_path).rows == 3
    assert len(list(evaluation_cases(str(output)))) == 3


def test_evaluation_publication_retries_preempted_outputs_without_mixing_stale_shards(corpus, tmp_path):
    output = corpus[1]
    for directory in ("normalize", "evaluation"):
        path = output / directory
        path.mkdir(parents=True)
        (path / "stale.parquet").write_bytes(b"preempted partial shard")
    (output / "manifest.json").write_text("incomplete conversion manifest")
    (output / "protocol.json").write_text("incomplete protocol")
    (output / "worker.log").write_text("executor evidence")
    manifest = publish(corpus, tmp_path)
    assert manifest.rows == 3
    assert len(list(evaluation_cases(str(output)))) == 3
    assert (output / "worker.log").read_text() == "executor evidence"
    assert not (output / "normalize" / "stale.parquet").exists()
    assert not (output / "evaluation" / "stale.parquet").exists()
    with pytest.raises(FileExistsError):
        publish(corpus, tmp_path)


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
    return chat_item(task).model_copy(update={"operation": TextGeneration()})


def convert_multimessage(row, context):
    task = convert_evaluation(row, context)
    return task.model_copy(
        update={
            "context": task.context.model_copy(
                update={"events": (*task.context.events, TextMessage(role="user", content="More"))}
            )
        }
    )


def test_raw_publication_withholds_release_for_multimessage_task(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    raw_protocol = protocol.model_copy(update={"operation": OperationKind.TEXT_GENERATION})
    with pytest.raises((ValueError, ZephyrWorkerError), match="requires one canonical user text message"):
        publish((source, output, replace(recipe, convert=convert_multimessage), raw_protocol), tmp_path, raw_item)
    assert not (output / MANIFEST_FILENAME).exists()
    assert publish((source, output, recipe, raw_protocol), tmp_path, raw_item).rows == 3
    assert len(list(evaluation_cases(str(output)))) == 3


def test_raw_reader_rejects_committed_multimessage_task(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    raw_protocol = protocol.model_copy(update={"operation": OperationKind.TEXT_GENERATION})
    manifest = publish((source, output, recipe, raw_protocol), tmp_path, raw_item)
    assert len(list(evaluation_cases(str(output)))) == 3
    shard = manifest.shards[0]
    file = output / shard.tasks.path
    table = pq.read_table(file)
    rows = table.to_pylist()
    task = json.loads(rows[0]["task_json"])
    task["context"]["events"].append(TextMessage(role="user", content="More").model_dump(mode="json"))
    rows[0]["task_json"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    marker = output / MANIFEST_FILENAME
    committed = json.loads(marker.read_text())
    committed["shards"][0]["tasks"]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    marker.write_text(json.dumps(committed))
    assert len(list(evaluation_cases(str(output), source_rows={"rows.jsonl:2"}))) == 1
    with pytest.raises(ValueError, match="requires one canonical user text message"):
        list(evaluation_cases(str(output), source_rows={"rows.jsonl:0"}))


def test_task_annotation_query_uses_projected_metadata_without_payload_decoding(corpus, tmp_path):
    manifest = publish(corpus, tmp_path)
    output = corpus[1]
    marker = output / MANIFEST_FILENAME
    committed = json.loads(marker.read_text())
    for kind, column in (("tasks", "task_json"), ("items", "evaluation_json")):
        file = output / getattr(manifest.shards[0], kind).path
        table = pq.read_table(file)
        rows = table.to_pylist()
        rows[0][column] = "must not decode payloads while querying annotations"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
        committed["shards"][0][kind]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    marker.write_text(json.dumps(committed))
    assert [row.source_row for row in task_rows(str(output), tags=("eval",), labels={"color": "red"})] == [
        "rows.jsonl:0",
        "rows.jsonl:1",
        "rows.jsonl:2",
    ]
    assert list(task_rows(str(output), labels={"color": "blue"})) == []
    assert [case.task.source.row for case in evaluation_cases(str(output), source_rows={"rows.jsonl:2"})] == [
        "rows.jsonl:2"
    ]


def test_selected_task_rejects_projection_disagreement(corpus, tmp_path):
    manifest = publish(corpus, tmp_path)
    output = corpus[1]
    file = output / manifest.shards[0].tasks.path
    table = pq.read_table(file)
    rows = table.to_pylist()
    rows[0]["labels"] = {"color": "blue"}
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), file)
    marker = output / MANIFEST_FILENAME
    committed = json.loads(marker.read_text())
    committed["shards"][0]["tasks"]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    marker.write_text(json.dumps(committed))
    with pytest.raises(ValueError, match="annotations differ"):
        list(evaluation_cases(str(output), source_rows={"rows.jsonl:0"}))


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(range(len(text)))

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        return self.encode("".join(message["content"] for message in conversation))


def publish_cohort(corpus, tmp_path, cohort):
    source, output, recipe, protocol = corpus
    rows = [
        {"prompt": "large" * 30, "color": "red"},
        {"prompt": "first red", "color": "red"},
        {"prompt": "second red", "color": "red"},
        {"prompt": "first blue", "color": "blue"},
        {"prompt": "second blue", "color": "blue"},
    ]
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    protocol = protocol.model_copy(update={"expected_population": len(rows), "cohort": cohort})
    publish((source, output, recipe, protocol), tmp_path)
    return str(output), protocol


def test_grouped_selection_refills_in_declared_round_robin_order_and_records_every_source_row(corpus, tmp_path):
    root, protocol = publish_cohort(
        corpus,
        tmp_path,
        CohortPolicy(
            order=CohortOrder.ROUND_ROBIN,
            limit_stage=LimitStage.ELIGIBLE,
            groups=(
                TaskGroup(tags=("eval",), labels={"color": "red"}, minimum_context=50),
                TaskGroup(labels={"color": "blue"}, minimum_context=50),
                TaskGroup(labels={"color": "green"}, minimum_context=1000),
            ),
        ),
    )
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="colors",
            context_window=100,
            max_tokens=20,
            limit=2,
            tokenizer=CharacterTokenizer(),
        )
    )
    assert [
        (decision.record.source_row, decision.record.rank) for decision in decisions if decision.case is not None
    ] == [("rows.jsonl:1", 2), ("rows.jsonl:3", 1)]
    assert [(decision.record.status, decision.record.reason) for decision in decisions] == [
        ("excluded", "context_window"),
        ("selected", None),
        ("uninspected", "source_limit"),
        ("selected", None),
        ("excluded", "source_limit"),
    ]
    assert [decision.case.task.context.events[0].content for decision in decisions if decision.case is not None] == [
        "first red",
        "first blue",
    ]


@pytest.mark.parametrize("limit_stage,selected", [(LimitStage.CANDIDATES, [1]), (LimitStage.ELIGIBLE, [1, 2])])
def test_source_selection_applies_limit_at_declared_stage(corpus, tmp_path, limit_stage, selected):
    root, protocol = publish_cohort(corpus, tmp_path, CohortPolicy(limit_stage=limit_stage))
    decisions = list(
        select_cases(
            root,
            protocol,
            dataset_id="colors",
            context_window=100,
            max_tokens=20,
            limit=2,
            tokenizer=CharacterTokenizer(),
        )
    )
    assert [decision.record.source_ordinal for decision in decisions if decision.case is not None] == selected
    assert len(decisions) == 5


def test_grouped_selection_rejects_missing_active_stratum(corpus, tmp_path):
    root, protocol = publish_cohort(
        corpus, tmp_path, CohortPolicy(groups=(TaskGroup(labels={"color": "green"}, minimum_context=50),))
    )
    with pytest.raises(ValueError, match="no examples for requested groups"):
        list(
            select_cases(
                root,
                protocol,
                dataset_id="colors",
                context_window=100,
                max_tokens=20,
                limit=2,
                tokenizer=CharacterTokenizer(),
            )
        )


def variant_item(task):
    return chat_item(task).model_copy(
        update={
            "operation": ChatGeneration(
                prompt_selection="fit_context",
                variants=(
                    ChatPromptVariant(
                        context=task.context.model_copy(
                            update={"events": (TextMessage(role="user", content="z" * 20),)}
                        ),
                        fewshot=0,
                    ),
                    ChatPromptVariant(context=task.context, fewshot=1),
                ),
            )
        }
    )


def test_selected_payload_reread_retains_prompt_variant_chosen_before_output_clamping(corpus, tmp_path):
    source, output, recipe, protocol = corpus
    (source / "rows.jsonl").write_text(json.dumps({"prompt": "f" * 30}) + "\n")
    protocol = protocol.model_copy(
        update={"expected_population": 1, "sampling": protocol.sampling.model_copy(update={"max_tokens": 80})}
    )
    publish((source, output, recipe, protocol), tmp_path, variant_item)
    (decision,) = select_cases(
        str(output),
        protocol,
        dataset_id="variants",
        context_window=100,
        max_tokens=80,
        limit=1,
        tokenizer=CharacterTokenizer(),
    )
    assert decision.case is not None
    assert decision.case.task.context.events[0].content == "z" * 20
    assert decision.record.max_tokens == 16
