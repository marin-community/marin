# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.reader import ReadView
from finestore.rl.curriculum import EmbeddingSpec
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet
from zephyr.stage_io import ZephyrWorkerError

from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline import task_embeddings
from taskcompendium.pipeline.alignment_publication import publish_alignments
from taskcompendium.pipeline.curriculum_alignment import (
    AnchorKind,
    CurriculumAnchor,
    CurriculumFacet,
    CurriculumVectors,
    rank_task_batch,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.review_requests import RequestObservation, RequestOutput
from taskcompendium.pipeline.semantics import (
    DescriptionModel,
    TaskInputFile,
    bounded_task_text,
    describe_tasks,
    prepare_semantic_inputs,
    task_semantic_input,
)
from taskcompendium.pipeline.task_alignment import align_tasks
from taskcompendium.pipeline.task_embeddings import (
    embed_texts,
    embedding_cache_partition,
    embedding_cache_path,
    task_text_records,
    text_record,
)


@pytest.fixture(autouse=True)
def cache_partitions(monkeypatch):
    monkeypatch.setattr(task_embeddings, "EMBEDDING_CACHE_SHARDS", 2)


@pytest.fixture
def context(tmp_path):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        yield context


@pytest.fixture
def spec():
    return EmbeddingSpec(
        model="fixture-harrier",
        revision="revision-1",
        tokenizer_revision="revision-1",
        dimensions=3,
        max_tokens=512,
        compute_dtype="float32",
    )


def task(identifier, prompt="Count two apples.", answer="817", dataset="private-source-label"):
    result = numeric_answer_task(
        RawRow(
            id=identifier,
            source=Source(dataset=dataset, revision="private-source-revision", importer_revision="1", row=identifier),
            data={},
        ),
        prompt=prompt,
        answer=answer,
        tolerance_abs=0,
        tolerance_rel=0,
        evidence={"private": "PRIVATE_EVIDENCE"},
    )
    assert isinstance(result, TaskSpec)
    return result


@dataclass(frozen=True)
class DescriptionTransport:
    calls: str
    fail_prompts: tuple[str, ...] = ()

    def __call__(self, requests):
        observations = []
        for request in reversed(requests):
            with Path(self.calls).open("a") as stream:
                stream.write(json.dumps(request) + "\n")
            public = request["body"]["messages"][1]["content"]
            if any(prompt in public for prompt in self.fail_prompts):
                observations.append(
                    RequestObservation(
                        (request["custom_id"],),
                        json.dumps({"custom_id": request["custom_id"], "error": {"code": "unavailable"}}),
                    )
                )
                continue
            missing = "UNSPECIFIED_OPERATION" in public
            description = {
                "status": "insufficient_context" if missing else "ready",
                "reason": "The requested operation is absent" if missing else "",
                "description": "" if missing else "Elementary arithmetic: count apples.",
            }
            if not missing and len(request["body"]["messages"]) == 2:
                description.pop("status")
            response = {
                "custom_id": request["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "description",
                                            "type": "function",
                                            "function": {
                                                "name": "describe_task",
                                                "arguments": json.dumps(description),
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    },
                },
            }
            observations.append(RequestObservation((request["custom_id"],), json.dumps(response)))
        return RequestOutput(tuple(requests), observations)


@dataclass(frozen=True)
class EmbeddingProvider:
    calls: str
    fail_after: int | None = None

    def embed(self, texts):
        calls = Path(self.calls)
        if self.fail_after is not None and calls.exists() and len(calls.read_text().splitlines()) >= self.fail_after:
            raise RuntimeError("Embedding provider unavailable")
        with calls.open("a") as stream:
            for text in texts:
                stream.write(json.dumps(text) + "\n")
        return np.asarray([[1.0, float(len(text) % 7), 0.5] for text in texts], dtype=np.float32)


def rows(paths):
    return [row for path in paths for row in load_parquet(path)]


def test_embedding_retry_reuses_completed_cache_batches(tmp_path, spec):
    records = []
    index = 0
    while len(records) < 520:
        row = text_record(f"Count case {index}.", spec)
        if embedding_cache_partition(row["embedding_key"]) == 0:
            records.append(row)
        index += 1
    cache = str(tmp_path / "vectors-cache")
    provider = EmbeddingProvider(str(tmp_path / "failed-calls.jsonl"), fail_after=256)
    arguments = dict(texts=Dataset.from_list(records), spec=spec, cache_root=cache)
    with ZephyrContext(max_workers=1, max_shard_failures=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with pytest.raises(ZephyrWorkerError):
            embed_texts(context, **arguments, embedder=provider, output_root=str(tmp_path / "failed"))
        successful = [json.loads(line) for line in Path(provider.calls).read_text().splitlines()]
        assert len(successful) == 256
    recovered = EmbeddingProvider(str(tmp_path / "recovered-calls.jsonl"))
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "recovery-chunks")) as context:
        result = embed_texts(context, **arguments, embedder=recovered, output_root=str(tmp_path / "recovered"))
    assert len(rows(result)) == 520
    retried = [json.loads(line) for line in Path(recovered.calls).read_text().splitlines()]
    assert len(retried) == 264
    assert not set(successful).intersection(retried)
    view = ReadView(embedding_cache_path(cache, embedding_cache_partition(records[0]["embedding_key"])))
    assert len(view.read_blobs([row["embedding_key"] for row in records])) == 520
    assert view.read_diagnostics().selected_shards == 1


def test_oversized_task_excerpts_preserve_operations_and_valid_utf8():
    text = "BEGIN_OPERATION\n" + "é" * 50000 + "\nMIDDLE_OPERATION\n" + "🙂" * 25000 + "\nEND_OPERATION"
    excerpt = bounded_task_text(text, 2048)
    assert len(excerpt.encode("utf-8")) <= 2048
    assert all(operation in excerpt for operation in ("BEGIN_OPERATION", "MIDDLE_OPERATION", "END_OPERATION"))
    assert "omitted" in excerpt and "�" not in excerpt


@pytest.mark.parametrize("failures", [1, 2])
def test_description_failures_preserve_tasks_and_enforce_global_skip_limit(context, tmp_path, failures):
    source = TaskInputFile("fixture", "revision", "unused", "train", "converted")
    records = [
        task_semantic_input(task(str(i), prompt=f"Count apples, case {i:02d}."), source).model_dump(mode="json")
        for i in range(25)
    ]
    inputs = tmp_path / "inputs.parquet"
    pq.write_table(pa.Table.from_pylist(records), inputs)
    output = tmp_path / "descriptions"
    transport = DescriptionTransport(str(tmp_path / "requests.jsonl"), tuple(f"case {i:02d}." for i in range(failures)))
    arguments = dict(
        input_paths=[str(inputs)],
        model=DescriptionModel("fixture", "1"),
        submit=transport,
        cache_root=str(tmp_path / "cache"),
        output_root=str(output),
    )
    if failures == 2:
        with pytest.raises(RuntimeError, match="Skipped 2/25"):
            describe_tasks(context, **arguments)
    else:
        paths = describe_tasks(context, **arguments)
        descriptions = rows(paths)
        assert len(descriptions) == 25
        skipped = [row for row in descriptions if row["status"] == "skipped"]
        assert len(skipped) == 1 and skipped[0]["reason"]
    counters = json.loads((output / "manifest.json").read_text())["counters"]
    assert counters["semantics/skipped_embeddings"] == failures


def test_instruction_projection_excludes_private_contract_and_ignores_source_identity():
    source = TaskInputFile("source-a", "key-a", "unused", "train", "admitted")
    first = task_semantic_input(task("a"), source)
    second = task_semantic_input(task("b", answer="999", dataset="other-private-source"), source)
    assert first.content_hash == second.content_hash
    assert first.task_hash != second.task_hash
    assert first.content_hash != task_semantic_input(task("c", prompt="Sort the apples."), source).content_hash
    public = json.loads(first.public_input)
    assert public["context"]["events"][0]["content"] == "Count two apples."
    assert not {"resources", "grader", "source", "tags"}.intersection(public)
    assert "PRIVATE_EVIDENCE" not in first.public_input
    assert "private-source" not in first.public_input
    assert "817" not in first.public_input


def test_zephyr_descriptions_embeddings_and_alignment_preserve_instances_and_reuse_cache(context, tmp_path, spec):
    raw = tmp_path / "tasks.parquet"
    long_prompt = "Public context details. " * 600 + "Count two apples."
    tasks = [
        task("a", prompt=long_prompt),
        task("b", prompt=long_prompt),
        task("c", prompt="UNSPECIFIED_OPERATION"),
        task("d", prompt="Count three apples."),
    ]
    pq.write_table(
        pa.Table.from_pylist([{"task_json": item.model_dump_json()} for item in tasks] + [{"task_json": None}]), raw
    )
    inputs = prepare_semantic_inputs(
        context,
        sources=[TaskInputFile("source-a", "curation-key", str(raw), "train", "admitted")],
        output_root=str(tmp_path / "inputs"),
        sample_size=None,
        seed=0,
    )
    transport = DescriptionTransport(str(tmp_path / "descriptions-called.jsonl"))
    description_cache = str(tmp_path / "description-cache")
    descriptions = describe_tasks(
        context,
        input_paths=inputs,
        model=DescriptionModel("model", "revision"),
        submit=transport,
        cache_root=description_cache,
        output_root=str(tmp_path / "descriptions"),
    )
    assert len(rows(inputs)) == 4
    assert len(rows(descriptions)) == 3
    requests = [json.loads(line) for line in Path(transport.calls).read_text().splitlines()]
    assert len(requests) == 5
    public_inputs = [json.loads(request["body"]["messages"][1]["content"]) for request in requests]
    assert any(item["context"]["events"][0]["content"] == long_prompt for item in public_inputs)
    replay = describe_tasks(
        context,
        input_paths=inputs,
        model=DescriptionModel("model", "revision"),
        submit=transport,
        cache_root=description_cache,
        output_root=str(tmp_path / "descriptions-replay"),
    )
    assert sorted(rows(replay), key=lambda row: row["content_hash"]) == sorted(
        rows(descriptions), key=lambda row: row["content_hash"]
    )
    assert len(Path(transport.calls).read_text().splitlines()) == 5

    provider = EmbeddingProvider(str(tmp_path / "embeddings-called.jsonl"))
    texts = [text for row in rows(descriptions) for text in task_text_records(row, spec)]
    vector_cache = str(tmp_path / "vectors-cache")
    vectors = embed_texts(
        context,
        texts=Dataset.from_list(texts),
        spec=spec,
        embedder=provider,
        cache_root=vector_cache,
        output_root=str(tmp_path / "vectors"),
    )
    assert len(rows(vectors)) == 1
    for vector in rows(vectors):
        assert np.linalg.norm(vector["embedding"]) == pytest.approx(1.0)

    anchors = (
        CurriculumAnchor(
            kind=AnchorKind.SUBJECT,
            subject_id="math",
            capability_id=None,
            facet=CurriculumFacet.SUBJECT_DOMAIN,
            text="Elementary arithmetic: count apples.",
        ),
        CurriculumAnchor(
            kind=AnchorKind.CAPABILITY,
            subject_id="math",
            capability_id="math.count",
            facet=CurriculumFacet.SUBJECT_DOMAIN,
            text="Count items",
        ),
    )
    anchor_texts = [text_record(anchor.text, spec) for anchor in anchors]
    anchor_paths = embed_texts(
        context,
        texts=Dataset.from_list(anchor_texts),
        spec=spec,
        embedder=provider,
        cache_root=vector_cache,
        output_root=str(tmp_path / "anchors"),
    )
    assert len(Path(provider.calls).read_text().splitlines()) == 2
    anchor_by_key = {row["embedding_key"]: row["embedding"] for row in rows(anchor_paths)}
    reference = CurriculumVectors(
        anchors, np.asarray([anchor_by_key[row["embedding_key"]] for row in anchor_texts], dtype=np.float32)
    )
    parked_cache = tmp_path / "parked-vector-cache"
    Path(vector_cache).rename(parked_cache)
    assigned = align_tasks(
        context,
        input_paths=inputs,
        description_paths=descriptions,
        vector_paths=vectors,
        curriculum=reference,
        curriculum_key="fixture",
        spec=spec,
        output_root=str(tmp_path / "aligned"),
    )
    parked_cache.rename(vector_cache)
    # Adding join inputs must not multiply the number of output files.
    assert len(assigned) <= context.max_workers
    output = {row["task_id"]: row for row in rows(assigned)}
    assert set(output) == {"a", "b", "c", "d"}
    assert output["a"]["description"] == "Elementary arithmetic: count apples."
    assert output["a"]["matches"] == output["b"]["matches"]
    assert output["a"]["matches"] == output["d"]["matches"]
    assert output["a"]["matches"]["subjects"][0]["membership_similarity"] == pytest.approx(1.0)
    assert output["a"]["preview_truncated"]
    assert output["a"]["instruction_characters"] > len(output["a"]["task_preview"])
    assert output["c"]["description_status"] == "insufficient_context"
    assert output["c"]["matches"] is None

    publication = publish_alignments(
        context, input_paths=inputs, alignment_paths=assigned, output_root=str(tmp_path / "release")
    )
    assert len(publication.paths) <= context.max_workers
    assert publication.sources[0].task_count == 4
    assert publication.sources[0].ready_count == 3
    assert publication.sources[0].insufficient_context_count == 1
    assert publication.unique_content_count == 3
    assert publication.unique_embedding_count == 1
    corrupt = tmp_path / "corrupt.parquet"
    pq.write_table(pa.Table.from_pylist([output["a"], output["a"], output["c"]]), corrupt)
    with pytest.raises(ZephyrWorkerError, match=r"Duplicate match|Incomplete alignment coverage"):
        publish_alignments(
            context, input_paths=inputs, alignment_paths=[str(corrupt)], output_root=str(tmp_path / "broken")
        )

    changed_anchor = text_record("Count and compare items", spec)
    embed_texts(
        context,
        texts=Dataset.from_list([*texts, *anchor_texts, changed_anchor]),
        spec=spec,
        embedder=provider,
        cache_root=vector_cache,
        output_root=str(tmp_path / "curriculum-edit"),
    )
    assert len(Path(provider.calls).read_text().splitlines()) == 3
    changed_spec = spec.model_copy(update={"max_tokens": 256})
    embed_texts(
        context,
        texts=Dataset.from_list([text_record("Count items", changed_spec)]),
        spec=changed_spec,
        embedder=provider,
        cache_root=vector_cache,
        output_root=str(tmp_path / "spec-edit"),
    )
    assert len(Path(provider.calls).read_text().splitlines()) == 4


def test_exact_matching_preserves_primary_candidates_and_weak_scores():
    anchors = (
        CurriculumAnchor(
            kind="subject",
            subject_id="math",
            capability_id=None,
            facet="subject_domain",
            text="Math",
        ),
        CurriculumAnchor(
            kind="capability",
            subject_id="math",
            capability_id="count",
            facet="subject_domain",
            text="Count",
        ),
        CurriculumAnchor(
            kind="capability",
            subject_id="math",
            capability_id="compare",
            facet="subject_domain",
            text="Compare",
        ),
        CurriculumAnchor(
            kind="subject",
            subject_id="practice",
            capability_id=None,
            facet="task_mechanic",
            text="Calculation",
        ),
        CurriculumAnchor(
            kind="capability",
            subject_id="practice",
            capability_id="calculate",
            facet="task_mechanic",
            text="Calculate",
        ),
    )
    references = np.asarray([[0, 0, 1], [0.6, 0, 0.8], [0, 0.6, 0.8], [0, 0, 1], [0.6, 0, 0.8]], dtype=np.float32)
    tasks = np.asarray([[0.6, 0, 0.8], [0.6, 0.6, 0.8], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
    neighbor = tuple(
        CurriculumAnchor(
            kind=kind,
            subject_id="neighbor",
            capability_id="neighbor.work" if kind == "capability" else None,
            facet="subject_domain",
            text="Adjacent domain",
        )
        for kind in ("subject", "capability")
    )
    output = rank_task_batch(
        CurriculumVectors((*anchors, *neighbor), np.concatenate([references, [[0, 0, 1], [0, 0, 1]]])),
        tasks,
    )
    assert {row.subject_id for row in output[0].subjects} == {"math", "practice"}
    math = next(row for row in output[0].subjects if row.subject_id == "math")
    assert math.capability_id == "count"
    assert output[0].capabilities[0].subject_id == "math"
    assert output[0].capabilities[0].capability_id == "count"
    assert [candidate.subject_id for candidate in math.subject_candidates] == ["math", "neighbor"]
    tied = next(row for row in output[1].subjects if row.subject_id == "math")
    assert tied.capability_id == "compare"
    assert tied.margin == pytest.approx(0)
    assert [candidate.capability_id for candidate in tied.candidates] == ["compare", "count"]
    assert all(row.candidates[0].similarity == pytest.approx(0.8) for row in output[2].subjects)
    assert output[2].capabilities[0].subject_id == "neighbor"
    assert all(row.membership_similarity == pytest.approx(-1) for row in output[3].subjects)
    assert all(row.capability_id for row in output[3].subjects)


@dataclass(frozen=True)
class ZeroEmbeddingProvider:
    def embed(self, texts):
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_invalid_provider_vectors_never_enter_persistent_cache(context, tmp_path, spec):
    row = text_record("Count items", spec)
    cache = str(tmp_path / "cache")
    with pytest.raises(ZephyrWorkerError, match="zero vector"):
        embed_texts(
            context,
            texts=Dataset.from_list([row]),
            spec=spec,
            embedder=ZeroEmbeddingProvider(),
            cache_root=cache,
            output_root=str(tmp_path / "failed"),
        )
    partition = embedding_cache_path(cache, embedding_cache_partition(row["embedding_key"]))
    assert ReadView(partition).read_blobs([row["embedding_key"]]) == {}
