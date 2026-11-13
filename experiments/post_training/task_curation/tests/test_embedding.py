# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from marin.execution.lazy import ArtifactStep, run
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source
from taskcompendium.pipeline.curriculum import embedding
from taskcompendium.pipeline.curriculum.embedding import text_record
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.semantics import DescriptionModel
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import driver
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.embedding import DescriptionEndpoint, curation_embedding_pipeline
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin
from experiments.post_training.task_curation.tests.numbers_pipeline import (
    NUMBERS_DATASET,
    NumbersConfig,
    numbers_pipeline,
)
from experiments.post_training.task_curriculum.models import CurriculumCatalog
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


@pytest.fixture(autouse=True)
def numeric_actor_census(request, monkeypatch):
    if request.node.get_closest_marker("data_integration") is not None:
        sources = driver.all_sources()
    else:
        sources = {}
        for source in skyrl_math.sources():
            if source.info.id not in {"HuggingFaceH4:aime24", "OpenAI:gsm8k"}:
                continue
            sources[source.info.id] = replace(
                source,
                info=replace(
                    source.info,
                    count=1,
                    within_source_dedup=deduplication_evidence(
                        source.dataset.name, source.dataset.revision, source.version
                    ),
                    daytona_machine_slots=0,
                    daytona_snapshot_count=0,
                    daytona_execution_failures=0,
                    daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
                ),
            )
    monkeypatch.setattr(driver, "all_sources", lambda: sources)
    monkeypatch.setattr("experiments.post_training.task_curation.sources.all_sources", lambda: sources)


@pytest.fixture(autouse=True)
def cache_partitions(monkeypatch):
    monkeypatch.setattr(embedding, "EMBEDDING_CACHE_SHARDS", 2)


class DescriptionServer(ThreadingHTTPServer):
    requests: list[dict]


class DescriptionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = cast(DescriptionServer, self.server)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(request)
        description = {
            "status": "ready",
            "reason": "",
            "description": "Elementary arithmetic: add two integers.",
        }
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "fixture",
                                    "type": "function",
                                    "function": {"name": "describe_task", "arguments": json.dumps(description)},
                                }
                            ],
                        },
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


@pytest.fixture
def description_server(monkeypatch):
    monkeypatch.setenv("GLM_BULK_TOKEN", "fixture-token")
    server = DescriptionServer(("127.0.0.1", 0), DescriptionHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@dataclass(frozen=True)
class EmbeddingEndpoint:
    calls: str
    dimensions: int = 3

    def embed(self, texts):
        with Path(self.calls).open("a") as stream:
            stream.write(json.dumps(texts) + "\n")
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors


def catalog():
    return CurriculumCatalog.model_validate(
        {
            "catalog_version": "fixture-1",
            "curricula": [
                {
                    "routing_facet": "subject_domain",
                    "curriculum": {
                        "version": "1",
                        "subject_id": "D01",
                        "subject_name": "Mathematics",
                        "sections": [
                            {
                                "kind": "group",
                                "id": "math",
                                "parent_id": None,
                                "name": "Arithmetic",
                                "scope": "Work with numbers",
                                "includes": ["count and calculate"],
                                "excludes": ["algebra"],
                            },
                            *[
                                {
                                    "kind": "capability",
                                    "id": f"math.{operation}",
                                    "parent_id": "math",
                                    "name": operation,
                                    "outcome": f"Perform {operation}",
                                    "includes": [operation],
                                    "excludes": ["unrelated work"],
                                    "prerequisites": [],
                                    "sample_tasks": [],
                                }
                                for operation in ("addition", "comparison")
                            ],
                        ],
                    },
                }
            ],
        }
    )


def test_curriculum_edit_reuses_task_artifacts_and_embeds_only_changed_anchor(tmp_path, description_server):
    curated = tmp_path / "curated"
    (curated / "final").mkdir(parents=True)
    (curated / "manifest.json").write_text(json.dumps({"intended_use": "train", "revision": "fixture-1"}))
    tasks = []
    for identifier in ("a", "b"):
        task = numeric_answer_task(
            RawRow(
                id=identifier,
                source=Source(dataset="fixture", revision="1", importer_revision="1", row=identifier),
                data={},
            ),
            prompt="Add two and three.",
            answer="5",
            tolerance_abs=0,
            tolerance_rel=0,
        )
        tasks.append({"task_id": identifier, "task_json": task.model_dump_json()})
    pq.write_table(pa.Table.from_pylist(tasks), curated / "final/tasks.parquet")
    first_catalog = catalog()
    changed = first_catalog.model_dump(mode="json")
    changed["catalog_version"] = "fixture-2"
    changed["curricula"][0]["curriculum"]["sections"][1]["name"] = "Integer addition"
    cache = tmp_path / "cache"
    provider = EmbeddingEndpoint(str(tmp_path / "embedding-calls.jsonl"))
    spec = EmbeddingSpec(
        model="fixture", revision="1", tokenizer_revision="1", dimensions=3, max_tokens=512, compute_dtype="float32"
    )
    runtime = CampaignRuntime()
    with (
        artifact_prefix(str(cache)),
        ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        stages = []
        for revision, payload in (("1", first_catalog.model_dump(mode="json")), ("2", changed)):
            directory = tmp_path / f"curriculum-{revision}"
            directory.mkdir()
            (directory / "curriculum.json").write_text(json.dumps(payload))
            stages.append(
                curation_embedding_pipeline(
                    curated={"Marin:source": ArtifactStep.adopt("curated-fixture", "2026.10.14.1", source=str(curated))},
                    curriculum=ArtifactStep.adopt("curriculum-fixture", f"2026.10.14.{revision}", source=str(directory)),
                    runtime=runtime,
                    description_model=DescriptionModel("fixture", "1"),
                    endpoint=DescriptionEndpoint(f"http://127.0.0.1:{description_server.server_port}/v1", "unused"),
                    spec=spec,
                    embedder_factory=lambda: provider,
                    cache_root=str(cache),
                    population="admitted",
                    sample_size=None,
                    seed=0,
                    inventory=[
                        RlDataSource(
                            config=NumbersConfig("source", tmp_path / "unused"),
                            pipeline=numbers_pipeline,
                            info=SourceInfo(
                                within_source_dedup=deduplication_evidence(
                                    NUMBERS_DATASET.name, NUMBERS_DATASET.revision, "1"
                                ),
                                id="Marin:source",
                                upstream_dataset="source",
                                upstream_name="source",
                                build_path="test",
                                origin=SourceOrigin.MARIN,
                                daytona_snapshot_count=0,
                                daytona_machine_slots=0,
                                daytona_execution_failures=0,
                                daytona_count_basis="Fixture numerical task requires no actor.",
                            ),
                        ).declaration
                    ],
                )
            )
        first, second = stages
        assert first.inputs.path() == second.inputs.path()
        assert first.descriptions.path() == second.descriptions.path()
        assert first.task_embeddings.path() == second.task_embeddings.path()
        assert first.curriculum_embeddings.path() != second.curriculum_embeddings.path()
        assert first.assignments.path() != second.assignments.path()
        built = run(first.publication, max_concurrent=1)[0]
        release = json.loads((Path(built.path) / "release.json").read_text())
        assert release["task_count"] == 2
        rows = [row for path in built.paths for row in load_parquet(path)]
        assert {row["task_id"] for row in rows} == {"a", "b"}
        assert all(row["matches"]["subjects"][0]["capability_id"] == "math.addition" for row in rows)
        batches = [json.loads(line) for line in Path(provider.calls).read_text().splitlines()]
        expected = {text_record(anchor.text, spec)["text"] for anchor in first_catalog.embedding_anchors()}
        assert expected <= {text for batch in batches for text in batch}
        texts_before = sum(map(len, batches))
        rebuilt = run(second.publication, max_concurrent=1)[0]
        assert json.loads((Path(rebuilt.path) / "release.json").read_text())["release_id"] != release["release_id"]
        assert {row["task_id"] for path in rebuilt.paths for row in load_parquet(path)} == {"a", "b"}
        assert len(description_server.requests) == 1
        batches = [json.loads(line) for line in Path(provider.calls).read_text().splitlines()]
    assert sum(map(len, batches)) == texts_before + 1


@pytest.mark.parametrize("representation", ["direct", "rewrite"])
def test_driver_converts_embeds_and_resumes_without_rereading_raw_inputs(
    tmp_path, monkeypatch, description_server, representation
):
    # Keep model inference at its provider boundary; execute conversion, HTTP, caches and export.
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.load_harrier",
        lambda config: EmbeddingEndpoint(str(tmp_path / "embeddings.jsonl"), config.dimensions),
    )
    staged = tmp_path / "input/data"
    staged.mkdir(parents=True)
    raw_file = staged / "train-0000.parquet"
    pq.write_table(pa.Table.from_pylist([{"problem": "Two plus three?", "answer": "5"}]), raw_file)
    curriculum = tmp_path / "curriculum"
    curriculum.mkdir()
    (curriculum / "curriculum.json").write_text(catalog().model_dump_json())
    output, cache = tmp_path / "output", tmp_path / "cache"
    common = [
        "--representation",
        representation,
        "--cache-dir",
        str(cache),
        "--embed",
        "coverage",
        "--max-workers",
        "1",
        "--catalog-root",
        str(curriculum),
        "--description-model-revision",
        "fixture",
        "--description-base-url",
        f"http://127.0.0.1:{description_server.server_port}/v1",
    ]
    runner = CliRunner()
    converted = runner.invoke(
        main,
        [
            *common,
            "--mode",
            "quick",
            "--source",
            "aime24",
            "--input-root",
            str(staged.parent),
            "--output-root",
            str(output),
            "--run",
        ],
    )
    assert converted.exit_code == 0, converted.output
    corpus = output / "corpus.json"
    assert json.loads(corpus.read_text())["sources"][0]["converted_rows"] == 1
    raw_file.unlink()
    resumed = runner.invoke(main, [*common, "--corpus", str(corpus), "--run"])
    assert resumed.exit_code == 0, resumed.output
    release_files = list((cache / "rl-semantics").glob("publication-*/*/release.json"))
    assert len(release_files) == 1
    release = json.loads(release_files[0].read_text())
    assert release["task_count"] == release["ready_count"] == 1
    assert release["population"] == "converted"
    assert len(description_server.requests) == int(representation == "rewrite")
    tasks = [row for path in release["paths"] for row in load_parquet(path)]
    assert len(tasks) == 1 and tasks[0]["intended_use"] == "eval"
    assert tasks[0]["matches"]["subjects"][0]["subject_id"] == "D01"


def test_publication_retains_registered_sources_without_placement(tmp_path, description_server):
    curated = tmp_path / "curated"
    (curated / "final").mkdir(parents=True)
    (curated / "manifest.json").write_text(json.dumps({"intended_use": "train", "revision": "fixture-1"}))
    task = numeric_answer_task(
        RawRow(
            id="a",
            source=Source(dataset="fixture", revision="1", importer_revision="1", row="a"),
            data={},
        ),
        prompt="Add two and three.",
        answer="5",
        tolerance_abs=0,
        tolerance_rel=0,
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_id": "a", "task_json": task.model_dump_json()}]), curated / "final/tasks.parquet"
    )
    directory = tmp_path / "curriculum"
    directory.mkdir()
    (directory / "curriculum.json").write_text(catalog().model_dump_json())
    cache = tmp_path / "cache"
    spec = EmbeddingSpec(
        model="fixture", revision="1", tokenizer_revision="1", dimensions=3, max_tokens=512, compute_dtype="float32"
    )
    runtime = CampaignRuntime()
    with (
        artifact_prefix(str(cache)),
        ZephyrContext(max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        stages = curation_embedding_pipeline(
            curated={"Marin:source": ArtifactStep.adopt("curated-fixture", "2026.10.14.1", source=str(curated))},
            curriculum=ArtifactStep.adopt("curriculum-fixture", "2026.10.14.1", source=str(directory)),
            runtime=runtime,
            description_model=DescriptionModel("fixture", "1"),
            endpoint=DescriptionEndpoint(f"http://127.0.0.1:{description_server.server_port}/v1", "unused"),
            spec=spec,
            embedder_factory=lambda: EmbeddingEndpoint(str(tmp_path / "embedding-calls.jsonl"), spec.dimensions),
            cache_root=str(cache),
            population="admitted",
            sample_size=None,
            seed=0,
            inventory=[
                RlDataSource(
                    config=NumbersConfig("source", tmp_path / "unused"),
                    pipeline=numbers_pipeline,
                    info=SourceInfo(
                        within_source_dedup=deduplication_evidence(NUMBERS_DATASET.name, NUMBERS_DATASET.revision, "1"),
                        id="Marin:source",
                        upstream_dataset="source",
                        upstream_name="source",
                        build_path="test",
                        origin=SourceOrigin.MARIN,
                        daytona_snapshot_count=0,
                        daytona_machine_slots=0,
                        daytona_execution_failures=0,
                        daytona_count_basis="Fixture numerical task requires no actor.",
                    ),
                ).declaration,
                RlDataSource(
                    config=NumbersConfig("source", tmp_path / "unused"),
                    pipeline=numbers_pipeline,
                    info=SourceInfo(
                        within_source_dedup=deduplication_evidence(
                            NUMBERS_DATASET.name, NUMBERS_DATASET.revision, "1", 7
                        ),
                        id="Marin:release",
                        upstream_dataset=("Marin:release").partition(":")[2],
                        upstream_name=("Marin:release").partition(":")[2],
                        build_path="test",
                        origin=SourceOrigin.MARIN,
                        daytona_snapshot_count=0,
                        daytona_machine_slots=0,
                        daytona_execution_failures=0,
                        daytona_count_basis="Fixture numerical task requires no actor.",
                        count=7,
                    ),
                ).declaration,
                RlDataSource(
                    config=NumbersConfig("source", tmp_path / "unused"),
                    pipeline=numbers_pipeline,
                    info=SourceInfo(
                        within_source_dedup=deduplication_evidence(NUMBERS_DATASET.name, NUMBERS_DATASET.revision, "1"),
                        id="Marin:no-count",
                        upstream_dataset=("Marin:no-count").partition(":")[2],
                        upstream_name=("Marin:no-count").partition(":")[2],
                        build_path="test",
                        origin=SourceOrigin.MARIN,
                        daytona_snapshot_count=0,
                        daytona_machine_slots=0,
                        daytona_execution_failures=0,
                        daytona_count_basis="Fixture numerical task requires no actor.",
                    ),
                ).declaration,
            ],
        )
        built = run(stages.publication, max_concurrent=1)[0]
    release = json.loads((Path(built.path) / "release.json").read_text())
    rows = {row["source_id"]: row for row in release["sources"]}
    assert rows["Marin:source"]["basis"] == "matched"
    assert rows["Marin:release"]["basis"] == "declared"
    assert rows["Marin:release"]["task_count"] == 7
    assert rows["Marin:release"]["intended_use"] is None
    assert rows["Marin:no-count"]["basis"] == "declared"
    assert rows["Marin:no-count"]["task_count"] == 1
    assert release["task_count"] == 1


@pytest.mark.data_integration
def test_corpus_resume_refreshes_registered_source_metadata(tmp_path, monkeypatch, description_server):
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.load_harrier",
        lambda config: EmbeddingEndpoint(str(tmp_path / "embeddings.jsonl"), config.dimensions),
    )
    staged = tmp_path / "input/data"
    staged.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"problem": "Two plus three?", "answer": "5"}]),
        staged / "train-0000.parquet",
    )
    curriculum = tmp_path / "curriculum"
    curriculum.mkdir()
    (curriculum / "curriculum.json").write_text(catalog().model_dump_json())
    output, cache = tmp_path / "output", tmp_path / "cache"
    common = [
        "--cache-dir",
        str(cache),
        "--embed",
        "coverage",
        "--max-workers",
        "1",
        "--catalog-root",
        str(curriculum),
        "--description-model-revision",
        "fixture",
        "--description-base-url",
        f"http://127.0.0.1:{description_server.server_port}/v1",
    ]
    runner = CliRunner()
    quick = runner.invoke(
        main,
        [
            *common,
            "--mode",
            "quick",
            "--source",
            "aime24",
            "--input-root",
            str(staged.parent),
            "--output-root",
            str(output),
            "--run",
        ],
    )
    assert quick.exit_code == 0, quick.output
    corpus_path = output / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    corpus["inventory"] = []
    corpus_path.write_text(json.dumps(corpus))
    resumed = runner.invoke(main, [*common, "--corpus", str(corpus_path), "--run"])
    assert resumed.exit_code == 0, resumed.output
    releases = [json.loads(path.read_text()) for path in (cache / "rl-semantics").glob("publication-*/*/release.json")]
    sources = {row["source_id"]: row for row in max(releases, key=lambda r: len(r["sources"]))["sources"]}
    assert sources["HuggingFaceH4:aime24"]["basis"] == "matched"
    declared = [row for row in sources.values() if row["basis"] == "declared"]
    assert declared and all(row["task_count"] > 0 and row["intended_use"] is None for row in declared)
    retired = read_registry_json(Path(__file__).parents[1] / "retired-inventory-only.json")
    assert {row["id"] for row in retired["entries"] if row["disposition"] != "quick_converted"}.isdisjoint(sources)
