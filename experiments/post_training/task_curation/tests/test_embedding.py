# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.lazy import ArtifactStep, run
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.semantics import DescriptionModel
from taskcompendium.pipeline.task_embeddings import EmbeddingSpec, text_record
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.curriculum import curriculum_anchors
from experiments.post_training.task_curation.embedding import DescriptionEndpoint, curation_embedding_pipeline
from experiments.post_training.task_curation.local import local_artifact_prefix
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo
from experiments.post_training.task_curation.source_metadata import source_declaration
from experiments.post_training.task_curriculum.models import CurriculumCatalog


class DescriptionServer(ThreadingHTTPServer):
    requests: list[dict]


class DescriptionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = cast(DescriptionServer, self.server)
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(request)
        identifier = request["tools"][0]["function"]["parameters"]["properties"]["task_id"]["const"]
        description = {
            "task_id": identifier,
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

    def embed(self, texts):
        with Path(self.calls).open("a") as stream:
            stream.write(json.dumps(texts) + "\n")
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


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
        assert isinstance(task, TaskSpec)
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
        local_artifact_prefix(str(cache)),
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
                    curated={"source": ArtifactStep.adopt("curated-fixture", "2026.10.14.1", source=str(curated))},
                    curriculum=ArtifactStep.adopt("curriculum-fixture", f"2026.10.14.{revision}", source=str(directory)),
                    runtime=runtime,
                    description_model=DescriptionModel("fixture", "1"),
                    endpoint=DescriptionEndpoint(f"http://127.0.0.1:{description_server.server_port}/v1", "unused"),
                    spec=spec,
                    embedder=provider,
                    cache_root=str(cache),
                    population="admitted",
                    sample_size=None,
                    seed=0,
                    inventory=[
                        source_declaration(RlDataSource(info=SourceInfo(id="source", title="Fixture", origin="fixture")))
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
        assert release["unique_content_count"] == release["unique_embedding_count"] == 1
        rows = [row for path in built.paths for row in load_parquet(path)]
        assert {row["task_id"] for row in rows} == {"a", "b"}
        assert all(row["matches"]["subjects"][0]["capability_id"] == "math.addition" for row in rows)
        batches = [json.loads(line) for line in Path(provider.calls).read_text().splitlines()]
        expected = {text_record(anchor.text, spec)["text"] for anchor in curriculum_anchors(first_catalog)}
        assert expected <= {text for batch in batches for text in batch}
        texts_before = sum(map(len, batches))
        rebuilt = run(second.publication, max_concurrent=1)[0]
        assert json.loads((Path(rebuilt.path) / "release.json").read_text())["release_id"] != release["release_id"]
        assert {row["task_id"] for path in rebuilt.paths for row in load_parquet(path)} == {"a", "b"}
        assert len(description_server.requests) == 1
        batches = [json.loads(line) for line in Path(provider.calls).read_text().splitlines()]
        assert sum(map(len, batches)) == texts_before + 1
