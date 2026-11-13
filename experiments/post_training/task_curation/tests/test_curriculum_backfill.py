# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.schema import arrow_schema
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.lazy import ArtifactStep, run
from marin.processing.embeddings.harrier import HARRIER_REPO, HARRIER_REVISION
from taskcompendium.convert.answers import numeric_answer_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import Source
from taskcompendium.pipeline.curriculum.alignment import CurriculumAnchor
from taskcompendium.pipeline.curriculum.embedding import TextEmbedder
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec
from taskcompendium.pipeline.models import RawRow
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from experiments.post_training.task_curation import curriculum_backfill
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.curriculum_backfill import (
    FrozenCoverageSource,
    prepare_source,
    prepare_sources,
    run_stage,
)
from experiments.post_training.task_curation.embedding import (
    DescriptionEndpoint,
    SemanticArtifact,
    curation_embedding_pipeline,
)
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin
from experiments.post_training.task_curation.tests.numbers_pipeline import (
    NUMBERS_DATASET,
    NumbersConfig,
    numbers_pipeline,
)
from experiments.post_training.task_curriculum.models import CurriculumCatalog
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def semantic_record(root, paths, identity):
    artifact = SemanticArtifact(path=str(root), paths=[str(path) for path in paths])
    write_record(
        ArtifactRecord(
            name="prepared",
            version="2026.11.13.1",
            fingerprint=identity,
            output_path=str(root),
            result_type=result_type_name(SemanticArtifact),
            result=artifact.result_payload(),
        )
    )


@pytest.fixture
def prepared_stages(tmp_path, monkeypatch):
    catalog = CurriculumCatalog.model_validate(
        {
            "catalog_version": "fixture",
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
                                "includes": ["calculate"],
                                "excludes": ["algebra"],
                            },
                            {
                                "kind": "capability",
                                "id": "math.add",
                                "parent_id": "math",
                                "name": "Addition",
                                "outcome": "Add numbers",
                                "includes": ["addition"],
                                "excludes": ["unrelated work"],
                                "prerequisites": [],
                                "sample_tasks": [],
                            },
                        ],
                    },
                }
            ],
        }
    )
    requested = tmp_path / "requested-catalog"
    requested.mkdir()
    (requested / "curriculum.json").write_text(catalog.model_dump_json())
    monkeypatch.setattr(
        curriculum_backfill,
        "TASK_CURRICULUM",
        ArtifactStep.adopt("fixture-catalog", "2026.11.13.1", source=str(requested)),
    )
    instructions = tmp_path / "shared-instructions"
    instructions.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"content_hash": "input", "description": "Add integers", "status": "ready", "reason": ""}]
        ),
        instructions / "texts.parquet",
    )
    semantic_record(instructions, [instructions / "texts.parquet"], "instructions-1")
    anchors = tmp_path / "shared-anchors"
    anchors.mkdir()
    (anchors / "curriculum.json").write_text(catalog.model_dump_json())
    (anchors / "nodes.json").write_text(json.dumps([node.model_dump(mode="json") for node in catalog.atlas_nodes()]))
    spec = EmbeddingSpec(
        model=HARRIER_REPO,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=1024,
        max_tokens=16384,
        compute_dtype="bfloat16",
    )
    (anchors / "embedding_spec.json").write_text(spec.model_dump_json())
    path = anchors / "anchors.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [anchor.model_dump(mode="json") for anchor in catalog.embedding_anchors()],
            schema=arrow_schema(CurriculumAnchor),
        ),
        path,
    )
    (anchors / "anchor_paths.json").write_text(json.dumps([str(path)]))
    semantic_record(anchors, [], "anchors-1")
    return instructions, anchors, requested


@pytest.fixture
def frozen_source(tmp_path):
    declaration = RlDataSource(
        config=NumbersConfig("source", tmp_path / "unused"),
        pipeline=numbers_pipeline,
        info=SourceInfo(
            origin=SourceOrigin.MARIN,
            upstream_dataset="source",
            upstream_name="source",
            build_path="test",
            within_source_dedup=deduplication_evidence(NUMBERS_DATASET.name, NUMBERS_DATASET.revision, "1", count=2),
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numerical tasks require no actor.",
        ),
    ).declaration
    specs = tmp_path / "specs"
    specs.mkdir()
    tasks = [
        numeric_answer_task(
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
        for identifier in ("a", "b", "c")
    ]
    pq.write_table(
        pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]), specs / "part.parquet"
    )
    drops = tmp_path / "drops.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_id": "b"}]), drops)
    evidence = tmp_path / "population.json"
    evidence.write_text(json.dumps({"dataset": NUMBERS_DATASET.name, "revision": NUMBERS_DATASET.revision}))
    source = FrozenCoverageSource(
        declaration=declaration,
        specs_dir=str(specs),
        task_column="task_spec",
        raw_count=3,
        task_specs_sha256=hashlib.sha256("".join(task.model_dump_json() for task in tasks).encode()).hexdigest(),
        task_spec_scan_sha256=hashlib.sha256("".join(task.model_dump_json() for task in tasks).encode()).hexdigest(),
        intended_use="eval",
        input_evidence=str(evidence),
        input_record_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        drop_manifest_uri=str(drops),
        drop_manifest_sha256=hashlib.sha256(drops.read_bytes()).hexdigest(),
        drop_count=1,
    )
    return source, tasks


@pytest.mark.parametrize("source_identity", ["current", "historical"])
def test_preparation_preserves_retained_identities_roles_and_reuses_completed_sources(
    tmp_path, frozen_source, source_identity
):
    source, tasks = frozen_source
    expected_source = source.declaration.id
    if source_identity == "historical":
        payload = source.model_dump(mode="json")
        payload["declaration"].update(id="MarinSkyRL:gsm8k", origin="OpenAI", upstream_dataset="gsm8k")
        source = FrozenCoverageSource.model_validate(payload)
        expected_source = "OpenAI:gsm8k"
    output = tmp_path / "public"
    prepare_sources([source], str(output), workers=1)
    root = output / "sources" / next((output / "sources").iterdir()).name
    public = json.loads((root / "public-inputs.json").read_text())
    rows = [row for path in public["paths"] for row in load_parquet(path)]
    assert {row["task_id"] for row in rows} == {"a", "c"}
    assert {row["source_id"] for row in rows} == {expected_source}
    assert {row["intended_use"] for row in rows} == {"eval"}
    assert {row["population"] for row in rows} == {"converted"}
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["input_rows"] == 3 and manifest["converted_rows"] == 2
    assert manifest["rejections"] == {"frozen_retention_exclusions": 1}
    assert all("Add two and three." in row["public_input"] for row in rows)
    original = (root / "public-inputs.json").read_bytes()
    prepare_sources([source], str(output), workers=1)
    assert (root / "public-inputs.json").read_bytes() == original
    assert len(list((root / "attempts").iterdir())) == 1
    expected = {task.id: canonical_sha256(task.model_dump(mode="json")) for task in tasks}
    assert all(row["task_hash"] == expected[row["task_id"]] for row in rows)


@pytest.mark.parametrize("reuse", ["fresh", "source", "corpus"])
def test_changed_task_bytes_with_same_identity_and_population_are_rejected(tmp_path, frozen_source, reuse):
    source, _ = frozen_source
    output = tmp_path / "public"
    if reuse == "source":
        prepare_source(source, str(output))
    elif reuse == "corpus":
        prepare_sources([source], str(output), workers=1)
    original_manifests = {path: path.read_bytes() for path in output.glob("sources/*/manifest.json")}
    path = next((tmp_path / "specs").glob("*.parquet"))
    table = pq.read_table(path)
    rows = table.to_pylist()
    encoded = rows[0][source.task_column]
    rows[0][source.task_column] = encoded.replace("Add two and three.", "Subtract two from three.")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="Frozen TaskSpec scan changed"):
        if reuse == "corpus":
            prepare_sources([source], str(output), workers=1)
        else:
            prepare_source(source, str(output))
    assert {path: path.read_bytes() for path in output.glob("sources/*/manifest.json")} == original_manifests


def test_scan_digest_has_an_explicit_convention_separate_from_historical_digest(tmp_path, frozen_source):
    source, _ = frozen_source
    source = source.model_copy(update={"task_specs_sha256": "f" * 64})
    prepared = prepare_source(source, str(tmp_path / "public"))
    assert prepared["stored_stream_sha256"] == source.task_spec_scan_sha256


def test_completed_corpus_revalidates_frozen_exclusions(tmp_path, frozen_source):
    source, _ = frozen_source
    output = tmp_path / "public"
    prepare_sources([source], str(output), workers=1)
    original = (output / "corpus.json").read_bytes()
    pq.write_table(pa.Table.from_pylist([{"task_id": "c"}]), source.drop_manifest_uri)
    with pytest.raises(ValueError, match="Frozen drop manifest changed"):
        prepare_sources([source], str(output), workers=1)
    assert (output / "corpus.json").read_bytes() == original


def test_new_source_lineage_forks_assignments_while_reusing_numeric_stages(
    tmp_path, frozen_source, caplog, prepared_stages
):
    source, _ = frozen_source
    evidence = tmp_path / "new-receipt.json"
    shutil.copyfile(source.input_evidence, evidence)
    changed = source.model_copy(update={"input_evidence": str(evidence)})
    plans = []
    caplog.set_level(logging.INFO)
    for index, frozen in enumerate((source, changed)):
        output = tmp_path / f"public-{index}"
        prepare_sources([frozen], str(output), workers=1)
        corpus_path = output / "corpus.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["prepared_descriptions"] = str(tmp_path / "shared-instructions")
        corpus["prepared_curriculum_embeddings"] = str(tmp_path / "shared-anchors")
        corpus_path.write_text(json.dumps(corpus))
        caplog.clear()
        run_stage(
            str(corpus_path), str(tmp_path / "artifacts"), str(tmp_path / "cache"), "coverage", "cpu", 1, "8g", False
        )
        plans.append(
            json.loads(
                next(
                    row.message.removeprefix("Stage plan: ")
                    for row in caplog.records
                    if row.message.startswith("Stage plan: ")
                )
            )
        )
    before, after = plans
    for stage in ("descriptions", "task_embeddings", "curriculum_embeddings"):
        assert before[stage] == after[stage]
    for stage in ("inputs", "assignments", "publication", "coverage"):
        assert before[stage] != after[stage]


@pytest.mark.parametrize("changed_stage", ["prepared_descriptions", "prepared_curriculum_embeddings"])
def test_changed_adopted_content_forks_downstream_artifacts(
    tmp_path, frozen_source, caplog, prepared_stages, changed_stage
):
    instructions, anchors, requested = prepared_stages
    source, _ = frozen_source
    output = tmp_path / "public"
    prepare_sources([source], str(output), workers=1)
    corpus_path = output / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    corpus.update(prepared_descriptions=str(instructions), prepared_curriculum_embeddings=str(anchors))
    plans = []
    caplog.set_level(logging.INFO)
    for index in range(2):
        if index:
            replacement = tmp_path / "replacement"
            shutil.copytree(instructions if changed_stage == "prepared_descriptions" else anchors, replacement)
            if changed_stage == "prepared_descriptions":
                pq.write_table(
                    pa.Table.from_pylist(
                        [{"content_hash": "input", "description": "Compare integers", "status": "ready", "reason": ""}]
                    ),
                    replacement / "texts.parquet",
                )
                semantic_record(replacement, [replacement / "texts.parquet"], "instructions-2")
            else:
                document = json.loads((requested / "curriculum.json").read_text())
                document["curricula"][0]["curriculum"]["sections"][-1]["outcome"] = "Subtract numbers"
                catalog = CurriculumCatalog.model_validate(document)
                (requested / "curriculum.json").write_text(catalog.model_dump_json())
                with pytest.raises(ValueError, match="Adopted anchors differ"):
                    run_stage(
                        str(corpus_path),
                        str(tmp_path / "artifacts"),
                        str(tmp_path / "cache"),
                        "coverage",
                        "cpu",
                        1,
                        "8g",
                        False,
                    )
                (replacement / "curriculum.json").write_text(catalog.model_dump_json())
                (replacement / "nodes.json").write_text(
                    json.dumps([node.model_dump(mode="json") for node in catalog.atlas_nodes()])
                )
                path = replacement / "anchors.parquet"
                pq.write_table(
                    pa.Table.from_pylist(
                        [anchor.model_dump(mode="json") for anchor in catalog.embedding_anchors()],
                        schema=arrow_schema(CurriculumAnchor),
                    ),
                    path,
                )
                (replacement / "anchor_paths.json").write_text(json.dumps([str(path)]))
                semantic_record(replacement, [], "anchors-2")
            corpus[changed_stage] = str(replacement)
        corpus_path.write_text(json.dumps(corpus))
        caplog.clear()
        run_stage(
            str(corpus_path), str(tmp_path / "artifacts"), str(tmp_path / "cache"), "coverage", "cpu", 1, "8g", False
        )
        plans.append(
            json.loads(
                next(
                    row.message.removeprefix("Stage plan: ")
                    for row in caplog.records
                    if row.message.startswith("Stage plan: ")
                )
            )
        )
    before, after = plans
    assert before["inputs"] == after["inputs"]
    if changed_stage == "prepared_descriptions":
        assert before["task_embeddings"] != after["task_embeddings"]
        assert before["curriculum_embeddings"] == after["curriculum_embeddings"]
    else:
        assert before["task_embeddings"] == after["task_embeddings"]
    for stage in ("assignments", "publication", "coverage"):
        assert before[stage] != after[stage]


def test_cuda_run_requires_a_parity_receipt_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="requires an accepted --parity-receipt"):
        run_stage(
            str(tmp_path / "missing-corpus.json"),
            str(tmp_path / "artifacts"),
            str(tmp_path / "cache"),
            "task_embeddings",
            "cuda",
            1,
            "8g",
            True,
        )


def test_unknown_exclusion_prevents_publication_and_keeps_partial_evidence(tmp_path, frozen_source):
    source, _ = frozen_source
    drops = tmp_path / "other-drops.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_id": "absent"}]), drops)
    changed = source.model_copy(
        update={"drop_manifest_uri": str(drops), "drop_manifest_sha256": hashlib.sha256(drops.read_bytes()).hexdigest()}
    )
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="population accounting"):
        prepare_sources([changed], str(output), workers=1)
    assert not (output / "corpus.json").exists()
    assert not list(output.glob("sources/*/manifest.json"))
    assert list(output.glob("sources/*/attempts/*/*.parquet"))


def unavailable_embedder() -> TextEmbedder:
    raise AssertionError("Instruction preparation must not run model inference")


def test_prepared_public_corpus_runs_direct_text_graph_without_private_views(tmp_path, frozen_source, monkeypatch):
    source, _ = frozen_source
    output = tmp_path / "prepared"
    prepare_sources([source], str(output), workers=1)
    corpus = json.loads((output / "corpus.json").read_text())
    source_root = corpus["sources"][0]["root"]
    runtime = CampaignRuntime()
    with (
        artifact_prefix(str(tmp_path / "artifacts")),
        local_client() as client,
        ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        runtime.activate(context),
    ):
        stages = curation_embedding_pipeline(
            curated={source.declaration.id: ArtifactStep.adopt("source", "2026.11.13.1", source=source_root)},
            curriculum=ArtifactStep.adopt("unused-curriculum", "2026.11.13.1", source=str(tmp_path / "unused")),
            runtime=runtime,
            description_model=None,
            endpoint=DescriptionEndpoint(None, "unused"),
            spec=EmbeddingSpec(
                model="fixture",
                revision="1",
                tokenizer_revision="1",
                dimensions=3,
                max_tokens=16384,
                compute_dtype="bfloat16",
            ),
            embedder_factory=unavailable_embedder,
            cache_root=str(tmp_path / "cache"),
            population="converted",
            sample_size=None,
            seed=0,
            inventory=[source.declaration],
            prepared_inputs=ArtifactStep.adopt(
                "prepared", "2026.11.13.1", source=corpus["prepared_inputs"], kind=SemanticArtifact
            ),
        )
        result = run(stages.descriptions, max_concurrent=1)[0]
    rows = [row for path in result.paths for row in load_parquet(path)]
    assert len(rows) == 1  # The two retained tasks have identical public instructions.
    assert rows[0]["status"] == "ready"
    assert "Add two and three." in rows[0]["description"]
    corpus["prepared_descriptions"] = result.path
    (output / "corpus.json").write_text(json.dumps(corpus))
    monkeypatch.setattr("experiments.post_training.task_curation.embedding._instructions", unavailable_embedder)
    run_stage(
        str(output / "corpus.json"),
        str(tmp_path / "regional-artifacts"),
        str(tmp_path / "regional-cache"),
        "descriptions",
        "cpu",
        1,
        "8g",
        True,
    )
