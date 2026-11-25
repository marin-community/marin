# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained census and corpus-assembly stages (R40 gate).

The fixtures build a real two-task frozen population and run the real census path —
inventory panel, oracle/static classification, public preparation — so the tests
exercise the same code path the release graph runs, not a parallel mock. The source
registry is the injected boundary (the graph resolves the real one at run time).
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput, QualifiedSourceDeclaration, SourceOrigin, source_identifier
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.release_stages import census as census_stage
from experiments.post_training.task_curation.release_stages import corpus_assembly as corpus_stage
from experiments.post_training.task_curation.release_stages import sizing
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

FIXTURE_IMAGE = "ubuntu:24.04@sha256:" + "a" * 64
SOURCE_ID = "Marin:fixture"
EMPTY_SOURCE_ID = "Marin:empty-fixture"


def _task_spec(task_id: str) -> TaskSpec:
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=FIXTURE_IMAGE,
        working_directory="/app",
        setup_commands=("true",),
        capabilities=(),
    )
    grader = ScriptGrader(
        argv=("true",),
        environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE
        ),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=requirements,
        execution_requirements=ExecutionRequirements(
            cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="fixture", revision="1", row=task_id, importer_revision="1"),
        resources=ResourceGroups(),
    )


def _write_population(tmp_path: Path, tasks: list[TaskSpec]) -> tuple[SourceTextPopulation, str]:
    """A frozen population artifact over ``tasks``; returns the artifact and its scan sha."""
    population_dir = tmp_path / "population"
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    population_dir.mkdir(parents=True, exist_ok=True)
    encoded = [json.dumps(task.model_dump(mode="json")) for task in tasks]
    scan_sha = hashlib.sha256("".join(encoded).encode()).hexdigest()
    rows = [{"task_spec": text, "runtime": json.dumps({"task_machine": {}})} for text in encoded]
    pq.write_table(pa.Table.from_pylist(rows), specs_dir / "part-00000.parquet", compression="zstd")
    population = SourceTextPopulation(
        path=str(population_dir),
        source_id=SOURCE_ID,
        dataset_id="fixture",
        dataset_revision="1",
        source_version="1",
        converter_revision="1",
        text_renderer_version="1",
        task_count=len(tasks),
        text_sha256=scan_sha,
        task_specs_sha256=scan_sha,
        main_output_dir=str(tmp_path / "main"),
        specs_output_dir=str(specs_dir),
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=str(population_dir),
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return population, scan_sha


def _declaration(tmp_path: Path, scan_sha: str, measured_tasks: int = 2) -> QualifiedSourceDeclaration:
    return QualifiedSourceDeclaration(
        id=source_identifier(SourceOrigin.MARIN, "fixture", None, None),
        display_name="Marin · fixture",
        origin="Marin",
        build_path="fixture",
        upstream_dataset="fixture",
        upstream_name="fixture",
        current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=measured_tasks,
        within_source_dedup=deduplication_evidence("fixture", "1", "1", count=measured_tasks),
        notes="",
        dataset_id="fixture",
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
        actor_census_input=ActorCensusInput(
            dataset_id="fixture",
            dataset_revision="1",
            converter_version="1",
            converter_revision="1",
            measured_tasks=measured_tasks,
            task_specs_sha256=scan_sha,
            evidence_url=str(tmp_path / "population" / ".artifact.json"),
        ),
    )


def _seal_json(tmp_path: Path, name: str, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _source_registry(declaration):
    """A callable source registry with the single fixture source, like ``all_sources``."""
    source = SimpleNamespace(info=SimpleNamespace(id=SOURCE_ID, oracle_filtration=None), declaration=declaration)
    return lambda: {SOURCE_ID: source}


def _census_inputs(tmp_path: Path, declaration, revision: str) -> census_stage.CensusInputs:
    document = {
        "kind": census_stage.INPUTS_KIND,
        "source_revision": revision,
        "rows": [
            {
                "source_id": SOURCE_ID,
                "declaration": declaration.model_dump(mode="json"),
                "population_path": str(tmp_path / "population"),
            }
        ],
    }
    return census_stage.CensusInputs(
        release_id="testrelease",
        source_revision=revision,
        inputs=_seal_json(tmp_path, "census-inputs.json", document),
        output_root=str(tmp_path / "release" / "census"),
        region_prefix=str(tmp_path) + "/",
        expected_sources=1,
        slice_output_root=str(tmp_path / "slice" / "census"),
    )


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def test_sizing_curves_scale_from_population():
    anchor = sizing.census_sizing(sizing._ANCHOR_TASKS)
    assert (anchor.cpu, anchor.memory_gb, anchor.disk_gb, anchor.timeout_s) == (16, 64, 200, 14_400)
    small = sizing.census_sizing(100)
    assert small.cpu == 16 and small.disk_gb == 50 and small.timeout_s == 3_600
    embed = sizing.embedding_sizing(sizing._ANCHOR_TASKS)
    assert (embed.cpu, embed.memory_gb, embed.workers) == (64, 256, 16)
    compose = sizing.curriculum_compose_sizing(sizing._ANCHOR_COMPOSE_TASKS)
    assert (compose.cpu, compose.memory_gb, compose.workers) == (32, 128, 8)
    with pytest.raises(ValueError, match="positive"):
        sizing.census_sizing(0)


def test_census_preflight_slice_measures_one_real_source(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    population, scan_sha = _write_population(tmp_path, tasks)
    del population
    declaration = _declaration(tmp_path, scan_sha)
    inputs = _census_inputs(tmp_path, declaration, revision="a" * 40)
    feeds = _feeds(tmp_path)
    proof = census_stage.run_census(
        inputs, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=feeds
    )
    assert proof["kind"] == "atlas-testrelease-admission-population-census"
    assert proof["tasks"] == 2 and proof["sources"] == 1 and proof["dry_run"] is True
    assert not (tmp_path / "slice" / "census" / "receipt.json").exists()
    assert (tmp_path / "slice" / "census" / "results").is_dir()
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert rows and json.loads(rows[0])["kind"] == census_stage.alerts.KIND_START


def test_census_write_seals_receipt_then_refuses_a_rerun(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    inputs = _census_inputs(tmp_path, declaration, revision="a" * 40)
    proof = census_stage.run_census(
        inputs, dry_run=False, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
    )
    receipt_path = tmp_path / "release" / "census" / "receipt.json"
    assert proof["receipt"]["uri"] == str(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["tasks"] == 2 and receipt["sources"] == 1
    assert receipt["resources"]["cpu"] == 16
    assert receipt["results"][0]["inventory"]["retained_tasks"] == 2
    before = receipt_path.read_bytes()
    # The inventory root is write-once: a rerun must fail instead of restaging.
    with pytest.raises(FileExistsError):
        census_stage.run_census(inputs, dry_run=False, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path))
    assert receipt_path.read_bytes() == before


def test_census_preflight_requires_slice_and_slice_root(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    inputs = _census_inputs(tmp_path, declaration, revision="a" * 40)
    with pytest.raises(ValueError, match="--only"):
        census_stage.run_census(inputs, dry_run=True, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path))
    bare = census_stage.CensusInputs(
        release_id=inputs.release_id,
        source_revision=inputs.source_revision,
        inputs=inputs.inputs,
        output_root=inputs.output_root,
        region_prefix=inputs.region_prefix,
        expected_sources=1,
    )
    with pytest.raises(ValueError, match="slice_output_root"):
        census_stage.run_census(
            bare, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )
    with pytest.raises(ValueError, match="write-mode"):
        census_stage.run_census(
            inputs, dry_run=False, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )


def test_census_refuses_bad_region_pin_count_and_inputs(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    inputs = _census_inputs(tmp_path, declaration, revision="a" * 40)
    registry = _source_registry(declaration)

    def variant(**overrides) -> census_stage.CensusInputs:
        fields = dict(
            release_id=inputs.release_id,
            source_revision=inputs.source_revision,
            inputs=inputs.inputs,
            output_root=inputs.output_root,
            region_prefix=inputs.region_prefix,
            expected_sources=inputs.expected_sources,
            slice_output_root=inputs.slice_output_root,
        )
        fields.update(overrides)
        return census_stage.CensusInputs(**fields)

    with pytest.raises(ValueError, match="region prefix"):
        census_stage.run_census(
            variant(region_prefix="s3://elsewhere-bucket/"),
            dry_run=True,
            only=SOURCE_ID,
            sources_fn=registry,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="expected"):
        census_stage.run_census(variant(expected_sources=2), dry_run=False, sources_fn=registry, feeds=_feeds(tmp_path))
    with pytest.raises(ValueError, match="digest mismatch"):
        census_stage.run_census(
            variant(inputs=ArtifactRef(uri=inputs.inputs.uri, sha256="0" * 64)),
            dry_run=True,
            only=SOURCE_ID,
            sources_fn=registry,
            feeds=_feeds(tmp_path),
        )


def test_census_refuses_unclassified_static_blocker(tmp_path, monkeypatch):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    inputs = _census_inputs(tmp_path, declaration, revision="a" * 40)

    def unclassified(task):
        return {"some_novel_blocker"}

    monkeypatch.setattr(census_stage, "task_blockers", unclassified)
    with pytest.raises(ValueError, match="unclassified static blockers"):
        census_stage.run_census(
            inputs, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )


def test_census_refuses_declaration_drift(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    document = {
        "kind": census_stage.INPUTS_KIND,
        "source_revision": "a" * 40,
        "rows": [
            {
                "source_id": SOURCE_ID,
                "declaration": {**declaration.model_dump(mode="json"), "task_count": 3},
                "population_path": str(tmp_path / "population"),
            }
        ],
    }
    inputs = census_stage.CensusInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        inputs=_seal_json(tmp_path, "drifted-inputs.json", document),
        output_root=str(tmp_path / "release" / "census"),
        region_prefix=str(tmp_path) + "/",
        expected_sources=1,
        slice_output_root=str(tmp_path / "slice" / "census"),
    )
    with pytest.raises(ValueError, match="does not match the current declaration"):
        census_stage.run_census(
            inputs, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )


def test_census_refuses_declared_count_drift(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    # The population carries two tasks while the declaration's census binding claims
    # three; the inventory verifies the binding before anything is measured.
    declaration = _declaration(tmp_path, scan_sha, measured_tasks=3)
    with pytest.raises(ValueError, match="differs from the current declaration"):
        census_stage.run_census(
            _census_inputs(tmp_path, declaration, revision="a" * 40),
            dry_run=False,
            sources_fn=_source_registry(declaration),
            feeds=_feeds(tmp_path),
        )


def _corpus_inputs(
    tmp_path: Path, declaration, census_receipt: ArtifactRef, revision: str
) -> corpus_stage.CorpusAssemblyInputs:
    document = {
        "kind": census_stage.INPUTS_KIND,
        "source_revision": revision,
        "rows": [
            {
                "source_id": SOURCE_ID,
                "declaration": declaration.model_dump(mode="json"),
                "population_path": str(tmp_path / "population"),
            }
        ],
    }
    return corpus_stage.CorpusAssemblyInputs(
        release_id="testrelease",
        source_revision=revision,
        inputs=_seal_json(tmp_path, "corpus-inputs.json", document),
        census_receipt=census_receipt,
        output_root=str(tmp_path / "release" / "corpus"),
        expected_sources=1,
        slice_output_root=str(tmp_path / "slice" / "corpus"),
        workers=1,
    )


def test_corpus_assembly_end_to_end_after_census(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    revision = "a" * 40
    census_inputs = _census_inputs(tmp_path, declaration, revision)
    census_proof = census_stage.run_census(
        census_inputs, dry_run=False, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
    )
    receipt = ArtifactRef(uri=census_proof["receipt"]["uri"], sha256=census_proof["receipt"]["sha256"])

    inputs = _corpus_inputs(tmp_path, declaration, receipt, revision)
    registry = _source_registry(declaration)
    slice_proof = corpus_stage.run_corpus_assembly(
        inputs, dry_run=True, only=SOURCE_ID, sources_fn=registry, feeds=_feeds(tmp_path)
    )
    assert slice_proof["tasks"] == 2 and slice_proof["dry_run"] is True
    assert not (tmp_path / "slice" / "corpus" / "corpus-proof.json").exists()

    proof = corpus_stage.run_corpus_assembly(inputs, dry_run=False, sources_fn=registry, feeds=_feeds(tmp_path))
    assert proof["kind"] == "atlas-testrelease-curriculum-corpus"
    assert proof["tasks"] == 2 and proof["sources"] == 1
    corpus = json.loads((tmp_path / "release" / "corpus" / "corpus.json").read_text())
    assert corpus["population"] == "converted"
    assert sum(row["task_count"] for row in corpus["inventory"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        corpus_stage.run_corpus_assembly(inputs, dry_run=False, sources_fn=registry, feeds=_feeds(tmp_path))


def test_corpus_assembly_refuses_stale_census_receipt(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    stale_receipt = _seal_json(
        tmp_path,
        "stale-census.json",
        {"kind": "atlas-otherrelease-admission-population-census", "results": []},
    )
    inputs = _corpus_inputs(tmp_path, declaration, stale_receipt, revision="a" * 40)
    with pytest.raises(ValueError, match="kind"):
        corpus_stage.run_corpus_assembly(
            inputs, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )


def test_corpus_assembly_refuses_a_pinned_scan_drift(tmp_path):
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    receipt = _seal_json(
        tmp_path,
        "census.json",
        {
            "kind": "atlas-testrelease-admission-population-census",
            "results": [
                {
                    "source_id": SOURCE_ID,
                    "population": {"specs_output_dir": str(tmp_path / "specs")},
                    "inventory": {"scanned_task_specs_sha256": "b" * 64},
                }
            ],
        },
    )
    inputs = _corpus_inputs(tmp_path, declaration, receipt, revision="a" * 40)
    # prepare_sources verifies the frozen scan digest; the census receipt pins a wrong
    # one, so the assembly must refuse before any corpus is written.
    with pytest.raises(ValueError, match="scan changed"):
        corpus_stage.run_corpus_assembly(
            inputs, dry_run=True, only=SOURCE_ID, sources_fn=_source_registry(declaration), feeds=_feeds(tmp_path)
        )


def test_census_records_an_empty_oracle_filtration(tmp_path):
    """A 0-admission oracle-filtered source ships the excluded, zero-task declaration.

    The census records the empty measurement from the oracle receipt without loading
    the pinned population, drawing an inventory, or preparing public inputs — the
    prior releases carried the same shape (the corpus omits these sources).
    """
    evidence = SimpleNamespace(
        original_count=13562,
        exclusion_counts={"failed_oracle": 13562},
        filtered_release=SimpleNamespace(repo="open-athena/filtered", revision="7af7"),
        validation=SimpleNamespace(uri="s3://bucket/validation"),
        admitted_count=0,
    )
    declaration = _declaration(tmp_path, "b" * 64)
    source = SimpleNamespace(info=SimpleNamespace(id=SOURCE_ID, oracle_filtration=evidence), declaration=declaration)
    row = census_stage.CensusSourceRow(
        source_id=SOURCE_ID,
        declaration=declaration.model_dump(mode="json"),
        population_path=str(tmp_path / "absent-population"),
    )
    result = census_stage.measure_source(
        row,
        str(tmp_path / "release" / "census"),
        "a" * 40,
        str(tmp_path) + "/",
        sources_fn=lambda: {SOURCE_ID: source},
        inventory_fn=lambda *args, **kwargs: pytest.fail("the empty census draws no inventory"),
    )
    assert result["oracle_empty"]["original_count"] == 13562
    assert result["oracle_empty"]["filtered_release"]["repo"] == "open-athena/filtered"
    assert result["inventory"] is None and result["population"] is None
    assert result["public_preparation"] is None and result["oracle_tasks"] == 0


def test_corpus_omits_empty_oracle_sources(tmp_path):
    """The corpus covers the shipped sources only; empty oracle filtrations ride as
    excluded, zero-task declarations without corpus presence."""
    tasks = [_task_spec("fixture-task-one"), _task_spec("fixture-task-two")]
    _, scan_sha = _write_population(tmp_path, tasks)
    declaration = _declaration(tmp_path, scan_sha)
    empty_evidence = SimpleNamespace(
        original_count=10,
        exclusion_counts={"failed_oracle": 10},
        filtered_release=SimpleNamespace(repo="open-athena/filtered", revision="7af7"),
        validation=SimpleNamespace(uri="s3://bucket/validation"),
        admitted_count=0,
    )
    empty_source = SimpleNamespace(
        info=SimpleNamespace(id=EMPTY_SOURCE_ID, oracle_filtration=empty_evidence), declaration=declaration
    )
    revision = "a" * 40
    document = {
        "kind": census_stage.INPUTS_KIND,
        "source_revision": revision,
        "rows": [
            {
                "source_id": SOURCE_ID,
                "declaration": declaration.model_dump(mode="json"),
                "population_path": str(tmp_path / "population"),
            },
            {
                "source_id": EMPTY_SOURCE_ID,
                "declaration": declaration.model_dump(mode="json"),
                "population_path": str(tmp_path / "absent"),
            },
        ],
    }
    census_inputs = census_stage.CensusInputs(
        release_id="testrelease",
        source_revision=revision,
        inputs=_seal_json(tmp_path, "census-inputs.json", document),
        output_root=str(tmp_path / "release" / "census"),
        region_prefix=str(tmp_path) + "/",
        expected_sources=2,
        slice_output_root=str(tmp_path / "slice" / "census"),
    )
    census_proof = census_stage.run_census(
        census_inputs,
        dry_run=False,
        sources_fn=lambda: {SOURCE_ID: _source_registry(declaration)()[SOURCE_ID], EMPTY_SOURCE_ID: empty_source},
        feeds=_feeds(tmp_path),
    )
    assert census_proof["tasks"] == 2 and census_proof["sources"] == 2
    receipt = ArtifactRef(uri=census_proof["receipt"]["uri"], sha256=census_proof["receipt"]["sha256"])

    inputs = corpus_stage.CorpusAssemblyInputs(
        release_id="testrelease",
        source_revision=revision,
        inputs=_seal_json(tmp_path, "corpus-inputs.json", document),
        census_receipt=receipt,
        output_root=str(tmp_path / "release" / "corpus"),
        expected_sources=2,
        slice_output_root=str(tmp_path / "slice" / "corpus"),
        workers=1,
    )
    proof = corpus_stage.run_corpus_assembly(
        inputs,
        dry_run=False,
        sources_fn=lambda: {SOURCE_ID: _source_registry(declaration)()[SOURCE_ID], EMPTY_SOURCE_ID: empty_source},
        feeds=_feeds(tmp_path),
    )
    assert proof["sources"] == 1 and proof["tasks"] == 2
    corpus = json.loads((tmp_path / "release" / "corpus" / "corpus.json").read_text())
    assert len(corpus["inventory"]) == 1
    assert sum(row["task_count"] for row in corpus["inventory"]) == 2
