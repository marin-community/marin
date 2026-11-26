# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained embedding and curriculum-compose stages (R40 gate).

The compose plan assertions run against real sealed release/coverage documents built
from the projection fixtures' shapes; the write path (Zephyr execution) is exercised
by the M3 preflight on real inputs, not by these tests.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import QualifiedSourceDeclaration, SourceOrigin, source_identifier
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.curriculum.coverage import CONFIDENCE_LEVELS
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, AlignmentSource, CurriculumCoverage

from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.release_stages import curriculum_compose as compose
from experiments.post_training.task_curation.release_stages import embed as embed_stage
from experiments.post_training.task_curation.release_stages.graph import _fresh_corpus_expectation
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

NODES = [
    {
        "subject_id": "math",
        "node_id": "subject:math",
        "parent_id": None,
        "kind": "subject",
        "name": "Math",
        "description": "Mathematics",
    },
    {
        "subject_id": "math",
        "node_id": "add",
        "parent_id": "subject:math",
        "kind": "capability",
        "name": "Addition",
        "description": "Add numbers",
    },
]
EMBEDDING_SPEC = {
    "model": "unused",
    "revision": "1",
    "tokenizer_revision": "1",
    "dimensions": 2,
    "max_tokens": 16,
    "compute_dtype": "float32",
}


def _declaration(identifier: str, revision: str = "1") -> QualifiedSourceDeclaration:
    return QualifiedSourceDeclaration(
        id=source_identifier(SourceOrigin.MARIN, identifier, None, None),
        display_name=f"Marin · {identifier}",
        origin="Marin",
        build_path="fixture",
        upstream_dataset=identifier,
        upstream_name=identifier,
        current_release={"repo": "fixture", "revision": revision, "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=3,
        within_source_dedup=deduplication_evidence("fixture", revision, "1", count=3),
        notes="",
        dataset_id="fixture",
        dataset_revision=revision,
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": revision, "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
    )


def _source(identifier: str, declaration, *, task_count: int = 3) -> AlignmentSource:
    return AlignmentSource(
        source_id=declaration.id,
        source_key=canonical_sha256(identifier),
        population="converted",
        intended_use="train",
        task_count=task_count,
        ready_count=task_count,
        insufficient_context_count=0,
        basis="matched",
        curation_root=str(Path(identifier)),
        input_rows=task_count,
        converted_rows=task_count,
    )


def _statistics(paths_root: Path, name: str, source_ids: list[str], offset: int) -> list[str]:
    statistics = [
        {"source_id": sid, "metric": metric, "subject_id": subject, "node_id": node, "instances": 1 + offset}
        for sid in source_ids
        for metric, subject, node in (
            ("denominator", "", ""),
            ("assigned", "", ""),
            ("placement", "math", "subject:math"),
            ("placement", "math", "add"),
            ("capability", "math", "subject:math"),
        )
    ]
    path = paths_root / name
    pq.write_table(pa.Table.from_pylist(statistics), path)
    return [str(path)]


def _release(tmp_path: Path, identifier: str, declaration, source, *, task_count: int) -> AlignmentRelease:
    assignments = tmp_path / f"{identifier}-assignments.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=pa.schema([("source_id", pa.string())])), assignments)
    lineage = {
        "alignment_root": str(tmp_path),
        "task_vectors_root": f"{identifier}-vectors",
        "curriculum_vectors_root": f"{identifier}-anchors",
    }
    return AlignmentRelease(
        release_id=canonical_sha256(
            {
                "lineage": lineage,
                "sources": [source.model_dump(mode="json")],
                "inventory": [declaration.model_dump(mode="json")],
            }
        ),
        created_at="2026-11-13T00:00:00Z",
        curriculum_key="same-curriculum",
        population="converted",
        sample_size=None,
        task_count=task_count,
        ready_count=task_count,
        insufficient_context_count=0,
        sources=[source],
        inventory=[declaration],
        nodes=NODES,
        paths=[str(assignments)],
        lineage=lineage,
        embedding_spec=EMBEDDING_SPEC,
    )


def _coverage(
    release: AlignmentRelease,
    statistics_paths: list[str],
    confidence_paths: list[str],
    *,
    distance: float = 0.45,
) -> CurriculumCoverage:
    identity = {
        "alignment_release_id": release.release_id,
        "subject_distance": distance,
        "capability_distance": 0.25,
        "source_sample_size": 1,
        "area_sample_size": 1,
        "version": COVERAGE_VERSION,
        "confidence_levels": [
            {"label": label, "subject_min_similarity": subject, "capability_min_similarity": capability}
            for label, subject, capability in CONFIDENCE_LEVELS
        ],
    }
    return CurriculumCoverage(
        **identity,
        coverage_id=canonical_sha256(identity),
        curriculum_key=release.curriculum_key,
        task_count=release.task_count,
        paths=statistics_paths,
        sample_paths=[],
        sample_task_count=0,
        confidence_paths=confidence_paths,
    )


def _seal_json(tmp_path: Path, name: str, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _composition(
    tmp_path: Path, *, drift=None, overlap=False, param_change=None, expected_tasks=2, removed=False, replacement=False
) -> tuple[dict, int]:
    """A sealed composition manifest over a one-source retained release plus one fresh.

    ``drift`` crafts a declaration copy in the current-target document that differs
    from the fresh inventory's own copy — the alignment-time inventory stays valid,
    the current declaration drifted. ``removed`` adds a second retained source the
    plan removes.
    """
    old_decl = _declaration("retained")
    fresh_decl = _declaration("admitted")
    if replacement:
        fresh_decl = _declaration("retained", "2").model_copy(
            update={"task_count": 2, "within_source_dedup": deduplication_evidence("fixture", "2", "1", count=2)}
        )
    removed_decl = _declaration("removed")
    current_decl = fresh_decl
    if drift == "task_count":
        current_decl = fresh_decl.model_copy(
            update={
                "task_count": 4,
                "within_source_dedup": deduplication_evidence("fixture", "1", "1", count=4),
            }
        )
    if drift == "display_name":
        current_decl = fresh_decl.model_copy(update={"url": "https://example.org/renamed"})
    old_source = _source("retained", old_decl)
    removed_source = _source("removed", removed_decl, task_count=1)
    fresh_source = _source("admitted", fresh_decl, task_count=2)
    if overlap:
        fresh_source = fresh_source.model_copy(update={"source_id": old_decl.id, "source_key": old_source.source_key})
    old_sources = [old_source] + ([removed_source] if removed else [])
    old_inventory = [old_decl] + ([removed_decl] if removed else [])
    old_task_count = 3 + (1 if removed else 0)
    old = _release(tmp_path, "retained", old_decl, old_source, task_count=old_task_count)
    if removed:
        old = old.model_copy(update={"sources": old_sources, "inventory": old_inventory, "task_count": old_task_count})
        old = compose.AlignmentRelease.model_validate(old.model_dump(mode="json"))
    fresh = _release(tmp_path, "admitted", fresh_decl, fresh_source, task_count=2)
    old_stats = _statistics(
        tmp_path, "old-statistics.parquet", [old_decl.id] + ([removed_decl.id] if removed else []), offset=0
    )
    fresh_stats = _statistics(tmp_path, "fresh-statistics.parquet", [fresh_source.source_id], offset=10)
    old_cov = _coverage(old, old_stats, old_stats)
    fresh_cov = _coverage(fresh, fresh_stats, fresh_stats, distance=0.45)
    if param_change:
        fresh_cov = fresh_cov.model_copy(update={"subject_distance": 0.9})
    if drift == "insufficient":
        fresh = fresh.model_copy(update={"sources": [fresh_source.model_copy(update={"insufficient_context_count": 1})]})
        fresh_cov = fresh_cov.model_copy(update={"alignment_release_id": fresh.release_id})
    targets_document = {current_decl.id: current_decl.model_dump(mode="json")}
    document = {
        "old_release": {
            "uri": str(tmp_path / "old-release.json"),
            "sha256": hashlib.sha256(json.dumps(old.model_dump(mode="json")).encode()).hexdigest(),
        },
        "old_coverage": {
            "uri": str(tmp_path / "old-coverage.json"),
            "sha256": hashlib.sha256(json.dumps(old_cov.model_dump(mode="json")).encode()).hexdigest(),
        },
        "fresh_release": {
            "uri": str(tmp_path / "fresh-release.json"),
            "sha256": hashlib.sha256(json.dumps(fresh.model_dump(mode="json")).encode()).hexdigest(),
        },
        "fresh_coverage": {
            "uri": str(tmp_path / "fresh-coverage.json"),
            "sha256": hashlib.sha256(json.dumps(fresh_cov.model_dump(mode="json")).encode()).hexdigest(),
        },
        "current_target_declarations": {
            "uri": str(tmp_path / "targets.json"),
            "sha256": hashlib.sha256(json.dumps(targets_document).encode()).hexdigest(),
        },
    }
    (tmp_path / "old-release.json").write_text(json.dumps(old.model_dump(mode="json")))
    (tmp_path / "old-coverage.json").write_text(json.dumps(old_cov.model_dump(mode="json")))
    (tmp_path / "fresh-release.json").write_text(json.dumps(fresh.model_dump(mode="json")))
    (tmp_path / "fresh-coverage.json").write_text(json.dumps(fresh_cov.model_dump(mode="json")))
    (tmp_path / "targets.json").write_text(json.dumps(targets_document))
    return document, expected_tasks


def _compose_inputs(tmp_path: Path, document: dict, expected_tasks: int) -> compose.CurriculumComposeInputs:
    return compose.CurriculumComposeInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        inputs=_seal_json(tmp_path, "composition-inputs.json", document),
        output_root=str(tmp_path / "release" / "compose"),
        expected_fresh_task_count=expected_tasks,
        removed_sources=(),
    )


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def test_compose_plan_accepts_an_additions_manifest(tmp_path):
    document, expected = _composition(tmp_path)
    plan = compose.compose_plan(document, expected, ())
    assert len(plan.targets) == 1
    assert plan.fresh_task_count == 2 and plan.retained_task_count == 3
    assert len(plan.sources) == 2 and len(plan.inventory) == 2


def test_compose_replaces_a_changed_population_without_carrying_its_old_count(tmp_path):
    document, expected = _composition(tmp_path, replacement=True, removed=True)
    plan = compose.compose_plan(document, expected, ())
    release = compose.compose_release(plan, document, {"uri": "/inputs.json", "sha256": "a" * 64})
    assert {row.source_id: row.task_count for row in release.sources} == {"Marin:retained": 2, "Marin:removed": 1}
    assert release.task_count == 3
    inventory = {row.id: row for row in release.inventory}
    assert inventory["Marin:retained"].dataset_revision == "2"
    assert plan.retained_task_count == 1
    assert plan.replaced_sources == {"Marin:retained"}


def test_compose_plan_drops_planned_removals_from_the_retained_curriculum(tmp_path):
    """A planned removal never survives into the composed release."""
    document, expected = _composition(tmp_path)
    retained_id = _declaration("retained").id
    plan = compose.compose_plan(document, expected, (retained_id,))
    assert plan.removed_sources == (retained_id,)
    assert [source.source_id for source in plan.sources] == list(plan.targets)
    assert [declaration.id for declaration in plan.inventory] == list(plan.targets)
    assert plan.retained_task_count == 0
    release = compose.compose_release(plan, document, {"uri": "/sealed/inputs.json", "sha256": "c" * 64})
    assert release.task_count == 2 and retained_id not in {source.source_id for source in release.sources}
    with pytest.raises(ValueError, match="absent from the retained curriculum"):
        compose.compose_plan(document, expected, ("Marin:never-retained",))


def test_compose_plan_reads_an_earlier_generation_retained_inventory(tmp_path):
    """A retained release sealed in the r39-era declaration form (with the removed name) upgrades."""
    document, expected = _composition(tmp_path)
    old_path = tmp_path / "old-release.json"
    old = json.loads(old_path.read_text())
    old["inventory"][0]["name"] = "retained-legacy-name"
    old_path.write_text(json.dumps(old))
    document["old_release"]["sha256"] = hashlib.sha256(old_path.read_bytes()).hexdigest()
    plan = compose.compose_plan(document, expected, ())
    assert plan.retained_task_count == 3 and "name" not in plan.inventory[0].model_dump(mode="json")


def test_compose_release_carries_lineage_and_totals(tmp_path):
    document, expected = _composition(tmp_path)
    plan = compose.compose_plan(document, expected, ())
    release = compose.compose_release(plan, document, {"uri": "/sealed/inputs.json", "sha256": "c" * 64})
    assert release.task_count == 5
    assert release.lineage["composition_inputs_uri"] == "/sealed/inputs.json"
    assert release.lineage["fresh_release_sha256"]
    assert len(release.paths) == 2
    assert compose.compose_coverage_identity(plan, release)["version"] == COVERAGE_VERSION


@pytest.mark.parametrize(
    ("drift", "overlap", "param_change", "expected_tasks", "message"),
    [
        ("task_count", False, None, 2, "changed its census binding or task count"),
        ("display_name", False, None, 2, "changed non-consumer fields"),
        ("insufficient", False, None, 2, "insufficient-context"),
        (None, True, None, 2, "current target declarations do not cover exactly the fresh and rebound inventory"),
        (None, False, "subject_distance", 2, "disagree on subject_distance"),
        (None, False, None, 3, "expected 3"),
    ],
)
def test_compose_plan_refusals(tmp_path, drift, overlap, param_change, expected_tasks, message):
    document, _ = _composition(
        tmp_path, drift=drift, overlap=overlap, param_change=param_change, expected_tasks=expected_tasks
    )
    with pytest.raises(ValueError, match=message):
        compose.compose_plan(document, expected_tasks, ())


def test_compose_run_dry_run_writes_nothing(tmp_path):
    document, expected = _composition(tmp_path)
    inputs = _compose_inputs(tmp_path, document, expected)
    proof = compose.run_compose(inputs, dry_run=True, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["composed_task_count"] == 5
    assert not (tmp_path / "release" / "compose").exists() or not any((tmp_path / "release" / "compose").iterdir())
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["kind"] == compose.alerts.KIND_START
    assert json.loads(rows[-1])["kind"] == compose.alerts.KIND_FINISH


def test_compose_run_refuses_bad_inputs_and_alerts(tmp_path):
    document, expected = _composition(tmp_path)
    inputs = compose.CurriculumComposeInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        inputs=ArtifactRef(uri=document["old_release"]["uri"], sha256="0" * 64),
        output_root=str(tmp_path / "release" / "compose"),
        expected_fresh_task_count=expected,
        removed_sources=(),
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        compose.run_compose(inputs, dry_run=True, feeds=_feeds(tmp_path))
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["kind"] == compose.alerts.KIND_FAIL
    empty = compose.CurriculumComposeInputs(
        release_id="testrelease",
        source_revision="a" * 40,
        inputs=ArtifactRef(uri=str(tmp_path / "missing.json"), sha256="0" * 64),
        output_root=str(tmp_path / "release" / "compose"),
        expected_fresh_task_count=0,
        removed_sources=(),
    )
    with pytest.raises(ValueError, match="positive"):
        compose.run_compose(empty, dry_run=True, feeds=_feeds(tmp_path))


def _embed_inputs(tmp_path: Path, corpus: dict, **overrides) -> embed_stage.EmbedInputs:
    defaults = dict(
        release_id="testrelease",
        source_revision="a" * 40,
        corpus=_seal_json(tmp_path, "corpus.json", corpus),
        artifact_root=str(tmp_path / "release" / "artifacts"),
        cache_root=str(tmp_path / "release" / "cache"),
        expected_source_count=1,
        expected_task_count=2,
    )
    defaults.update(overrides)
    return embed_stage.EmbedInputs(**defaults)


def _corpus_document() -> dict:
    declaration = _declaration("retained").model_dump(mode="json")
    declaration["task_count"] = 2
    return {
        "selection_key": canonical_sha256("fixture"),
        "population": "converted",
        "sources": [{"source_id": declaration["id"], "root": "/unused"}],
        "inventory": [declaration],
        "prepared_inputs": "/unused/inputs",
    }


def test_embed_inputs_pair_parity_and_runtime(tmp_path):
    parity = ArtifactRef(uri=str(tmp_path / "parity.json"), sha256="0" * 64)
    with pytest.raises(ValueError, match="parity receipt and the typed accelerator"):
        _embed_inputs(tmp_path, _corpus_document(), parity_receipt=parity).validate()
    with pytest.raises(ValueError, match="parity receipt and the typed accelerator"):
        _embed_inputs(tmp_path, _corpus_document(), gpu_runtime="h100").validate()
    inputs = _embed_inputs(tmp_path, _corpus_document(), parity_receipt=parity, gpu_runtime="h100")
    inputs.validate()
    assert inputs.device == "cuda"
    cpu = _embed_inputs(tmp_path, _corpus_document())
    cpu.validate()
    assert cpu.device == "cpu"


def test_embed_verifies_corpus_counts_and_digest(tmp_path):
    corpus = _corpus_document()
    inputs = _embed_inputs(tmp_path, corpus)
    assert embed_stage.verify_corpus(corpus, inputs) == 2
    short = {**corpus, "inventory": [{**corpus["inventory"][0], "task_count": 1}]}
    with pytest.raises(ValueError, match="census total"):
        embed_stage.verify_corpus(short, inputs)
    wrong_population = {**corpus, "population": "raw"}
    with pytest.raises(ValueError, match="converted"):
        embed_stage.verify_corpus(wrong_population, inputs)
    tampered = embed_stage.EmbedInputs(
        release_id=inputs.release_id,
        source_revision=inputs.source_revision,
        corpus=ArtifactRef(uri=inputs.corpus.uri, sha256="0" * 64),
        artifact_root=inputs.artifact_root,
        cache_root=inputs.cache_root,
        expected_source_count=1,
        expected_task_count=2,
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        embed_stage.read_corpus(tampered.corpus)


def test_embed_dry_run_records_plan_without_executing(tmp_path):
    corpus = _corpus_document()
    inputs = _embed_inputs(tmp_path, corpus)
    proof = embed_stage.run_embed(inputs, dry_run=True, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["tasks"] == 2 and proof["device"] == "cpu"
    assert proof["workers"] >= 2 and proof["resources"]["cpu"] >= 8
    assert not any((tmp_path / "release" / "artifacts").rglob("*"))
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["kind"] == embed_stage.alerts.KIND_FINISH


def test_fresh_corpus_expectation_counts_the_shipped_targets():
    """Embed's expectation is the shipped fresh set: zero-admission targets ship nothing.

    The census proof's per-source results carry the oracle-empty marker; the
    expectation counts only the targets the census did not filter to empty and
    sums their retained tasks.
    """
    plan = fixture_plan(targets=("Marin:fixture", "Marin:fixture-empty"), expected_population_tasks=10)
    census_proof = {
        "results": [
            {"source_id": "Marin:fixture", "inventory": {"retained_tasks": 4}},
            {
                "source_id": "Marin:fixture-empty",
                "oracle_empty": {"original_count": 7},
                "inventory": {"retained_tasks": 0},
            },
            {"source_id": "Marin:not-a-target", "inventory": {"retained_tasks": 99}},
            # A non-target zero-admission source carries no census inventory at all;
            # curriculum compose reads the same expectation (R40 refused on this row).
            {"source_id": "Marin:empty-non-target", "oracle_empty": {"original_count": 3}, "inventory": None},
        ]
    }
    expectation = _fresh_corpus_expectation(plan, census_proof)
    assert expectation == {"expected_source_count": 1, "expected_task_count": 4}


def test_end_to_end_targets_scoped_and_removals_dropped(tmp_path):
    """The structural-scoping chain on a small fixture plan.

    Embed's expectation counts only the plan's targets (the removed source's
    census row is oracle-empty and a non-target row is ignored); the compose drops
    the planned removal from the retained curriculum so nothing survives into the
    composed release; and a removed source that would survive — as a fresh target
    — refuses.
    """
    plan = fixture_plan(targets=("Marin:admitted",), removed_sources=("Marin:removed",))
    census_proof = {
        "results": [
            {"source_id": "Marin:admitted", "inventory": {"retained_tasks": 2}},
            {
                "source_id": "Marin:removed",
                "oracle_empty": {"original_count": 5},
                "inventory": {"retained_tasks": 0},
            },
            {"source_id": "Marin:not-in-the-plan", "inventory": {"retained_tasks": 99}},
        ]
    }
    # 1. Only targets are embedded: the shipped fresh set is the one target.
    expectation = _fresh_corpus_expectation(plan, census_proof)
    assert expectation == {"expected_source_count": 1, "expected_task_count": 2}

    # 2. The compose drops the planned removal: the composed release carries only
    #    the retained survivor and the fresh target.
    document, expected = _composition(tmp_path, removed=True)
    removed_id = _declaration("removed").id
    compose_plan = compose.compose_plan(document, expected, (removed_id,))
    assert removed_id not in {source.source_id for source in compose_plan.sources}
    assert removed_id not in {declaration.id for declaration in compose_plan.inventory}
    assert compose_plan.retained_task_count == 3
    release = compose.compose_release(compose_plan, document, {"uri": "/sealed/inputs.json", "sha256": "c" * 64})
    assert release.task_count == 5 and removed_id not in {source.source_id for source in release.sources}

    # 3. A removed source left in retained fails: one that is also a fresh target
    #    cannot be dropped from the retained curriculum, so the compose refuses.
    overlap_document, overlap_expected = _composition(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="also a planned removal"):
        compose.compose_plan(overlap_document, overlap_expected, (_declaration("retained").id,))


def test_archived_assignment_counts_cover_planned_removals(tmp_path):
    """The filter counts the archive before projection, so a removed source still counts."""
    document, _ = _composition(tmp_path, removed=True)
    archived = compose.read_retained_release(document["old_release"]).sources
    retained_id, removed_id = _declaration("retained").id, _declaration("removed").id
    files = [
        {"counts": [{"source_id": retained_id, "status": "ready", "count": 3}]},
        {"counts": [{"source_id": removed_id, "status": "ready", "count": 1}]},
    ]
    compose.check_archived_assignment_counts(files, archived)
    with pytest.raises(ValueError, match="retained assignment counts disagree"):
        compose.check_archived_assignment_counts(files[:1], archived)
    files[0]["counts"][0]["status"] = "insufficient_context"
    with pytest.raises(ValueError, match="readiness differs"):
        compose.check_archived_assignment_counts(files, archived)


def test_confidence_export_binds_the_release_and_coverage_ids(tmp_path):
    """The curriculum load's confidence reader binds the document by release and coverage id."""
    document, _ = _composition(tmp_path)
    fresh = compose.AlignmentRelease.model_validate_json(Path(document["fresh_release"]["uri"]).read_text())
    coverage = compose.CurriculumCoverage.model_validate_json(Path(document["fresh_coverage"]["uri"]).read_text())
    (source,) = fresh.sources
    vectors = tmp_path / "confidence.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"source_id": source.source_id, "metric": metric, "subject_id": "", "node_id": "", "instances": [2] * 5}
                for metric in ("denominator", "assigned")
            ]
        ),
        vectors,
    )
    coverage = coverage.model_copy(update={"confidence_paths": [str(vectors)]})
    release_reference = {"uri": "memory://compose/release.json", "sha256": "a" * 64, "release_id": fresh.release_id}
    coverage_reference = {
        "uri": "memory://compose/coverage.json",
        "sha256": "b" * 64,
        "coverage_id": coverage.coverage_id,
    }
    ref, validated = compose.confidence_export(
        str(tmp_path / "composed"), fresh, coverage, release_reference, coverage_reference
    )
    exported = json.loads(Path(ref.uri).read_text())
    assert (exported["release"], exported["coverage"]) == (release_reference, coverage_reference)
    assert validated > 0
