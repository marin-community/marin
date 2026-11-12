# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical scans use every public conversation and never private grader resources."""

import json
from contextlib import ExitStack

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.decon import AllSourceDropSets, NGramConfig, build_eval_bloom, decon_to_parquet
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.lexical_artifacts import (
    LexicalDropSets,
    SourceLexicalMarks,
    SourceTextPopulation,
)
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.lexical_scan import ReportConfig, measure_marks
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.published_census import CensusConfig, ContractCensus, measure_population
from experiments.post_training.task_curation.published_integrity import measure_integrity


def lexical_task(identifier: str, public: str, private: str) -> dict:
    return TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="pin", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=public),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="The fixture has no executable grader"),
        resources=ResourceGroups(oracle=(inline_resource("private.txt", private.encode()),)),
    ).model_dump(mode="json")


def test_full_task_text_marks_public_eval_copy_and_excludes_private_reference(tmp_path):
    leaked = " ".join(f"DistinctiveEvalWord{index}" for index in range(40))
    unrelated = " ".join(f"IndependentTaskWord{index}" for index in range(40))
    first = lexical_task("public-copy", leaked, "private witness")
    second = lexical_task("private-only", unrelated, leaked)
    population = write_text_population([(first, {}), (second, {})], str(tmp_path / "normalized"))
    assert population.task_count == 2
    texts = pq.read_table(tmp_path / "normalized/outputs/main/part-000000.parquet").to_pylist()
    assert {r["id"]: r["text"] for r in texts} == {
        "public-copy": "User:\n" + leaked,
        "private-only": "User:\n" + unrelated,
    }
    specs = pq.read_table(tmp_path / "normalized/specs/part-000000.parquet").to_pylist()
    assert {json.loads(r["task_spec"])["id"] for r in specs} == {"public-copy", "private-only"}
    evals = tmp_path / "evals"
    evals.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": "eval-control", "text": leaked}]), evals / "eval.parquet")
    config = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)
    with ExitStack() as cleanup:
        client = LocalClient()
        cleanup.callback(client.shutdown)
        resources = ResourceConfig(cpu=1, ram="1g")
        with ZephyrContext(client=client, max_workers=1, resources=resources) as context:
            bloom = build_eval_bloom(
                eval_data_sources=[str(evals)],
                output_path=str(tmp_path / "bloom"),
                ngram=config,
                estimated_doc_count=10000,
                max_workers=1,
                worker_resources=resources,
                zephyr_context=context,
            )
            result = decon_to_parquet(
                normalized_data=NormalizedData(
                    main_output_dir=population.main_output_dir, dup_output_dir="", counters={}
                ),
                prebuilt_bloom_dir=bloom.bloom_dir,
                output_path=str(tmp_path / "marks"),
                ngram=config,
                max_workers=1,
                worker_resources=resources,
                zephyr_context=context,
            )
    flags = pq.read_table(tmp_path / "marks/outputs/main/part-000000.parquet").to_pylist()
    assert {r["id"]: r["contaminated"] for r in flags} == {"public-copy": True, "private-only": False}
    assert result.counters["decon/contaminated"] == result.counters["decon/clean"] == 1

    with artifact_prefix(str(tmp_path)):
        # Persist the same artifacts consumed by the report; no gallery/sample path is used.
        text_population = SourceTextPopulation(
            path=str(tmp_path / "normalized"),
            source_id="fixture",
            dataset_id="fixture/dataset",
            dataset_revision="pin",
            source_version="1",
            converter_revision="converter-pin",
            text_renderer_version="renderer-v1",
            task_count=population.task_count,
            text_sha256=population.text_sha256,
            task_specs_sha256=population.task_specs_sha256,
            main_output_dir=population.main_output_dir,
            specs_output_dir=population.specs_output_dir,
            converter_rejections={},
            admission_rejections={},
        )
        write_record(
            ArtifactRecord(
                output_path=text_population.path,
                result_type=result_type_name(type(text_population)),
                result=text_population.result_payload(),
            )
        )
        refreshed = measure_integrity(CensusConfig(text_population.path, str(tmp_path / "integrity-refresh")))
        assert refreshed.task_count == 2
        assert refreshed.summary["required_shared_tasks"] == 0
        assert refreshed.summary["absent_contract_tasks"] == 2
        marks = SourceLexicalMarks(
            path=str(tmp_path / "marks"), source_id="fixture", population_path=text_population.path, attributes=result
        )
        write_record(
            ArtifactRecord(
                output_path=marks.path, result_type=result_type_name(type(marks)), result=marks.result_payload()
            )
        )
        drops = LexicalDropSets(
            path=str(tmp_path / "df"),
            attributes=AllSourceDropSets(
                output_dir=str(tmp_path / "df/local"),
                global_output_dir=str(tmp_path / "df/global"),
                num_sources=1,
                n_global_dropped=0,
                counters={},
            ),
        )
        write_record(
            ArtifactRecord(
                output_path=drops.path, result_type=result_type_name(type(drops)), result=drops.result_payload()
            )
        )
        report = measure_marks(
            ReportConfig([marks.path], drops.path, bloom.bloom_dir, "eval-control-v1", "pilot", str(tmp_path / "report"))
        )
        evidence = report.measurements["fixture"]
        assert evidence["measured_tasks"] == 2 and evidence["contaminated_tasks"] == 1
        assert evidence["contaminated_fraction"] == 0.5 and evidence["receipt"] is None
        attributed = pq.read_table(next((tmp_path / "report").glob("*/task_flags/*.parquet"))).to_pylist()
        assert {row["id"]: row["matched_eval_ids"] for row in attributed} == {
            "public-copy": ["eval-control"],
            "private-only": [],
        }
        dropped = pq.read_table(next((tmp_path / "report").glob("*/drop_manifest.parquet"))).to_pylist()
        assert [row["id"] for row in dropped] == ["public-copy"]


def test_census_reads_every_partition_and_measures_empty_actor_environments(tmp_path):
    files = write_text_population(
        ((lexical_task(str(index), "A text-only task", "private"), {}) for index in range(1001)),
        str(tmp_path / "population"),
    )
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id="two-partitions",
        dataset_id="private/dataset",
        dataset_revision="a" * 40,
        source_version="1",
        converter_revision="converter-pin",
        text_renderer_version="renderer-v1",
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
    )
    with artifact_prefix(str(tmp_path)):
        write_record(
            ArtifactRecord(
                output_path=population.path,
                result_type=result_type_name(type(population)),
                result=population.result_payload(),
            )
        )
        result = measure_population(CensusConfig(population.path, str(tmp_path / "census")))
    assert result.task_count == 1001
    assert result.daytona_safety == "not_applicable"
    assert result.daytona_machine_slots == result.daytona_snapshot_count == result.daytona_execution_failures == 0
    assert result.empty_declared_environment_tasks == {"agent": 1001, "grader": 1001}
    assert result.integrity_contracts["absent_contract_tasks"] == 1001
    assert result.integrity_contracts["shared_missing_contract_tasks"] == 0


def test_contract_census_separates_native_families_and_required_shared_missing():
    from taskcompendium.models import ScriptGrader  # noqa: PLC0415

    environment = EnvironmentRequirements(
        command_semantics="linux_process",
        docker_image="example/image@sha256:" + "0" * 64,
        capabilities=("git_repository",),
    )
    shared = lexical_task("shared", "Edit the repository", "private")
    shared["environment_requirements"] = environment.model_dump(mode="json")
    shared["grader"] = ScriptGrader(argv=("true",), environment=environment).model_dump(mode="json")
    counts = ContractCensus()
    for family in ("pytest", "shell"):
        # Native publisher metadata is opaque to the grading-boundary assessment.
        contract = {
            "id": "native-shared",
            "version": 1,
            "dependency_state": "agent_mutable",
            "native": {"family": family},
        }
        counts.observe({**shared, "integrity_contract": contract})
    counts.observe(shared)
    counts.observe(lexical_task("separate", "Reply only", "private"))
    groups = counts.rows()
    assert {r["native_family"] for r in groups if r["present"]} == {"pytest", "shell"}
    assert sum(r["count"] for r in groups if not r["present"]) == 2
    assert counts.required_shared == 3 and counts.shared_missing == 1
