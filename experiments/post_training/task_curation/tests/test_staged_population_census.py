# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A staged-population census binds the release pin and registers lexically.

Coordinator ruling 2026-11-25 ~21:05 EST (option A): the staged censuses
re-measure at the staged lexical population artifacts — the same task bytes,
the release pin preserved through the population's dataset pin, and the digest
semantics the lexical registration and SAMPLE panel freezing expect. This test
pins that chain end to end: census -> receipt -> evidence check -> lexical
registration.
"""

import json
from pathlib import Path

from finestore.rl.curation import LexicalContaminationEvidence, LexicalUse
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.identity import task_spec_json
from taskcompendium.pipeline.models import RawRow, Source

from experiments.post_training.task_curation.census_binding import (
    check_receipt_evidence,
    receipt_from_census,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import (
    RENDERER_VERSION,
    write_text_population,
)
from experiments.post_training.task_curation.lexical_policy import LexicalInputUse
from experiments.post_training.task_curation.lexical_registration import measurement_receipts
from experiments.post_training.task_curation.staged_census import measure_staged_population

SOURCE_ID = "TaskTrove:test__test-cohort-v1"
RELEASE_REPO = "open-athena/test__test-cohort-v1-00000000-taskspec-20261124g"
RELEASE_REVISION = "b" * 40
CONVERTER_REVISION = "c" * 40
PROTOCOL = "harbor-cohort-taskspec-v1"


def task_document(row: int) -> dict:
    """One real TaskSpec document the way the conversion releases carry them."""
    source = Source(dataset="fixture", revision="pinned", row=f"fixture/tasks.parquet:{row}", importer_revision="1")
    task = exact_answer_task(
        RawRow("fixture", source, {}), prompt=f"Name color {row}", answers=("red",), ignore_case=False
    )
    return json.loads(task_spec_json(task))


def stage_population(tmp_path: Path) -> SourceTextPopulation:
    """Stage a tiny one-task population the way the release-population staging does."""
    documents = [(task_document(0), {})]
    output = tmp_path / "population"
    files = write_text_population(documents, output_path=str(output))
    population = SourceTextPopulation(
        path=str(output),
        source_id=SOURCE_ID,
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        source_version=PROTOCOL,
        converter_revision=CONVERTER_REVISION,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    payload_fields = {name for name in SourceTextPopulation.model_fields if name != "path"}
    write_record(
        ArtifactRecord(
            output_path=str(output),
            result_type=result_type_name(SourceTextPopulation),
            result=population.model_dump(mode="json", include=payload_fields),
        )
    )
    return population


def test_staged_population_census_binds_the_release_pin_and_registers_lexically(tmp_path: Path):
    population = stage_population(tmp_path)
    staged = measure_staged_population(
        SOURCE_ID,
        RELEASE_REPO,
        RELEASE_REVISION,
        converter_version=PROTOCOL,
        converter_revision=CONVERTER_REVISION,
        acceptance_job="acceptance-test-cohort",
        population_record=str(population.path),
    )
    # The census binds the release pin and the population's own digest semantics.
    assert staged.record["dataset_id"] == RELEASE_REPO
    assert staged.record["dataset_revision"] == RELEASE_REVISION
    assert staged.record["task_specs_sha256"] == population.task_specs_sha256
    assert staged.record["task_count"] == population.task_count == 1

    # The receipt's evidence check passes against the census artifact, as in the runner.
    census_out = tmp_path / "census"
    census_out.mkdir()
    census = {**staged.census, "path": str(census_out)}
    (census_out / ".artifact.json").write_text(
        json.dumps({"result": {**staged.record, "path": str(census_out)}}, indent=2, sort_keys=True) + "\n"
    )
    receipt = receipt_from_census(census, None)
    check_receipt_evidence(receipt)

    # The lexical registration joins the census receipt, the population and the Use declaration.
    evidence = LexicalContaminationEvidence(
        version=1,
        input_dataset_id=RELEASE_REPO,
        input_dataset_revision=RELEASE_REVISION,
        converter_revision=CONVERTER_REVISION,
        text_renderer_version=RENDERER_VERSION,
        measured_tasks=population.task_count,
        contaminated_tasks=0,
        contaminated_fraction=0.0,
        max_overlap=0.0,
        mean_max_overlap=0.0,
        p95_max_overlap=0.0,
        matched_evals={},
        eval_corpus_version="aa-index-v4.1.1-v3",
        eval_bloom_uri="s3://bloom/example",
        eval_hash_index_uri="s3://bloom/example/_bloom/eval_hash_index.parquet",
        feature_filter_version=4,
        ngram_length=13,
        overlap_threshold=0.5,
        min_matched_features=2,
        drop_set_uris=["s3://drops/example"],
        task_flags_uri="s3://flags/example",
        evidence_url="s3://lexical/example/.artifact.json",
    )
    use = LexicalInputUse(
        source_id=SOURCE_ID,
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        converter_revision=CONVERTER_REVISION,
        use=LexicalUse.TRAIN,
    )
    receipts = measurement_receipts(
        {SOURCE_ID: evidence},
        [population],
        [use],
        {SOURCE_ID: receipt},
    )
    registered = receipts[SOURCE_ID]
    assert registered.input.dataset_id == RELEASE_REPO
    assert registered.input.dataset_revision == RELEASE_REVISION
    assert registered.input.task_specs_sha256 == population.task_specs_sha256
    assert registered.use is LexicalUse.TRAIN
    assert registered.measurement == evidence
