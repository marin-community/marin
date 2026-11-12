# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark self-matches are measured while only training enters release review."""

import json
import sys
from dataclasses import replace

import pyarrow as pa
import pytest
from finestore.rl.curation import DeconReceiptStatus, LexicalContaminationEvidence, LexicalUse
from finestore.schema import arrow_schema
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.evaluation import EvalDataset
from experiments.post_training.task_curation.lexical_policy import main, measurement_for_use, source_lexical_use
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.sources import all_sources
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


def overlap_measurement(dataset: str) -> LexicalContaminationEvidence:
    document = decontamination_evidence(dataset, "a" * 40, 2).model_dump(mode="json")
    document.update(
        receipt=None,
        contaminated_tasks=1,
        contaminated_fraction=0.5,
        max_overlap=1,
        mean_max_overlap=0.5,
        p95_max_overlap=0.95,
        matched_evals={"HumanEval": 1},
    )
    return LexicalContaminationEvidence.model_validate(document)


@pytest.mark.parametrize("use", [LexicalUse.EVAL, LexicalUse.BENCHMARK])
def test_benchmark_self_match_preserves_population_pin_and_eval_evidence(use):
    raw = overlap_measurement("upstream/benchmark")
    measured = measurement_for_use(raw, use)
    table = pa.Table.from_pylist([measured.model_dump(mode="json")], schema=arrow_schema(LexicalContaminationEvidence))
    restored = LexicalContaminationEvidence.model_validate(table.to_pylist()[0])
    assert restored.receipt.status is DeconReceiptStatus.BENCHMARK_MEASURED
    assert restored.receipt.retained_tasks == 2 and restored.receipt.dropped_tasks == 0
    assert restored.receipt.release is None
    assert (
        restored.input_dataset_id == raw.input_dataset_id
        and restored.input_dataset_revision == raw.input_dataset_revision
    )
    assert restored.matched_evals == {"HumanEval": 1} and restored.contaminated_tasks == 1
    assert "does not certify a training release" in restored.admission_blocker(
        raw.input_dataset_id, raw.input_dataset_revision
    )


def test_evaluation_use_comes_from_recipe_even_without_evaluation_tags():
    source = next(source for source in all_sources().values() if isinstance(source.config, EvalDataset))
    source = replace(source, info=replace(source.info, tags=()))
    assert source_lexical_use(source) is LexicalUse.EVAL


def test_report_keeps_benchmarks_out_of_the_training_review_table(tmp_path, monkeypatch):
    records = {key: overlap_measurement("upstream/" + key).model_dump(mode="json") for key in ("train", "eval")}
    measured = tmp_path / "measured.json"
    measured.write_text(
        json.dumps(
            {
                "measurements": records,
                "drop_manifests": {"train": "s3://test/train-drops", "eval": "s3://test/eval-flags"},
            }
        )
    )
    uses = tmp_path / "uses.json"
    uses.write_text(
        json.dumps(
            [
                {
                    "source_id": key,
                    "dataset_id": record["input_dataset_id"],
                    "dataset_revision": record["input_dataset_revision"],
                    "converter_revision": record["converter_revision"],
                    "use": key,
                }
                for key, record in records.items()
            ]
        )
    )
    output = tmp_path / "outcomes.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["lexical_policy", "--measurements", str(measured), "--source-uses", str(uses), "--output", str(output)],
    )
    main()
    result = json.loads(output.read_text())
    assert [row["source_id"] for row in result["training_review"]] == ["train"]
    assert result["measurements"]["train"]["receipt"] is None
    assert result["measurements"]["eval"]["receipt"]["status"] == "measured (benchmark, not remediated)"
    assert result["measurements"]["eval"]["receipt"]["dropped_tasks"] == 0


def test_training_use_keeps_benchmark_tagged_sources_in_remediation_review():
    source = next(source for source in all_sources().values() if isinstance(source.config, CurationRecipe))
    source = replace(
        source,
        config=replace(source.config, intended_use=IntendedUse.TRAIN),
        info=replace(source.info, tags=("benchmark",)),
    )
    assert source_lexical_use(source) is LexicalUse.TRAIN
    assert measurement_for_use(overlap_measurement(source.dataset.name), source_lexical_use(source)).receipt is None
