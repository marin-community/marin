# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lexical decontamination: the stage measures the kept population and drops contaminated training rows."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from zephyr.context import ZephyrContext

from taskcompendium.pipeline.lexical import (
    TASK_FLAGS_SCHEMA,
    DecontaminationEngine,
    MarkedPopulation,
    TextPopulation,
    decontaminate_population,
    population_summary,
)
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.source_processing import (
    SourcePipelineConfig,
    SourceProcessingMode,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, FilterPolicy

from .pipeline_stages import convert_svamp, fixture_recipe, svamp_task
from .test_pipeline import BatchService

CONTAMINATED_PHRASE = "the quick brown fixture fox"
MATCHED_HASH = int.from_bytes(b"lexical-fixture", "big") % (1 << 64)


class FixtureEngine:
    """Marks every task whose rendered text contains the fixture eval phrase."""

    def __init__(self, phrase: str = CONTAMINATED_PHRASE, family: str = "fixture-benchmark") -> None:
        self.phrase = phrase
        self.family = family

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "eval_corpus_version": "fixture-corpus-v1",
            "eval_bloom_uri": "fixture://bloom",
            "eval_hash_index_uri": "fixture://index",
            "feature_filter_version": 4,
            "ngram_length": 13,
            "overlap_threshold": 0.5,
            "min_matched_features": 2,
        }

    def mark_population(self, population: TextPopulation, output_path: str, *, context: Any) -> MarkedPopulation:
        flags_dir = Path(output_path) / "flags"
        flags_dir.mkdir(parents=True, exist_ok=True)
        rows = 0
        for path in sorted(Path(population.path).glob("*.parquet")):
            flagged = [
                {
                    "id": record["id"],
                    "max_overlap": 1.0 if self.phrase in record["text"] else 0.0,
                    "contaminated": self.phrase in record["text"],
                    "matched_hashes": [MATCHED_HASH] if self.phrase in record["text"] else [],
                    "matched_eval_ids": ["fixture:eval-item"] if self.phrase in record["text"] else [],
                    "matched_eval_families": [self.family] if self.phrase in record["text"] else [],
                }
                for record in pq.read_table(path).to_pylist()
            ]
            rows += len(flagged)
            pq.write_table(pa.Table.from_pylist(flagged, schema=TASK_FLAGS_SCHEMA), flags_dir / path.name)
        assert rows == population.task_count
        return MarkedPopulation(
            flags_dir=str(flags_dir),
            attributes={"drop_set_uris": ["fixture://drops/local", "fixture://drops/global"]},
        )


class ShortFlagsEngine(FixtureEngine):
    """A broken engine whose mark loses rows; the stage must refuse it."""

    def mark_population(self, population: TextPopulation, output_path: str, *, context: Any) -> MarkedPopulation:
        marked = super().mark_population(population, output_path, context=context)
        paths = sorted(Path(marked.flags_dir).glob("*.parquet"))
        trimmed = pq.read_table(paths[-1]).slice(0, 1)
        pq.write_table(trimmed, paths[-1])
        return marked


def source_rows(count: int, contaminated: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = [
        {"Body": f"Aya has {index} apples.", "Question": "How many?", "Answer": str(index)}
        for index in range(1, count + 1)
    ]
    for index in contaminated:
        rows[index - 1]["Body"] = f"Aya has {index} apples. {CONTAMINATED_PHRASE}"
    return rows


def write_jsonl(source: Path, rows: list[dict[str, Any]]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def pipeline_config(mode: SourceProcessingMode, decontamination: DecontaminationEngine | None) -> SourcePipelineConfig:
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    return SourcePipelineConfig(
        mode,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        None,
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
        machines=None,
        decontamination=decontamination,
    )


def run_pipeline(recipe, source: Path, output: Path, config: SourcePipelineConfig):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(output.parent / "chunks")) as context:
        return run_source_pipeline(
            recipe, context, str(source), str(output), config, mode=config.mode, canonical_source=recipe.name
        )


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(Path(path).glob("*.parquet"))
    return [row for file in files for row in pq.read_table(file).to_pylist()]


def write_checked_audit(path: Path, task_ids: tuple[str, ...]) -> None:
    """A minimal filtered audit: every row kept, with a task body the fixture phrase can match."""
    (path / "audit").mkdir(parents=True)
    rows = []
    data = {"Body": f"Aya has 1 apple. {CONTAMINATED_PHRASE}", "Question": "How many?", "Answer": "1"}
    for task_id in task_ids:
        task = svamp_task(task_id, data)
        rows.append(
            {"task_id": task_id, "filter_status": "keep", "filter_reasons": [], "task_json": task.model_dump_json()}
        )
    pq.write_table(pa.Table.from_pylist(rows), path / "audit" / "part-00000.parquet")
    (path / "manifest.json").write_text(
        json.dumps({"input_rows": len(task_ids), "dispositions": {"keep": len(task_ids)}})
    )


def test_full_train_source_drops_contaminated_rows_and_records_the_measurement(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(5, contaminated=(2, 4)))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    result = run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, FixtureEngine()))

    manifest = json.loads(Path(result.manifest_path).read_text())
    evidence = manifest["lexical"]
    assert evidence["measured_tasks"] == 5
    assert evidence["contaminated_tasks"] == 2
    assert evidence["contaminated_fraction"] == 0.4
    assert evidence["matched_evals"] == {"fixture-benchmark": 2}
    assert evidence["eval_corpus_version"] == "fixture-corpus-v1"
    assert evidence["drop_set_uris"] == ["fixture://drops/local", "fixture://drops/global"]
    assert evidence["text_renderer_version"] == "taskcompendium-format-conversation-v1"
    assert evidence["ngram_length"] == 13

    all_ids = {row["task_id"] for row in parquet_rows(Path(result.normalize_path))}
    dropped = {row["id"] for row in parquet_rows(output / "lexical" / "drop_manifest.parquet")}
    kept_ids = {row["task_id"] for row in parquet_rows(Path(result.final_path))}
    assert len(dropped) == 2
    assert kept_ids == all_ids - dropped
    assert manifest["admission_counts"] == {"admitted": 3, "rejected": 2}

    flags = parquet_rows(output / "lexical" / "task_flags")
    assert len(flags) == 5
    assert {row["id"] for row in flags if row["contaminated"]} == dropped
    review_rows = {row["task_id"]: row for row in parquet_rows(Path(result.review_path))}
    rejected = [row for row in review_rows.values() if row["filter_status"] == "reject"]
    assert {row["task_id"] for row in rejected} == dropped
    assert all("lexical:contaminated" in row["filter_reasons"] for row in rejected)
    assert result.lexical_path == str(output / "lexical")
    measurement = json.loads((output / "lexical" / "measurement.json").read_text())
    assert measurement["measured_tasks"] == 5
    telemetry = json.loads((output / "telemetry.json").read_text())
    assert {phase["phase"] for phase in telemetry["phases"]} >= {"lexical_text", "lexical_mark", "lexical_gate"}


def test_full_eval_source_keeps_every_row_and_still_records_the_measurement(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(4, contaminated=(1,)))
    output = tmp_path / "output"
    recipe = replace(fixture_recipe(convert_svamp, rubric=None), intended_use=IntendedUse.EVAL)
    result = run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, FixtureEngine()))

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["lexical"]["contaminated_tasks"] == 1
    assert manifest["lexical"]["measured_tasks"] == 4
    assert len(parquet_rows(Path(result.final_path))) == 4
    assert manifest["admission_counts"] == {"admitted": 4}
    assert {row["filter_status"] for row in parquet_rows(Path(result.review_path))} == {"keep"}


def test_source_without_an_engine_skips_lexical_decontamination(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(3, contaminated=(1,)))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    result = run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, None))

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["lexical"] is None
    assert result.lexical_path is None
    assert len(parquet_rows(Path(result.final_path))) == 3


class StubTelemetry:
    """Records named phases with placeholder executions, for direct stage calls."""

    def phase(self, name: str):
        return _NullPhase()


class _NullPhase:
    def __init__(self) -> None:
        self.executions: list[Any] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_stage_refuses_a_mark_that_lost_population_rows(tmp_path):
    audit = tmp_path / "checked"
    write_checked_audit(audit, ("task-a", "task-b"))
    with ZephyrContext(max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with pytest.raises(ValueError, match="flagged"):
            decontaminate_population(
                str(audit),
                str(tmp_path / "lexical-view"),
                str(tmp_path / "lexical-scratch"),
                source_name="fixture",
                dataset_id="fixture",
                dataset_revision="1",
                converter_revision="1",
                train=True,
                engine=ShortFlagsEngine(),
                context=context,
                telemetry=StubTelemetry(),
            )


def test_population_summary_counts_overlaps_and_families(tmp_path):
    flags_dir = tmp_path / "flags"
    flags_dir.mkdir()
    rows = [
        {
            "id": "a",
            "max_overlap": 1.0,
            "contaminated": True,
            "matched_hashes": [1],
            "matched_eval_ids": ["e1"],
            "matched_eval_families": ["math"],
        },
        {
            "id": "b",
            "max_overlap": 0.5,
            "contaminated": False,
            "matched_hashes": [],
            "matched_eval_ids": [],
            "matched_eval_families": [],
        },
        {
            "id": "c",
            "max_overlap": 0.0,
            "contaminated": True,
            "matched_hashes": [2],
            "matched_eval_ids": ["e2"],
            "matched_eval_families": ["code"],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=TASK_FLAGS_SCHEMA), flags_dir / "part-00000.parquet")
    summary = population_summary(str(flags_dir))
    assert summary["measured_tasks"] == 3
    assert summary["contaminated_tasks"] == 2
    assert summary["max_overlap"] == 1.0
    assert summary["matched_evals"] == {"math": 1, "code": 1}
