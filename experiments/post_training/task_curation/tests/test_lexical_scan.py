# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen drop-set adoption for lexical re-measurement at the retained DF state."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.decon import AllSourceDropSets
from marin.execution.artifact import write_artifact

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_scan import (
    population_key,
    require_frozen_drop_sets_coverage,
    scan_steps,
)

DROP_SCHEMA = pa.schema([("hash", pa.uint64())])


def _population(tmp_path: Path, source_id: str, revision: str, text_sha: str) -> SourceTextPopulation:
    return SourceTextPopulation(
        path=str(tmp_path / "text" / population_key(source_id)),
        source_id=source_id,
        dataset_id="upstream/dataset",
        dataset_revision=revision,
        source_version="6",
        converter_revision="1509fc8638f6be3840b5c82b53348f93152a9c31",
        text_renderer_version="taskcompendium-format-conversation-v1",
        task_count=10,
        text_sha256=text_sha,
        task_specs_sha256=text_sha,
        main_output_dir=str(tmp_path / "main"),
        specs_output_dir=str(tmp_path / "specs"),
        converter_rejections={},
        admission_rejections={},
    )


def _write_drop_artifact(tmp_path: Path, covered: list[str]) -> str:
    output_dir = tmp_path / "df"
    for key in [*covered, "_global"]:
        directory = output_dir / key
        directory.mkdir(parents=True)
        pq.write_table(pa.table({"hash": pa.array([], pa.uint64())}, schema=DROP_SCHEMA), directory / "drop.parquet")
    attributes = AllSourceDropSets(
        output_dir=str(output_dir),
        global_output_dir=str(output_dir / "_global"),
        num_sources=len(covered),
        n_global_dropped=0,
        counters={},
    )
    path = str(tmp_path / "df-artifact")
    write_artifact({"attributes": attributes.model_dump(mode="json")}, path)
    return path


def test_frozen_drop_sets_coverage_passes_when_every_population_is_covered(tmp_path):
    populations = [
        _population(tmp_path, "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean", "4" * 40, "a" * 64),
        _population(tmp_path, "di-zhang-fdu:aime_1983_2024", "3" * 40, "b" * 64),
    ]
    covered = [population_key(p.source_id) for p in populations]
    path = _write_drop_artifact(tmp_path, covered)
    frozen = require_frozen_drop_sets_coverage(path, populations)
    assert frozen.num_sources == 2


def test_frozen_drop_sets_coverage_names_the_missing_population(tmp_path):
    populations = [
        _population(tmp_path, "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean", "4" * 40, "a" * 64),
        _population(tmp_path, "di-zhang-fdu:aime_1983_2024", "3" * 40, "b" * 64),
    ]
    path = _write_drop_artifact(tmp_path, [population_key(populations[0].source_id)])
    with pytest.raises(ValueError, match=population_key(populations[1].source_id)):
        require_frozen_drop_sets_coverage(path, populations)


def test_scan_steps_adopts_the_frozen_drop_sets_as_the_mark_dependency(tmp_path):
    populations = [_population(tmp_path, "di-zhang-fdu:aime_1983_2024", "3" * 40, "b" * 64)]
    frozen_path = _write_drop_artifact(tmp_path, [population_key(populations[0].source_id)])
    report = scan_steps(
        populations,
        "bloom-path",
        "atlas-eval-coverage-v2",
        "retained",
        "2026.11.23.1",
        2,
        CampaignRuntime(),
        frozen_drop_sets=frozen_path,
    )
    adopted = [dep for dep in report.deps if dep.name.endswith("/df-frozen")]
    assert len(adopted) == 1
    assert adopted[0].adopt_source == frozen_path
    assert adopted[0].artifact_type.__name__ == "LexicalDropSets"
