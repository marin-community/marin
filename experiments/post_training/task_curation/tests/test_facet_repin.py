# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import task_spec_json
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage

from experiments.post_training.task_curation.facet_repin import SOURCE_ID, prepare_evidence, prepare_population
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_published import PublishedPopulation
from experiments.post_training.task_curation.sample_qc import panel_task_runtimes
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


@pytest.mark.parametrize("corrupt_row", [False, True])
def test_published_repair_preserves_task_and_runtime_bytes_before_admission(tmp_path, corrupt_row):
    template = TaskSpec.model_validate(json.loads(prebuilt_task("facet")["lowered_task_spec"])["task"])
    tasks = [
        template.model_copy(
            update={"id": name, "context": ConversationInput(events=(TextMessage(role="user", content=name),))}
        )
        for name in ("facet-a", "facet-b")
    ]
    runtime = [{"actor_profile": "frozen-a"}, {"actor_profile": "frozen-b"}]
    rows = [
        {
            "task_spec": task_spec_json(task),
            "task_sha256": hashlib.sha256(task_spec_json(task).encode()).hexdigest(),
            "runtime": json.dumps(packet),
        }
        for task, packet in zip(tasks, runtime, strict=True)
    ]
    if corrupt_row:
        rows[0]["task_sha256"] = "0" * 64
    parquet = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = PublishedPopulation(
        source_id=SOURCE_ID,
        dataset_id="open-athena/facet-fixture",
        dataset_revision="a" * 40,
        converter_version="fixture",
        converter_revision="b" * 40,
        tasks_parquet_uri=str(parquet),
        tasks_parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        task_count=2,
        converter_rejections={},
        admission_rejections={},
    )
    root = tmp_path / "population"
    if corrupt_row:
        with pytest.raises(ValueError, match="row hash"):
            prepare_population(parquet, metadata, str(root))
        assert not (root / ".artifact.json").exists()
        return
    result = prepare_evidence(parquet, metadata, str(tmp_path / "evidence"))
    population = SourceTextPopulation.model_validate(result["population"])
    inventory = SampleInventory.model_validate(result["inventory"])
    assert result["actor"]["input"]["dataset_revision"] == metadata.dataset_revision
    assert result["grading"]["input"] == result["actor"]["input"]
    assert result["derived"]["input"] == result["actor"]["input"]
    frozen_tasks, frozen_runtime = panel_task_runtimes(inventory.qc_panel())
    expected = {task.id: (task_spec_json(task), packet) for task, packet in zip(tasks, runtime, strict=True)}
    assert {
        task.id: (task_spec_json(task), packet) for task, packet in zip(frozen_tasks, frozen_runtime, strict=True)
    } == expected
    assert inventory.sample_seed == 0 and inventory.sample_size == inventory.retained_tasks == 2
    assert inventory.dataset_revision == metadata.dataset_revision
    assert population.task_specs_sha256 == hashlib.sha256("".join(row["task_spec"] for row in rows).encode()).hexdigest()
