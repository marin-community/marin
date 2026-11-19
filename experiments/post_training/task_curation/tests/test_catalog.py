# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from collections import Counter
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.curation import CurationCatalog

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation.count_inputs import parquet_counts
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.resource_receipts import resource_recipe, write_fixture_receipt
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def test_parquet_count_export_identifies_the_counted_input(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(pa.table({"question": ["one", "two", "three"]}), snapshot / "tasks.parquet")
    metadata = snapshot / ".cache/huggingface/download/tasks.parquet.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("pinned-input\nfile-etag\n0\n")
    assert parquet_counts(snapshot, "pinned-input", "*.parquet")["tasks.parquet"] == 3
    with pytest.raises(ValueError, match="expected revision"):
        parquet_counts(snapshot, "another-revision", "*.parquet")


def test_cli_catalog_preserves_archived_packets_and_reuses_curation_for_new_rubric(
    tmp_path, monkeypatch, review_server, qa_reviewer
):
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    staged = tmp_path / "input"
    staged.mkdir()
    (staged / "test.jsonl").write_text(
        "\n".join(json.dumps({"problem": f"What is {value} plus one?", "answer": str(value + 1)}) for value in range(3))
    )
    cache = tmp_path / "cache"
    source = next(s for s in skyrl_math.sources() if s.name == "math500")
    source = replace(
        source,
        config=resource_recipe(
            replace(source.config, source=replace(source.config.source, repo="open-athena/math500-fixture"))
        ),
        info=replace(
            source.info,
            within_source_dedup=deduplication_evidence(
                "open-athena/math500-fixture", source.dataset.revision, source.version, 3
            ),
            daytona_machine_slots=0,
            daytona_snapshot_count=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
        ),
    )
    source = with_admission_census(source)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.runnable_sources",
        lambda declared_source=source: {declared_source.name: declared_source},
    )
    receipt = tmp_path / "decontamination.json"
    receipt.write_text(decontamination_evidence(source.dataset.name, source.dataset.revision, 3).model_dump_json())
    args = [
        *write_fixture_receipt(source.config, staged, tmp_path / "oracle-filter.json"),
        "--decontamination-receipt",
        str(receipt),
        "--mode",
        "full",
        "--cache-dir",
        str(cache),
        "--source",
        "math500",
        "--input-root",
        str(staged),
        "--max-workers",
        "1",
        "--quality-sample-size",
        "2",
        "--assessment-sample-size",
        "2",
        "--review-concurrency",
        "1",
        "--run",
    ]
    runner = CliRunner()
    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    root = str(cache / "campaigns/catalog")
    catalog = CurationCatalog(root)
    snapshot = catalog.current()
    source = next(row for row in catalog.sources(snapshot.snapshot_id) if row.declaration.name == "math500")
    key = source.assessment_key
    assert key is not None
    items = list(catalog.items(key))
    assert len(items) == 2 and all(item.verdict == "good" for item in items)
    assert source.curation.admission_counts.admitted == 3
    assert all(row.declaration.pipeline is not None for row in catalog.sources(snapshot.snapshot_id))
    packet = catalog.packet(key, items[0].task_id)
    assert packet["task"]["id"] == items[0].task_id

    calls = Counter(request["tool_choice"]["function"]["name"] for request in review_server.requests)
    assert calls["review_task"] > 0 and calls["assess_task"] == 2
    repeated = runner.invoke(main, args)
    assert repeated.exit_code == 0, repeated.output
    assert Counter(request["tool_choice"]["function"]["name"] for request in review_server.requests) == calls
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("Check whether the requested operation is unambiguous.")
    changed = runner.invoke(main, [*args, "--assessment-rubric", str(rubric)])
    assert changed.exit_code == 0, changed.output
    changed_catalog = CurationCatalog(root)
    changed_source = next(
        row for row in changed_catalog.sources(changed_catalog.current().snapshot_id) if row.id == source.id
    )
    assert changed_source.assessment_key != key
    assert Counter(request["tool_choice"]["function"]["name"] for request in review_server.requests) == calls + Counter(
        {"assess_task": 2}
    )

    rubric.write_text("Check answer units as well.")
    review_server.assessment_available = False
    failed = runner.invoke(main, [*args, "--assessment-rubric", str(rubric)])
    assert failed.exit_code != 0
    failed_catalog = CurationCatalog(root)
    failed_source = next(
        row for row in failed_catalog.sources(failed_catalog.current().snapshot_id) if row.id == source.id
    )
    # The missing SAMPLE Result prevents publication; the selected prior evidence remains intact.
    assert failed_catalog.current().snapshot_id == changed_catalog.current().snapshot_id
    assert failed_source.assessment_key == changed_source.assessment_key
    assert failed_source.latest_attempt_status == "complete"
    assert catalog.packet(key, items[0].task_id) == packet

    attempt = next(catalog.attempts(snapshot.snapshot_id))
    shutil.rmtree(attempt.paths.inputs_root)
    shutil.rmtree(attempt.paths.judgments_root)
    assert CurationCatalog(root).packet(key, items[0].task_id) == packet
