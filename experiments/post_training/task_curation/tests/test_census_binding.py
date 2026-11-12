# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configured source identity controls whether a census can publish a verdict."""

from dataclasses import replace

import pyarrow as pa
import pytest
from finestore.rl.curation import ActorCensusInput, DaytonaMeasurement, DaytonaSafety, SourceDeclaration
from finestore.schema import arrow_schema

from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def census_receipt(source) -> ActorCensusReceipt:
    return ActorCensusReceipt(
        source_id=source.info.id,
        input=ActorCensusInput(
            dataset_id=source.dataset.name,
            dataset_revision=source.dataset.revision,
            converter_version=source.version,
            converter_revision="a" * 40,
            measured_tasks=1,
            task_specs_sha256="b" * 64,
            evidence_url="s3://fixture/census/.artifact.json",
        ),
        measurement=DaytonaMeasurement(
            dataset_id=source.dataset.name,
            dataset_revision=source.dataset.revision,
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Full source census",
            daytona_unsafe_reasons=[],
            daytona_unsafe_details={},
            daytona_unsafe_counts={},
        ),
        integrity_contracts=None,
    )


def test_bound_zero_actor_receipt_survives_catalog_arrow_roundtrip(tmp_path):
    source = number_source(tmp_path / "input")
    bound = bind_actor_census(source, census_receipt(source))
    declaration = bound.declaration
    table = pa.Table.from_pylist([declaration.model_dump(mode="json")], schema=arrow_schema(SourceDeclaration))
    restored = SourceDeclaration.model_validate(table.to_pylist()[0])
    assert restored.daytona_safety is DaytonaSafety.NOT_APPLICABLE
    assert restored.actor_census_input.converter_version == source.version
    assert restored.actor_census_input.measured_tasks == 1
    changed = replace(bound, config=replace(bound.config, version="2"))
    with pytest.raises(ValueError, match="Actor census input or converter"):
        _ = changed.declaration


@pytest.mark.parametrize("changed", ["dataset", "revision", "converter", "missing"])
def test_changed_input_or_missing_receipt_clears_stale_inline_census(tmp_path, changed):
    source = number_source(tmp_path / "input")
    receipt = census_receipt(source)
    if changed in {"dataset", "revision"}:
        dataset = replace(
            source.dataset, **({"name": "open-athena/replaced"} if changed == "dataset" else {"revision": "c" * 40})
        )
        source = replace(source, config=replace(source.config, dataset=dataset))
    elif changed == "converter":
        source = replace(source, config=replace(source.config, version="2"))
    else:
        receipt = None
    bound = bind_actor_census(source, receipt)
    assert bound.catalog_declaration is None
    assert not bound.has_actor_census
    assert bound.info.daytona_snapshot_count is None
    assert bound.info.actor_census_input is None
