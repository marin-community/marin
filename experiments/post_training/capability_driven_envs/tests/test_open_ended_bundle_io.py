# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io

import pytest
from pydantic import ValidationError

from experiments.post_training.capability_driven_envs.open_ended_bundle_io import (
    OpenEndedBundleRecord,
    bundle_record,
    load_record_json,
    read_bundle_parquet,
    record_from_mapping,
    record_json,
    write_bundle_parquet,
)
from experiments.post_training.capability_driven_envs.tests.test_open_ended import bundle


def test_bundle_record_round_trips_canonical_json_and_detects_payload_identity_changes():
    original = bundle()
    record = bundle_record(original)

    assert load_record_json(record_json(record)).bundle() == original
    assert record_from_mapping(record.model_dump(mode="json")) == record

    changed = record.model_copy(update={"bundle_id": "other-bundle"})
    with pytest.raises(ValidationError, match="identity differs"):
        OpenEndedBundleRecord.model_validate(changed.model_dump())


def test_bundle_parquet_round_trips_in_stable_bundle_id_order():
    second = bundle().model_copy(update={"bundle_id": "z-bundle"})
    first = bundle().model_copy(update={"bundle_id": "a-bundle"})
    stream = io.BytesIO()

    write_bundle_parquet(stream, (second, first))
    stream.seek(0)
    restored = read_bundle_parquet(stream)

    assert [item.bundle_id for item in restored] == ["a-bundle", "z-bundle"]
    assert restored == (first, second)


def test_bundle_parquet_rejects_duplicate_bundle_ids():
    stream = io.BytesIO()

    with pytest.raises(ValueError, match="repeat bundle_id"):
        write_bundle_parquet(stream, (bundle(), bundle()))
