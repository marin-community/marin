# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from taskforge.content_hash import digest
from taskforge.stage40_build.step_store import (
    SDK_VERSION,
    CacheStatus,
    RLEnvBuildStepKey,
    RLEnvBuildStepStore,
    memoized,
)


@dataclass(frozen=True)
class Fixture:
    path: str
    lines: tuple[str, ...]


KEY = RLEnvBuildStepKey(SDK_VERSION, "source", {"proposal": digest({"id": "a"})}, "model")
OUTPUT = Fixture(path="workspace/numbers.txt", lines=("6", "7"))


@pytest.mark.parametrize("scheme", ["", "file://"], ids=["local path", "file url"])
def test_a_stored_output_loads_back_and_items_share_its_blob(tmp_path, scheme):
    root = f"{scheme}{tmp_path}/store"
    first = RLEnvBuildStepStore(root, "item-a")
    assert first.load(KEY, "fixtures", Fixture) is None

    stored = first.store(KEY, "fixtures", OUTPUT)
    RLEnvBuildStepStore(root, "item-b").store(KEY, "fixtures", OUTPUT)
    loaded = RLEnvBuildStepStore(root, "item-a").load(KEY, "fixtures", Fixture)

    assert stored.inputs == KEY.inputs
    assert loaded is not None
    record, output = loaded
    assert output == OUTPUT
    assert record == stored
    assert stored.output_digest == digest(OUTPUT)
    assert [path.name for path in (tmp_path / "store" / "blobs").iterdir()] == [stored.output_digest]
    assert (tmp_path / "store" / "items" / "item-b" / "steps" / "fixtures" / f"{KEY.digest}.json").exists()


async def test_a_memoized_helper_computes_once_and_hits_on_the_second_call(tmp_path):
    store = RLEnvBuildStepStore(str(tmp_path), "item-a")
    key = RLEnvBuildStepKey(SDK_VERSION, "helper source", {"path": digest(OUTPUT.path)}, "model")
    calls = []

    async def compute() -> Fixture:
        calls.append(None)
        return OUTPUT

    first_status, first_record, first_output = await memoized(store, "fixtures.render", key, Fixture, compute)
    again_status, again_record, again_output = await memoized(store, "fixtures.render", key, Fixture, compute)

    assert (first_status, again_status) == (CacheStatus.MISS, CacheStatus.HIT)
    assert again_record == first_record
    assert again_output == first_output == OUTPUT
    assert len(calls) == 1


@dataclass(frozen=True)
class Readings:
    values: tuple[int, ...]


async def test_a_miss_returns_the_stored_form_of_the_output(tmp_path):
    """A consumer sees the same value on a miss as on a hit, whatever the step returned."""
    store = RLEnvBuildStepStore(str(tmp_path), "item-a")
    key = RLEnvBuildStepKey(SDK_VERSION, "readings source", {}, "model")

    async def compute() -> Readings:
        return Readings(values=[3, 1, 2])  # type: ignore[arg-type]  # a list where the type says tuple

    _, _, missed = await memoized(store, "readings", key, Readings, compute)
    _, _, hit = await memoized(store, "readings", key, Readings, compute)

    assert missed == hit == Readings(values=(3, 1, 2))
    assert isinstance(missed.values, tuple)


def test_a_blob_altered_on_disk_is_refused(tmp_path):
    store = RLEnvBuildStepStore(str(tmp_path), "item-a")
    record = store.store(KEY, "fixtures", OUTPUT)
    (tmp_path / "blobs" / record.output_digest).write_text("{}")

    with pytest.raises(ValueError, match="does not match its digest"):
        store.load(KEY, "fixtures", Fixture)


@pytest.mark.parametrize("item_id", ["../escape", "a/b"])
def test_an_item_id_unsafe_as_a_directory_name_is_refused(tmp_path, item_id):
    with pytest.raises(ValueError, match="must match"):
        RLEnvBuildStepStore(str(tmp_path), item_id)
