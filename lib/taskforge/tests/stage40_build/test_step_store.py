# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from taskforge.content_hash import digest
from taskforge.stage40_build.step_store import SDK_VERSION, CacheStatus, RLEnvBuildStepKey, RLEnvBuildStepStore


@dataclass(frozen=True)
class Fixture:
    path: str
    lines: tuple[str, ...]


KEY = RLEnvBuildStepKey(SDK_VERSION, "source", {"proposal": digest({"id": "a"})}, "model")
OUTPUT = Fixture(path="workspace/numbers.txt", lines=("6", "7"))


@pytest.mark.parametrize("scheme", ["", "file://"], ids=["local path", "file url"])
def test_a_stored_output_loads_back_as_a_hit_and_items_share_its_blob(tmp_path, scheme):
    root = f"{scheme}{tmp_path}/store"
    first = RLEnvBuildStepStore(root, "item-a")
    assert first.load(KEY, "fixtures", Fixture) is None

    stored = first.store(KEY, "fixtures", OUTPUT)
    RLEnvBuildStepStore(root, "item-b").store(KEY, "fixtures", OUTPUT)
    loaded = RLEnvBuildStepStore(root, "item-a").load(KEY, "fixtures", Fixture)

    assert stored.status is CacheStatus.MISS
    assert stored.inputs == KEY.inputs
    assert loaded is not None
    record, output = loaded
    assert output == OUTPUT
    assert record.status is CacheStatus.HIT
    assert record.output_digest == stored.output_digest == digest(OUTPUT)
    assert [path.name for path in (tmp_path / "store" / "blobs").iterdir()] == [stored.output_digest]
    assert (tmp_path / "store" / "items" / "item-b" / "steps" / "fixtures" / f"{KEY.digest}.json").exists()


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
