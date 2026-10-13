# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Any

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from levanter.data.image import ImageConversationUrlDataSource, ImageDatasetSourceConfig, image_source_urls, load_image


def test_image_references_preserve_pixels(tmp_path: Path) -> None:
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    image = Image.fromarray(pixels)
    path = tmp_path / "image.png"
    image.save(path)
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    for reference in (image, pixels, str(path), {"bytes": encoded.getvalue()}, {"bytes": None, "path": str(path)}):
        np.testing.assert_array_equal(np.asarray(load_image(reference)), pixels)


def test_conversation_source_preserves_other_modalities(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    record = {"messages": [{"role": "user", "content": [{"type": "audio"}]}], "audio": ["sound.wav"], "videos": ["clip.mp4"]}
    path.write_text(json.dumps(record) + "\n")
    source = ImageConversationUrlDataSource([str(path)])
    loaded = list(source.open_shard(source.shard_names[0]))
    assert loaded == [{**record, "images": []}]


def test_hf_source_iteration_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    consumed: list[int] = []

    def records() -> Iterator[dict[str, str]]:
        for index in range(5):
            consumed.append(index)
            yield {"image": f"{index}.png", "text": str(index)}

    def load_dataset(*args: Any, **kwargs: Any) -> Iterator[dict[str, str]]:
        assert kwargs["streaming"] is True
        return records()

    monkeypatch.setattr("levanter.data.image.datasets.load_dataset", load_dataset)
    iterator = ImageDatasetSourceConfig(id="fixture", stream=True).doc_iterator("train")
    assert consumed == []
    assert next(iterator) == {"image": "0.png", "text": "0"}
    assert consumed == [0]


def test_directory_and_brace_shards_are_reopenable(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first, second = tmp_path / "a.parquet", nested / "b.parquet"
    first.touch()
    second.touch()
    assert image_source_urls([str(tmp_path)]) == sorted([str(first), str(second)])
    assert image_source_urls([str(tmp_path / "{a,missing}.parquet")]) == [str(first), str(tmp_path / "missing.parquet")]
