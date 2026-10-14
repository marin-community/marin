# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Any

import json
import asyncio
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from levanter.store.cache import CacheOptions
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import CLIPImageProcessor, CLIPProcessor, PreTrainedTokenizerFast

from levanter.data.image import (
    processor_with_tokenizer,
    ConversationDatasetSourceConfig,
    ImageMixtureDatasetConfig,
    image_caption_to_conversation,
    BatchImageProcessor,
    ImageConversationUrlDataSource,
    ImageDatasetSourceConfig,
    image_source_urls,
    load_image,
)


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
    record = {
        "messages": [{"role": "user", "content": [{"type": "audio"}]}],
        "audio": ["sound.wav"],
        "videos": ["clip.mp4"],
    }
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
    assert image_source_urls([str(tmp_path / "{a,missing}.parquet")]) == [
        str(first),
        str(tmp_path / "missing.parquet"),
    ]


def test_processor_replacement_uses_configured_modality_tokens() -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "<visual>": 1, "<clip>": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    image_processor = CLIPImageProcessor()
    original = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
    replaced = processor_with_tokenizer(original, tokenizer, image_token="<visual>", video_token="<clip>")
    assert isinstance(replaced, CLIPProcessor)
    assert replaced is not original
    assert replaced.image_token_id == 1
    assert replaced.video_token_id == 2
    assert replaced.image_processor is image_processor


@pytest.fixture
def caption_processor():
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "user", "assistant", "<image>", "red", "square"]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<image>"],
    )
    processor = CLIPProcessor(
        image_processor=CLIPImageProcessor(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}),
        tokenizer=tokenizer,
    )
    processor.chat_template = (
        "{% for message in messages %}<|im_start|> {{ message['role'] }} "
        "{% for item in message['content'] %}{% if item['type'] == 'image' %}<image> "
        "{% else %}{{ item['text'] }} {% endif %}{% endfor %}<|im_end|> {% endfor %}"
    )
    return processor


@pytest.mark.parametrize("max_length", [6, 7, 8, 10])
def test_caption_source_preserves_images_and_truncated_targets(tmp_path, caption_processor, max_length):
    image = tmp_path / "red.png"
    Image.new("RGB", (4, 4), "red").save(image)
    records = tmp_path / "captions.jsonl"
    records.write_text(json.dumps({"image": str(image), "text": "red square"}) + "\n")
    source = ImageDatasetSourceConfig(train_urls=[str(records)]).get_shard_source("train")
    batch = list(source.open_shard(source.shard_names[0]))
    processed = BatchImageProcessor(caption_processor, max_length=max_length, max_num_patches=3)(batch)[0]
    assert processed["pixel_values"] is not None
    expected_pixels = caption_processor.image_processor(images=Image.open(image), return_tensors="np")["pixel_values"][
        0
    ]
    np.testing.assert_allclose(processed["pixel_values"][0], expected_pixels)
    np.testing.assert_array_equal(processed["pixel_values"][1:], 0)
    ids = processed["input_ids"]
    tokenizer = caption_processor.tokenizer
    expected_mask = np.isin(ids, tokenizer.convert_tokens_to_ids(["red", "square"]))
    # The assistant terminator is supervised only when it survives truncation.
    if max_length >= 9:
        expected_mask[8] = True
    np.testing.assert_array_equal(processed["loss_mask"], expected_mask)


@pytest.mark.parametrize("source_kind", ["caption", "conversation"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_image_mixture_preserves_patch_limit_and_unpadded_targets(
    tmp_path, caption_processor, source_kind, use_cache, monkeypatch
):
    image = tmp_path / "red.png"
    Image.new("RGB", (4, 4), "red").save(image)
    records = tmp_path / "records.jsonl"
    rows = [{"image": str(image), "text": text} for text in ("red", "red square")]
    if source_kind == "conversation":
        conversations = [image_caption_to_conversation(row) for row in rows]
        rows = [{"dialog": row["messages"], "pictures": row["images"]} for row in conversations]
        source = ConversationDatasetSourceConfig(
            train_urls=[str(records)], messages_key="dialog", images_key="pictures"
        )
    else:
        source = ImageDatasetSourceConfig(train_urls=[str(records)])
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *args, **kwargs: caption_processor)
    config = ImageMixtureDatasetConfig(
        processor="local-test-processor",
        configs={"tiny": source},
        train_weights={"tiny": 1},
        cache_dir=str(tmp_path / "cache"),
        mixture_block_size=2,
        use_cache=use_cache,
        padding=not use_cache,
        max_length=12,
    )
    # Different target lengths require single-example batches when padding is disabled.
    with set_current_client(LocalClient()):
        dataset = config.train_set(options=CacheOptions(batch_size=1), max_num_patches=3)
        first = asyncio.run(dataset.get_batch([0]))[0]
        second = asyncio.run(dataset.get_batch([1]))[0]
    expected_pixels = caption_processor.image_processor(images=Image.open(image), return_tensors="np")["pixel_values"][
        0
    ]
    for item, length in zip(sorted([first, second], key=lambda item: int(item["attention_mask"].sum())), [8, 9]):
        expected_length = length if use_cache else 12
        assert item["input_ids"].shape == (expected_length,)
        assert item["pixel_values"].shape == (4, 3, 4, 4)
        np.testing.assert_allclose(item["pixel_values"][0], expected_pixels)
        np.testing.assert_array_equal(item["grid_mask"], [True, False, False, False])
        np.testing.assert_array_equal(
            item["loss_mask"], [0] * 6 + [1] * (length - 6) + [0] * (expected_length - length)
        )
    if use_cache:
        restored = config.train_set(options=CacheOptions(batch_size=1), max_num_patches=3)
        np.testing.assert_array_equal(asyncio.run(restored.get_batch([1]))[0]["input_ids"], second["input_ids"])
