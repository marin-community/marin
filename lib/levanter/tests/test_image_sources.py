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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    CLIPImageProcessor,
    CLIPProcessor,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    PreTrainedTokenizerFast,
)

from levanter.data.image import (
    processor_with_tokenizer,
    ConversationDatasetSourceConfig,
    ImageMixtureDatasetConfig,
    image_caption_to_conversation,
    BatchMultimodalProcessor,
    ImageConversationUrlDataSource,
    ImageDatasetSourceConfig,
    ImageTextUrlDataSource,
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


@pytest.mark.parametrize("file_format", ["json", "jsonl", "parquet"])
def test_conversation_source_preserves_other_modalities(tmp_path: Path, file_format: str) -> None:
    path = tmp_path / f"records.{file_format}"
    record = {
        "messages": [{"role": "user", "content": [{"type": "audio"}]}],
        "audio": ["sound.wav"],
        "videos": ["clip.mp4"],
    }
    if file_format == "parquet":
        pq.write_table(pa.Table.from_pylist([record]), path)
    elif file_format == "json":
        path.write_text(json.dumps([record]))
    else:
        path.write_text(json.dumps(record) + "\n")
    source = ImageConversationUrlDataSource([str(path)])
    loaded = list(source.open_shard(source.shard_names[0]))
    assert loaded == [{**record, "images": []}]


def test_parquet_image_source_resumes_across_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    table = pa.table({"picture": ["a.png", "b.png", "c.png"], "caption": ["a", "b", "c"]})
    pq.write_table(table, path, row_group_size=1)
    source = ImageTextUrlDataSource([str(path)], image_key="picture", text_key="caption")
    assert list(source.open_shard_at_row(source.shard_names[0], 1)) == [
        {"image": "b.png", "text": "b"},
        {"image": "c.png", "text": "c"},
    ]


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
    processed = BatchMultimodalProcessor(caption_processor, max_length=max_length, max_num_patches=3)(batch)[0]
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


def test_any_resolution_patch_cap_rejects_incompatible_grid(caption_processor) -> None:
    processor = BatchMultimodalProcessor(
        caption_processor,
        max_length=12,
        max_num_patches=1,
        grid_pinpoints=[[8, 8]],
        tile_size=4,
        vision_feature_height=1,
    )
    record = image_caption_to_conversation({"image": Image.new("RGB", (8, 8)), "text": "red"})
    with pytest.raises(ValueError, match="requires 4 any-resolution patches"):
        processor([record])


@pytest.mark.parametrize("tile_size", [4, 8])
def test_processor_dimensions_match_processed_pixels_and_cache_schema(caption_processor, tile_size) -> None:
    caption_processor.image_processor = CLIPImageProcessor(
        size={"shortest_edge": tile_size}, crop_size={"height": tile_size, "width": tile_size}
    )
    processor = BatchMultimodalProcessor(caption_processor, max_length=12, max_num_patches=2)
    record = image_caption_to_conversation({"image": Image.new("RGB", (16, 16), "red"), "text": "red"})
    output = processor([record])[0]
    pixels = output["pixel_values"]
    exemplar = processor.output_exemplar["pixel_values"]
    assert pixels is not None and exemplar is not None
    assert pixels.shape == exemplar.shape == (3, 3, tile_size, tile_size)
    expected = caption_processor.image_processor(images=record["images"], return_tensors="np")["pixel_values"][0]
    np.testing.assert_allclose(pixels[0], expected)
    np.testing.assert_array_equal(pixels[1:], 0)


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
    # The dataset must preserve different target lengths even when callers request them together.
    with set_current_client(LocalClient()):
        dataset = config.train_set(options=CacheOptions(batch_size=1), max_num_patches=3)
        first, second = asyncio.run(dataset.get_batch([0, 1]))
    expected_pixels = caption_processor.image_processor(images=Image.open(image), return_tensors="np")["pixel_values"][
        0
    ]
    for item, length in zip(sorted([first, second], key=lambda item: int(item["attention_mask"].sum())), [8, 9]):
        expected_length = length if use_cache else 12
        assert item["input_ids"].shape == (expected_length,)
        assert item["pixel_values"] is not None
        assert item["pixel_values"].shape == (4, 3, 4, 4)
        np.testing.assert_allclose(item["pixel_values"][0], expected_pixels)
        np.testing.assert_array_equal(item["grid_mask"], [True, False, False, False])
        np.testing.assert_array_equal(
            item["loss_mask"], [0] * 6 + [1] * (length - 6) + [0] * (expected_length - length)
        )
    if use_cache:
        restored = config.train_set(options=CacheOptions(batch_size=1), max_num_patches=3)
        np.testing.assert_array_equal(asyncio.run(restored.get_batch([1]))[0]["input_ids"], second["input_ids"])


def test_any_resolution_uses_processor_encoder_patch_size(caption_processor) -> None:
    hf_processor = LlavaNextProcessor(
        image_processor=LlavaNextImageProcessor(
            size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}, image_grid_pinpoints=[[8, 8]]
        ),
        tokenizer=caption_processor.tokenizer,
        patch_size=2,
        vision_feature_select_strategy="full",
        num_additional_image_tokens=0,
        chat_template=caption_processor.chat_template,
    )
    processor = BatchMultimodalProcessor(hf_processor, max_length=64)
    record = image_caption_to_conversation({"image": Image.new("RGB", (8, 8), "red"), "text": "red"})
    output = processor([record])[0]
    assert output["pixel_values"].shape == (5, 3, 4, 4)
    np.testing.assert_array_equal(output["grid_mask"], [True] * 5)
    # Four base tokens followed by the four 2x2 tile grids in spatial row order.
    np.testing.assert_array_equal(
        output["unpad_indices"], [0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13, 16, 17, 14, 15, 18, 19]
    )
