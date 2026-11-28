# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from dataclasses import replace
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
    LlavaOnevisionImageProcessorPil,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
    PreTrainedTokenizerFast,
    Siglip2ImageProcessorPil,
    Siglip2Processor,
)

from levanter.data.image import (
    ConversationDatasetSourceConfig,
    ImageCaptionDatasetSourceConfig,
    ImageCaptionProcessor,
    ImageTextConversationProcessor,
    ImageTextMixtureDatasetConfig,
    image_source_urls,
    image_caption_to_conversation,
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
    source = ConversationDatasetSourceConfig(train_urls=[str(path)]).get_shard_source("train")
    assert source is not None
    loaded = list(source.open_shard(source.shard_names[0]))
    assert loaded == [{**record, "images": []}]


def test_parquet_image_source_resumes_across_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    table = pa.table({"picture": ["a.png", "b.png", "c.png"], "caption": ["a", "b", "c"]})
    pq.write_table(table, path, row_group_size=1)
    config = ImageCaptionDatasetSourceConfig(train_urls=[str(path)], image_key="picture", text_key="caption")
    source = config.get_shard_source("train")
    assert source is not None
    rows = list(source.open_shard_at_row(source.shard_names[0], 1))
    assert rows == [
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
    iterator = ImageCaptionDatasetSourceConfig(id="fixture", stream=True).doc_iterator("train")
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


@pytest.fixture
def caption_processor():
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["[UNK]", "[PAD]", "<turn>", "</turn>", "user", "assistant", "<image>", "red", "square"]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<turn>", "</turn>", "<image>"],
    )
    processor = CLIPProcessor(
        image_processor=CLIPImageProcessor(size={"shortest_edge": 4}, crop_size={"height": 4, "width": 4}),
        tokenizer=tokenizer,
    )
    processor.chat_template = (
        "{% for message in messages %}<turn> {{ message['role'] }} "
        "{% macro content(items) %}{% for item in items %}{% if item['type'] == 'image' %}<image> "
        "{% else %}{{ item['text'] }} {% endif %}{% endfor %}</turn>{% endmacro %}"
        "{% if message.role == 'assistant' %}{% generation %}{{ content(message.content) }}{% endgeneration %}"
        "{% else %}{{ content(message.content) }}{% endif %} {% endfor %}"
    )
    return processor


@pytest.mark.parametrize("max_length", [1, 2, 4])
def test_caption_source_preserves_images_and_text_targets(tmp_path, caption_processor, max_length):
    image = tmp_path / "red.png"
    Image.new("RGB", (4, 4), "red").save(image)
    records = tmp_path / "captions.jsonl"
    records.write_text(json.dumps({"image": str(image), "text": "red square"}) + "\n")
    source = ImageCaptionDatasetSourceConfig(train_urls=[str(records)]).get_shard_source("train")
    assert source is not None
    batch = list(source.open_shard(source.shard_names[0]))
    processed = ImageCaptionProcessor(caption_processor, max_length=max_length)(batch)[0]
    expected_pixels = caption_processor.image_processor(images=Image.open(image), return_tensors="np")["pixel_values"][
        0
    ]
    np.testing.assert_allclose(processed["pixel_values"][0], expected_pixels)
    np.testing.assert_array_equal(processed["loss_mask"], processed["attention_mask"])


def test_siglip2_caption_preserves_native_patch_inputs(caption_processor):
    hf_processor = Siglip2Processor(
        image_processor=Siglip2ImageProcessorPil(patch_size=4, max_num_patches=16),
        tokenizer=caption_processor.tokenizer,
    )
    image = Image.new("RGB", (12, 8), "blue")
    record = {"image": image, "text": "red square"}
    processor = ImageCaptionProcessor(hf_processor, max_length=12)
    result = processor([record])[0]
    expected = hf_processor(
        images=[image],
        text=["red square"],
        padding="max_length",
        max_length=12,
        return_attention_mask=True,
        truncation=True,
        return_tensors="np",
    )
    for field in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
        np.testing.assert_array_equal(result[field], expected[field])
    np.testing.assert_array_equal(result["input_ids"], expected["input_ids"][0])
    np.testing.assert_array_equal(result["loss_mask"], expected["attention_mask"][0])


def test_assistant_mask_requires_generation_block(caption_processor):
    caption_processor.chat_template = "{{ messages[0].role }}"
    with pytest.raises(ValueError, match="generation block"):
        ImageTextConversationProcessor(caption_processor, max_length=12)


@pytest.mark.parametrize("tile_size", [4, 8])
def test_processor_dimensions_match_processed_pixels_and_cache_schema(caption_processor, tile_size) -> None:
    caption_processor.image_processor = CLIPImageProcessor(
        size={"shortest_edge": tile_size}, crop_size={"height": tile_size, "width": tile_size}
    )
    processor = ImageCaptionProcessor(caption_processor, max_length=12)
    record = {"image": Image.new("RGB", (16, 16), "red"), "text": "red"}
    output = processor([record])[0]
    pixels = output["pixel_values"]
    exemplar = processor.output_exemplar["pixel_values"]
    assert pixels.shape == exemplar.shape == (1, 3, tile_size, tile_size)
    expected = caption_processor.image_processor(images=[record["image"]], return_tensors="np")["pixel_values"][0]
    np.testing.assert_allclose(pixels[0], expected)


@pytest.mark.parametrize("source_kind", ["caption", "conversation"])
def test_image_mixture_caches_processor_pixels_and_unpadded_targets(
    tmp_path, caption_processor, source_kind, monkeypatch
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
        source = ImageCaptionDatasetSourceConfig(train_urls=[str(records)])
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *args, **kwargs: caption_processor)
    config = ImageTextMixtureDatasetConfig(
        processor="local-test-processor",
        configs={"tiny": source},
        train_weights={"tiny": 1},
        cache_dir=str(tmp_path / "cache"),
        mixture_block_size=2,
        padding=False,
        max_length=12,
    )
    # The dataset must preserve different target lengths even when callers request them together.
    with set_current_client(LocalClient()):
        dataset = config.train_set(options=CacheOptions(batch_size=1))
        first, second = asyncio.run(dataset.get_batch([0, 1]))
    expected_pixels = caption_processor.image_processor(images=Image.open(image), return_tensors="np")["pixel_values"][
        0
    ]
    expected_lengths = [8, 9] if source_kind == "conversation" else [1, 2]
    for item, length in zip(
        sorted([first, second], key=lambda item: int(item["attention_mask"].sum())), expected_lengths
    ):
        assert item["input_ids"].shape == (length,)
        assert item["pixel_values"].shape == (1, 3, 4, 4)
        np.testing.assert_allclose(item["pixel_values"][0], expected_pixels)
        if source_kind == "conversation":
            np.testing.assert_array_equal(item["loss_mask"], [0] * 6 + [1] * (length - 6))
        else:
            np.testing.assert_array_equal(item["loss_mask"], item["attention_mask"])
    restored = config.train_set(options=CacheOptions(batch_size=1))
    np.testing.assert_array_equal(asyncio.run(restored.get_batch([1]))[0]["input_ids"], second["input_ids"])
    if source_kind == "conversation":
        caption_processor.chat_template = caption_processor.chat_template.replace("{% generation %}", "").replace(
            "{% endgeneration %}", ""
        )
        with set_current_client(LocalClient()):
            dataset = replace(config, mask_prompt=False).train_set(options=CacheOptions(batch_size=1))
            row = asyncio.run(dataset.get_batch([0]))[0]
        np.testing.assert_array_equal(row["loss_mask"], row["attention_mask"])


@pytest.mark.parametrize("onevision", [False, True], ids=["llava-next", "onevision"])
def test_multi_image_expansion_preserves_processor_features_and_assistant_targets(
    caption_processor, onevision
) -> None:
    if onevision:
        hf_processor = LlavaOnevisionProcessor(
            image_processor=LlavaOnevisionImageProcessorPil(
                size={"height": 8, "width": 8}, image_grid_pinpoints=[[8, 8]]
            ),
            tokenizer=caption_processor.tokenizer,
            video_processor=LlavaOnevisionVideoProcessor(),
            num_image_tokens=4,
            chat_template=caption_processor.chat_template,
        )
    else:
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
    processor = ImageTextConversationProcessor(hf_processor, max_length=128)
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")]
    record = image_caption_to_conversation({"image": images[0], "text": "red"})
    record["images"] = images
    record["messages"][0]["content"].append({"type": "image"})
    output = processor([record])[0]
    prompt = hf_processor.apply_chat_template(record["messages"], tokenize=False)
    expected = hf_processor(images=[images], text=[prompt], return_tensors="np")
    np.testing.assert_array_equal(output["pixel_values"], expected["pixel_values"])
    np.testing.assert_array_equal(output["image_sizes"], expected["image_sizes"])
    ids = output["input_ids"]
    np.testing.assert_array_equal(ids[output["attention_mask"].astype(bool)], expected["input_ids"][0])
    assert (ids == hf_processor.image_token_id).sum() > 2
    selected = ids[output["loss_mask"].astype(bool)]
    assert selected.tolist() == hf_processor.tokenizer.convert_tokens_to_ids(["red", "</turn>"])


def test_image_source_rejects_hf_id_with_urls() -> None:
    with pytest.raises(ValueError, match="either a Hugging Face dataset ID or split URLs"):
        ImageCaptionDatasetSourceConfig(id="org/dataset", train_urls=["records.jsonl"])
