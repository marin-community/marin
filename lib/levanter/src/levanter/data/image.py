# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


"""Image and conversation sources, preprocessing, batching, and caching."""

import abc
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, cast

import datasets
import jax
import numpy as np
import requests
from datasets.fingerprint import Hasher
from draccus import field
from jaxtyping import PRNGKeyArray
from PIL import Image
from rigging.filesystem.storage_path import StoragePath, prefix_join
from typing_extensions import TypedDict

from levanter.compat.hf_checkpoints import load_processor
from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.sharded_datasource import (
    ShardedDataSource,
    UrlDataSource,
    datasource_from_hf_or_none,
)
from levanter.store.cache import CacheOptions, TreeCache, build_or_load_cache
from levanter.tokenizers import chat_template_has_generation_block
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import (  # noqa: E402
    PreTrainedTokenizerBase,
    ProcessorMixin,
)


class ConversationMessage(TypedDict):
    role: str
    content: list[dict[str, Any]]


class ConversationDict(TypedDict, total=False):
    messages: list[ConversationMessage]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class ImageCaptionDict(TypedDict):
    image: Any
    text: str


def image_source_urls(urls: Sequence[str]) -> list[str]:
    """Expand shard patterns and find Parquet files under directories."""
    result: list[str] = []
    for specification in urls:
        for path in StoragePath(specification).expand_glob():
            if path.isdir():
                result.extend(str(match) for match in (path / "**/*.parquet").glob())
            else:
                result.append(str(path))
    return sorted(result)


ImageTextDict = dict[str, np.ndarray]


def load_image_from_path_or_url(path_or_url: str) -> Image.Image:
    """Read an RGB image from a local path, HTTP URL, or storage URL."""
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    elif path_or_url.startswith(("gs://", "s3://", "hf://")):
        with StoragePath(path_or_url).open("rb") as f:
            image = Image.open(f)
            image.load()
    else:
        image = Image.open(path_or_url)

    return image.convert("RGB")


def load_image(image_data: Any) -> Image.Image:
    """Decode an RGB image from a path, array, PIL image, or Hugging Face image record."""
    match image_data:
        case Image.Image():
            return image_data.convert("RGB")
        case str():
            return load_image_from_path_or_url(image_data)
        case np.ndarray():
            return Image.fromarray(image_data).convert("RGB")
        case {"bytes": bytes() as encoded}:
            with Image.open(BytesIO(encoded)) as image:
                return image.convert("RGB")
        case {"path": str() as path}:
            return load_image_from_path_or_url(path)
        case _:
            raise ValueError(f"Unsupported image data: {type(image_data)}")


def _processor_fields(processor: ProcessorMixin, encoded: Mapping[str, Any]) -> ImageTextDict:
    """Remove the batch axis from text fields while preserving grouped image fields."""
    sequence_keys = {*processor.tokenizer.model_input_names, "input_ids", "attention_mask", "mm_token_type_ids"}
    return {key: np.asarray(value[0] if key in sequence_keys else value) for key, value in encoded.items()}


class ImageCaptionProcessor(BatchProcessor[ImageCaptionDict, ImageTextDict]):
    """Prepare contrastive image-caption processor outputs for caching."""

    def __init__(self, processor: ProcessorMixin, *, max_length: int, padding: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.padding = padding

    def _process(self, caption: str, image: Image.Image, *, truncate: bool) -> ImageTextDict:
        options: dict[str, Any] = dict(
            return_tensors="np",
            return_attention_mask=True,
            padding="max_length" if truncate and self.padding else False,
            truncation=truncate,
            max_length=self.max_length if truncate else None,
        )
        encoded = self.processor(text=[caption], images=[image], **options)
        result = _processor_fields(self.processor, encoded)
        result["loss_mask"] = result["attention_mask"].astype(np.float32)
        return result

    @cached_property
    def output_exemplar(self) -> ImageTextDict:
        return self._process("caption", Image.new("RGB", (32, 32)), truncate=False)

    def __call__(self, batch: Sequence[ImageCaptionDict]) -> Sequence[ImageTextDict]:
        return [self._process(item["text"], load_image(item["image"]), truncate=True) for item in batch]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "image-caption",
            "processor": self.processor.to_dict(),
            "tokenizer": self.processor.tokenizer.name_or_path,
            "padding_side": self.processor.tokenizer.padding_side,
            "truncation_side": self.processor.tokenizer.truncation_side,
            "max_length": self.max_length,
            "padding": self.padding,
        }

    @property
    def num_cpus(self) -> int:
        return 2


class ImageTextConversationProcessor(BatchProcessor[ConversationDict, ImageTextDict]):
    """Prepare image-text conversations and assistant loss masks for caching."""

    def __init__(
        self,
        processor: ProcessorMixin,
        *,
        max_length: int,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        add_generation_prompt: bool = False,
        mask_prompt: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.messages_key = messages_key
        self.images_key = images_key
        self.add_generation_prompt = add_generation_prompt
        self.mask_prompt = mask_prompt
        self.template = processor.chat_template or processor.tokenizer.chat_template
        if isinstance(self.template, dict):
            self.template = self.template["default"]
        if self.template is None:
            raise ValueError("Conversation preprocessing requires a chat template")
        if mask_prompt and not chat_template_has_generation_block(self.template):
            raise ValueError("Assistant masking requires a chat template with a generation block")

    def _process(self, messages, images, *, truncate: bool) -> ImageTextDict:
        tokenizer = self.processor.tokenizer
        template_args = dict(chat_template=self.template, add_generation_prompt=self.add_generation_prompt)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, **template_args)
        text_options: dict[str, Any] = dict(
            return_tensors="np",
            return_attention_mask=True,
            padding="max_length" if truncate and self.padding else False,
            truncation=truncate,
            max_length=self.max_length if truncate else None,
        )
        text_options["add_special_tokens"] = False
        # Keep all images for this prompt in one sample, as in ProcessorMixin.apply_chat_template.
        encoded = self.processor(text=[prompt], images=[images] if images else None, **text_options)
        result = _processor_fields(self.processor, encoded)
        ids = result["input_ids"]
        active = result["attention_mask"].astype(bool)
        image_ids = self.processor.image_token_ids
        text = active & ~np.isin(ids, image_ids)
        loss = text.astype(np.float32)
        if self.mask_prompt:
            # Image expansion shifts token positions. Match the remaining text to its template mask.
            original = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, **template_args
            )
            original_ids = np.asarray(original["input_ids"])
            original_text = ~np.isin(original_ids, image_ids)
            count = int(text.sum())
            start = int(original_text.sum()) - count if tokenizer.truncation_side == "left" else 0
            keep = slice(start, start + count)
            if not np.array_equal(ids[text], original_ids[original_text][keep]):
                raise ValueError("Cannot align the processor's text tokens with the chat template's assistant mask")
            loss[text] = np.asarray(original["assistant_masks"])[original_text][keep]
        result["loss_mask"] = loss
        return result

    @cached_property
    def output_exemplar(self) -> ImageTextDict:
        image = Image.new("RGB", (32, 32))
        record = image_caption_to_conversation({"image": image, "text": "caption"})
        return self._process(record["messages"], [image], truncate=False)

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Sequence[ImageTextDict]:
        results = []
        for item in batch:
            if item.get("audio") or item.get("videos"):
                raise ValueError("Image preprocessing accepts image and text content")
            messages = item[self.messages_key]
            images = [load_image(image) for image in item.get(self.images_key, [])]
            fields = self._process(messages, images, truncate=True)
            if not images:
                # TreeCache requires the same fields and ranks for text-only rows.
                for key, exemplar in self.output_exemplar.items():
                    if key not in fields:
                        fields[key] = exemplar[:0]
            results.append(fields)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format": "image-text-conversation",
            "processor": self.processor.to_dict(),
            "tokenizer": self.processor.tokenizer.name_or_path,
            "padding_side": self.processor.tokenizer.padding_side,
            "truncation_side": self.processor.tokenizer.truncation_side,
            "template": self.template,
            "max_length": self.max_length,
            "padding": self.padding,
            "mask_prompt": self.mask_prompt,
            "add_generation_prompt": self.add_generation_prompt,
        }

    @property
    def num_cpus(self) -> int:
        return 2


def image_caption_to_conversation(record: ImageCaptionDict) -> ConversationDict:
    """Represent an image-caption pair as a user image and assistant caption."""
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": record["text"]}]},
        ],
        "images": [record["image"]],
    }


def _raw_source(
    config: "ImageTextSource", split: str, columns: list[str] | None = None
) -> ShardedDataSource[dict] | None:
    """Build a split source; column projection applies only to URL-backed records."""
    if config.id is not None:
        return datasource_from_hf_or_none(config.id, split=split, name=config.name, streaming=config.stream)
    urls = config.urls_for_split(split)
    return UrlDataSource(urls, columns=columns) if urls else None


def _raw_records(config: "ImageTextSource", split: str, columns: list[str] | None = None) -> Iterator[dict]:
    if config.id is not None:
        rows = datasets.load_dataset(config.id, split=split, name=config.name, streaming=config.stream)
        for row in rows:
            yield cast(dict, row)
    elif (source := _raw_source(config, split, columns)) is not None:
        yield from source


def _image_split_urls(split: str, split_urls: Mapping[str, Sequence[str]]) -> List[str]:
    if split not in split_urls:
        raise ValueError(f"Unknown split: {split}")
    return image_source_urls(split_urls[split])


def _validate_source_location(id: str | None, train_urls: Sequence[str], validation_urls: Sequence[str]) -> None:
    if id is not None and (train_urls or validation_urls):
        raise ValueError("Configure either a Hugging Face dataset ID or split URLs, not both")


@dataclass
class ImageCaptionDatasetSourceConfig:
    """Read one image and caption per record."""

    id: Optional[str] = None  # Hugging Face dataset ID or path
    name: Optional[str] = None  # Hugging Face dataset configuration

    stream: bool = True
    image_key: str = "image"
    text_key: str = "text"

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore

    def __post_init__(self) -> None:
        _validate_source_location(self.id, self.train_urls, self.validation_urls)

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[ImageCaptionDict]]:
        """Return image-caption records for the split, or None if it has no data."""
        source = _raw_source(self, split, [self.image_key, self.text_key])
        return None if source is None else source.map(self._caption_fields)

    def _caption_fields(self, record: dict) -> ImageCaptionDict:
        return {"image": record[self.image_key], "text": record[self.text_key]}

    def doc_iterator(self, split: str) -> Iterator[ImageCaptionDict]:
        """Yield records with image and text fields."""
        for row in _raw_records(self, split, [self.image_key, self.text_key]):
            yield self._caption_fields(row)

    def urls_for_split(self, split: str) -> List[str]:
        return _image_split_urls(
            split, {self.train_split: self.train_urls, self.validation_split: self.validation_urls}
        )


@dataclass
class ConversationDatasetSourceConfig:
    """Read chat turns with one image reference per placeholder, in order."""

    id: Optional[str] = None  # Hugging Face dataset ID or path
    name: Optional[str] = None  # Hugging Face dataset configuration

    stream: bool = True
    messages_key: str = "messages"
    images_key: str = "images"

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore

    def __post_init__(self) -> None:
        _validate_source_location(self.id, self.train_urls, self.validation_urls)

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[ConversationDict]]:
        """Return the split as conversations, or None if it has no data."""
        source = _raw_source(self, split)
        return None if source is None else source.map(self._conversation)

    def _conversation(self, record: dict) -> ConversationDict:
        return {
            "messages": record[self.messages_key],
            "images": record.get(self.images_key, []),
            "audio": record.get("audio", []),
            "videos": record.get("videos", []),
        }

    def doc_iterator(self, split: str) -> Iterator[ConversationDict]:
        """Iterate over conversations with normalized field names."""
        for row in _raw_records(self, split):
            yield self._conversation(row)

    def urls_for_split(self, split: str) -> List[str]:
        return _image_split_urls(
            split, {self.train_split: self.train_urls, self.validation_split: self.validation_urls}
        )


ImageTextSource = ImageCaptionDatasetSourceConfig | ConversationDatasetSourceConfig


@dataclass
class ImageTextTaskConfig(abc.ABC):
    """Base configuration for image-text tasks."""

    processor: str
    max_length: int = 2048
    padding: bool = True
    mask_prompt: bool = True

    @cached_property
    def the_processor(self) -> ProcessorMixin:
        return load_processor(self.processor)

    @cached_property
    def pad_token_id(self) -> int:
        return self.the_processor.tokenizer.pad_token_id

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.the_processor.tokenizer

    @abc.abstractmethod
    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
    ) -> AsyncDataset[ImageTextDict]:
        pass

    @abc.abstractmethod
    def validation_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        pass


@dataclass(kw_only=True)
class ImageTextIODatasetConfig(ImageTextTaskConfig):
    """Preprocess a normalized image source into a persistent cache."""

    source: ImageTextSource
    cache_dir: str

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
    ) -> TreeCache[ImageTextDict]:
        dataset = self.build_or_load_cache(self.source.train_split, options)
        if dataset is None:
            raise ValueError("No training set!")
        return dataset

    def validation_sets(self) -> Mapping[str, TreeCache[ImageTextDict]]:
        dataset = self.build_or_load_cache(self.source.validation_split)
        return {"": dataset} if dataset is not None else {}

    def build_or_load_cache(
        self,
        split: str,
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> Optional[TreeCache[ImageTextDict]]:
        source = self.source.get_shard_source(split)
        if source is None:
            return None
        match self.source:
            case ImageCaptionDatasetSourceConfig():
                processor = ImageCaptionProcessor(self.the_processor, max_length=self.max_length, padding=self.padding)
            case ConversationDatasetSourceConfig():
                processor = ImageTextConversationProcessor(
                    self.the_processor,
                    max_length=self.max_length,
                    padding=self.padding,
                    mask_prompt=self.mask_prompt,
                )
        fingerprint = Hasher.hash(processor.metadata)
        cache_dir = prefix_join(self.cache_dir, split)
        return build_or_load_cache(
            prefix_join(cache_dir, f"processor-inputs/{fingerprint}"), source, processor, options=cache_options
        )


@dataclass
class ImageTextMixtureDatasetConfig(ImageTextTaskConfig):
    """Mix cached image-caption and conversation datasets by weight."""

    cache_dir: str = "cache/"

    configs: Dict[str, ImageTextSource] = field(default_factory=dict)
    train_weights: Dict[str, float] = field(default_factory=dict)
    """Relative sampling weights, keyed by source name."""
    shuffle: bool | int = False
    """Use True for a full shuffle, False to keep source order, or an integer to shuffle within eras of that size."""
    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    mixture_block_size: int = 2048

    def __post_init__(self) -> None:
        if not self.configs:
            raise ValueError("At least one dataset must be provided")

        if self.configs.keys() != self.train_weights.keys():
            raise ValueError(
                f"The keys in configs and weights must be the same; got {self.configs.keys()} and"
                f" {self.train_weights.keys()}"
            )

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> AsyncDataset[ImageTextDict]:
        image_datasets = self.training_sets(options=options)

        if key is None:
            key = jax.random.PRNGKey(0)

        mix_key, shuffle_key = jax.random.split(key)

        if self.shuffle:
            shuffled = {}
            for (name, ds), source_key in zip(image_datasets.items(), key_iterator(shuffle_key)):
                if self.shuffle is True:
                    shuffled[name] = ds.shuffle(source_key)
                else:
                    shuffled[name] = ds.era_shuffle(self.shuffle, key=source_key)
            image_datasets = shuffled

        return MixtureDataset(
            datasets=image_datasets,
            weights={name: self.train_weights[name] for name in image_datasets},
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )

    def training_sets(
        self, options: CacheOptions = CacheOptions.default()
    ) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        return self.build_caches("train", options=options)

    def validation_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        return self.build_caches("validation")

    def build_caches(
        self,
        split: str,
        options: CacheOptions = CacheOptions.default(),
    ) -> Dict[str, TreeCache[ImageTextDict]]:
        caches = {}
        for name, source in self.configs.items():
            if self.train_weights[name] == 0 and split == "train":
                continue
            dataset = ImageTextIODatasetConfig(
                processor=self.processor,
                max_length=self.max_length,
                padding=self.padding,
                mask_prompt=self.mask_prompt,
                source=source,
                cache_dir=prefix_join(self.cache_dir, name),
            )
            source_split = (
                source.train_split if split == "train" else source.validation_split if split == "validation" else split
            )
            cache = dataset.build_or_load_cache(source_split, options)
            if cache is not None:
                caches[name] = cache
        return caches

    @property
    def sources(self) -> Mapping[str, ImageTextSource]:
        return self.configs
