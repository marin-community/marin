# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


"""Image and conversation sources, preprocessing, batching, and caching."""

import abc
import logging
from dataclasses import dataclass
from functools import cached_property, partial
from io import BytesIO
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union, cast

import datasets
import jax
import numpy as np
from draccus import field

from levanter.data.mixture import MixtureDataset, StopStrategy
from jaxtyping import PRNGKeyArray
from typing_extensions import TypedDict

from levanter.compat.hf_checkpoints import load_processor
from levanter.data.dataset import AsyncDataset
from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import RepeatedDataset
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


import requests  # noqa: E402
from PIL import Image  # noqa: E402
from rigging.filesystem.storage_path import StoragePath, prefix_join  # noqa: E402

logger = logging.getLogger("levanter.data.image")


class ConversationMessage(TypedDict):
    role: str
    content: list[dict[str, Any]]


class ConversationDict(TypedDict, total=False):
    messages: list[ConversationMessage]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


def image_source_urls(urls: Sequence[str]) -> list[str]:
    """Expand shard patterns and recursively discover parquet files in directories."""
    result: list[str] = []
    for specification in urls:
        for path in StoragePath(specification).expand_glob():
            if path.isdir():
                result.extend(str(match) for match in (path / "**/*.parquet").glob())
            else:
                result.append(str(path))
    return sorted(result)


def conversation_fields(record: Mapping[str, Any], messages_key: str, images_key: str) -> ConversationDict:
    """Normalize source field names and retain optional modality references."""
    return {
        "messages": record[messages_key],
        "images": record.get(images_key, []),
        "audio": record.get("audio", []),
        "videos": record.get("videos", []),
    }


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


class BatchMultimodalProcessor(BatchProcessor[ConversationDict, ImageTextDict]):
    """Cache native processor arrays and a loss mask aligned to the expanded token sequence."""

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
        if mask_prompt and self.template is not None and not chat_template_has_generation_block(self.template):
            raise ValueError("Assistant masking requires a chat template with a generation block")

    def _process(self, messages, images, *, truncate: bool) -> ImageTextDict:
        tokenizer = self.processor.tokenizer
        template_args = dict(chat_template=self.template, add_generation_prompt=self.add_generation_prompt)
        if self.template is None:
            match messages:
                case [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(prompt)}]},
                ] if (
                    len(images) == 1
                ):
                    pass
                case _:
                    raise ValueError(
                        "A chat template is required for conversations beyond a single image-caption pair"
                    )
        else:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, **template_args)
        text_options: dict[str, Any] = dict(
            return_tensors="np",
            add_special_tokens=self.template is None,
            return_attention_mask=True,
            padding="max_length" if truncate and self.padding else False,
            truncation=truncate,
            max_length=self.max_length if truncate else None,
        )
        encoded = self.processor(text=[prompt], images=images or None, **text_options)
        # Image tensors keep the processor's dimensions, including any image, tile, or patch axes.
        sequence_keys = {*tokenizer.model_input_names, "input_ids", "attention_mask", "mm_token_type_ids"}
        result = {key: np.asarray(value[0] if key in sequence_keys else value) for key, value in encoded.items()}
        ids = result["input_ids"]
        active = result["attention_mask"].astype(bool)
        image_ids = self.processor.image_token_ids
        text = active & ~np.isin(ids, image_ids)
        loss = text.astype(np.float32)
        if self.mask_prompt and self.template is not None:
            # The template marks the original text. Image expansion may insert many tokens
            # before an assistant turn, so align only the surviving text tokens.
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
            "processor": self.processor.to_dict(),
            "template": self.template,
            "max_length": self.max_length,
            "padding": self.padding,
            "mask_prompt": self.mask_prompt,
            "add_generation_prompt": self.add_generation_prompt,
        }

    @property
    def num_cpus(self) -> int:
        return 2


def image_caption_to_conversation(record: Dict[str, Any]) -> ConversationDict:
    """Represent an image-caption pair as a user image and assistant caption."""
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": record["text"]}]},
        ],
        "images": [record["image"]],
    }


def _image_split_urls(split: str, split_urls: Mapping[str, Sequence[str]]) -> List[str]:
    if split not in split_urls:
        raise ValueError(f"Unknown split: {split}")
    return image_source_urls(list(split_urls[split]))


@dataclass
class ImageDatasetSourceConfig:
    """Configuration for a simple image-text dataset source (single image + text pairs)."""

    id: Optional[str] = None  # HuggingFace dataset id or path
    name: Optional[str] = None  # Dataset configuration name

    stream: bool = True  # Whether to use streaming
    image_key: str = "image"
    text_key: str = "text"

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore
    cache_dir: Optional[str] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[ConversationDict]]:
        """Get a sharded data source for the specified split."""
        source = self._caption_source(split)
        return None if source is None else source.map(image_caption_to_conversation)

    def _caption_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        if self.id is not None:
            source = datasource_from_hf_or_none(self.id, split=split, name=self.name, streaming=self.stream)
            if source is None:
                return None
        else:
            urls = self.urls_for_split(split)
            if not urls:
                return None
            source = UrlDataSource(urls, columns=[self.image_key, self.text_key])
        return source.map(self._caption_fields)

    def _caption_fields(self, record: dict) -> dict:
        return {"image": record[self.image_key], "text": record[self.text_key]}

    def doc_iterator(self, split: str) -> Iterator[Dict[str, Any]]:
        """Iterate over normalized image-caption pairs."""
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield self._caption_fields(cast(dict, doc))
        else:
            source = self._caption_source(split)
            if source is not None:
                yield from source

    def urls_for_split(self, split: str) -> List[str]:
        return _image_split_urls(
            split, {self.train_split: self.train_urls, self.validation_split: self.validation_urls}
        )


@dataclass
class ConversationDatasetSourceConfig:
    """Read conversations whose image placeholders consume image references in order."""

    id: Optional[str] = None  # HuggingFace dataset id or path
    name: Optional[str] = None  # Dataset configuration name

    stream: bool = True  # Whether to use streaming
    messages_key: str = "messages"  # Key for messages field
    images_key: str = "images"  # Key for images field

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type: ignore
    cache_dir: Optional[str] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[ConversationDict]]:
        """Get a sharded data source for the specified split."""
        if self.id is not None:
            source = datasource_from_hf_or_none(self.id, split=split, name=self.name, streaming=self.stream)
            if source is None:
                return None
        else:
            urls = self.urls_for_split(split)
            if not urls:
                return None
            source = UrlDataSource(urls)
        return source.map(partial(conversation_fields, messages_key=self.messages_key, images_key=self.images_key))

    def doc_iterator(self, split: str) -> Iterator[ConversationDict]:
        """Iterate over conversations with normalized field names."""
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield conversation_fields(cast(dict, doc), self.messages_key, self.images_key)
        else:
            source = self.get_shard_source(split)
            if source is not None:
                yield from source

    def urls_for_split(self, split: str) -> List[str]:
        return _image_split_urls(
            split, {self.train_split: self.train_urls, self.validation_split: self.validation_urls}
        )


@dataclass
class ImageTaskConfig(abc.ABC):
    """Base configuration for image-text tasks."""

    processor: str
    max_length: int = 2048
    padding: bool = True

    @cached_property
    def loaded_processor(self) -> ProcessorMixin:
        return load_processor(self.processor)

    @cached_property
    def pad_token_id(self) -> int:
        return self.loaded_processor.tokenizer.pad_token_id

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.loaded_processor.tokenizer

    @abc.abstractmethod
    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
    ) -> AsyncDataset[ImageTextDict]:
        pass

    @abc.abstractmethod
    def validation_sets(self) -> Mapping[str, AsyncDataset[ImageTextDict]]:
        pass


class ProcessedImageCache(AsyncDataset[ImageTextDict]):
    """
    Cache for preprocessed image-text data.
    """

    def __init__(self, cache: TreeCache[ImageTextDict]) -> None:
        super().__init__()
        self.cache = cache

    async def async_len(self) -> int:
        return await self.cache.async_len()

    def is_finite(self) -> bool:
        return self.cache.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ImageTextDict]:
        return await self.cache.get_batch(indices)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        source: ShardedDataSource[ConversationDict],
        processor: ProcessorMixin,
        max_length: int,
        padding: bool = True,
        messages_key: str = "messages",
        images_key: str = "images",
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> "ProcessedImageCache":
        batch_processor = BatchMultimodalProcessor(
            processor,
            max_length=max_length,
            padding=padding,
            messages_key=messages_key,
            images_key=images_key,
        )
        cache = build_or_load_cache(
            prefix_join(cache_dir, "processor-inputs-v1"), source, batch_processor, options=cache_options
        )

        if cache.is_finished:
            logger.info(f"Cache {cache_dir} is complete.")
        else:
            logger.info(f"Cache {cache_dir} is incomplete.")

        return ProcessedImageCache(cache)


@dataclass(kw_only=True)
class ImageIODatasetConfig(ImageTaskConfig):
    """Preprocess a normalized image source into a persistent cache."""

    source: Union[ImageDatasetSourceConfig, ConversationDatasetSourceConfig]
    cache_dir: str

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
    ) -> ProcessedImageCache:
        dataset = self.build_or_load_cache(self.source.train_split, options)
        if dataset is None:
            raise ValueError("No training set!")
        return dataset

    def validation_sets(self) -> Mapping[str, ProcessedImageCache]:
        dataset = self.build_or_load_cache(self.source.validation_split)
        return {"": dataset} if dataset is not None else {}

    def build_or_load_cache(
        self,
        split: str,
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> Optional[ProcessedImageCache]:
        source = self.source.get_shard_source(split)
        if source is None:
            return None
        return ProcessedImageCache.build_or_load(
            prefix_join(self.cache_dir, split),
            source,
            self.loaded_processor,
            max_length=self.max_length,
            padding=self.padding,
            cache_options=cache_options,
        )


@dataclass
class ImageMixtureDatasetConfig(ImageTaskConfig):
    """Mix cached image-caption and conversation datasets by weight."""

    cache_dir: Optional[str] = "cache/"

    configs: Dict[str, Union[ImageDatasetSourceConfig, ConversationDatasetSourceConfig]] = field(default_factory=dict)
    """Configuration of each dataset source (URLs, HF dataset ID, etc.)"""
    train_weights: Dict[str, float] = field(default_factory=dict)
    """Weights for each dataset source. They will be normalized to sum to 1."""
    shuffle: bool | int = False
    """Shuffle each source before mixing: True for a full shuffle, False to preserve order,
    or an integer for the per-source shuffle era length."""
    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    mixture_block_size: int = 2048
    """Block size for the mixture dataset."""

    def __post_init__(self) -> None:
        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if set(self.configs.keys()) != set(self.train_weights.keys()):
            raise ValueError(
                f"The keys in configs and weights must be the same; got {self.configs.keys()} and"
                f" {self.train_weights.keys()}"
            )

    def train_set(
        self,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[ImageTextDict]:
        image_datasets = self.training_sets(options=options)

        if key is None:
            key = jax.random.PRNGKey(0)

        mix_key, shuffle_key = jax.random.split(key)

        # Shuffle components, not the overall mixture, to preserve "stable batch" property
        def shuffle_ds(ds: AsyncDataset[ImageTextDict], key: PRNGKeyArray) -> AsyncDataset[ImageTextDict]:
            if self.shuffle is True:
                ds = ds.shuffle(key)
            elif isinstance(self.shuffle, int):
                ds = ds.era_shuffle(self.shuffle, key=key)
            return ds

        if self.shuffle:
            out_datasets = {}
            key_iter = key_iterator(shuffle_key)
            for name, ds in image_datasets.items():
                out_datasets[name] = shuffle_ds(ds, next(key_iter))
            image_datasets = out_datasets

        if epochs and epochs > 0:
            logger.info(f"Wrapping each dataset in RepeatedDataset with max_epochs={epochs}")
            epoch_wrapped_datasets = {}
            for name, ds in image_datasets.items():
                epoch_wrapped_datasets[name] = RepeatedDataset(ds, epochs=epochs)
            image_datasets = epoch_wrapped_datasets

        mixture = MixtureDataset(
            datasets=image_datasets,
            weights={name: self.train_weights[name] for name in image_datasets},
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )

        return mixture

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
    ) -> Dict[str, ProcessedImageCache]:
        caches = {}
        for name, source in self.configs.items():
            if self.train_weights[name] == 0 and split == "train":
                continue
            cache_dir = source.cache_dir
            if cache_dir is None:
                if self.cache_dir is None:
                    raise ValueError(f"No cache directory configured for {name}")
                cache_dir = prefix_join(self.cache_dir, name)
            dataset = ImageIODatasetConfig(
                processor=self.processor,
                max_length=self.max_length,
                padding=self.padding,
                source=source,
                cache_dir=cache_dir,
            )
            source_split = (
                source.train_split if split == "train" else source.validation_split if split == "validation" else split
            )
            cache = dataset.build_or_load_cache(source_split, options)
            if cache is not None:
                caches[name] = cache
        return caches

    @property
    def sources(self) -> Mapping[str, Union[ImageDatasetSourceConfig, ConversationDatasetSourceConfig]]:
        return self.configs
