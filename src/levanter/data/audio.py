# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import braceexpand
import datasets
import equinox as eqx
import fsspec
import jax
import numpy as np
from draccus import field
from jaxtyping import PRNGKeyArray
from typing_extensions import TypedDict

import haliax as hax
from haliax import Axis

from levanter.compat.hf_checkpoints import load_processor, load_tokenizer
from levanter.data import AsyncDataset
from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.metrics_monitor import LoggerMetricsMonitor, LoggingMetricsMonitor, MetricsMonitor
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.sharded_datasource import AudioTextUrlDataSource, ShardedDataSource, WrappedHFDataSource
from levanter.data.text import BatchTokenizer

# intercept the logging nonsense here
from levanter.models.asr_model import AudioTextExample
from levanter.store.cache import CacheOptions, TreeCache, build_or_load_cache
from levanter.utils.hf_utils import HfTokenizer
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()  # noqa
from transformers import (  # noqa
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    SequenceFeatureExtractor,
)


logger = logging.getLogger("levanter.data.audio")

AudioTextDict = TypedDict(
    "AudioTextDict",
    {
        "input_features": np.ndarray,
        "input_ids": np.ndarray,
        "attention_mask": np.ndarray,
    },
)

AudioTextDict_exemplar = {
    "input_features": np.zeros((1, 1), dtype=np.float32),
    "input_ids": np.zeros((0,), dtype=np.int32),
    "attention_mask": np.zeros((0,), dtype=np.int32),
}


class BatchAudioProcessor(BatchProcessor[Tuple[np.ndarray, int, str], AudioTextDict]):
    """
    A batch processor that converts raw audio into the expected inputs of a model.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        tokenizer: HfTokenizer,
        enforce_bos=True,
        enforce_eos=True,
        *,
        override_resources=None,
        max_length=448,
        padding=True,
    ):
        self.feature_extractor: SequenceFeatureExtractor = processor.feature_extractor
        self.bt = BatchTokenizer(
            tokenizer,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            override_resources=override_resources,
            return_attention_mask=True,
            padding="max_length" if padding else False,
            max_length=max_length,
        )

        self.override_resources = override_resources

    def __call__(self, batch: Sequence[Tuple[np.ndarray, int, str]]) -> Sequence[AudioTextDict]:
        """
        Process a batch of data.
        """
        audio_batch: Sequence[np.ndarray]
        sampling_rates: Sequence[int]
        text_batch: Sequence[str]
        audio_batch, sampling_rates, text_batch = list(zip(*batch))
        uniq_sampling_rates: set[int] = set(sampling_rates)
        assert len(uniq_sampling_rates) == 1, "Sampling rates should be standardized"
        audio_features: BatchFeature = self.feature_extractor(audio_batch, sampling_rate=uniq_sampling_rates.pop())
        audio_features["input_features"] = np.array(audio_features["input_features"])
        text_features: list[dict] = self.bt([{"text": text} for text in text_batch])
        text_features = [
            {k: np.array(tf[k], dtype=np.int32) for k in ["input_ids", "attention_mask"]} for tf in text_features
        ]

        # debatch and return
        out = []
        for i, text in enumerate(text_features):
            out.append(
                {
                    "input_features": audio_features["input_features"][i],
                    "input_ids": text["input_ids"],
                    "attention_mask": text["attention_mask"],
                }
            )

        return out  # type: ignore

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.bt.metadata,
            "processor": self.feature_extractor.to_dict(),
        }

    @property
    def output_exemplar(self):
        return AudioTextDict_exemplar

    @property
    def num_cpus(self) -> int:
        """The number of CPUs this processor needs to run."""
        return self.bt.num_cpus

    @property
    def num_gpus(self) -> int:
        return self.bt.num_gpus


@dataclass
class AudioDatasetSourceConfig:
    """This class represents a dataset source with URLs or hf name/id."""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset

    plaintext: bool = False
    stream: bool = True  # whether to use streaming when doing hf
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset
    audio_key: str = "audio"  # key for the text field in the jsonl file or hf dataset
    sampling_rate: int = 16_000

    train_split: str = "train"
    validation_split: str = "validation"
    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore
    cache_dir: str = "cache/"

    def get_shard_source(self, split) -> Optional[ShardedDataSource[Tuple[np.ndarray, int, str]]]:
        if self.id is not None:
            try:
                ds = WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                # if the message starts with Bad split, then just return None
                if str(e).startswith("Bad split"):
                    logger.warning(f"Splits {split} not found for {self.id} {self.name}")
                    return None
                else:
                    raise

            if len(ds.shard_names) == 0:
                return None

            def decode(x):
                text = x[self.text_key]
                audio_pointer = x[self.audio_key]
                audio = AudioTextUrlDataSource.resolve_audio_pointer(audio_pointer, self.sampling_rate)
                return (audio["array"], audio["sampling_rate"], text)

            return ds.map(decode)
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return AudioTextUrlDataSource(split_urls, self.text_key, self.audio_key, sampling_rate=self.sampling_rate)

    def doc_iterator(self, split: str) -> Iterator[Tuple[np.ndarray, int, str]]:
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield (doc[self.audio_key]["array"], doc[self.audio_key]["sampling_rate"], doc[self.text_key])
        else:
            urls = self.urls_for_split(split)

            yield from AudioTextUrlDataSource(urls, self.text_key, self.audio_key, sampling_rate=self.sampling_rate)

    def urls_for_split(self, split):
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split {split}")

        def fsspec_expand_glob(url):
            if "*" in url:
                fs = fsspec.core.url_to_fs(url)[0]
                return fs.glob(url)
            else:
                return [url]

        urls = [globbed for pat in urls for url in braceexpand.braceexpand(pat) for globbed in fsspec_expand_glob(url)]
        return urls


@dataclass
class AudioTaskConfig(abc.ABC):
    processor: str = "openai/whisper-tiny"
    tokenizer: Optional[str] = None
    enforce_bos: bool = True  # whether to append bos even if the tokenizer doesn't
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't
    max_length: int = 448

    @cached_property
    def the_processor(self) -> ProcessorMixin:
        return load_processor(self.processor)

    @cached_property
    def pad_token_id(self) -> int:
        return self.the_tokenizer.pad_token_id

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            return self.the_processor.tokenizer
        else:
            return load_tokenizer(self.tokenizer)

    @cached_property
    def the_feature_extractor(self) -> SequenceFeatureExtractor:
        return self.the_processor.feature_extractor

    @abc.abstractmethod
    def train_set(
        self,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> AsyncDataset[AudioTextDict]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[np.ndarray]]:
        pass


class ProcessedAudioCache(AsyncDataset[AudioTextDict]):
    """
    Represents a cache of data with both pre-processed audio and tokenized text, which is a directory of parquet files with a ledger file.
    """

    def __init__(self, cache: TreeCache[AudioTextDict]):
        super().__init__()
        self.cache = cache

    async def async_len(self) -> int:
        return await self.cache.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.cache.final_length_is_known()

    def is_finite(self) -> bool:
        return self.cache.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.cache.current_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[AudioTextDict]:
        return await self.cache.get_batch(indices)

    # def _convert_to_example(self, storage: AudioTextStorageBatch) -> AudioTextDict:
    #     storage["input_features"] = storage["input_features"].reshape(storage["audio_shape"])
    #     del storage["audio_shape"]
    #     return storage

    @staticmethod
    def build_or_load(
        cache_dir: str,
        source: ShardedDataSource[Tuple[np.ndarray, int, str]],
        processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizerBase,
        enforce_bos=True,
        enforce_eos=True,
        monitors=None,
        await_finished=True,
        override_resources=None,
        max_length=448,
        cache_options: CacheOptions = CacheOptions.default(),
        split: str = "",
    ) -> "ProcessedAudioCache":
        bp = BatchAudioProcessor(
            processor,
            tokenizer,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            override_resources=override_resources,
            max_length=max_length,
        )
        monitors = monitors or []
        cache = build_or_load_cache(
            cache_dir, source, bp, await_finished=await_finished, monitors=monitors, options=cache_options
        )
        if cache.is_finished:
            logger.info(f"Cache {cache_dir} is complete.")
        else:
            logger.info(
                f"Cache {cache_dir} is incomplete. This will block until at least one chunk per process is complete."
            )

        return ProcessedAudioCache(cache)

    @staticmethod
    def load(cache_dir):
        """
        Load a ProcessedAudioCache from a directory. If the ledger file is not present, this will raise a
        FileNotFoundError.

        NOTE: ATM this attempts to migrate old caches to the new format, but this will be removed in the future.

        :param cache_dir:
        :return:
        """

        try:
            # TODO: populate cache config
            cache = TreeCache.load(cache_dir, AudioTextDict_exemplar, options=None)
            return ProcessedAudioCache(cache)
        except FileNotFoundError:
            raise FileNotFoundError(f"{cache_dir} is not a complete cache")
        except Exception:
            logger.exception("error loading cache")
            raise


@dataclass
class AudioIODatasetConfig(AudioDatasetSourceConfig, AudioTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    def train_set(
        self,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> ProcessedAudioCache:
        ds = self.build_or_load_cache(self.train_split, monitors=monitors)
        if ds is None:
            raise ValueError("No training set!")
        return ds

    def validation_set(self, monitors: Union[bool, List[MetricsMonitor]] = True) -> Optional[ProcessedAudioCache]:
        return self.build_or_load_cache(self.validation_split, monitors=monitors)

    def validation_sets(self, monitors: Union[bool, List[MetricsMonitor]] = True) -> Mapping[str, ProcessedAudioCache]:
        if self._has_validation_set:
            validation_set = self.validation_set(monitors)
            if validation_set is not None:
                return {"": validation_set}
        return {}

    @cached_property
    def _has_validation_set(self):
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            dataset = datasets.load_dataset(
                self.id, name=self.name, streaming=self.stream, split=self.validation_split
            )
            try:
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def build_or_load_cache(
        self,
        split: str,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        logger_name: Optional[str] = None,
        cache_options: CacheOptions = CacheOptions.default(),
    ) -> Optional[ProcessedAudioCache]:
        split_cache_dir = os.path.join(self.cache_dir, split)
        name = logger_name or os.path.basename(self.cache_dir)

        try:
            return ProcessedAudioCache.load(split_cache_dir)
        except FileNotFoundError:
            pass

        source = self.get_shard_source(split)
        if source is None:
            logger.info(f"No data for {split}")
            return None

        logger.info(f"Building cache for {split}...")

        if monitors is True:
            monitors = [
                LoggingMetricsMonitor(prefix=f"preprocessing/{name}/{split}", commit=False),
                LoggerMetricsMonitor(f"preprocessing.{name}.{split}"),
            ]
        elif monitors is False:
            monitors = []

        return ProcessedAudioCache.build_or_load(
            split_cache_dir,
            source,
            self.the_processor,
            self.the_tokenizer,
            enforce_bos=self.enforce_bos,
            enforce_eos=self.enforce_eos,
            monitors=monitors,
            await_finished=(split == "validation"),
            max_length=self.max_length,
            cache_options=cache_options,
        )


class AudioTextDataset(MappedAsyncDataset[AudioTextDict, AudioTextExample]):
    def __init__(
        self,
        dataset: AsyncDataset[AudioTextDict],
        TextPos: Axis,
        AudioPos: hax.AxisSelector,
        KPos: Axis,
        key: Optional[PRNGKeyArray] = None,
        ignore_index: Optional[int] = None,
    ):
        self.dataset = dataset
        self.TextPos = TextPos
        self.AudioPos = AudioPos
        self.KPos = KPos
        self.key = key
        self.ignore_id = ignore_index

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _convert_example(inputs: AudioTextDict) -> "AudioTextExample":
            tokens = hax.named(inputs["input_ids"], self.TextPos)
            audio_features = hax.named(inputs["input_features"], self.AudioPos)
            return AudioTextExample.init(audio_features, tokens, ignore_id=self.ignore_id)

        super().__init__(self.dataset, _convert_example)

    # def __iter__(self) -> Iterator[AudioTextExample]:
    #
    #
    #     with use_cpu_device():
    #
    #
    #         for example in self.dataset:
    #             converted_example = _convert_example(example)
    #             yield converted_example


@dataclass
class AudioMixtureDatasetConfig(AudioTaskConfig):
    """This class represents a mixture of datasets with their associated weights."""

    cache_dir: Optional[str] = "cache/"

    # data source configs and weights
    configs: Dict[str, AudioDatasetSourceConfig] = field(default_factory=dict)
    """ configuration of each dataset source (urls, hf dataset id, etc.) """
    train_weights: Dict[str, float] = field(default_factory=dict)
    """ weights for each dataset source. They will be normalized to sum to 1. """
    shuffle: bool | int = False
    """whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length"""
    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    mixture_block_size: int = 2048
    """ block size for the mixture dataset."""

    def __post_init__(self):
        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if set(self.configs.keys()) != set(self.train_weights.keys()):
            raise ValueError(
                f"The keys in configs and weights must be the same;got {self.configs.keys()} and"
                f" {self.train_weights.keys()}"
            )

    def train_set(
        self,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        options: CacheOptions = CacheOptions.default(),
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> AsyncDataset[AudioTextDict]:
        audio_datasets = self.training_sets(monitors)

        if key is None:
            key = jax.random.PRNGKey(0)

        mix_key, shuffle_key = jax.random.split(key)

        # We shuffle the components and not the overall mixture because this lets us preserve
        # the "stable batch" property of the mixture dataset.
        def shuffle_ds(ds, key):
            if self.shuffle is True:
                ds = ds.shuffle(key)
            elif isinstance(self.shuffle, int):
                ds = ds.era_shuffle(self.shuffle, key=key)

            return ds

        if self.shuffle:
            out_datasets = {}
            key_iter = key_iterator(shuffle_key)
            for name, ds in audio_datasets.items():
                out_datasets[name] = shuffle_ds(ds, next(key_iter))
            audio_datasets = out_datasets

        mixture = MixtureDataset(
            datasets=audio_datasets,
            weights=self.train_weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=2048,
        )

        return mixture

    def training_sets(self, monitors: Union[bool, List[MetricsMonitor]] = True) -> Mapping[str, ProcessedAudioCache]:
        doc_caches = self.build_caches("train", monitors=monitors)
        return doc_caches

    def validation_sets(
        self, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[np.ndarray]]:
        doc_caches = self.build_caches("validation", monitors=monitors)
        return doc_caches

    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Dict[str, ProcessedAudioCache]:
        # this is a bit gross, but we want to forward all "Task" config fields to the AudioIODatasetConfig for building.
        # We do this by just grabbing all the fields from the AudioTaskConfig and forwarding them.
        task_config_fields = set(x.name for x in dataclasses.fields(AudioTaskConfig))
        task_config_dict = {k: v for k, v in self.__dict__.items() if k in task_config_fields and k != "cache_dir"}

        caches = {}
        for name, source_config in self.configs.items():
            weight = self.train_weights.get(name, 0)

            if weight == 0 and split == "train":
                continue

            source_config_dict = dict(**source_config.__dict__)

            if source_config.cache_dir is None:
                # replace with the main cache dir/{name}
                if self.cache_dir is None:
                    raise ValueError(
                        "If the 'main' cache_dir is None, then all component cache_dirs must be non-None, but"
                        f"{name}'s cache_dir is None."
                    )
                cache_dir = os.path.join(self.cache_dir, name)
                source_config_dict["cache_dir"] = cache_dir

            dataset = AudioIODatasetConfig(
                **source_config_dict,
                **task_config_dict,
            )
            if split == "train":
                cache = dataset.build_or_load_cache(dataset.train_split, monitors)
            elif split == "validation":
                cache = dataset.build_or_load_cache(dataset.validation_split, monitors)
            else:
                cache = dataset.build_or_load_cache(split, monitors)
            # drop the data source and corresponding weight if the cache is not built
            if cache is None:
                logger.warning(f"Skipping {name} for split {split} because no source was provided")
            else:
                caches[name] = cache

        # in practice it works best if we block on validation caches
        if split == "validation":
            for cache in caches.values():
                cache.cache.await_finished()

        else:
            logger.info(f"Not waiting for {split} caches to finish building")

        return caches

    @property
    def sources(self) -> Mapping[str, AudioDatasetSourceConfig]:
        return self.configs
