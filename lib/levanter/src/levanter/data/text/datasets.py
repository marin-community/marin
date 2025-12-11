# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from draccus import field
from haliax import Axis
from jaxtyping import PRNGKeyArray

import levanter
from levanter.data import AsyncDataset
from levanter.data.dataset import EpochDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy, rescale_mixture_schedule_for_batch_schedule
from levanter.data.packing import GreedyPrepackedDataset
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.data.sharded_datasource import (
    ShardedDataSource,
    UrlDataSource,
    WrappedHFDataSource,
)
from levanter.data.text.cache import build_lm_dataset_cache, load_lm_dataset_cache
from levanter.data.text.formats import (
    ChatLmDatasetFormat,
    LmDatasetFormatBase,
    ProcessedChatDict,
    TextLmDatasetFormat,
)
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions, TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStore
from levanter.utils import fsspec_utils
from levanter.utils.hf_utils import HfTokenizer
from levanter.utils.jax_utils import key_iterator
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.utils.logging import silence_transformer_nag


# Metrics monitoring removed; keep alias for type hints.
MetricsMonitor = Any

silence_transformer_nag()  # noqa

T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"


class TokenSeqDataset(AsyncDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int):
        super().__init__()
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self._store: Optional[TreeStore] = doc_cache.store
        self._cached_len: Optional[int] = None

    async def async_len(self) -> int:
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = self.doc_cache.store
        return self._store.tree["input_ids"]

    async def final_length_is_known(self) -> bool:
        return await self.doc_cache.final_length_is_known()

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        store = await self._await_token_cache()
        return store.data_size // self.seq_len

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        token_arrays = await self._await_token_cache()
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        ds_len = await self.wait_until_len_at_least(max(indices) + 1)
        if ds_len is not None and ds_len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        out = await asyncio.gather(*out)
        return out

    async def wait_until_len_at_least(self, length: int) -> int:
        # length is brutally slow to compute, so we cache it
        if self._cached_len is not None and self._cached_len >= length:
            return self._cached_len

        # TODO: would be better to listen for cache updates
        length = await super().wait_until_len_at_least(length)
        self._cached_len = length
        return length


class CausalLmDataset(MappedAsyncDataset[np.ndarray, LmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[np.ndarray],
        Pos: Axis,
        *,
        eos_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.eos_id = eos_id

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(tokens):
            tokens = hax.named(tokens, self.Pos)
            example = LmExample.causal(tokens=tokens, eos_id=eos_id)

            example = jax.lax.with_sharding_constraint(example, sharding)

            return example

        super().__init__(self.dataset, _create_lm_example)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


@dataclass(frozen=True)
class LmDatasetSourceConfigBase(abc.ABC):
    """This class represents a dataset source with URLs or hf name/id."""

    tags: Optional[List[str]] = None
    """tags for the dataset. Typically the name of the dataset in the config will be added as a tag as well"""
    cache_dir: str | None = None  # Optionally override the cache dir for this component
    format: LmDatasetFormatBase = field(default_factory=TextLmDatasetFormat)
    """format of the dataset."""

    @abc.abstractmethod
    def get_shard_source(self, split) -> Optional[ShardedDataSource[dict]]:
        raise NotImplementedError

    def load_cache(
        self, split, tokenizer: HfTokenizer, override_cache_dir: str | None = None, enforce_eos=True
    ) -> TreeCache[dict]:
        base_cache = override_cache_dir if override_cache_dir is not None else self.cache_dir
        if base_cache is None:
            raise ValueError("cache_dir must be set or override_cache_dir must be provided")
        return load_lm_dataset_cache(os.path.join(base_cache, split), self.format, tokenizer, enforce_eos=enforce_eos)


@dataclass(frozen=True)
class HfDatasetSourceConfig(LmDatasetSourceConfigBase):
    """
    This class represents a dataset source with hf id and optional name.
    """

    id: str = dataclasses.field(kw_only=True)
    name: Optional[str] = None  # name for hf dataset
    stream: bool = True  # whether to use streaming when doing hf

    def get_shard_source(self, split) -> Optional[ShardedDataSource[dict]]:
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

            return ds


@dataclass(frozen=True)
class UrlDatasetSourceConfig(LmDatasetSourceConfigBase):
    train_urls: list[str] = ()  # type: ignore
    validation_urls: list[str] = ()  # type:ignore

    def get_shard_source(self, split) -> Optional[ShardedDataSource[dict]]:
        split_urls = self.urls_for_split(split)

        if len(split_urls) == 0:
            return None

        return UrlDataSource(split_urls)

    def urls_for_split(self, split):
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split {split}")

        # it's ok for there to be no urls for a split, but if there are, they need to be findable
        if len(urls) == 0:
            return []

        if len(urls) == 0:
            raise ValueError(f"No urls found for split {split}")
        return urls


@dataclass(frozen=True)
class LMTaskConfig(abc.ABC):
    tokenizer: str = "gpt2"
    vocab_size: Optional[int] = None  # if using the passthrough tokenizer, this is required

    # config related to caching
    cache_dir: Optional[str] = "cache/"
    cache_options: CacheOptions = field(default_factory=CacheOptions)
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't

    chat_template: str | None = None  # If set, use this template for chat datasets. Otherwise, use the tokenizer's.

    shuffle: bool | int = False
    """whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length"""
    permutation_type: Literal["feistel", "linear"] | None = None
    """
    Type of permutation to use for shuffle.

    If None, defaults to linear, but this will change in the future since Feistel is better.
    """

    @cached_property
    def the_tokenizer(self) -> HfTokenizer:
        if self.tokenizer == "passthrough":
            return PassthroughTokenizer(self.vocab_size)
        else:
            return load_tokenizer(self.tokenizer)

    @abc.abstractmethod
    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        pass

    @abc.abstractmethod
    def train_sets(
        self,
        Pos: Axis,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @abc.abstractmethod
    def validation_sets(self, Pos: Axis) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @abc.abstractmethod
    def build_caches(self, split: str) -> Mapping[str, TreeCache[dict]]:
        pass

    @property
    @abc.abstractmethod
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        pass

    def tagged_eval_sets(self, Pos: Axis) -> list[Tuple[AsyncDataset[LmExample], List[str]]]:
        tags = {name: (config.tags or []) + [name] for name, config in self.sources.items()}
        eval_sets = self.validation_sets(Pos)

        return [(eval_sets[name], tags[name]) for name in eval_sets]


def dataset_for_format(
    format: LmDatasetFormatBase,
    Pos: Axis,
    cache: TreeCache[dict],
    *,
    eos_id: int | None,
) -> AsyncDataset[LmExample]:
    match format:
        case TextLmDatasetFormat():
            return CausalLmDataset(TokenSeqDataset(cache, Pos.size), Pos, eos_id=eos_id)
        case ChatLmDatasetFormat(pack=pack, mask_user_turns=mask_user_turns):
            effective_pack = pack
            if effective_pack is None:
                effective_pack = True
            if effective_pack == "pad":
                raise NotImplementedError("Padding mode is not implemented yet.")
            if isinstance(effective_pack, int):
                max_segments = effective_pack
            else:
                max_segments = 64 if effective_pack else 1
            return ChatDataset(cache, Pos, max_segments_per_example=max_segments, mask_user_turns=mask_user_turns)  # type: ignore
        case _:
            raise ValueError(f"Unknown format {format}")


@dataclass(frozen=True)
class SingleDatasetLMConfigBase(LmDatasetSourceConfigBase, LMTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    cache_dir: Optional[str] = "cache/"

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        del batch_schedule  # unused

        cache = self.build_or_load_cache("train")
        if cache is None:
            raise ValueError("No training set!")
        else:
            ds = dataset_for_format(self.format, Pos, cache, eos_id=self.the_tokenizer.eos_token_id)

        perm_type = self.permutation_type
        if perm_type is None:
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        if self.shuffle is True:
            ds = ds.shuffle(key, perm_type=perm_type)
        elif isinstance(self.shuffle, int) and self.shuffle > 0:
            ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

        if epochs:
            logger.info("Wrapping dataset in epoch dataset")
            ds = EpochDataset(ds, max_epochs=epochs)

        return ds

    def train_sets(
        self,
        Pos: Axis,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        return {
            # we don't care about BatchSchedule in this class
            "": self.train_set(Pos, BatchSchedule(32), key=key, epochs=epochs)
        }

    def validation_set(
        self,
        Pos: Axis,
    ) -> AsyncDataset[LmExample] | None:
        cache = self.build_or_load_cache("validation")
        if cache is None:
            return None

        return dataset_for_format(self.format, Pos, cache, eos_id=self.the_tokenizer.eos_token_id)

    def validation_sets(self, Pos: Axis) -> Mapping[str, AsyncDataset[LmExample]]:
        validation_set = self.validation_set(Pos)
        if validation_set is not None:
            return {"": validation_set}
        else:
            return {}

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return {"": self}

    def build_caches(self, split: str) -> Mapping[str, TreeCache[dict]]:
        out = {}
        cache = self.build_or_load_cache(split)
        if cache is not None:
            out[""] = cache
        return out

    def build_or_load_cache(self, split: str) -> Optional[TreeCache[dict]]:
        tokenizer = self.the_tokenizer
        cache_dir = self.cache_dir
        source = self.get_shard_source(split)
        format = self.format
        enforce_eos = self.enforce_eos
        options = self.cache_options

        if cache_dir is None:
            raise ValueError("cache_dir cannot be None")

        cache_dir = os.path.join(cache_dir, split)

        if fsspec_utils.exists(cache_dir):
            try:
                return load_lm_dataset_cache(cache_dir, format, tokenizer, enforce_eos)
            except FileNotFoundError:
                pass

        if source is None:
            logger.warning(f"Skipping {split} because no source was provided")
            return None

        return build_lm_dataset_cache(cache_dir, source, format, tokenizer, options, enforce_eos)


@dataclass(frozen=True)
class UrlSingleDatasetLMConfig(SingleDatasetLMConfigBase, UrlDatasetSourceConfig):
    pass


@dataclass(frozen=True)
class HfSingleDatasetLMConfig(SingleDatasetLMConfigBase, HfDatasetSourceConfig):
    pass


SingleDatasetLMConfig: TypeAlias = UrlSingleDatasetLMConfig | HfSingleDatasetLMConfig
LMDatasetSourceConfig: TypeAlias = UrlDatasetSourceConfig | HfDatasetSourceConfig


@dataclass(frozen=True)
class LMMixtureDatasetConfig(LMTaskConfig):
    """A mixture of language model datasets that supports dynamic weight changes during training.

    Weights can be specified either as a single dictionary for constant mixing ratios,
    or as a list of (step, weights) tuples to change mixing ratios during training.
    """

    cache_dir: Optional[str] = "cache/"

    configs: Dict[str, LMDatasetSourceConfig] = field(default_factory=dict)
    """ Configuration of each dataset source (urls, hf dataset id, etc.) """

    train_weights: Union[Dict[str, float], List[Tuple[int, Dict[str, float]]]] = field(default_factory=dict)
    """ Dataset mixing weights. Either a constant dict[name->weight] or list of (step, weights) tuples """

    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)

    # Configuration for Simulated Epoching
    target_budget: Optional[int] = None
    experiment_budget: Optional[int] = None

    mixture_block_size: int = 2048
    """Block size for deterministic mixing. In each block, a given dataset will have exactly the same number
    of samples, equal to the expected number of samples in the mixture, rounding in the expected way."""

    max_train_batches: Optional[Dict[str, int]] = None
    """ Maximum number of batches to use from each dataset for training (using the initial batch size)"""

    num_validation_sequences: Optional[Dict[str, int]] = None
    """ Number of validation sequences to sample from the training set for each dataset"""

    def __post_init__(self):
        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if isinstance(self.train_weights, dict):
            if not all(name in self.configs for name in self.train_weights):
                raise ValueError(
                    f"Weight keys {self.train_weights.keys()} must be subset of config keys {self.configs.keys()}"
                )
        elif isinstance(self.train_weights, list):
            for step, weights in self.train_weights:
                if not all(name in self.configs for name in weights):
                    raise ValueError(
                        f"Weight keys {weights.keys()} must be subset of config keys {self.configs.keys()}"
                    )
        else:
            raise ValueError(f"Invalid train_weights type: {type(self.train_weights)}")

        if self.max_train_batches is not None or self.num_validation_sequences is not None:
            assert (
                self.experiment_budget is None and self.target_budget is None
            ), "max_train_batches and num_validation_sequences and simulated data budget cannot all be set"

    def build_token_datasets(self, caches: Mapping[str, TreeCache[dict]], Pos: Axis):
        token_datasets = {
            name: dataset_for_format(
                self.configs[name].format,
                Pos,
                cache,
                eos_id=self.the_tokenizer.eos_token_id,
            )
            for name, cache in caches.items()
        }

        return token_datasets

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        mix_key, shuffle_key = jax.random.split(key)

        weights = self.train_weights
        if isinstance(weights, Sequence):
            weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

        initial_batch_size = batch_schedule.batch_size_at_step(0)

        causal_datasets = self.train_sets(Pos, key=shuffle_key, epochs=epochs, initial_batch_size=initial_batch_size)

        mixture = MixtureDataset(
            datasets=causal_datasets,
            weights=weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )

        return mixture

    def train_sets(
        self,
        Pos: Axis,
        *,
        initial_batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("train")
        datasets = self.build_token_datasets(doc_caches, Pos)

        if epochs:
            raise ValueError("Epochs are not supported for mixture datasets")

        if key is None:
            key = jax.random.PRNGKey(0)

        # We shuffle the components and not the overall mixture because this lets us preserve
        # the "stable batch" property of the mixture dataset.
        perm_type = self.permutation_type
        if perm_type is None and self.shuffle is not False:
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        def shuffle_ds(ds, key):
            if self.shuffle is True:
                ds = ds.shuffle(key, perm_type=perm_type)
            elif isinstance(self.shuffle, int) and self.shuffle > 0:
                ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

            return ds

        if self.shuffle:
            key_iter = key_iterator(key)
            datasets = {name: shuffle_ds(ds, next(key_iter)) for name, ds in datasets.items()}

        if (
            self.experiment_budget is not None and self.target_budget is not None
        ) and self.experiment_budget > self.target_budget:
            raise ValueError(
                f"Experiment budget should be smaller than target budget, got {self.experiment_budget} >"
                f" {self.target_budget}"
            )
        if self.experiment_budget is not None and self.target_budget is not None:
            simulated_data_ratio = self.experiment_budget / self.target_budget
            sliced_datasets: Dict[str, AsyncDataset[LmExample]] = {}
            for name, ds in datasets.items():
                # Note(Will): This blocks on datasets being fully processed even for small simulated runs making simulating data size slightly latency inducing but I think that's ok
                true_length_of_dataset = len(ds.as_sync_dataset())
                simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
                sliced_datasets[name] = ds.slice_dataset(end_index=simulated_length_of_dataset)
            datasets = sliced_datasets

        if self.num_validation_sequences is not None:
            for name, ds in datasets.items():
                if name in self.num_validation_sequences:
                    num_sequences = self.num_validation_sequences[name]
                    len_dataset = len(ds.as_sync_dataset())
                    # Reserve the last N sequences for validation and use the rest for training
                    logger.info(
                        f"Reserving {num_sequences} sequences from {name} training set of size {len_dataset} for"
                        " validation"
                    )
                    datasets[name] = ds.slice_dataset(start_index=0, end_index=len_dataset - num_sequences)

        if self.max_train_batches is not None:
            assert (
                initial_batch_size is not None
            ), "initial_batch_size must be provided if max_train_batches is provided"

            for name, ds in datasets.items():
                if name in self.max_train_batches:
                    num_sequences = self.max_train_batches[name] * initial_batch_size
                    len_dataset = len(ds.as_sync_dataset())
                    assert (
                        num_sequences <= len_dataset
                    ), f"Max sequences for {name} ({num_sequences}) is greater than the dataset size ({len_dataset})"
                    logger.info(f"Selecting {num_sequences} sequences from {name} training set of size {len_dataset}")
                    datasets[name] = ds.slice_dataset(end_index=num_sequences)

        return datasets

    def validation_sets(self, Pos: Axis) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("validation")
        validation_datasets = self.build_token_datasets(doc_caches, Pos)

        if self.num_validation_sequences is not None:
            train_doc_caches = self.build_caches("train")
            train_datasets = self.build_token_datasets(train_doc_caches, Pos)

            for name, num_sequences in self.num_validation_sequences.items():
                len_dataset = len(train_datasets[name].as_sync_dataset())
                logger.info(
                    f"Selecting {num_sequences} sequences from {name} training set of size {len_dataset} for"
                    " validation"
                )
                validation_dataset = train_datasets[name].slice_dataset(
                    start_index=len_dataset - num_sequences, end_index=len_dataset
                )

                if name in validation_datasets:
                    logger.warning(f"Validation dataset {name} already exists, overwriting")

                validation_datasets[name] = validation_dataset

        return validation_datasets

    def build_caches(self, split: str) -> Dict[str, TreeCache[dict]]:
        caches: dict[str, TreeCache[dict]] = {}
        for name, source_config in self.configs.items():
            # Skip datasets with zero weight in all stages
            if isinstance(self.train_weights, dict):
                has_nonzero_weight = self.train_weights.get(name, 0) > 0
            elif isinstance(self.train_weights, list):
                has_nonzero_weight = any(weights.get(name, 0) > 0 for _, weights in self.train_weights)
            else:
                raise ValueError(f"Invalid train_weights type: {type(self.train_weights)}")

            if not has_nonzero_weight and split == "train":
                continue

            if source_config.cache_dir is None:
                # replace with the main cache dir/{name}
                if self.cache_dir is None:
                    raise ValueError(
                        "If the 'main' cache_dir is None, then all component cache_dirs must be non-None, but"
                        f"{name}'s cache_dir is None."
                    )
                cache_dir = os.path.join(self.cache_dir, name)
            else:
                cache_dir = source_config.cache_dir

            source = source_config.get_shard_source(split)

            # drop the data source and corresponding weight if the cache is not built
            if source is None:
                try:
                    caches[name] = load_lm_dataset_cache(
                        os.path.join(cache_dir, split),
                        source_config.format,
                        self.the_tokenizer,
                        self.enforce_eos,
                    )
                except FileNotFoundError:
                    logger.warning(f"No source for {name} in {split} split and no cache either, skipping")
                    continue
            else:
                caches[name] = build_lm_dataset_cache(
                    os.path.join(cache_dir, split),
                    source,
                    source_config.format,
                    self.the_tokenizer,
                    self.cache_options,
                    self.enforce_eos,
                )

        return caches

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return self.configs


class ChatDataset(MappedAsyncDataset[tuple[ProcessedChatDict, ProcessedChatDict], LmExample]):
    """
    A dataset that yields multiturn chat examples from a cache of processed chat data.


    Args:
        cache: The cache of processed chat data.
        Pos: The position axis.
        max_segments_per_example: The maximum number of segments to pack into a single example. Set to 1 to disable packing.
        slice_strategy: The strategy to use when an example is too long.
    """

    def __init__(
        self,
        cache: TreeCache[ProcessedChatDict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
        mask_user_turns: bool = True,
    ):
        # NB the GreedyPackedDataset returns a tuple, where the first has the packed leaves
        # and the second has the segment ids
        self.packed: GreedyPrepackedDataset[ProcessedChatDict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
        self.mask_user_turns = mask_user_turns

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(e: tuple[ProcessedChatDict, ProcessedChatDict]) -> LmExample:
            example, seg_ids = e
            tokens = hax.named(example["input_ids"], self.Pos)

            if mask_user_turns:
                # mask is 1 on the position of the assistant tokens
                mask = example["assistant_masks"]
                # loss_weight by convention is 1 on the positions where we compute loss, i.e. shifted back 1
                mask = jnp.roll(mask, -1, axis=-1)
                loss_weight = hax.named(mask, self.Pos)
            else:
                loss_weight = None

            seg_ids = hax.named(seg_ids["input_ids"], self.Pos)

            out = LmExample.causal(tokens=tokens, loss_weight=loss_weight, segment_ids=seg_ids)
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_lm_example)


def count_corpus_sizes(
    config: LMMixtureDatasetConfig | SingleDatasetLMConfig,
    prefix: str = "data/stats/",
    seq_len: int = 4096,
) -> dict:
    """
    Counts the number of tokens in each dataset in the config.

    Args:
        config: the config to count the sizes of
        prefix: prefix to use for all metric keys. Defaults to "data/stats/"
        seq_len: sequence length to assume when computing per-sequence stats (padding/truncation);
            defaults to 4096.

    Returns:
        dict containing statistics about the datasets, with keys flattened using /
    """
    stats = {}

    train_caches = config.build_caches("train")

    sources: Mapping[str, LmDatasetSourceConfigBase]
    if isinstance(config, SingleDatasetLMConfigBase):
        sources = {"": config}
    else:
        sources = config.sources

    Pos = hax.Axis("position", seq_len)

    weights: dict[str, float]
    if isinstance(config, LMMixtureDatasetConfig):
        if isinstance(config.train_weights, list):
            logger.warning("Stats are computed using the first stage of the mixture schedule.")
            # TODO: improve this
            train_weights = config.train_weights[0][1]
        else:
            train_weights = config.train_weights
        total_weight = sum(train_weights.values())

        weights = {name: weight / total_weight for name, weight in train_weights.items()}
    else:
        weights = {name: 1.0 for name in train_caches}

    for name, cache in train_caches.items():
        source = sources[name]
        metric_prefix = f"{prefix}train/{name}/"

        stats[f"{metric_prefix}total_tokens"] = cache.store.tree["input_ids"].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree["input_ids"].num_rows

        train_set = dataset_for_format(source.format, Pos, cache, eos_id=None)
        train_seqs = len(train_set.as_sync_dataset())
        stats[f"{metric_prefix}total_seqs"] = train_seqs

        padding_fraction = 1 - (cache.store.tree["input_ids"].data_size / (train_seqs * seq_len))
        if padding_fraction < 0:
            stats[f"{metric_prefix}truncation_fraction"] = -padding_fraction
        else:
            stats[f"{metric_prefix}padding_fraction"] = padding_fraction

        if isinstance(config, LMMixtureDatasetConfig):
            weight = weights.get(name, 0.0)
            stats[f"{metric_prefix}weight"] = weight
            stats[f"{metric_prefix}normalized_weight"] = weights[name]
            stats[f"{metric_prefix}approx_global_tokens_per_epoch"] = train_seqs * seq_len / weight

    validation_caches = config.build_caches("validation")
    for name, cache in validation_caches.items():
        source = sources[name]
        metric_prefix = f"{prefix}validation/{name}/"

        stats[f"{metric_prefix}total_tokens"] = cache.store.tree["input_ids"].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree["input_ids"].num_rows

        validation_set = dataset_for_format(source.format, Pos, cache, eos_id=None)
        stats[f"{metric_prefix}total_seqs"] = len(validation_set.as_sync_dataset())

    return stats


if __name__ == "__main__":

    @levanter.config.main()
    def main(config: LMMixtureDatasetConfig):
        stats = count_corpus_sizes(config)

        print("TRAIN")
        for key, value in stats.items():
            if key.startswith("data/stats/train/"):
                name = key.split("/")[3]
                metric = key.split("/")[4]
                print(f"{name} {metric}: {value}")

        print("\nVALIDATION")
        for key, value in stats.items():
            if key.startswith("data/stats/validation/"):
                name = key.split("/")[3]
                metric = key.split("/")[4]
                print(f"{name} {metric}: {value}")

    main()
