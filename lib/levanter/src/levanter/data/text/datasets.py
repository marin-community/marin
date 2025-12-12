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
from draccus import ChoiceRegistry, field
from haliax import Axis
from jaxtyping import PRNGKeyArray

import levanter
from levanter.data import AsyncDataset
from levanter.data.dataset import MappedAsyncDataset
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
class LmDatasetSourceConfigBase(ChoiceRegistry):
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

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "url"


@LmDatasetSourceConfigBase.register_subclass("hf")
@dataclass(frozen=True)
class HfDatasetSourceConfig(LmDatasetSourceConfigBase):
    """
    This class represents a dataset source with hf id and optional name.
    """

    id: str = dataclasses.field(kw_only=True)
    name: Optional[str] = None  # name for hf dataset
    stream: bool = True  # whether to use streaming when doing hf
    splits: Optional[List[str]] = None

    def get_shard_source(self, split) -> Optional[ShardedDataSource[dict]]:
        if self.splits is not None and split not in self.splits:
            logger.warning(f"Splits {split} not found for {self.id} {self.name}")
            return None
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


@LmDatasetSourceConfigBase.register_subclass("url")
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


LmDatasetSourceConfig: TypeAlias = UrlDatasetSourceConfig | HfDatasetSourceConfig
LMDatasetSourceConfig: TypeAlias = LmDatasetSourceConfig


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


@dataclass(frozen=True)
class DatasetComponent:
    """A single dataset component with optional source and cache."""

    source: LmDatasetSourceConfigBase | None = None
    cache_dir: str | None = None
    format: LmDatasetFormatBase = field(default_factory=TextLmDatasetFormat)
    pack: bool | int | Literal["pad"] | None = None
    tags: Optional[List[str]] = None


def _effective_pack(component: DatasetComponent) -> bool | int | Literal["pad"]:
    if component.pack is not None:
        return component.pack
    fmt = component.format
    if isinstance(fmt, TextLmDatasetFormat):
        return False
    if isinstance(fmt, ChatLmDatasetFormat):
        return True if fmt.pack is None else fmt.pack
    return False


class PackedTokenDataset(MappedAsyncDataset[tuple[dict, dict], LmExample]):
    """Packed version of token dataset using GreedyPrepackedDataset."""

    def __init__(
        self,
        cache: TreeCache[dict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
    ):
        self.packed: GreedyPrepackedDataset[dict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(e: tuple[dict, dict]) -> LmExample:
            example, seg_ids = e
            tokens = hax.named(example["input_ids"], self.Pos)
            loss_weight = hax.ones_like(tokens)
            seg_ids_named = hax.named(seg_ids["input_ids"], self.Pos)
            out = LmExample.causal(tokens=tokens, loss_weight=loss_weight, segment_ids=seg_ids_named)
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_lm_example)


class ChatDataset(MappedAsyncDataset[tuple[ProcessedChatDict, ProcessedChatDict], LmExample]):
    """
    A dataset that yields multiturn chat examples from a cache of processed chat data.
    """

    def __init__(
        self,
        cache: TreeCache[ProcessedChatDict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
        mask_user_turns: bool = True,
    ):
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
                mask = example["assistant_masks"]
                mask = jnp.roll(mask, -1, axis=-1)
                loss_weight = hax.named(mask, self.Pos)
            else:
                loss_weight = None

            seg_ids_named = hax.named(seg_ids["input_ids"], self.Pos)

            out = LmExample.causal(tokens=tokens, loss_weight=loss_weight, segment_ids=seg_ids_named)
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_lm_example)


def dataset_for_component(
    component: DatasetComponent,
    Pos: Axis,
    cache: TreeCache[dict],
    *,
    eos_id: int | None,
) -> AsyncDataset[LmExample]:
    pack = _effective_pack(component)
    fmt = component.format
    if isinstance(fmt, TextLmDatasetFormat):
        if pack == "pad":
            raise NotImplementedError("Padding mode not yet implemented.")
        if pack:
            max_segments = 64 if pack is True else int(pack)
            return PackedTokenDataset(cache, Pos, max_segments_per_example=max_segments)
        else:
            return CausalLmDataset(TokenSeqDataset(cache, Pos.size), Pos, eos_id=eos_id)
    elif isinstance(fmt, ChatLmDatasetFormat):
        effective_pack = pack
        if effective_pack == "pad":
            raise NotImplementedError("Padding mode not yet implemented.")
        max_segments = (
            64 if effective_pack is True else (int(effective_pack) if isinstance(effective_pack, int) else 1)
        )
        mask_user_turns = fmt.mask_user_turns
        return ChatDataset(cache, Pos, max_segments_per_example=max_segments, mask_user_turns=mask_user_turns)  # type: ignore
    else:
        raise ValueError(f"Unknown format {fmt}")


def _component_cache_dir(name: str, component: DatasetComponent, default_root: str | None) -> str:
    base = component.cache_dir if component.cache_dir is not None else default_root
    if base is None:
        raise ValueError(f"No cache_dir provided for component {name}")
    if component.cache_dir is None:
        return os.path.join(base, name)
    return base


@dataclass(frozen=True)
class LmDataConfig(LMTaskConfig):
    """Unified LM data config built from components."""

    components: Dict[str, DatasetComponent] = field(default_factory=dict)
    train_weights: Union[Dict[str, float], List[Tuple[int, Dict[str, float]]]] | None = None

    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    target_budget: Optional[int] = None
    experiment_budget: Optional[int] = None
    mixture_block_size: int = 2048
    max_train_batches: Optional[Dict[str, int]] = None
    num_validation_sequences: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.components and self.train_weights is None:
            object.__setattr__(self, "train_weights", {name: 1.0 for name in self.components})

        weights = self.train_weights
        if weights is not None:
            if isinstance(weights, dict):
                if not all(name in self.components for name in weights):
                    raise ValueError("Weight keys must be subset of component keys.")
            elif isinstance(weights, list):
                for _, w in weights:
                    if not all(name in self.components for name in w):
                        raise ValueError("Weight keys must be subset of component keys.")
            else:
                raise ValueError(f"Invalid train_weights type: {type(weights)}")

        if self.max_train_batches is not None or self.num_validation_sequences is not None:
            assert (
                self.experiment_budget is None and self.target_budget is None
            ), "max_train_batches/num_validation_sequences and simulated data budget cannot all be set"

    def build_token_datasets(self, caches: Mapping[str, TreeCache[dict]], Pos: Axis):
        return {
            name: dataset_for_component(self.components[name], Pos, cache, eos_id=self.the_tokenizer.eos_token_id)
            for name, cache in caches.items()
        }

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
        if isinstance(weights, list):
            weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)
        initial_batch_size = batch_schedule.batch_size_at_step(0)
        datasets = self.train_sets(Pos, key=shuffle_key, epochs=epochs, initial_batch_size=initial_batch_size)
        mixture = MixtureDataset(
            datasets=datasets,
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

        perm_type = self.permutation_type
        if perm_type is None and self.shuffle is not False:
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        def shuffle_ds(ds, k):
            if self.shuffle is True:
                ds = ds.shuffle(k, perm_type=perm_type)
            elif isinstance(self.shuffle, int) and self.shuffle > 0:
                ds = ds.era_shuffle(self.shuffle, key=k, perm_type=perm_type)
            return ds

        if self.shuffle:
            key_iter = key_iterator(key)
            datasets = {name: shuffle_ds(ds, next(key_iter)) for name, ds in datasets.items()}

        if (
            self.experiment_budget is not None and self.target_budget is not None
        ) and self.experiment_budget > self.target_budget:
            raise ValueError(
                f"Experiment budget should be smaller than target budget, got {self.experiment_budget} > {self.target_budget}"
            )
        if self.experiment_budget is not None and self.target_budget is not None:
            simulated_data_ratio = self.experiment_budget / self.target_budget
            sliced_datasets: Dict[str, AsyncDataset[LmExample]] = {}
            for name, ds in datasets.items():
                true_length_of_dataset = len(ds.as_sync_dataset())
                simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
                sliced_datasets[name] = ds.slice_dataset(end_index=simulated_length_of_dataset)
            datasets = sliced_datasets

        if self.num_validation_sequences is not None:
            for name, ds in datasets.items():
                if name in self.num_validation_sequences:
                    num_sequences = self.num_validation_sequences[name]
                    len_dataset = len(ds.as_sync_dataset())
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
                validation_dataset = train_datasets[name].slice_dataset(
                    start_index=len_dataset - num_sequences, end_index=len_dataset
                )
                validation_datasets[name] = validation_dataset

        return validation_datasets

    def build_caches(self, split: str) -> Dict[str, TreeCache[dict]]:
        caches: dict[str, TreeCache[dict]] = {}
        weights = self.train_weights
        for name, component in self.components.items():
            if split == "train" and weights is not None:
                if isinstance(weights, dict):
                    has_nonzero = weights.get(name, 0) > 0
                else:
                    has_nonzero = any(w.get(name, 0) > 0 for _, w in weights)
                if not has_nonzero:
                    continue

            cache_root = _component_cache_dir(name, component, self.cache_dir)
            source = component.source

            if source is None:
                try:
                    caches[name] = load_lm_dataset_cache(
                        os.path.join(cache_root, split), component.format, self.the_tokenizer, self.enforce_eos
                    )
                except FileNotFoundError:
                    raise ValueError(f"No source and no cache found for component {name} split {split}")
                continue

            shard_source = source.get_shard_source(split)
            if shard_source is None:
                cache_path = os.path.join(cache_root, split)
                if not fsspec_utils.exists(cache_path):
                    logger.warning(f"No source for {name} in {split} split and no cache at {cache_path}, skipping")
                    continue
                caches[name] = load_lm_dataset_cache(
                    cache_path, component.format, self.the_tokenizer, self.enforce_eos
                )
                continue

            caches[name] = build_lm_dataset_cache(
                os.path.join(cache_root, split),
                shard_source,
                component.format,
                self.the_tokenizer,
                self.cache_options,
                self.enforce_eos,
            )

        return caches

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return {name: comp.source for name, comp in self.components.items() if comp.source is not None}

    def tagged_eval_sets(self, Pos: Axis) -> list[Tuple[AsyncDataset[LmExample], List[str]]]:
        eval_sets = self.validation_sets(Pos)
        tagged = []
        for name, ds in eval_sets.items():
            tags = (self.components[name].tags or []) + [name]
            tagged.append((ds, tags))
        return tagged


LMMixtureDatasetConfig: TypeAlias = LmDataConfig


def count_corpus_sizes(
    config: LmDataConfig,
    prefix: str = "data/stats/",
    seq_len: int = 4096,
) -> dict:
    stats = {}
    train_caches = config.build_caches("train")
    Pos = hax.Axis("position", seq_len)

    weights: dict[str, float]
    if isinstance(config.train_weights, list):
        logger.warning("Stats are computed using the first stage of the mixture schedule.")
        train_weights = config.train_weights[0][1]
    else:
        train_weights = config.train_weights or {name: 1.0 for name in train_caches}
    total_weight = sum(train_weights.values()) if train_weights else 1.0
    weights = {name: weight / total_weight for name, weight in (train_weights or {}).items()}

    for name, cache in train_caches.items():
        metric_prefix = f"{prefix}train/{name}/"
        stats[f"{metric_prefix}total_tokens"] = cache.store.tree["input_ids"].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree["input_ids"].num_rows
        train_set = dataset_for_component(config.components[name], Pos, cache, eos_id=None)
        train_seqs = len(train_set.as_sync_dataset())
        stats[f"{metric_prefix}total_seqs"] = train_seqs
        padding_fraction = 1 - (cache.store.tree["input_ids"].data_size / (train_seqs * seq_len))
        if padding_fraction < 0:
            stats[f"{metric_prefix}truncation_fraction"] = -padding_fraction
        else:
            stats[f"{metric_prefix}padding_fraction"] = padding_fraction
        if name in weights:
            weight = weights.get(name, 0.0)
            stats[f"{metric_prefix}weight"] = weight
            stats[f"{metric_prefix}normalized_weight"] = weights[name]
            stats[f"{metric_prefix}approx_global_tokens_per_epoch"] = train_seqs * seq_len / max(weight, 1e-8)

    validation_caches = config.build_caches("validation")
    for name, cache in validation_caches.items():
        metric_prefix = f"{prefix}validation/{name}/"
        stats[f"{metric_prefix}total_tokens"] = cache.store.tree["input_ids"].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree["input_ids"].num_rows
        validation_set = dataset_for_component(config.components[name], Pos, cache, eos_id=None)
        stats[f"{metric_prefix}total_seqs"] = len(validation_set.as_sync_dataset())

    return stats


if __name__ == "__main__":

    @levanter.config.main()
    def main(config: LmDataConfig):
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
