# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import dataclasses
import functools
import zlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, TypeAlias, TypeVar

import equinox as eqx
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
from levanter.data.text.examples import (
    GrugLmExample,
    named_lm_example_from_grug,
)
from levanter.data.text.formats import (
    ChatLmDatasetFormat,
    LmDatasetFormatBase,
    PrebuiltLmDatasetFormat,
    ProcessedChatDict,
    TextLmDatasetFormat,
)
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions, TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStore
from levanter.utils import fsspec_utils
from levanter.tokenizers import MarinTokenizer, load_tokenizer as load_marin_tokenizer
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()  # noqa

T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")


class TokenSeqDataset(AsyncDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self._store: TreeStore | None = doc_cache.store

    async def async_len(self) -> int:
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = self.doc_cache.store
        return self._store.tree["input_ids"]

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        if not indices:
            return []

        token_arrays = await self._await_token_cache()
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        ds_len = await self.async_len()
        if ds_len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        out = await asyncio.gather(*out)
        return out


def _single_cpu_sharding() -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])


class NamedLmDataset(MappedAsyncDataset[GrugLmExample, LmExample]):
    """Adapter that wraps unnamed examples into Levanter's NamedArray-based LmExample."""

    def __init__(self, dataset: AsyncDataset[GrugLmExample], Pos: Axis):
        self.dataset = dataset
        self.Pos = Pos
        sharding = _single_cpu_sharding()

        @functools.partial(eqx.filter_jit)
        def _to_named(example: GrugLmExample) -> LmExample:
            out = named_lm_example_from_grug(example, Pos)
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(dataset, _to_named)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


class CausalLmDataset(MappedAsyncDataset[np.ndarray, GrugLmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[np.ndarray],
        Pos: Axis,
        *,
        eos_id: int | None = None,
        block_cross_document_attention: bool = True,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.eos_id = eos_id
        self.block_cross_document_attention = block_cross_document_attention

        sharding = _single_cpu_sharding()

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(tokens: jax.Array) -> GrugLmExample:
            example = GrugLmExample.causal(
                tokens=tokens,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )

            example = jax.lax.with_sharding_constraint(example, sharding)

            return example

        super().__init__(self.dataset, _create_lm_example)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


def _identity_loss_weight(loss_weight: np.ndarray) -> np.ndarray:
    return loss_weight


class PrebuiltLmDataset(MappedAsyncDataset[dict, GrugLmExample]):
    """
    A dataset that maps prebuilt cache entries to GrugLmExample instances.
    """

    def __init__(
        self,
        dataset: AsyncDataset[dict],
        Pos: Axis,
        *,
        input_ids_key: str,
        loss_weights_key: str | None,
        loss_weight_transform: Callable[[np.ndarray], np.ndarray] | None,
        eos_id: int | None = None,
        block_cross_document_attention: bool = True,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.eos_id = eos_id
        self.block_cross_document_attention = block_cross_document_attention
        self.input_ids_key = input_ids_key
        self.loss_weights_key = loss_weights_key
        self.loss_weight_transform = loss_weight_transform or _identity_loss_weight

        sharding = _single_cpu_sharding()

        if loss_weights_key is None:

            @functools.partial(eqx.filter_jit)
            def _create_lm_example(tokens: jax.Array) -> GrugLmExample:
                example = GrugLmExample.causal(
                    tokens=tokens,
                    eos_id=eos_id,
                    block_cross_document_attention=block_cross_document_attention,
                )
                example = jax.lax.with_sharding_constraint(example, sharding)
                return example

            def _map(example: dict) -> GrugLmExample:
                return _create_lm_example(example[input_ids_key])

        else:

            @functools.partial(eqx.filter_jit)
            def _create_lm_example(tokens: jax.Array, loss_weight: jax.Array) -> GrugLmExample:
                example = GrugLmExample.causal(
                    tokens=tokens,
                    loss_weight=loss_weight,
                    eos_id=eos_id,
                    block_cross_document_attention=block_cross_document_attention,
                )
                example = jax.lax.with_sharding_constraint(example, sharding)
                return example

            def _map(example: dict) -> GrugLmExample:
                loss_weight = example[loss_weights_key]
                loss_weight = self.loss_weight_transform(loss_weight)
                return _create_lm_example(example[input_ids_key], loss_weight)

        super().__init__(self.dataset, _map)


@dataclass(frozen=True)
class LmDatasetSourceConfigBase(ChoiceRegistry):
    """This class represents a dataset source with URLs or hf name/id."""

    tags: list[str] | None = None
    """tags for the dataset. Typically the name of the dataset in the config will be added as a tag as well"""
    cache_dir: str | None = None  # Optionally override the cache dir for this component
    format: LmDatasetFormatBase = field(default_factory=TextLmDatasetFormat)
    """format of the dataset."""

    @abc.abstractmethod
    def get_shard_source(self, split) -> ShardedDataSource[dict] | None:
        raise NotImplementedError

    def load_cache(
        self, split, tokenizer: MarinTokenizer, override_cache_dir: str | None = None, enforce_eos=True
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
    name: str | None = None  # name for hf dataset
    stream: bool = True  # whether to use streaming when doing hf
    splits: list[str] | None = None

    def get_shard_source(self, split) -> ShardedDataSource[dict] | None:
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
    train_urls: list[str] = field(default_factory=list)
    validation_urls: list[str] = field(default_factory=list)

    def get_shard_source(self, split) -> ShardedDataSource[dict] | None:
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
        return urls


@dataclass(frozen=True)
class DatasetComponentBase(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "cached"


@DatasetComponentBase.register_subclass("cached")
@dataclass(frozen=True)
class DatasetComponent(DatasetComponentBase):
    """A single cache-backed dataset component with optional source."""

    source: LmDatasetSourceConfigBase | None = None
    cache_dir: str | None = None
    format: LmDatasetFormatBase = field(default_factory=TextLmDatasetFormat)
    pack: bool | int | Literal["pad"] | None = None
    tags: list[str] | None = None
    split: str = "validation"


@DatasetComponentBase.register_subclass("direct")
@dataclass(frozen=True)
class DirectDatasetComponent(DatasetComponentBase):
    """A programmatic dataset component that supplies AsyncDataset examples directly."""

    datasets: Mapping[str, AsyncDataset[GrugLmExample]]
    tags: list[str] | None = None


@DatasetComponentBase.register_subclass("hierarchical_cached")
@dataclass(frozen=True)
class HierarchicalMixtureDatasetComponent(DatasetComponentBase):
    """A top-level component backed by a weighted mixture of child cache-backed components."""

    components: dict[str, DatasetComponent]
    train_weights: dict[str, float]
    token_counts: dict[str, int] | None = None
    tags: list[str] | None = None

    def __post_init__(self):
        if not self.components:
            raise ValueError("HierarchicalMixtureDatasetComponent requires at least one child component.")

        unknown_weights = set(self.train_weights) - set(self.components)
        if unknown_weights:
            raise ValueError(
                f"Train weight keys must be a subset of child component keys, got unknown keys {sorted(unknown_weights)}"
            )

        if self.token_counts is not None:
            missing_token_counts = set(self.components) - set(self.token_counts)
            if missing_token_counts:
                raise ValueError(
                    "Token counts must cover every child component, " f"missing {sorted(missing_token_counts)}"
                )


def _effective_pack(component: DatasetComponent) -> bool | int | Literal["pad"]:
    if component.pack is not None:
        return component.pack
    fmt = component.format
    if isinstance(fmt, TextLmDatasetFormat):
        return False
    if isinstance(fmt, ChatLmDatasetFormat):
        return True if fmt.pack is None else fmt.pack
    return False


class LazyAsyncDataset(AsyncDataset[T_co]):
    """Create an AsyncDataset lazily on first access."""

    def __init__(
        self,
        factory: Callable[[], AsyncDataset[T_co]],
        *,
        finite_length: int | None = None,
        assume_finite: bool = False,
    ):
        if finite_length is not None and not assume_finite:
            assume_finite = True
        self._factory = factory
        self._dataset: AsyncDataset[T_co] | None = None
        self._finite_length = finite_length
        self._assume_finite = assume_finite
        self._init_lock = asyncio.Lock()

    async def _dataset_async(self) -> AsyncDataset[T_co]:
        if self._dataset is None:
            async with self._init_lock:
                if self._dataset is None:
                    self._dataset = await asyncio.to_thread(self._factory)
        return self._dataset

    async def async_len(self) -> int:
        if self._finite_length is not None:
            return self._finite_length
        dataset = await self._dataset_async()
        return await dataset.async_len()

    def is_finite(self) -> bool:
        if self._finite_length is not None or self._assume_finite:
            return True
        if self._dataset is not None:
            return self._dataset.is_finite()
        return False

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        dataset = await self._dataset_async()
        return await dataset.get_batch(indices)

    async def getitem_async(self, index: int) -> T_co:
        dataset = await self._dataset_async()
        return await dataset.getitem_async(index)


class PackedTokenDataset(MappedAsyncDataset[tuple[dict, dict], GrugLmExample]):
    """Packed version of token dataset using GreedyPrepackedDataset."""

    def __init__(
        self,
        cache: TreeCache[dict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
        block_cross_document_attention: bool = True,
    ):
        self.packed: GreedyPrepackedDataset[dict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos
        self.block_cross_document_attention = block_cross_document_attention

        sharding = _single_cpu_sharding()

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(e: tuple[dict, dict]) -> GrugLmExample:
            example, seg_ids = e
            tokens = example["input_ids"]
            loss_weight = jnp.ones_like(tokens, dtype=jnp.float32)
            seg_ids_raw = seg_ids["input_ids"]
            out = GrugLmExample.causal(
                tokens=tokens,
                loss_weight=loss_weight,
                segment_ids=seg_ids_raw,
                block_cross_document_attention=block_cross_document_attention,
            )
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_lm_example)


class ChatDataset(MappedAsyncDataset[tuple[ProcessedChatDict, ProcessedChatDict], GrugLmExample]):
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
        block_cross_document_attention: bool = True,
    ):
        self.packed: GreedyPrepackedDataset[ProcessedChatDict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos
        self.block_cross_document_attention = block_cross_document_attention

        sharding = _single_cpu_sharding()
        self.mask_user_turns = mask_user_turns

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(e: tuple[ProcessedChatDict, ProcessedChatDict]) -> GrugLmExample:
            example, seg_ids = e
            tokens = example["input_ids"]

            if mask_user_turns:
                mask = example["assistant_masks"]
                mask = jnp.roll(mask, -1, axis=-1)
                loss_weight = mask.astype(jnp.float32)
            else:
                loss_weight = None

            seg_ids_raw = seg_ids["input_ids"]

            out = GrugLmExample.causal(
                tokens=tokens,
                loss_weight=loss_weight,
                segment_ids=seg_ids_raw,
                block_cross_document_attention=block_cross_document_attention,
            )
            out = jax.lax.with_sharding_constraint(out, sharding)
            return out

        super().__init__(self.packed, _create_lm_example)


def dataset_for_component(
    component: DatasetComponent,
    Pos: Axis,
    cache: TreeCache[dict],
    *,
    eos_id: int | None,
    block_cross_document_attention: bool,
) -> AsyncDataset[GrugLmExample]:
    pack = _effective_pack(component)
    fmt = component.format
    if isinstance(fmt, TextLmDatasetFormat):
        if pack == "pad":
            raise NotImplementedError("Padding mode not yet implemented.")
        if pack:
            max_segments = 64 if pack is True else int(pack)
            return PackedTokenDataset(
                cache,
                Pos,
                max_segments_per_example=max_segments,
                block_cross_document_attention=block_cross_document_attention,
            )
        else:
            return CausalLmDataset(
                TokenSeqDataset(cache, Pos.size),
                Pos,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )
    elif isinstance(fmt, ChatLmDatasetFormat):
        effective_pack = pack
        if effective_pack == "pad":
            raise NotImplementedError("Padding mode not yet implemented.")
        max_segments = (
            64 if effective_pack is True else (int(effective_pack) if isinstance(effective_pack, int) else 1)
        )
        mask_user_turns = fmt.mask_user_turns
        return ChatDataset(
            cache,
            Pos,
            max_segments_per_example=max_segments,
            mask_user_turns=mask_user_turns,
            block_cross_document_attention=block_cross_document_attention,
        )  # type: ignore
    elif isinstance(fmt, PrebuiltLmDatasetFormat):
        return PrebuiltLmDataset(
            cache,
            Pos,
            input_ids_key=fmt.input_ids_key,
            loss_weights_key=fmt.loss_weights_key,
            loss_weight_transform=fmt.loss_weight_transform,
            eos_id=eos_id,
            block_cross_document_attention=block_cross_document_attention,
        )
    else:
        raise ValueError(f"Unknown format {fmt}")


def _stable_dataset_key(name: str, split: str) -> PRNGKeyArray:
    seed = zlib.crc32(f"{name}:{split}".encode("utf-8")) & 0xFFFFFFFF
    return jax.random.PRNGKey(seed)


def _stable_simulated_epoch_subset_key(name: str, split: str, subset_seed: int) -> PRNGKeyArray:
    base_key = jax.random.PRNGKey(subset_seed)
    fold_value = zlib.crc32(f"simulated_epoch_subset:{name}:{split}".encode("utf-8")) & 0xFFFFFFFF
    return jax.random.fold_in(base_key, fold_value)


def _stable_child_order(name: str, split: str, child_names: Sequence[str]) -> list[str]:
    return sorted(
        child_names,
        key=lambda child_name: zlib.crc32(f"{name}:{split}:{child_name}".encode("utf-8")) & 0xFFFFFFFF,
    )


def _sequence_count_from_token_count(component: DatasetComponent, token_count: int, seq_len: int) -> int | None:
    if not isinstance(component.format, TextLmDatasetFormat):
        return None
    if _effective_pack(component):
        return None
    return token_count // seq_len


def _finite_length_for_hierarchical_component(
    component: HierarchicalMixtureDatasetComponent,
    *,
    seq_len: int,
) -> int | None:
    if component.token_counts is None:
        return None

    total_length = 0
    for child_name, child_component in component.components.items():
        child_length = _sequence_count_from_token_count(child_component, component.token_counts[child_name], seq_len)
        if child_length is None:
            return None
        if component.train_weights.get(child_name, 0.0) > 0:
            total_length += child_length

    return total_length


def _component_cache_dir(name: str, component: DatasetComponent, default_root: str | None) -> str:
    base = component.cache_dir if component.cache_dir is not None else default_root
    if base is None:
        raise ValueError(f"No cache_dir provided for component {name}")
    if component.cache_dir is None:
        return os.path.join(base, name)
    return base


def _split_into_trainval_sets(
    dataset: "AsyncDataset[LmExample]", num_validation_sequences: int, *, shuffle: bool = True
) -> tuple["AsyncDataset[LmExample]", "AsyncDataset[LmExample]"]:
    """Split a dataset into train/val portions, optionally shuffling first.

    When shuffle is True, a deterministic shuffle is applied before
    splitting so that the validation set is a random subset. Uses a fixed key so
    that train_sets() and validation_sets() produce the same permutation,
    guaranteeing disjoint splits even though they are constructed independently.

    When shuffle is False, the split is positional: the last
    num_validation_sequences go to validation and the rest to training.
    """
    logger.info(f"Splitting dataset into train/val sets. Shuffle before split: {shuffle}")
    length = len(dataset.as_sync_dataset())
    if shuffle:
        split_key = jax.random.PRNGKey(0)
        dataset = dataset.shuffle(split_key, perm_type="feistel")
    train_ds = dataset.slice_dataset(start_index=0, end_index=length - num_validation_sequences)
    val_ds = dataset.slice_dataset(start_index=length - num_validation_sequences, end_index=length)
    return train_ds, val_ds


@dataclass(frozen=True)
class BlockShuffleConfig:
    """Configuration for hierarchical block shuffling."""

    io_block_size: int
    window_blocks: int
    perm_type: Literal["feistel", "linear"] = "feistel"


@dataclass(frozen=True)
class LmDataConfig:
    """Unified LM data config built from components."""

    tokenizer: str = "gpt2"
    vocab_size: int | None = None  # if using the passthrough tokenizer, this is required

    # config related to caching
    cache_dir: str | None = "cache/"
    cache_options: CacheOptions = field(default_factory=CacheOptions)
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't
    auto_build_caches: bool = True
    """Whether to build dataset caches automatically when they are missing.

    If False, any attempt to access a cache that does not already exist will raise
    a FileNotFoundError instead of building the cache on the fly.
    """

    chat_template: str | None = None  # If set, use this template for chat datasets. Otherwise, use the tokenizer's.

    shuffle: bool | int | BlockShuffleConfig = False
    """Shuffle policy.

    - `True`: full permutation shuffle
    - `False`: no shuffle
    - positive `int`: era shuffle with this era length
    - `BlockShuffleConfig`: hierarchical block shuffle
    """
    permutation_type: Literal["feistel", "linear"] | None = None
    """
    Type of permutation to use for shuffle.

    If None, defaults to linear, but this will change in the future since Feistel is better.
    """

    block_cross_document_attention: bool = True
    """Whether to block attention across document boundaries.

    If True (default), attention is blocked across documents using segment ids derived from EOS tokens.
    If False, full causal attention is allowed across packed documents.
    """

    components: dict[str, DatasetComponentBase] = field(default_factory=dict)
    train_weights: dict[str, float] | list[tuple[int, dict[str, float]]] | None = None

    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)
    target_budget: int | None = None
    experiment_budget: int | None = None
    simulated_epoch_subset_seed: int | None = None
    mixture_block_size: int = 2048
    max_train_batches: dict[str, int] | None = None
    num_validation_sequences: dict[str, int] | None = None
    shuffle_before_trainval_split: bool = True
    """Whether to shuffle the dataset before splitting off validation sequences.

    When True (default), a deterministic shuffle is applied before the train/val
    split so that the validation set is a random subset rather than a positional
    slice (e.g. the last N sequences). Set to False to preserve the original
    dataset ordering for the split. Only relevant when num_validation_sequences
    is set.
    """

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

    @cached_property
    def the_tokenizer(self) -> MarinTokenizer:
        if self.tokenizer == "passthrough":
            return PassthroughTokenizer(self.vocab_size)
        else:
            return load_marin_tokenizer(self.tokenizer)

    def _has_nonzero_weight(self, name: str) -> bool:
        weights = self.train_weights
        if weights is None:
            return True
        if isinstance(weights, dict):
            return weights.get(name, 0) > 0
        return any(w.get(name, 0) > 0 for _, w in weights)

    def _cache_for_component(self, name: str, component: DatasetComponent, split: str) -> TreeCache[dict] | None:
        cache_root = _component_cache_dir(name, component, self.cache_dir)
        source = component.source

        if source is None:
            try:
                return load_lm_dataset_cache(
                    os.path.join(cache_root, split),
                    component.format,
                    self.the_tokenizer,
                    self.enforce_eos,
                )
            except FileNotFoundError as exc:
                raise ValueError(f"No source and no cache found for component {name} split {split}") from exc

        shard_source = source.get_shard_source(split)
        if shard_source is None:
            cache_path = os.path.join(cache_root, split)
            if not fsspec_utils.exists(cache_path):
                logger.warning("No source for %s in %s split and no cache at %s, skipping", name, split, cache_path)
                return None
            return load_lm_dataset_cache(
                cache_path,
                component.format,
                self.the_tokenizer,
                self.enforce_eos,
            )

        cache_path = os.path.join(cache_root, split)
        if not self.auto_build_caches:
            if not fsspec_utils.exists(cache_path):
                raise FileNotFoundError(f"Cache not found at {cache_path} and auto_build_caches is disabled")
            return load_lm_dataset_cache(
                cache_path,
                component.format,
                self.the_tokenizer,
                self.enforce_eos,
            )

        return build_lm_dataset_cache(
            cache_path,
            shard_source,
            component.format,
            self.the_tokenizer,
            self.cache_options,
            self.enforce_eos,
        )

    def _build_token_dataset_for_component(
        self,
        name: str,
        component: DatasetComponentBase,
        Pos: Axis,
        *,
        split: str,
        caches: Mapping[str, TreeCache[dict]] | None,
    ) -> AsyncDataset[GrugLmExample] | None:
        if isinstance(component, DirectDatasetComponent):
            direct = component.datasets.get(split)
            if direct is None:
                if split == "train":
                    raise ValueError(f"Direct dataset format missing {split} split for component {name}")
                logger.warning("Direct dataset format missing %s split for component %s", split, name)
                return None
            return direct

        if isinstance(component, DatasetComponent):
            cache = caches.get(name) if caches is not None else self._cache_for_component(name, component, split)
            if cache is None:
                if split == "train":
                    raise ValueError(f"No cache available for component {name} in {split} split")
                return None

            return dataset_for_component(
                component,
                Pos,
                cache,
                eos_id=self.the_tokenizer.eos_token_id,
                block_cross_document_attention=self.block_cross_document_attention,
            )

        if isinstance(component, HierarchicalMixtureDatasetComponent):
            ordered_child_names = _stable_child_order(name, split, list(component.components))

            def build_child_datasets() -> dict[str, AsyncDataset[GrugLmExample]]:
                child_datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
                for child_name in ordered_child_names:
                    child_component = component.components[child_name]
                    dataset = self._build_token_dataset_for_component(
                        f"{name}/{child_name}",
                        child_component,
                        Pos,
                        split=split,
                        caches=None,
                    )
                    if dataset is None:
                        continue
                    child_datasets[child_name] = dataset
                return child_datasets

            def build_lazy_train_child_datasets() -> dict[str, AsyncDataset[GrugLmExample]]:
                child_datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
                for child_name in ordered_child_names:
                    child_component = component.components[child_name]
                    if component.train_weights.get(child_name, 0.0) <= 0:
                        continue

                    child_finite_length = None
                    if component.token_counts is not None:
                        child_finite_length = _sequence_count_from_token_count(
                            child_component,
                            component.token_counts[child_name],
                            Pos.size,
                        )

                    def build_child_dataset(
                        child_name: str = child_name,
                        child_component: DatasetComponentBase = child_component,
                    ) -> AsyncDataset[GrugLmExample]:
                        dataset = self._build_token_dataset_for_component(
                            f"{name}/{child_name}",
                            child_component,
                            Pos,
                            split=split,
                            caches=None,
                        )
                        if dataset is None:
                            raise ValueError(f"No dataset available for hierarchical child {name}/{child_name}")
                        return dataset

                    child_datasets[child_name] = LazyAsyncDataset(
                        build_child_dataset,
                        finite_length=child_finite_length,
                        assume_finite=True,
                    )

                return child_datasets

            def build_hierarchical_mixture(
                child_datasets: Mapping[str, AsyncDataset[GrugLmExample]],
            ) -> AsyncDataset[GrugLmExample]:
                if split == "train" and not child_datasets:
                    raise ValueError(f"No child datasets available for hierarchical component {name}")
                if not child_datasets:
                    raise ValueError(f"No datasets available for hierarchical component {name}")

                child_weights = {
                    child_name: weight
                    for child_name in ordered_child_names
                    for weight in [component.train_weights.get(child_name, 0.0)]
                    if child_name in child_datasets and weight > 0
                }
                # Hierarchical domains use metadata-derived finite lengths, so the
                # runtime child sampler can restart individual children without
                # forcing startup-time length inference over tiny-weight shards.
                return MixtureDataset(
                    datasets=child_datasets,
                    weights=child_weights,
                    stop_strategy=StopStrategy.RESTART_STRATEGY,
                    key=_stable_dataset_key(name, split),
                    block_size=self.mixture_block_size,
                    randomize_blocks=False,
                )

            def build_nested_dataset() -> AsyncDataset[GrugLmExample]:
                return build_hierarchical_mixture(build_lazy_train_child_datasets())

            finite_length = _finite_length_for_hierarchical_component(
                component,
                seq_len=Pos.size,
            )
            if split != "train":
                child_datasets = build_child_datasets()
                if not child_datasets:
                    logger.warning(
                        "No datasets available for hierarchical component %s in %s split, skipping", name, split
                    )
                    return None
                finite_length = sum(len(dataset.as_sync_dataset()) for dataset in child_datasets.values())
                return LazyAsyncDataset(
                    lambda: build_hierarchical_mixture(child_datasets),
                    finite_length=finite_length,
                )
            return LazyAsyncDataset(build_nested_dataset, finite_length=finite_length)

        raise ValueError(f"Unsupported component type for {name}: {type(component)}")

    def _has_hierarchical_components(self) -> bool:
        return any(
            isinstance(component, HierarchicalMixtureDatasetComponent) for component in self.components.values()
        )

    def build_token_datasets(
        self,
        caches: Mapping[str, TreeCache[dict]] | None,
        Pos: Axis,
        *,
        split: str,
    ):
        datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
        for name, component in self.components.items():
            if split == "train" and not self._has_nonzero_weight(name):
                continue
            dataset = self._build_token_dataset_for_component(name, component, Pos, split=split, caches=caches)
            if dataset is None:
                continue
            datasets[name] = dataset

        return datasets

    @staticmethod
    def _position_axis(seq_len: int) -> Axis:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        return Axis("position", seq_len)

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        mix_key, shuffle_key = jax.random.split(key)
        weights = self.train_weights
        if isinstance(weights, list):
            weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)
        initial_batch_size = batch_schedule.batch_size_at_step(0)
        datasets = self.train_sets(Pos, key=shuffle_key, initial_batch_size=initial_batch_size)
        mixture = MixtureDataset(
            datasets=datasets,
            weights=weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )
        return NamedLmDataset(mixture, Pos)

    def train_sets(
        self,
        Pos: Axis,
        *,
        initial_batch_size: int | None = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[GrugLmExample]]:
        doc_caches = None if self._has_hierarchical_components() else self.build_caches("train")
        datasets = self.build_token_datasets(doc_caches, Pos, split="train")

        if self.num_validation_sequences is not None:
            for name, ds in datasets.items():
                if name in self.num_validation_sequences:
                    train_ds, _ = _split_into_trainval_sets(
                        ds, self.num_validation_sequences[name], shuffle=self.shuffle_before_trainval_split
                    )
                    datasets[name] = train_ds

        if key is None:
            key = jax.random.PRNGKey(0)

        shuffle_cfg = self.shuffle
        perm_type = self.permutation_type
        if perm_type is None and shuffle_cfg is not False and not isinstance(shuffle_cfg, BlockShuffleConfig):
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        def shuffle_ds(ds, k):
            if isinstance(shuffle_cfg, BlockShuffleConfig):
                ds = ds.block_shuffle(
                    io_block_size=shuffle_cfg.io_block_size,
                    window_blocks=shuffle_cfg.window_blocks,
                    key=k,
                    perm_type=shuffle_cfg.perm_type,
                )
            elif shuffle_cfg is True:
                ds = ds.shuffle(k, perm_type=perm_type)
            elif isinstance(shuffle_cfg, int) and not isinstance(shuffle_cfg, bool) and shuffle_cfg > 0:
                ds = ds.era_shuffle(shuffle_cfg, key=k, perm_type=perm_type)
            return ds

        if (
            self.experiment_budget is not None and self.target_budget is not None
        ) and self.experiment_budget > self.target_budget:
            raise ValueError(
                f"Experiment budget should be smaller than target budget, got {self.experiment_budget} > {self.target_budget}"
            )

        def slice_for_simulated_epoching(
            ds_by_name: Mapping[str, AsyncDataset[GrugLmExample]],
            *,
            subset_seed: int | None,
        ) -> dict[str, AsyncDataset[GrugLmExample]]:
            assert self.experiment_budget is not None
            assert self.target_budget is not None

            simulated_data_ratio = self.experiment_budget / self.target_budget
            sliced_datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
            for name, ds in ds_by_name.items():
                subset_dataset = ds
                if subset_seed is not None and shuffle_cfg:
                    subset_dataset = shuffle_ds(
                        ds,
                        _stable_simulated_epoch_subset_key(name, "train", subset_seed),
                    )

                true_length_of_dataset = len(ds.as_sync_dataset())
                simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
                sliced_datasets[name] = subset_dataset.slice_dataset(end_index=simulated_length_of_dataset)

            return sliced_datasets

        if self.experiment_budget is not None and self.target_budget is not None:
            if self.simulated_epoch_subset_seed is None:
                if shuffle_cfg:
                    key_iter = key_iterator(key)
                    datasets = {name: shuffle_ds(ds, next(key_iter)) for name, ds in datasets.items()}
                datasets = slice_for_simulated_epoching(datasets, subset_seed=None)
            else:
                datasets = slice_for_simulated_epoching(
                    datasets,
                    subset_seed=self.simulated_epoch_subset_seed,
                )
                if shuffle_cfg:
                    key_iter = key_iterator(key)
                    datasets = {name: shuffle_ds(ds, next(key_iter)) for name, ds in datasets.items()}
        elif shuffle_cfg:
            key_iter = key_iterator(key)
            datasets = {name: shuffle_ds(ds, next(key_iter)) for name, ds in datasets.items()}

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

    def train_grug_sets(
        self,
        *,
        seq_len: int,
        initial_batch_size: int | None = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[GrugLmExample]]:
        """Build train datasets that emit array-first [GrugLmExample][]."""
        return self.train_sets(
            self._position_axis(seq_len),
            initial_batch_size=initial_batch_size,
            key=key,
        )

    def _validation_datasets_unwrapped(self, Pos: Axis) -> dict[str, AsyncDataset[GrugLmExample]]:
        doc_caches = None if self._has_hierarchical_components() else self.build_caches("validation")
        validation_datasets = self.build_token_datasets(doc_caches, Pos, split="validation")

        if self.num_validation_sequences is not None:
            train_doc_caches = None if self._has_hierarchical_components() else self.build_caches("train")
            train_datasets = self.build_token_datasets(train_doc_caches, Pos, split="train")

            for name, num_sequences in self.num_validation_sequences.items():
                _, val_ds = _split_into_trainval_sets(
                    train_datasets[name], num_sequences, shuffle=self.shuffle_before_trainval_split
                )
                validation_datasets[name] = val_ds

        return validation_datasets

    def validation_sets(self, Pos: Axis) -> Mapping[str, AsyncDataset[LmExample]]:
        validation_datasets = self._validation_datasets_unwrapped(Pos)
        return {name: NamedLmDataset(ds, Pos) for name, ds in validation_datasets.items()}

    def validation_grug_sets(self, *, seq_len: int) -> Mapping[str, AsyncDataset[GrugLmExample]]:
        """Build validation datasets that emit array-first [GrugLmExample][]."""
        Pos = self._position_axis(seq_len)
        return self._validation_datasets_unwrapped(Pos)

    def build_caches(self, split: str) -> dict[str, TreeCache[dict]]:
        caches: dict[str, TreeCache[dict]] = {}
        for name, component in self.components.items():
            if split == "train" and not self._has_nonzero_weight(name):
                continue

            if isinstance(component, DirectDatasetComponent):
                continue

            if isinstance(component, HierarchicalMixtureDatasetComponent):
                raise ValueError(
                    "HierarchicalMixtureDatasetComponent does not correspond to a single cache. "
                    "Build datasets directly instead of calling build_caches()."
                )

            if not isinstance(component, DatasetComponent):
                raise ValueError(f"Unsupported component type for {name}: {type(component)}")
            cache = self._cache_for_component(name, component, split)
            if cache is None:
                continue
            caches[name] = cache

        return caches

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        sources: dict[str, LmDatasetSourceConfigBase] = {}
        for name, comp in self.components.items():
            if isinstance(comp, DatasetComponent) and comp.source is not None:
                sources[name] = comp.source
        return sources

    def tagged_eval_sets(self, Pos: Axis) -> list[tuple[AsyncDataset[LmExample], list[str]]]:
        eval_sets = self.validation_sets(Pos)
        tagged = []
        for name, ds in eval_sets.items():
            tags = (self.components[name].tags or []) + [name]
            tagged.append((ds, tags))
        return tagged

    def tagged_eval_grug_sets(self, *, seq_len: int) -> list[tuple[AsyncDataset[GrugLmExample], list[str]]]:
        """Build tagged validation datasets for array-first evaluators."""
        eval_sets = self.validation_grug_sets(seq_len=seq_len)
        tagged = []
        for name, ds in eval_sets.items():
            tags = (self.components[name].tags or []) + [name]
            tagged.append((ds, tags))
        return tagged


LMMixtureDatasetConfig: TypeAlias = LmDataConfig


def _get_token_key_for_component(component: DatasetComponentBase) -> str:
    """Get the appropriate token key based on component format."""
    if isinstance(component, DatasetComponent):
        return component.format.token_data_key
    return "input_ids"


def count_corpus_sizes(
    config: LmDataConfig,
    prefix: str = "data/stats/",
    seq_len: int = 4096,
) -> dict:
    stats = {}
    train_caches = config.build_caches("train")
    Pos = Axis("position", seq_len)

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
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        stats[f"{metric_prefix}total_tokens"] = cache.store.tree[token_key].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree[token_key].num_rows
        train_set = dataset_for_component(
            component,
            Pos,
            cache,
            eos_id=None,
            block_cross_document_attention=config.block_cross_document_attention,
        )
        train_seqs = len(train_set.as_sync_dataset())
        stats[f"{metric_prefix}total_seqs"] = train_seqs
        padding_fraction = 1 - (cache.store.tree[token_key].data_size / (train_seqs * seq_len))
        if padding_fraction < 0:
            stats[f"{metric_prefix}truncation_fraction"] = -padding_fraction
        else:
            stats[f"{metric_prefix}padding_fraction"] = padding_fraction
        if name in weights:
            weight = weights.get(name, 0.0)
            stats[f"{metric_prefix}weight"] = weight
            stats[f"{metric_prefix}normalized_weight"] = weights[name]
            stats[f"{metric_prefix}approx_global_tokens_per_pass"] = train_seqs * seq_len / max(weight, 1e-8)

    validation_caches = config.build_caches("validation")
    for name, cache in validation_caches.items():
        metric_prefix = f"{prefix}validation/{name}/"
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        stats[f"{metric_prefix}total_tokens"] = cache.store.tree[token_key].data_size
        stats[f"{metric_prefix}total_docs"] = cache.store.tree[token_key].num_rows
        validation_set = dataset_for_component(
            component,
            Pos,
            cache,
            eos_id=None,
            block_cross_document_attention=config.block_cross_document_attention,
        )
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
