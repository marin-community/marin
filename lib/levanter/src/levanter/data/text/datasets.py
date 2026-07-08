# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NotRequired, TypeAlias, TypeVar, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from draccus import ChoiceRegistry, field
from haliax import Axis
from jaxtyping import PRNGKeyArray
from rigging.filesystem import StoragePath, prefix_join
from rigging.timing import log_time

import levanter
from levanter.data import AsyncDataset
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.mixture import (
    ConcatDataset,
    MixtureDataset,
    StopStrategy,
    rescale_mixture_schedule_for_batch_schedule,
)
from levanter.data.packing import GreedyPrepackedDataset
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.data.sharded_datasource import (
    ShardedDataSource,
    UrlDataSource,
    datasource_from_hf_or_none,
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
    SupervisedLmDatasetFormat,
    SupervisedTextProcessor,
    TextLmDatasetFormat,
)
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions, TreeCache
from levanter.tokenizers import MarinTokenizer, load_tokenizer as load_marin_tokenizer
from levanter.utils.jax_utils import key_iterator
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()  # noqa

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

logger = logging.getLogger("levanter.data.text")


class TokenSeqDict(TypedDict):
    input_ids: np.ndarray
    loss_weights: NotRequired[np.ndarray]


class TokenSeqDataset(AsyncDataset[TokenSeqDict]):
    """
    A dataset that yields fixed-length token sequences from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int, loss_weights_key: str | None = None):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self.loss_weights_key = loss_weights_key

    async def async_len(self) -> int:
        return await self.doc_cache.async_flat_field_length("input_ids") // self.seq_len

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[TokenSeqDict]:
        if not indices:
            return []

        ds_len = await self.async_len()
        if ds_len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")

        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        token_batch = await self.doc_cache.get_flat_field_batch("input_ids", offsets, self.seq_len)
        if self.loss_weights_key is None:
            return [{"input_ids": tokens} for tokens in token_batch]

        weight_batch = await self.doc_cache.get_flat_field_batch(self.loss_weights_key, offsets, self.seq_len)
        return [
            {"input_ids": tokens, "loss_weights": weights}
            for tokens, weights in zip(token_batch, weight_batch, strict=True)
        ]


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


class CausalLmDataset(MappedAsyncDataset[TokenSeqDict, GrugLmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[TokenSeqDict],
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
        def _create_lm_example(example_dict: TokenSeqDict) -> GrugLmExample:
            example = GrugLmExample.causal(
                tokens=example_dict["input_ids"],
                loss_weight=example_dict.get("loss_weights"),
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
                # pyrefly: ignore[bad-return]  # eqx.filter_jit wrapper types the call as returning Unknown
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
                # pyrefly: ignore[bad-return, bad-argument-count]  # eqx.filter_jit wrapper hides the real signature
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
        return load_lm_dataset_cache(prefix_join(base_cache, split), self.format, tokenizer, enforce_eos=enforce_eos)

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
    revision: str | None = None  # revision, branch, or tag for hf dataset
    stream: bool = True  # whether to use streaming when doing hf
    splits: list[str] | None = None

    def get_shard_source(self, split) -> ShardedDataSource[dict] | None:
        if self.splits is not None and split not in self.splits:
            logger.warning(f"Splits {split} not found for {self.id} {self.name}")
            return None
        if self.id is not None:
            return datasource_from_hf_or_none(
                self.id, split=split, name=self.name, revision=self.revision, streaming=self.stream
            )


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
    flat_cache: bool = False
    """Treat ``cache_dir`` as the cache root directly, without appending ``/<split>``."""


@DatasetComponentBase.register_subclass("direct")
@dataclass(frozen=True)
class DirectDatasetComponent(DatasetComponentBase):
    """A programmatic dataset component that supplies AsyncDataset examples directly."""

    datasets: Mapping[str, AsyncDataset[GrugLmExample]]
    tags: list[str] | None = None


@DatasetComponentBase.register_subclass("concat")
@dataclass(frozen=True)
class ConcatDatasetComponent(DatasetComponentBase):
    """A logical component formed by concatenating cache-backed children."""

    children: dict[str, DatasetComponent]
    tags: list[str] | None = None


def _effective_pack(component: DatasetComponent) -> bool | int | Literal["pad"]:
    if component.pack is not None:
        return component.pack
    fmt = component.format
    if isinstance(fmt, TextLmDatasetFormat):
        return False
    if isinstance(fmt, ChatLmDatasetFormat):
        return True if fmt.pack is None else fmt.pack
    return False


class PackedTokenDataset(MappedAsyncDataset[tuple[dict, dict], GrugLmExample]):
    """Packed version of token dataset using GreedyPrepackedDataset."""

    def __init__(
        self,
        cache: TreeCache[dict],
        Pos: Axis,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
        loss_weights_key: str | None = None,
        block_cross_document_attention: bool = True,
    ):
        self.packed: GreedyPrepackedDataset[dict] = GreedyPrepackedDataset(
            cache.jagged_array_tree(),
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos
        self.block_cross_document_attention = block_cross_document_attention
        self.loss_weights_key = loss_weights_key

        sharding = _single_cpu_sharding()

        if loss_weights_key is None:

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
                    max_segments=max_segments_per_example + 1,
                    block_cross_document_attention=block_cross_document_attention,
                )
                out = jax.lax.with_sharding_constraint(out, sharding)
                return out

        else:

            @functools.partial(eqx.filter_jit)
            def _create_lm_example(e: tuple[dict, dict]) -> GrugLmExample:
                example, seg_ids = e
                tokens = example["input_ids"]
                loss_weight = example[loss_weights_key]
                seg_ids_raw = seg_ids["input_ids"]
                out = GrugLmExample.causal(
                    tokens=tokens,
                    loss_weight=loss_weight,
                    segment_ids=seg_ids_raw,
                    max_segments=max_segments_per_example + 1,
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
            cache.jagged_array_tree(),
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
                max_segments=max_segments_per_example + 1,
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
        return CausalLmDataset(
            TokenSeqDataset(cache, Pos.size),
            Pos,
            eos_id=eos_id,
            block_cross_document_attention=block_cross_document_attention,
        )
    elif isinstance(fmt, SupervisedLmDatasetFormat):
        loss_weights_key = SupervisedTextProcessor.loss_weights_key
        if pack == "pad":
            raise NotImplementedError("Padding mode not yet implemented.")
        if pack:
            max_segments = 64 if pack is True else int(pack)
            return PackedTokenDataset(
                cache,
                Pos,
                max_segments_per_example=max_segments,
                loss_weights_key=loss_weights_key,
                block_cross_document_attention=block_cross_document_attention,
            )
        return CausalLmDataset(
            TokenSeqDataset(cache, Pos.size, loss_weights_key=loss_weights_key),
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


def _component_cache_dir(name: str, component: DatasetComponent, default_root: str | None) -> str:
    base = component.cache_dir if component.cache_dir is not None else default_root
    if base is None:
        raise ValueError(f"No cache_dir provided for component {name}")
    if component.cache_dir is None:
        return prefix_join(base, name)
    return base


def _split_into_trainval_sets(
    dataset: "AsyncDataset[T]", num_validation_sequences: int, *, shuffle: bool = True
) -> tuple["AsyncDataset[T]", "AsyncDataset[T]"]:
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


DEFAULT_LM_DATA_SHUFFLE = BlockShuffleConfig(
    io_block_size=256,
    window_blocks=512,
    perm_type="feistel",
)
"""Default hierarchical block-shuffle policy for LM training data."""


# A classified dataset component from `build_caches`: (name, loaded cache, deferred
# build args). Exactly one of the two trailing fields is non-None.
_ClassifiedComponent: TypeAlias = tuple[
    str, "TreeCache[dict] | None", "tuple[str, ShardedDataSource, LmDatasetFormatBase] | None"
]


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

    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE
    """Shuffle policy.

    - `True`: full permutation shuffle
    - `False`: no shuffle
    - `BlockShuffleConfig`: hierarchical block shuffle
    """
    permutation_type: Literal["feistel", "linear"] = "feistel"
    """
    Type of permutation to use for full shuffle.
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

    def build_token_datasets(self, caches: Mapping[str, TreeCache[dict]], Pos: Axis, *, split: str):
        datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
        for name, component in self.components.items():
            if split == "train" and not self._has_nonzero_weight(name):
                continue

            if isinstance(component, DirectDatasetComponent):
                direct = component.datasets.get(split)
                if direct is None:
                    if split == "train":
                        raise ValueError(f"Direct dataset format missing {split} split for component {name}")
                    logger.warning("Direct dataset format missing %s split for component %s", split, name)
                    continue
                datasets[name] = direct
                continue

            if isinstance(component, ConcatDatasetComponent):
                child_datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
                for child_name, child in component.children.items():
                    child_key = f"{name}/{child_name}"
                    cache = caches.get(child_key)
                    if cache is None:
                        if split == "train":
                            raise ValueError(f"No cache available for concat child {child_key} in {split} split")
                        continue
                    child_datasets[child_name] = dataset_for_component(
                        child,
                        Pos,
                        cache,
                        eos_id=self.the_tokenizer.eos_token_id,
                        block_cross_document_attention=self.block_cross_document_attention,
                    )
                if child_datasets:
                    datasets[name] = ConcatDataset(child_datasets)
                continue

            if not isinstance(component, DatasetComponent):
                raise ValueError(f"Unsupported component type for {name}: {type(component)}")

            cache = caches.get(name)
            if cache is None:
                if split == "train":
                    raise ValueError(f"No cache available for component {name} in {split} split")
                continue

            datasets[name] = dataset_for_component(
                component,
                Pos,
                cache,
                eos_id=self.the_tokenizer.eos_token_id,
                block_cross_document_attention=self.block_cross_document_attention,
            )

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
        doc_caches = self.build_caches("train")
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
            return ds

        if shuffle_cfg:
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
            sliced_datasets: dict[str, AsyncDataset[GrugLmExample]] = {}
            for name, ds in datasets.items():
                true_length_of_dataset = len(ds.as_sync_dataset())
                simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
                sliced_datasets[name] = ds.slice_dataset(end_index=simulated_length_of_dataset)
            datasets = sliced_datasets

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
        doc_caches = self.build_caches("validation")
        validation_datasets = self.build_token_datasets(doc_caches, Pos, split="validation")

        if self.num_validation_sequences is not None:
            train_doc_caches = self.build_caches("train")
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
        items: list[tuple[str, "DatasetComponent"]] = []
        for name, component in self.components.items():
            if split == "train" and not self._has_nonzero_weight(name):
                continue
            if isinstance(component, DirectDatasetComponent):
                continue
            if isinstance(component, ConcatDatasetComponent):
                for child_name, child in component.children.items():
                    items.append((f"{name}/{child_name}", child))
                continue
            if not isinstance(component, DatasetComponent):
                raise ValueError(f"Unsupported component type for {name}: {type(component)}")
            items.append((name, component))

        if not items:
            return {}

        # Loads are pure GCS metadata reads and parallelize cleanly. Builds may
        # enter `_distributed_build_cache`, which uses unidentified jax
        # collectives paired across processes by dispatch order — running
        # multiple of those concurrently can cross-wire status broadcasts or
        # hang. Classify each component in the pool, then build any misses
        # serially in the original component order.
        def _load_or_defer(item: tuple[str, "DatasetComponent"]) -> _ClassifiedComponent:
            name, component = item
            cache_root = _component_cache_dir(name, component, self.cache_dir)
            if component.flat_cache:
                if split != "train":
                    return name, None, None
                cache_path = cache_root
            else:
                cache_path = prefix_join(cache_root, split)
            source = component.source

            if source is None:
                try:
                    cache = load_lm_dataset_cache(cache_path, component.format, self.the_tokenizer, self.enforce_eos)
                except FileNotFoundError:
                    raise ValueError(f"No source and no cache found for component {name} split {split}")
                return name, cache, None

            shard_source = source.get_shard_source(split)
            cache_exists = StoragePath(cache_path).exists()

            if shard_source is None:
                if not cache_exists:
                    logger.warning(f"No source for {name} in {split} split and no cache at {cache_path}, skipping")
                    return name, None, None
                cache = load_lm_dataset_cache(cache_path, component.format, self.the_tokenizer, self.enforce_eos)
                return name, cache, None

            if not self.auto_build_caches:
                if not cache_exists:
                    raise FileNotFoundError(f"Cache not found at {cache_path} and auto_build_caches is disabled")
                cache = load_lm_dataset_cache(cache_path, component.format, self.the_tokenizer, self.enforce_eos)
                return name, cache, None

            if cache_exists:
                try:
                    cache = load_lm_dataset_cache(cache_path, component.format, self.the_tokenizer, self.enforce_eos)
                    return name, cache, None
                except FileNotFoundError:
                    logger.warning(
                        f"Cache dir at {cache_path} exists but is unloadable (likely a "
                        "partial build from a killed prior cache-build job); auto_build_caches "
                        "is on, so falling through to rebuild."
                    )
            return name, None, (cache_path, shard_source, component.format)

        caches: dict[str, TreeCache[dict]] = {}
        to_build: list[tuple[str, tuple[str, ShardedDataSource, LmDatasetFormatBase]]] = []
        max_workers = min(32, len(items))
        classified: dict[int, _ClassifiedComponent] = {}
        with (
            log_time(f"build_caches[{split}] over {len(items)} components"),
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="build_caches") as pool,
        ):
            # Consume results as each future completes rather than in submission
            # order (`pool.map`), so a worker that raises — e.g. a tokenizer
            # staging failure — surfaces immediately and aborts the process
            # instead of being stranded behind a slower sibling's still-pending
            # load. Key by submission index so the serial build below runs in the
            # original component order, which must be identical across hosts for
            # `_distributed_build_cache`'s dispatch-order-paired collectives.
            index_of = {pool.submit(_load_or_defer, item): index for index, item in enumerate(items)}
            for future in as_completed(index_of):
                classified[index_of[future]] = future.result()

        for index in range(len(items)):
            name, cache, build_args = classified[index]
            if cache is not None:
                caches[name] = cache
            elif build_args is not None:
                to_build.append((name, build_args))

        for name, (cache_path, shard_source, fmt) in to_build:
            caches[name] = build_lm_dataset_cache(
                cache_path, shard_source, fmt, self.the_tokenizer, self.cache_options, self.enforce_eos
            )
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
    stats: dict[str, int | float] = {}
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
        total_tokens = cache.flat_field_length(token_key)
        stats[f"{metric_prefix}total_tokens"] = total_tokens
        stats[f"{metric_prefix}total_docs"] = cache.flat_field_num_rows(token_key)
        train_set = dataset_for_component(
            component,
            Pos,
            cache,
            eos_id=None,
            block_cross_document_attention=config.block_cross_document_attention,
        )
        train_seqs = len(train_set.as_sync_dataset())
        stats[f"{metric_prefix}total_seqs"] = train_seqs
        if train_seqs == 0 or seq_len == 0:
            continue
        padding_fraction = 1 - (total_tokens / (train_seqs * seq_len))
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
        stats[f"{metric_prefix}total_tokens"] = cache.flat_field_length(token_key)
        stats[f"{metric_prefix}total_docs"] = cache.flat_field_num_rows(token_key)
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

    main()  # pyrefly: ignore[missing-argument]
