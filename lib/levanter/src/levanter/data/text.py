# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import dataclasses
import functools
import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import regex
import tensorstore as ts
from draccus import ChoiceRegistry, field
from haliax import Axis
from jaxtyping import PRNGKeyArray
from tokenizers import normalizers

import levanter
from levanter.data import AsyncDataset
from levanter.data.dataset import EpochDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy, rescale_mixture_schedule_for_batch_schedule
from levanter.data.packing import GreedyPrepackedDataset
from levanter.data.passthrough_tokenizer import PassthroughTokenizer
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheMetadata, CacheOptions, TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStore
from levanter.utils import fsspec_utils
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer

# intercept the logging nonsense here
from levanter.utils.logging import silence_transformer_nag  # noqa

silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_tokenizer  # noqa
from levanter.data._preprocessor import BatchProcessor, IdentityProcessor, U, dict_from_record_batch  # noqa
from levanter.data.sharded_datasource import (  # noqa
    JsonlDataSource,
    ShardedDataSource,
    TextUrlDataSource,
    UrlDataSource,
    WrappedHFDataSource,
)
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.store.cache import build_or_load_cache  # noqa


# Metrics monitoring removed; keep alias for type hints.
MetricsMonitor = Any
from levanter.utils.jax_utils import key_iterator, use_cpu_device  # noqa


T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"

DEFAULT_IGNORE_INDEX = -100  # Mirrors pytorch's default ignore index


class GenericTokenSeqDataset(AsyncDataset[T_co], Generic[T_co]):
    """
    A dataset that yields fixed-length sequences from an underlying TreeCache.

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
        token_arrays = await self._await_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_cache(self, key: str = "input_ids") -> JaggedArrayStore:
        if self._store is None:
            self._store = self.doc_cache.store
        return self._store.tree[key]

    async def final_length_is_known(self) -> bool:
        return await self.doc_cache.final_length_is_known()

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        store = await self._await_cache()
        return store.data_size // self.seq_len

    async def wait_until_len_at_least(self, length: int) -> int:
        # length is brutally slow to compute, so we cache it
        if self._cached_len is not None and self._cached_len >= length:
            return self._cached_len

        # TODO: would be better to listen for cache updates
        length = await super().wait_until_len_at_least(length)
        self._cached_len = length
        return length


class TokenSeqDataset(GenericTokenSeqDataset[np.ndarray]):
    """A dataset that yields sequences of tokens from a cache."""

    async def get_batch(self, indices: Sequence[int]) -> Sequence[np.ndarray]:
        token_arrays = await self._await_cache()
        ds_len = await self.wait_until_len_at_least(max(indices) + 1)
        if ds_len is not None and ds_len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        return await asyncio.gather(*out)


class WeightedTokenSeqDataset(GenericTokenSeqDataset[dict[str, np.ndarray]]):
    """A dataset that yields sequences of tokens and loss weights from a cache."""

    async def get_batch(self, indices: Sequence[int]) -> Sequence[dict[str, np.ndarray]]:
        token_arrays = await self._await_cache("input_ids")
        weight_arrays = await self._await_cache("loss_weight")

        ds_len = await self.wait_until_len_at_least(max(indices) + 1)
        if ds_len is not None and ds_len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")

        offsets = np.array(indices, dtype=np.int64) * self.seq_len

        with ts.Batch():
            token_reads = []
            weight_reads = []
            for offset in offsets:
                token_reads.append(token_arrays.data[offset : offset + self.seq_len].read())
                weight_reads.append(weight_arrays.data[offset : offset + self.seq_len].read())

        tokens = await asyncio.gather(*token_reads)
        weights = await asyncio.gather(*weight_reads)

        return [{"input_ids": t, "loss_weight": w} for t, w in zip(tokens, weights)]


def standard_extractor(data: np.ndarray, Pos: Axis) -> dict[str, hax.NamedArray]:
    """Extract tokens from a numpy array."""
    return {"tokens": hax.named(data, Pos)}


def weighted_extractor(data: dict[str, np.ndarray], Pos: Axis) -> dict[str, hax.NamedArray]:
    """Extract tokens and loss weights from a dict."""
    return {
        "tokens": hax.named(data["input_ids"], Pos),
        "loss_weight": hax.named(data["loss_weight"], Pos),
    }


class CausalLmDataset(MappedAsyncDataset[Any, LmExample]):
    def __init__(
        self,
        dataset: AsyncDataset,
        Pos: Axis,
        *,
        extractor: Callable[[Any, Axis], dict[str, hax.NamedArray]] = standard_extractor,
        ignore_index: Optional[int] = None,
        eos_id: Optional[int] = None,
        block_cross_document_attention: bool = True,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.ignore_id = ignore_index
        self.eos_id = eos_id
        self.block_cross_document_attention = block_cross_document_attention

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(data):
            extracted = extractor(data, self.Pos)
            example = LmExample.causal(
                **extracted,
                ignore_id=self.ignore_id,
                eos_id=eos_id,
                block_cross_document_attention=block_cross_document_attention,
            )
            example = jax.lax.with_sharding_constraint(example, sharding)
            return example

        super().__init__(self.dataset, _create_lm_example)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


class WeightedCausalLmDataset(CausalLmDataset):
    """A dataset that creates LmExamples with per-token loss weights."""

    def __init__(
        self,
        dataset: AsyncDataset[dict[str, np.ndarray]],
        Pos: Axis,
        *,
        ignore_index: Optional[int] = None,
        eos_id: Optional[int] = None,
        block_cross_document_attention: bool = True,
    ):
        super().__init__(
            dataset,
            Pos,
            extractor=weighted_extractor,
            ignore_index=ignore_index,
            eos_id=eos_id,
            block_cross_document_attention=block_cross_document_attention,
        )


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


LONG_STRING_WORKAROUND = 10_000

ws = regex.compile(r"\s")


class BaseBatchTokenizer(BatchProcessor[dict, dict]):
    """Base class for tokenizer-based batch processors."""

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "text",
        *,
        override_resources=None,
    ):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.override_resources = override_resources

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0


class BatchTokenizer(BaseBatchTokenizer):
    """
    A batch processor that tokenizes a batch of strings using a tokenizer.
    By default, this will append eos to the end of the string, even if the tokenizer doesn't.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field="text",
        enforce_bos=True,
        enforce_eos=True,
        *,
        override_resources=None,
        _workaround_len=LONG_STRING_WORKAROUND,
        return_attention_mask=False,
        padding=False,
        max_length=None,
    ):
        super().__init__(tokenizer, text_field, override_resources=override_resources)
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

        # see if the tokenizer appends bos/eos
        # if we don't have an eos/bos token in the tokenizer, skip
        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        # HF's BPE-based tokenizers do not, but the bert and roberta ones do
        # TODO: this doesn't necessarily ensure it, I guess, but eh
        if enforce_eos or enforce_bos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id and enforce_eos
            should_append_bos = input_ids[0] != tokenizer.bos_token_id and enforce_bos
        else:
            should_append_eos = False
            should_append_bos = False

        self._need_to_add_eos = should_append_eos
        self._need_to_add_bos = should_append_bos
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        batch_text = [example[self.text_field] for example in batch]

        if self._need_to_add_bos:
            batch_text = [self.tokenizer.bos_token + " " + d for d in batch_text]

        if self._need_to_add_eos:
            batch_text = [d + " " + self.tokenizer.eos_token for d in batch_text]

        if self._needs_long_sequence_workaround:
            batch_text, needs_merge = self._break_for_long_sequences(batch_text)
        else:
            needs_merge = []

        if self.padding is not False:
            encoding = self.tokenizer(
                batch_text,
                return_attention_mask=self.return_attention_mask,
                verbose=False,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
            )  # type: ignore
        else:
            encoding = self.tokenizer(
                batch_text, return_attention_mask=self.return_attention_mask, verbose=False
            )  # type: ignore

        if needs_merge:
            new_encoding = self._merge_split_encodings(batch_text, encoding, needs_merge)
            encoding = BatchEncoding(new_encoding)

        # debatch the encoding
        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]

        return unbatched

    def _break_for_long_sequences(self, batch):
        orig_lengths = [len(d) for d in batch]
        # break any strings that are longer than LONG_STRING_WORKAROUND characters into smaller chunks
        orig_batch = batch
        batch = []
        needs_merge = []
        for i, d in enumerate(orig_batch):
            needs_merge.append(False)
            orig_len = orig_lengths[i]
            while len(d) > self._workaround_len:
                # we'd rather break strings at whitespace, so find the first whitespace
                match = ws.search(d, self._workaround_len)
                # this is vanishingly unlikely, but if we can't find a whitespace, just break it at the limit
                if match is None:
                    split = len(d)
                else:
                    split = match.start()

                batch.append(d[:split])
                needs_merge.append(True)

                d = d[split:]
                orig_len -= split

            batch.append(d)
        return batch, needs_merge

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        return dict(**self.tokenizer("hi there", return_attention_mask=self.return_attention_mask, verbose=False))

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        # merge the encodings back together
        # we might need to merge multiple encodings together
        # needs merge marks the first n-1 encodings that need to be merged for each document
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], np.ndarray):
                assert len(v) == len(batch)
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        v_out.append(np.concatenate(vs_to_merge))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(np.concatenate(vs_to_merge))

                new_encoding[k] = v_out
            elif isinstance(v[0], list):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(list(chain(*vs_to_merge)))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(list(chain(*vs_to_merge)))
                new_encoding[k] = v_out
            else:
                raise ValueError(f"Unknown type {type(v[0])}")
        return new_encoding

    # TODO remove this when it's resolved https://github.com/huggingface/tokenizers/issues/1495
    @cached_property
    def _needs_long_sequence_workaround(self):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            if normalizer is None:
                return False
            # if there's a "Replace" normalizer, then we need to do the workaround
            # inexplicably there's no way to see inside a Sequence so we also have to assume it needs it
            return isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))
        else:
            return False


class DNABatchTokenizer(BaseBatchTokenizer):
    """
    A batch processor that tokenizes DNA sequences with soft-masking support.

    Assigns loss weights based on character case:
    - Uppercase (ACGT): weight = 1.0
    - Lowercase (acgt): weight = soft_mask_weight

    Assumptions:
    - Character-level tokenizer (1:1 character-to-token mapping)
    - All sequences have the same length (no padding/truncation)
    - Model context size matches sequence length (see experiment configs)
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        text_field: str = "seq",
        soft_mask_weight: float = 1.0,
        *,
        override_resources=None,
    ):
        super().__init__(tokenizer, text_field, override_resources=override_resources)
        self.soft_mask_weight = soft_mask_weight

    def __call__(self, batch: Sequence[dict]) -> list[dict]:
        results = []
        for example in batch:
            text = example[self.text_field]
            loss_weight = [1.0 if c.isupper() else self.soft_mask_weight for c in text]
            encoding = self.tokenizer(text, return_attention_mask=False, verbose=False)
            results.append(
                {
                    "input_ids": np.array(encoding["input_ids"], dtype=np.int32),
                    "loss_weight": np.array(loss_weight, dtype=np.float32),
                }
            )
        return results

    @property
    def output_exemplar(self) -> dict:
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "loss_weight": np.zeros((0,), dtype=np.float32),
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "soft_mask_weight": self.soft_mask_weight,
        }


class LmDatasetFormatBase(abc.ABC, ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "text"


@LmDatasetFormatBase.register_subclass("text")
@dataclass(frozen=True)
class TextLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for raw text examples.

    Attributes:
        text_key: Field name containing the raw text or tokens.
    """

    text_key: str = "text"  # key for the text field in the jsonl file


@LmDatasetFormatBase.register_subclass("chat")
@dataclass(frozen=True)
class ChatLmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for multi-turn chat transcripts.

    Attributes:
        messages_field: Field name containing the ordered list of chat messages.
        chat_template: Overrides the tokenizer's chat template when provided.
        system_prompt: Field name carrying an optional system instruction to prepend.
        chat_template_kwargs: Field name containing optional keyword arguments passed to the chat template.
        pack: Whether to allow example packing for efficient batching.
        mask_user_turns: Mask user tokens from the training loss when True.
    """

    messages_field: str = "messages"  # key for the messages field in the jsonl file
    chat_template: str | None = None
    system_prompt: str | None = None
    chat_template_kwargs: str | None = "chat_template_kwargs"
    pack: bool = True
    mask_user_turns: bool = True


@LmDatasetFormatBase.register_subclass("dna")
@dataclass(frozen=True)
class DNALmDatasetFormat(LmDatasetFormatBase):
    """Dataset configuration for DNA sequences with soft-masking support.

    Supports position-wise loss weighting based on character case:
    - Uppercase nucleotides (ACGT): full loss weight (1.0)
    - Lowercase nucleotides (acgt): reduced loss weight (soft_mask_weight)

    This is useful for down-weighting repetitive elements in genomic data,
    as pioneered by GPN and adopted by PlantCaduceus and Evo 2.

    Attributes:
        text_key: Field name containing the DNA sequence.
        soft_mask_weight: Loss weight for lowercase (soft-masked) positions.
    """

    text_key: str = "seq"
    soft_mask_weight: float = 1.0


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
    auto_build_caches: bool = True
    """Whether to build dataset caches automatically when they are missing.

    If False, any attempt to access a cache that does not already exist will raise
    a FileNotFoundError instead of building the cache on the fly. This is useful
    when running in environments where cache construction is undesirable (e.g.,
    to avoid expensive preprocessing during training jobs).
    """

    chat_template: str | None = None  # If set, use this template for chat datasets. Otherwise, use the tokenizer's.

    ignore_token_id: Optional[int] = DEFAULT_IGNORE_INDEX

    shuffle: bool | int = False
    """whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length"""
    permutation_type: Literal["feistel", "linear"] | None = None
    """
    Type of permutation to use for shuffle.

    If None, defaults to linear, but this will change in the future since Feistel is better.
    """

    block_cross_document_attention: bool = True
    """Whether to block attention across document boundaries.

    If True (default), attention is blocked across documents using segment ids derived from
    EOS tokens. This prevents tokens from one document attending to tokens from a different
    document within a packed sequence.

    If False, full causal attention is allowed across packed documents, meaning tokens can
    attend to all previous tokens regardless of document boundaries.
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


def preprocessor_for_format(
    format: LmDatasetFormatBase, tokenizer: HfTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
) -> BatchProcessor[dict, dict]:
    match format:
        case TextLmDatasetFormat(text_key=key):
            return BatchTokenizer(tokenizer, enforce_bos=enforce_bos, enforce_eos=enforce_eos, text_field=key)
        case ChatLmDatasetFormat(
            messages_field=m,
            chat_template=ct,
            system_prompt=sp,
            chat_template_kwargs=ct_kwargs,
            mask_user_turns=mt,
        ):
            return ChatProcessor(
                tokenizer,
                messages_field=m,
                chat_template=ct,
                system_prompt_field=sp,
                chat_template_kwargs_field=ct_kwargs,
                mask_user_turns=mt,
            )  # type: ignore
        case DNALmDatasetFormat(text_key=key, soft_mask_weight=weight):
            return DNABatchTokenizer(
                tokenizer,
                text_field=key,
                soft_mask_weight=weight,
            )
        case _:
            raise ValueError(f"Unknown format {format}")


def dataset_for_format(
    format: LmDatasetFormatBase,
    Pos: Axis,
    cache: TreeCache[dict],
    *,
    eos_id: int | None,
    ignore_index: int | None,
    block_cross_document_attention: bool = True,
) -> AsyncDataset[LmExample]:
    match format:
        case TextLmDatasetFormat():
            return CausalLmDataset(
                TokenSeqDataset(cache, Pos.size),
                Pos,
                eos_id=eos_id,
                ignore_index=ignore_index,
                block_cross_document_attention=block_cross_document_attention,
            )
        case ChatLmDatasetFormat(pack=pack, mask_user_turns=mask_user_turns):
            return MultiturnChatDataset(cache, Pos, max_segments_per_example=64 if pack else 1, mask_user_turns=mask_user_turns)  # type: ignore
        case DNALmDatasetFormat():
            return WeightedCausalLmDataset(
                WeightedTokenSeqDataset(cache, Pos.size),
                Pos,
                eos_id=eos_id,
                ignore_index=ignore_index,
                block_cross_document_attention=block_cross_document_attention,
            )
        case _:
            raise ValueError(f"Unknown format {format}")


def build_lm_dataset_cache(
    cache_dir: str,
    source: ShardedDataSource[dict],
    format: LmDatasetFormatBase,
    tokenizer: HfTokenizer,
    options: CacheOptions = CacheOptions.default(),
    enforce_eos=True,
) -> TreeCache[dict]:
    """
    Creates a cache for a dataset. If the cache already exists, it will be loaded. Otherwise, it will be built.

    Args:
        cache_dir: the path to the cache e.g. gs://my-bucket/cache/train
        source: the source of the data.
        format: the format of the data
        tokenizer: the tokenizer
        options: the cache options to control how it's built
        enforce_eos: whether to enforce EOS

    Returns:

    """
    # name is the final two components of the path
    name = os.path.join(*cache_dir.split("/")[-2:])

    processor = preprocessor_for_format(format, tokenizer, enforce_bos=True, enforce_eos=enforce_eos)
    try:
        return TreeCache.load(
            cache_dir,
            exemplar=processor.output_exemplar,
            options=CacheMetadata(preprocessor_metadata=processor.metadata),
        )
    except FileNotFoundError:
        pass

    logger.info(f"Building cache for {name}...")
    return build_or_load_cache(
        cache_dir,
        source,
        processor,
        options=options,
    )


def load_lm_dataset_cache(
    cache_dir: str,
    format: LmDatasetFormatBase,
    tokenizer: HfTokenizer,
    enforce_eos=True,
) -> TreeCache[dict]:
    """Similar to build_lm_dataset_cache, but just loads the cache. Raises an error if the cache doesn't exist."""

    processor = preprocessor_for_format(format, tokenizer, enforce_bos=True, enforce_eos=enforce_eos)
    cache = TreeCache.load(
        cache_dir,
        exemplar=processor.output_exemplar,
        options=CacheMetadata(preprocessor_metadata=processor.metadata),
    )
    return cache


@dataclass(frozen=True)
class SingleDatasetLMConfigBase(LmDatasetSourceConfigBase, LMTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    cache_dir: Optional[str] = "cache/"
    auto_build_caches: bool = True

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
            ds = dataset_for_format(
                self.format,
                Pos,
                cache,
                eos_id=self.the_tokenizer.eos_token_id,
                ignore_index=self.ignore_token_id,
                block_cross_document_attention=self.block_cross_document_attention,
            )

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

        return dataset_for_format(
            self.format,
            Pos,
            cache,
            eos_id=self.the_tokenizer.eos_token_id,
            ignore_index=self.ignore_token_id,
            block_cross_document_attention=self.block_cross_document_attention,
        )

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
        auto_build = self.auto_build_caches

        if cache_dir is None:
            raise ValueError("cache_dir cannot be None")

        cache_dir = os.path.join(cache_dir, split)

        cache_exists = fsspec_utils.exists(cache_dir)

        if cache_exists:
            try:
                return load_lm_dataset_cache(cache_dir, format, tokenizer, enforce_eos)
            except FileNotFoundError:
                if not auto_build:
                    raise
                # fall through to rebuild if allowed

        if not auto_build:
            raise FileNotFoundError(f"Cache not found at {cache_dir} and auto_build_caches is disabled")

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
    auto_build_caches: bool = True

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
                ignore_index=self.ignore_token_id,
                block_cross_document_attention=self.block_cross_document_attention,
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

            cache_path = os.path.join(cache_dir, split)

            # easy path: cache already exists
            try:
                caches[name] = load_lm_dataset_cache(
                    cache_path,
                    source_config.format,
                    self.the_tokenizer,
                    self.enforce_eos,
                )
                continue
            except FileNotFoundError:
                # Will build below
                pass

            # now see if we can/need to build the cache
            try:
                source = source_config.get_shard_source(split)
                if source is None:
                    logger.warning(f"No source for {name} in {split} split, skipping")
                    continue

                elif not self.auto_build_caches:
                    raise FileNotFoundError(f"Cache not found at {cache_path} and auto_build_caches is disabled")
                else:
                    caches[name] = build_lm_dataset_cache(
                        cache_path,
                        source,
                        source_config.format,
                        self.the_tokenizer,
                        self.cache_options,
                        self.enforce_eos,
                    )
            except Exception as e:
                logger.exception(f"Error building/loading cache for dataset {name} {split} {cache_path}: {e}")
                raise

        return caches

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return self.configs


ProcessedChatDict = TypedDict(
    "ProcessedChatDict",
    {
        "input_ids": np.ndarray,
        "assistant_masks": np.ndarray,
    },
)


class ChatProcessor(BatchProcessor[dict, ProcessedChatDict]):
    """
    A batch processor that converts chat data into the expected inputs of a model using a chat template.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        chat_template: str | None = None,
        messages_field: str = "messages",
        system_prompt_field: str | None = "system",
        chat_template_kwargs_field: str | None = "chat_template_kwargs",
        mask_user_turns: bool = True,
    ):
        if chat_template is None and tokenizer.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")
        self.tokenizer = tokenizer
        self.chat_template = chat_template or tokenizer.chat_template
        self.messages_field = messages_field
        self.system_prompt_field = system_prompt_field
        self.chat_template_kwargs_field = chat_template_kwargs_field

        if self.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")

        # check for {%generation%} in the template
        # cribbed from https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1687
        if mask_user_turns and not re.search(r"\{%-?\s*generation\s*-?%}", self.chat_template):
            raise ValueError(
                "Chat template must contain {%generation%} to indicate the position of the assistant message "
                "if mask_user_turns is True. However, the provided template does not contain this tag: "
                " ```{chat_template}```. "
                "See https://levanter.readthedocs.io/en/latest/reference/Data-Formats.html#chat-templates"
                " for more details."
            )

    def __call__(self, batch: Sequence[dict]) -> Sequence[ProcessedChatDict]:
        # Extract messages from the specified field, optionally injecting a system prompt
        messages: list[list[dict[str, Any]]] = []
        chat_kwargs_list: list[Mapping[str, Any] | None] = []
        for example in batch:
            example_messages = example[self.messages_field]
            # Copy to avoid mutating the original structure
            normalized_messages = list(example_messages)

            if self.system_prompt_field is not None and self.system_prompt_field in example:
                system_content = example[self.system_prompt_field]
                if system_content is not None:
                    if isinstance(system_content, Mapping):
                        system_message = dict(system_content)
                        system_message["role"] = "system"
                        if "content" not in system_message:
                            raise ValueError(
                                "System prompt mapping must include a 'content' field when provided as a mapping."
                            )
                    else:
                        system_message = {"role": "system", "content": system_content}
                    normalized_messages = [system_message, *normalized_messages]

            messages.append(normalized_messages)

            example_kwargs: Mapping[str, Any] | None = None
            if self.chat_template_kwargs_field is not None and self.chat_template_kwargs_field in example:
                raw_kwargs = example[self.chat_template_kwargs_field]
                if raw_kwargs is not None:
                    if not isinstance(raw_kwargs, Mapping):
                        raise ValueError("chat_template_kwargs must be provided as a mapping when present.")
                    example_kwargs = dict(raw_kwargs)
            chat_kwargs_list.append(example_kwargs)

        use_per_example_kwargs = any(kwargs for kwargs in chat_kwargs_list)

        if not use_per_example_kwargs:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                chat_template=self.chat_template,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
        else:
            input_ids_batches: list[Sequence[int]] = []
            assistant_mask_batches: list[Sequence[int]] = []

            for conversation, example_kwargs in zip(messages, chat_kwargs_list):
                kwargs_dict = dict(example_kwargs) if example_kwargs is not None else {}

                for forbidden in ("tokenize", "return_assistant_tokens_mask", "return_dict"):
                    if forbidden in kwargs_dict:
                        raise ValueError(
                            f"chat_template_kwargs may not override '{forbidden}' because the processor relies on it."
                        )

                chat_template_override = kwargs_dict.pop("chat_template", self.chat_template)
                if chat_template_override is None:
                    raise ValueError("Chat template must be provided either in the dataset format or per example.")

                apply_kwargs = {
                    **kwargs_dict,
                    "tokenize": True,
                    "return_assistant_tokens_mask": True,
                    "return_dict": True,
                    "chat_template": chat_template_override,
                }

                tokenized_single = self.tokenizer.apply_chat_template(
                    [conversation],
                    **apply_kwargs,
                )

                input_ids_batches.extend(tokenized_single["input_ids"])
                assistant_mask_batches.extend(tokenized_single["assistant_masks"])

            tokenized = {"input_ids": input_ids_batches, "assistant_masks": assistant_mask_batches}

        masks = tokenized["assistant_masks"]
        for seq, mask_for_seq in zip(batch, masks):
            if not np.any(mask_for_seq):
                raise ValueError(f"Chat did not contain an assistant message for sequence {seq}")

        out: list[ProcessedChatDict] = []
        for ids, mask in zip(tokenized["input_ids"], masks):
            out.append(
                {
                    "input_ids": np.array(ids, dtype=np.int32),
                    "assistant_masks": np.array(mask, dtype=np.int32),
                }
            )

        return out

    @property
    def output_exemplar(self):
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "assistant_masks": np.zeros((0,), dtype=np.int32),
        }

    @property
    def num_cpus(self) -> int:
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "chat_template": self.chat_template,
            "messages_field": self.messages_field,
            "system_prompt_field": self.system_prompt_field,
            "chat_template_kwargs_field": self.chat_template_kwargs_field,
        }


class MultiturnChatDataset(MappedAsyncDataset[tuple[ProcessedChatDict, ProcessedChatDict], LmExample]):
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

        train_set = dataset_for_format(source.format, Pos, cache, eos_id=None, ignore_index=None)
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

        validation_set = dataset_for_format(source.format, Pos, cache, eos_id=None, ignore_index=None)
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
