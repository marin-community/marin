# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import dataclasses
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import regex
import tensorstore as ts
from draccus import ChoiceRegistry, field
from jaxtyping import PRNGKeyArray
from tokenizers import normalizers

import haliax as hax
from haliax import Axis

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
from levanter.data.metrics_monitor import LoggerMetricsMonitor, LoggingMetricsMonitor, MetricsMonitor  # noqa
from levanter.data.sharded_datasource import (  # noqa
    JsonlDataSource,
    ShardedDataSource,
    TextUrlDataSource,
    UrlDataSource,
    WrappedHFDataSource,
)
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.store.cache import build_or_load_cache  # noqa
from levanter.utils.jax_utils import key_iterator, use_cpu_device  # noqa


T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"

DEFAULT_IGNORE_INDEX = -100  # Mirrors pytorch's default ignore index


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
        self._store: Optional[TreeStore] = None
        self._cached_len: Optional[int] = None

    async def async_len(self) -> int:
        await self.doc_cache.finished()
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = await self.doc_cache.store_async()
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
        ignore_index: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.ignore_id = ignore_index
        self.eos_id = eos_id

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _create_lm_example(tokens):
            tokens = hax.named(tokens, self.Pos)
            example = LmExample.causal(tokens=tokens, ignore_id=self.ignore_id, eos_id=eos_id)

            return example

        super().__init__(self.dataset, _create_lm_example)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


LONG_STRING_WORKAROUND = 10_000

ws = regex.compile(r"\s")


class BatchTokenizer(BatchProcessor[dict, dict]):
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
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.override_resources = override_resources
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


CANONICAL_INPUT_FIELD = "prompt"
CANONICAL_OUTPUT_FIELD = "response"


class LmDatasetFormatBase(abc.ABC, ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "text"


@LmDatasetFormatBase.register_subclass("text")
@dataclass(frozen=True)
class TextLmDatasetFormat(LmDatasetFormatBase):
    text_key: str = "text"  # key for the text field in the jsonl file


@LmDatasetFormatBase.register_subclass("chat")
@dataclass(frozen=True)
class ChatLmDatasetFormat(LmDatasetFormatBase):
    messages_field: str = "messages"  # key for the messages field in the jsonl file
    single_turn: bool = False
    chat_template: str | None = None
    pack: bool = True
    mask_user_turns: bool = True


@LmDatasetFormatBase.register_subclass("supervised")
@dataclass(frozen=True)
class SupervisedLmDatasetFormat(LmDatasetFormatBase):
    input_field: str = CANONICAL_INPUT_FIELD  # key for the input field in the jsonl file
    output_field: str = CANONICAL_OUTPUT_FIELD  # key for the output field in the jsonl file
    separate_with: str | int | None = None  # string to separate input and output with
    pack: bool = True
    mask_inputs: bool = True


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

    ignore_token_id: Optional[int] = DEFAULT_IGNORE_INDEX

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
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        pass

    @abc.abstractmethod
    def train_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @abc.abstractmethod
    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, TreeCache[dict]]:
        pass

    @property
    @abc.abstractmethod
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        pass

    def tagged_eval_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> list[Tuple[AsyncDataset[LmExample], List[str]]]:
        tags = {name: (config.tags or []) + [name] for name, config in self.sources.items()}
        eval_sets = self.validation_sets(Pos, monitors)

        return [(eval_sets[name], tags[name]) for name in eval_sets]


def preprocessor_for_format(
    format: LmDatasetFormatBase, tokenizer: HfTokenizer, *, enforce_eos: bool = True, enforce_bos: bool = True
) -> BatchProcessor[dict, dict]:
    match format:
        case TextLmDatasetFormat(text_key=key):
            return BatchTokenizer(tokenizer, enforce_bos=enforce_bos, enforce_eos=enforce_eos, text_field=key)
        case ChatLmDatasetFormat(messages_field=m, single_turn=s_turn, chat_template=ct, mask_user_turns=mt):
            if s_turn:
                if ct is not None:
                    raise NotImplementedError("Don't currently support chat templates for single turn chat")
                return SingleTurnChatProcessor(tokenizer, messages_field=m)  # type: ignore
            else:
                return ChatProcessor(tokenizer, messages_field=m, chat_template=ct, mask_user_turns=mt)  # type: ignore
        case SupervisedLmDatasetFormat(input_field=i, output_field=o, separate_with=s):
            return SupervisedProcessor(tokenizer, input_field=i, output_field=o, separate_with=s)  # type: ignore
        case _:
            raise ValueError(f"Unknown format {format}")


def dataset_for_format(
    format: LmDatasetFormatBase,
    Pos: Axis,
    cache: TreeCache[dict],
    *,
    eos_id: int | None,
    ignore_index: int | None,
) -> AsyncDataset[LmExample]:
    match format:
        case TextLmDatasetFormat():
            return CausalLmDataset(TokenSeqDataset(cache, Pos.size), Pos, eos_id=eos_id, ignore_index=ignore_index)
        case ChatLmDatasetFormat(single_turn=single_turn, pack=pack, mask_user_turns=mask_user_turns):
            if single_turn:
                # We treat single turn like supervised
                return SupervisedDataset(cache, Pos, max_segments_per_example=64 if pack else 1, mask_inputs=mask_user_turns)  # type: ignore
            else:
                return MultiturnChatDataset(cache, Pos, max_segments_per_example=64 if pack else 1, mask_user_turns=mask_user_turns)  # type: ignore
        case SupervisedLmDatasetFormat(pack=pack, mask_inputs=mask_inputs):
            return SupervisedDataset(cache, Pos, max_segments_per_example=64 if pack else 1, mask_inputs=mask_inputs)  # type: ignore
        case _:
            raise ValueError(f"Unknown format {format}")


class SupervisedSourceConfigBase(Protocol):
    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        raise NotImplementedError

    input_field: str
    output_field: str
    tags: Optional[List[str]]
    cache_dir: str


@dataclass(frozen=True)
class SupervisedHfSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    id: str
    name: str | None = None

    streaming: bool = True

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        return WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.streaming).map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


@dataclass(frozen=True)
class SupervisedUrlSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    train_urls: list[str] = dataclasses.field(default_factory=list)
    validation_urls: list[str] = dataclasses.field(default_factory=list)

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        urls = self.train_urls if split == "train" else self.validation_urls
        if not urls:
            return None

        source = UrlDataSource(urls, columns=[self.input_field, self.output_field])
        return source.map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


SupervisedSourceConfig: TypeAlias = Union[SupervisedHfSourceConfig, SupervisedUrlSourceConfig]

# for compatibility with old configs
LMSupervisedDatasetConfig: TypeAlias = SupervisedUrlSourceConfig


def _preprocess_supervised_example(
    batch, tokenizer: PreTrainedTokenizerBase, input_field: str, output_field: str
) -> dict:
    sources = [example[input_field] for example in batch]

    targets = [example[output_field] for example in batch]
    # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how alpaca does it.
    examples = [s + t for s, t in zip(sources, targets)]
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)
    examples_tokenized = tokenizer(examples, padding=False, truncation=True)

    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def _prepare_supervised_examples(ex: list[dict], tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis) -> list[LmExample]:
    """
    Prepare examples for training. This function converts the (cached) encodings into an LmExample.

    It goes through the following steps:

    1. Pad the batch to the maximum length.
    2. Mask out the input and prompt if requested.
    3. Create an LmExample with the input_ids as the input and the next token as the target.
    """
    lens = np.array([ex["sources_len"] for ex in ex])

    ex_pad = tokenizer.pad(
        ex,
        padding="max_length",
        max_length=Pos.size,
    )

    input_ids = ex_pad["input_ids"]
    truncated = [ids[-Pos.size :] for ids in input_ids]

    out = []
    for ids, length in zip(truncated, lens):
        causal = _mk_sup_example_jit(Pos, hax.named(ids, Pos), length, tokenizer.pad_token_id, tokenizer.eos_token_id)

        out.append(causal)

    return out


@functools.partial(jax.jit, static_argnums=(0, 3, 4))
def _mk_sup_example_jit(Pos, input_ids: hax.NamedArray, sources_len, pad_token_id, eos_id):
    # mask out padding and anything before the start of the target
    loss_mask = hax.arange(Pos) >= sources_len - 1
    # don't predict the padding
    targets = hax.roll(input_ids, -1, Pos)
    loss_mask = loss_mask & (targets != pad_token_id)
    loss_mask = loss_mask & (1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.bool_))
    return LmExample.causal(input_ids, loss_mask=loss_mask, eos_id=eos_id)


def mk_supervised_dataset(
    config: SupervisedSourceConfigBase, split: str, tokenizer: HfTokenizer, Pos: hax.Axis
) -> AsyncDataset[LmExample]:

    source = config.get_shard_source(split)

    if source is None:
        raise ValueError("No training data source found")

    processor = SupervisedProcessor(tokenizer, config.input_field, config.output_field)

    cached_dataset = build_or_load_cache(
        config.cache_dir,
        source,
        processor,
        await_finished=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


@dataclass(frozen=True)
class ChatUrlDataSourceConfig:
    """Config for loading JSONL files in OpenAI chat format for supervised fine-tuning."""

    cache_dir: str
    train_urls: List[str] = field(default_factory=list)
    validation_urls: List[str] = field(default_factory=list)

    # Chat format specific fields
    messages_field: str = "messages"
    input_role: str = "user"
    output_role: str = "assistant"

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        """Gets ShardedDataSource for either training or validation data."""
        urls = self.validation_urls if split == "validation" else self.train_urls

        if not urls:
            return None

        # Use the datasource_from_chat_jsonl_single_turn function from sharded_datasource
        return datasource_from_chat_jsonl_single_turn(
            urls, messages_field=self.messages_field, input_role=self.input_role, output_role=self.output_role
        )


def preprocess_legacy_chat_template(batch, tokenizer: PreTrainedTokenizerBase, should_append_eos: bool) -> dict:
    """
    Preprocess chat examples to match the format of preprocess_supervised_example.
    Returns a dict with input_ids and sources_len like the supervised case.

    Args:
        batch: List of dicts with input/output pairs
        tokenizer: HuggingFace tokenizer
        should_append_eos: Whether we need to manually add EOS (True if tokenizer doesn't do it automatically)
    """
    # Get sources (inputs) and targets (outputs) from the batch
    sources = [example["input"] for example in batch]
    targets = [example["output"] for example in batch]

    # Add EOS only if needed (tokenizer doesn't do it automatically)
    if should_append_eos:
        targets = [t + tokenizer.eos_token for t in targets]

    # Tokenize sources alone first to get the source lengths
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)

    # Combine source and target for full examples
    full_examples = [f"{s}{t}" for s, t in zip(sources, targets)]
    examples_tokenized = tokenizer(full_examples, padding=False, truncation=True)

    # Get source lengths to mask loss appropriately
    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def mk_single_turn_cached_sft_dataset(
    config: ChatUrlDataSourceConfig, tokenizer: HfTokenizer, Pos: hax.Axis
) -> AsyncDataset[dict]:
    """Creates a dataset from JSONL files containing chat format data for SFT."""
    source = config.get_shard_source("train")
    if source is None:
        raise ValueError("No training data source found")

    # Set up example structure matching supervised case
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    input_ids = tokenizer("hi there")["input_ids"]
    should_append_eos = input_ids[-1] != tokenizer.eos_token_id
    logger.info(f"Manual EOS Needed: {should_append_eos}")

    # Process the dataset
    dataset = source.map_batches(
        lambda ex: preprocess_legacy_chat_template(ex, tokenizer, should_append_eos),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    # Cache the processed data
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)
    return cached_dataset


def build_lm_dataset_cache(
    cache_dir: str,
    source: ShardedDataSource[dict],
    format: LmDatasetFormatBase,
    tokenizer: HfTokenizer,
    options: CacheOptions = CacheOptions.default(),
    enforce_eos=True,
    monitors: Union[bool, List[MetricsMonitor]] = True,
):
    """
    Creates a cache for a dataset. If the cache already exists, it will be loaded. Otherwise, it will be built.

    Args:
        cache_dir: the path to the cache e.g. gs://my-bucket/cache/train
        source: the source of the data.
        format: the format of the data
        tokenizer: the tokenizer
        options: the cache options to control how it's built
        enforce_eos: whether to enforce EOS
        monitors: the metrics monitors to use

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

    if source is None:
        logger.info(f"No data for {name}")
        return None

    logger.info(f"Building cache for {name}...")
    if monitors is True:
        monitors = [
            LoggingMetricsMonitor(prefix=f"preprocessing/{name}", commit=False),
            LoggerMetricsMonitor(f"preprocessing.{name}"),
        ]
    elif monitors is False:
        monitors = []
    return build_or_load_cache(
        cache_dir,
        source,
        processor,
        monitors=monitors,
        await_finished=False,
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


def mk_chat_sft_dataset(
    config: ChatUrlDataSourceConfig, tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis
) -> AsyncDataset[LmExample]:
    """Creates a dataset from JSONL files containing chat format data for SFT."""
    source = config.get_shard_source("train")
    if source is None:
        raise ValueError("No training data source found")

    # Set up example structure matching supervised case
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    input_ids = tokenizer("hi there")["input_ids"]
    should_append_eos = input_ids[-1] != tokenizer.eos_token_id
    logger.info(f"Manual EOS Needed: {should_append_eos}")

    # Process the dataset
    dataset = source.map_batches(
        lambda ex: preprocess_legacy_chat_template(ex, tokenizer, should_append_eos),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    # Cache the processed data
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)

    # Ensure padding token is set (needed by _prepare_supervised_example)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Reuse the supervised prepare function directly
    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


@dataclass(frozen=True)
class SingleDatasetLMConfigBase(LmDatasetSourceConfigBase, LMTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    cache_dir: Optional[str] = "cache/"

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        del batch_schedule  # unused

        cache = self.build_or_load_cache("train", monitors=monitors)
        if cache is None:
            raise ValueError("No training set!")
        else:
            ds = dataset_for_format(
                self.format, Pos, cache, eos_id=self.the_tokenizer.eos_token_id, ignore_index=self.ignore_token_id
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
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        return {
            # we don't care about BatchSchedule in this class
            "": self.train_set(Pos, BatchSchedule(32), monitors, key=key, epochs=epochs)
        }

    def validation_set(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
    ) -> AsyncDataset[LmExample] | None:
        cache = self.build_or_load_cache("validation", monitors=monitors)
        if cache is None:
            return None

        return dataset_for_format(
            self.format, Pos, cache, eos_id=self.the_tokenizer.eos_token_id, ignore_index=self.ignore_token_id
        )

    def validation_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        validation_set = self.validation_set(Pos, monitors)
        if validation_set is not None:
            return {"": validation_set}
        else:
            return {}

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return {"": self}

    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, TreeCache[dict]]:
        out = {}
        cache = self.build_or_load_cache(split, monitors)
        if cache is not None:
            out[""] = cache
        return out

    def build_or_load_cache(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Optional[TreeCache[dict]]:
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

        return build_lm_dataset_cache(cache_dir, source, format, tokenizer, options, enforce_eos, monitors)


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
                ignore_index=self.ignore_token_id,
            )
            for name, cache in caches.items()
        }

        return token_datasets

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        mix_key, shuffle_key = jax.random.split(key)

        weights = self.train_weights
        if isinstance(weights, Sequence):
            weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

        initial_batch_size = batch_schedule.batch_size_at_step(0)

        causal_datasets = self.train_sets(
            Pos, monitors, key=shuffle_key, epochs=epochs, initial_batch_size=initial_batch_size
        )

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
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        initial_batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("train", monitors=monitors)
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

    def validation_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("validation", monitors=monitors)
        validation_datasets = self.build_token_datasets(doc_caches, Pos)

        if self.num_validation_sequences is not None:
            train_doc_caches = self.build_caches("train", monitors=monitors)
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

    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Dict[str, TreeCache[dict]]:
        caches = {}
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
                    monitors,
                )

        # In practice, it works best if we block on validation caches
        if split == "validation":
            for cache in caches.values():
                cache.await_finished()
        else:
            logger.info(f"Not waiting for {split} caches to finish building")

        return caches

    @property
    def sources(self) -> Mapping[str, LmDatasetSourceConfigBase]:
        return self.configs


def datasource_from_chat_jsonl_single_turn(
    urls: Sequence[str], messages_field: str = "messages", input_role: str = "user", output_role: str = "assistant"
) -> "ShardedDataSource[dict]":
    """Creates a ShardedDataSource from JSONL files containing chat messages.

    Args:
        urls: Sequence of URLs or glob patterns pointing to JSONL files
        messages_field: Field name containing the messages in each JSON object
        input_role: Role identifier for input messages
        output_role: Role identifier for output messages

    Returns:
        ShardedDataSource configured for chat data
    """
    return SingleTurnChatJsonlDataSource(urls, messages_field, input_role, output_role)


class SingleTurnChatJsonlDataSource(JsonlDataSource):
    """DataSource that reads JSONL files containing OpenAI chat format messages."""

    def __init__(self, urls: Sequence[str], messages_field: str, input_role: str, output_role: str):
        super().__init__(urls)
        self.messages_field = messages_field
        self.input_role = input_role
        self.output_role = output_role

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            for line in f:
                if i >= row:
                    data = json.loads(line)
                    messages = data[self.messages_field]

                    # Extract input/output from messages
                    input_msg = next(m["content"] for m in messages if m["role"] == self.input_role)
                    output_msg = next(m["content"] for m in messages if m["role"] == self.output_role)

            yield {CANONICAL_INPUT_FIELD: input_msg, CANONICAL_OUTPUT_FIELD: output_msg}


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
        mask_user_turns: bool = True,
    ):
        if chat_template is None and tokenizer.chat_template is None:
            raise ValueError("No chat template provided and tokenizer has no default chat template")
        self.tokenizer = tokenizer
        self.chat_template = chat_template or tokenizer.chat_template
        self.messages_field = messages_field

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
        # Extract messages from the specified field
        messages = [example[self.messages_field] for example in batch]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            chat_template=self.chat_template,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
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
        # TODO: do better with blocking
        cache.await_finished()
        self.packed: GreedyPrepackedDataset[ProcessedChatDict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )
        self.Pos = Pos

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
        self.mask_user_turns = mask_user_turns

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _create_lm_example(e: tuple[ProcessedChatDict, ProcessedChatDict]) -> LmExample:
            example, seg_ids = e
            tokens = hax.named(example["input_ids"], self.Pos)

            if mask_user_turns:
                # mask is 1 on the position of the assistant tokens
                mask = example["assistant_masks"]
                # loss_mask by convention is 1 on the positions where we compute loss, i.e. shifted back 1
                mask = jnp.roll(mask, -1, axis=-1)
                loss_mask = hax.named(mask, self.Pos)
            else:
                loss_mask = None

            seg_ids = hax.named(seg_ids["input_ids"], self.Pos)

            return LmExample.causal(tokens=tokens, loss_mask=loss_mask, segment_ids=seg_ids)

        super().__init__(self.packed, _create_lm_example)


ProcessedSupervisedDict = TypedDict(
    "ProcessedSupervisedDict",
    {
        "input_ids": np.ndarray,
        "sources_len": np.ndarray,
    },
)


class SupervisedProcessor(BatchProcessor[dict, ProcessedSupervisedDict]):
    """
    A batch processor that converts supervised data into the expected inputs of a model.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        input_field: str,
        output_field: str,
        separate_with: str | int | None = None,
    ):
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.output_field = output_field

        if isinstance(separate_with, int):
            separate_with = tokenizer.convert_ids_to_tokens(separate_with)
        self.separate_with = separate_with

    def __call__(self, batch: Sequence[dict]) -> ProcessedSupervisedDict:
        sources = [example[self.input_field] for example in batch]
        targets = [example[self.output_field] for example in batch]

        # Add sep if needed
        if self.separate_with is not None:
            sources = [s + self.separate_with for s in sources]

        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = self.tokenizer(sources, padding=False, truncation=True)
        examples_tokenized = self.tokenizer(examples, padding=False, truncation=True)
        source_lens = [len(s) for s in sources_tokenized["input_ids"]]

        return {
            "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],  # type: ignore
            "sources_len": np.array(source_lens, dtype=np.int32),
        }

    @property
    def output_exemplar(self):
        return {
            "input_ids": np.zeros((0,), dtype=np.int32),
            "sources_len": np.zeros((0,), dtype=np.int32),
        }

    @property
    def num_cpus(self) -> int:
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "input_field": self.input_field,
            "output_field": self.output_field,
            "separate_with": self.separate_with,
        }


class SupervisedDataset(MappedAsyncDataset[tuple[ProcessedSupervisedDict, ProcessedSupervisedDict], LmExample]):
    """
    A dataset that yields packed supervised examples from a cache of processed supervised data.
    """

    def __init__(
        self,
        cache: TreeCache[ProcessedSupervisedDict],
        Pos: Axis,
        max_segments_per_example: int | None = 64,
        mask_inputs: bool = True,
        slice_strategy: Literal["left", "right", "raise"] = "right",
    ):
        self.mask_inputs = mask_inputs
        # TODO: do better with blocking
        cache.await_finished()
        self.packed = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
        )

        def _create_lm_example(ex_pair: tuple[ProcessedSupervisedDict, ProcessedSupervisedDict]) -> LmExample:
            ex, seg_ids = ex_pair
            tokens = hax.named(ex["input_ids"], Pos)
            segment_ids = seg_ids["input_ids"]

            if self.mask_inputs:
                sequence_mask = self._make_sequence_mask(segment_ids, ex["sources_len"])
                loss_mask = hax.named(sequence_mask, Pos)
            else:
                # Use default loss mask
                loss_mask = None

            return LmExample.causal(tokens, loss_mask=loss_mask, segment_ids=hax.named(segment_ids, Pos))

        super().__init__(self.packed, _create_lm_example)

    @staticmethod
    def _make_sequence_mask(segment_ids: np.ndarray, segment_source_len: np.ndarray) -> np.ndarray:
        """
        Constructs a mask hiding input tokens for a packed supervised example.

        Args:
          segment_ids: shape [N], arbitrary integer IDs for each token.
          segment_source_len: array mapping each segment_id to the number of input tokens for that segment.

        Returns:
          mask of shape [N] (dtype=int8). 1 => token is output, 0 => token is input.
        """
        N = len(segment_ids)
        positions = np.empty(N, dtype=int)  # positions[i] = position of token i within its segment

        # We'll store a running counter for each segment in a dict
        segment_counters: dict[int, int] = {}

        # unique segment ids
        unique_seg_ids, seg_idxes = np.unique(segment_ids, return_index=True)
        unique_seg_ids = unique_seg_ids[np.argsort(seg_idxes)]

        source_len_dict = {}
        for seg_id, source_len in zip(unique_seg_ids, segment_source_len):
            seg_id = int(seg_id)
            source_len_dict[seg_id] = source_len
            if seg_id == -1:
                break

        # Single pass: assign positions
        for i, seg_id in enumerate(segment_ids):
            seg_id = int(seg_id)
            pos = segment_counters.get(seg_id, 0)  # how many tokens we've seen so far for seg_id
            positions[i] = pos
            segment_counters[seg_id] = pos + 1

        # Build mask: tokens are "output" (mask=1) if positions[i] >= segment_source_len[ seg_id ]
        mask = np.zeros(N, dtype=np.int32)
        for i, seg_id in enumerate(segment_ids):
            input_len = source_len_dict[int(seg_id)]
            if positions[i] >= input_len:
                mask[i] = 1

        # also don't predict the padding, which is -1
        mask[segment_ids == -1] = 0

        return mask


class SingleTurnChatProcessor(BatchProcessor[dict, ProcessedSupervisedDict]):
    """
    A batch processor that converts chat data into single turn supervised examples.
    This omits any turn after the first pair.
    """

    def __init__(self, tokenizer: HfTokenizer, messages_field: str = "messages"):
        self.tokenizer = tokenizer
        self.messages_field = messages_field
        self.supervised_processor = SupervisedProcessor(tokenizer, CANONICAL_INPUT_FIELD, CANONICAL_OUTPUT_FIELD)

    def __call__(self, batch: Sequence[dict]) -> ProcessedSupervisedDict:
        batch = [example[self.messages_field] for example in batch]

        # Extract input/output from messages
        input_msg = [next(m["content"] for m in messages if m["role"] == "user") for messages in batch]
        output_msg = [next(m["content"] for m in messages if m["role"] == "assistant") for messages in batch]
        batch = [{CANONICAL_INPUT_FIELD: i, CANONICAL_OUTPUT_FIELD: o} for i, o in zip(input_msg, output_msg)]

        return self.supervised_processor(batch)

    @property
    def output_exemplar(self):
        return self.supervised_processor.output_exemplar

    @property
    def num_cpus(self) -> int:
        return self.supervised_processor.num_cpus

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            **self.supervised_processor.metadata,
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "messages_field": self.messages_field,
        }


def cached_token_count(cache_path: str, field: str = "input_ids") -> int:
    """Return the total number of tokens stored in a finished :class:`TreeCache`.

    This simply loads the cache and reads the ``data_size`` for ``field``.  It
    assumes the cache contains the given field of token ids and that the cache
    is already finished.

    Args:
        cache_path: Path to the on-disk cache directory.
        field: Name of the field containing token ids. Defaults to ``"input_ids"``.

    Returns:
        The total number of tokens in the cache.
    """

    cache = TreeCache.load(cache_path, {field: np.zeros((0,), dtype=np.int32)})
    return cache.store.tree[field].data_size


def count_corpus_sizes(config: LMMixtureDatasetConfig | SingleDatasetLMConfig, prefix: str = "data/stats/") -> dict:
    """
    Counts the number of tokens in each dataset in the config.

    Args:
        config: the config to count the sizes of
        prefix: prefix to use for all metric keys. Defaults to "data/stats/"

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

    seq_len = 4096
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
        cache.await_finished()
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
        cache.await_finished()
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
