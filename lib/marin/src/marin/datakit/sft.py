# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare masked Harmony conversation stores for packed Snowball SFT."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import fsspec
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from levanter.data.text.datasets import (
    ChatDataset,
    ConcatDatasetComponent,
    DatasetComponent,
    DatasetComponentBase,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import ChatLmDatasetFormat, ChatProcessor
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import load_parquet

from marin.datakit.chat_render import chat_training_record
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import Artifact, write_artifact
from marin.processing.tokenize.store_builder import build_from_datasets, write_stats_json


@dataclass(frozen=True)
class SftInput:
    name: str
    path: str


class SftSourceCounts(BaseModel):
    conversations: int = 0
    tokens: int = 0
    assistant_tokens: int = 0
    overlength_conversations: int = 0
    overlength_tokens: int = 0


class SftTokenStore(Artifact):
    cache_path: str
    tokenizer: str
    max_length: int
    seed: int
    sources: dict[str, SftSourceCounts]
    packed_sequences: int


def _read_source_file(source: SftInput) -> Iterator[dict]:
    for row in load_parquet(source.path):
        yield {**chat_training_record(row), "source": source.name}


def _shuffle_key(record: dict, seed: int) -> str:
    identity = json.dumps([seed, record["source"], record["id"]], ensure_ascii=False)
    return hashlib.sha256(identity.encode()).hexdigest()


def _keep_rows(_key: str, rows: Iterator[dict]) -> Iterator[dict]:
    # Retain equal text across sources during the shuffle.
    yield from rows


def _tokenize_conversations(
    batches: Iterator[list[dict]],
    shard: ShardInfo,
    *,
    tokenizer: str,
    max_length: int,
    output_path: str,
) -> Iterator[dict]:
    processor = ChatProcessor(
        load_tokenizer(tokenizer),
        chat_template=MARIN_CHAT_TEMPLATE,
        system_prompt_field=None,
        mask_user_turns=True,
    )
    counts: dict[str, SftSourceCounts] = {}
    for batch in batches:
        for row, encoded in zip(batch, processor(batch), strict=True):
            count = counts.setdefault(row["source"], SftSourceCounts())
            length = len(encoded["input_ids"])
            if length > max_length:
                count.overlength_conversations += 1
                count.overlength_tokens += length
                continue
            count.conversations += 1
            count.tokens += length
            count.assistant_tokens += int(np.count_nonzero(encoded["assistant_masks"]))
            yield {"id": row["id"], **encoded}
    path = prefix_join(output_path, f"source-counts/{shard.shard_idx:05d}.json")
    StoragePath(path).write_text(json.dumps({name: count.model_dump() for name, count in counts.items()}))


def build_sft_store(
    sources: Sequence[SftInput],
    *,
    output_path: str,
    tokenizer: str,
    max_length: int,
    seed: int,
    num_shards: int,
    max_workers: int,
) -> SftTokenStore:
    """Shuffle normalized Harmony sources into a masked chat token store.

    Assistant analysis and final spans receive loss; user text is context only.
    Conversations longer than ``max_length`` are excluded and counted. Source
    shares follow retained data volume without resampling.
    """
    if max_length < 2 or num_shards < 1 or max_workers < 1:
        raise ValueError("max_length must be >= 2; num_shards and max_workers must be positive")
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("SFT sources must be nonempty and have distinct names")
    files = []
    for source in sources:
        paths = fsspec.open_files(prefix_join(source.path, "*.parquet"), mode="rb")
        if not paths:
            raise FileNotFoundError(f"No normalized Parquet shards for {source.name}: {source.path}")
        files.extend(SftInput(source.name, path.full_name) for path in paths)
    rows = (
        Dataset.from_list(files)
        .flat_map(_read_source_file)
        .group_by(key=partial(_shuffle_key, seed=seed), reducer=_keep_rows, num_output_shards=num_shards)
    )
    tokenized = rows.window(16).map_shard(
        partial(_tokenize_conversations, tokenizer=tokenizer, max_length=max_length, output_path=output_path)
    )
    cache_path = prefix_join(output_path, "train")
    ledger = build_from_datasets(
        ctx=ZephyrContext(
            name="sft-store", resources=ResourceConfig(cpu=2, ram="16g", disk="20g"), max_workers=max_workers
        ),
        dataset=tokenized,
        output_path=cache_path,
        batch_size=128,
        skip_existing=True,
    )
    counts = {source.name: SftSourceCounts() for source in sources}
    for shard in range(num_shards):
        path = prefix_join(output_path, f"source-counts/{shard:05d}.json")
        for name, values in json.loads(StoragePath(path).read_text()).items():
            previous = counts[name]
            counts[name] = SftSourceCounts(**{key: getattr(previous, key) + value for key, value in values.items()})
    if sum(count.conversations for count in counts.values()) != ledger.total_num_rows:
        raise ValueError("SFT source counts do not match the token store")
    if sum(count.tokens for count in counts.values()) != ledger.field_counts.get("input_ids", 0):
        raise ValueError("SFT token counts do not match the token store")
    write_stats_json(cache_path, ledger)
    packed_sequences = 0
    if ledger.total_num_rows:
        cache = TreeCache.load(
            cache_path,
            {"input_ids": np.zeros(0, dtype=np.int32), "assistant_masks": np.zeros(0, dtype=np.int32)},
        )
        packed_sequences = len(ChatDataset(cache, Axis("position", max_length)).as_sync_dataset())
    result = SftTokenStore(
        path=output_path,
        cache_path=cache_path,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        sources=counts,
        packed_sequences=packed_sequences,
    )
    write_artifact(result.result_payload(), output_path)
    return result


def sft_data_config(stores: Mapping[str, SftTokenStore], *, minimum_weight: float) -> LmDataConfig:
    """Weight sources by packed sequences and pool sources below ``minimum_weight``.

    ``minimum_weight`` is relative to the SFT share, before adding pretraining.
    Use at least ``1 / (sft_fraction * mixture_block_size)`` to retain each component.
    Empty stores contribute no training examples. Pooling concatenates already
    packed sources, preserving their conversation boundaries and sequence counts.
    """
    if not stores or not 0 < minimum_weight <= 1:
        raise ValueError("SFT stores must be nonempty and minimum_weight must be in (0, 1]")
    tokenizers = {store.tokenizer for store in stores.values()}
    lengths = {store.max_length for store in stores.values()}
    if len(tokenizers) != 1 or len(lengths) != 1:
        raise ValueError("SFT stores must share a tokenizer and packing context length")
    total = sum(store.packed_sequences for store in stores.values())
    if total == 0:
        raise ValueError("No conversations fit the SFT context length")
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    pooled: dict[str, DatasetComponent] = {}
    pooled_sequences = 0
    for name, store in sorted(stores.items()):
        if store.packed_sequences == 0:
            continue
        component = DatasetComponent(
            source=UrlDatasetSourceConfig(train_urls=[], validation_urls=[]),
            cache_dir=store.cache_path,
            flat_cache=True,
            format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE, mask_user_turns=True),
            pack=store.max_length,
        )
        if store.packed_sequences / total < minimum_weight:
            pooled[name] = component
            pooled_sequences += store.packed_sequences
        else:
            components[name] = component
            weights[name] = store.packed_sequences / total
    if pooled and pooled_sequences / total < minimum_weight:
        smallest = min(weights, key=weights.__getitem__)
        pooled[smallest] = components.pop(smallest)
        pooled_sequences += stores[smallest].packed_sequences
        del weights[smallest]
    mixture_components: dict[str, DatasetComponentBase] = {
        f"sft/source/{name}": component for name, component in components.items()
    }
    mixture_weights = {f"sft/source/{name}": weight for name, weight in weights.items()}
    if pooled:
        mixture_components["sft/pooled"] = ConcatDatasetComponent(children=pooled)
        mixture_weights["sft/pooled"] = pooled_sequences / total
    return LmDataConfig(
        tokenizer=next(iter(tokenizers)),
        cache_dir=None,
        components=mixture_components,
        train_weights=mixture_weights,
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
    )
