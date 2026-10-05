# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema and mixture conversion for the materialized September 2026 SFT stores."""

from collections.abc import Mapping

from levanter.data.text.datasets import (
    ConcatDatasetComponent,
    DatasetComponent,
    DatasetComponentBase,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from marin.execution.artifact import Artifact
from pydantic import BaseModel


class SftSourceCounts(BaseModel):
    conversations: int = 0
    tokens: int = 0
    overlength_conversations: int = 0
    overlength_tokens: int = 0


class SftTokenStore(Artifact):
    """Recorded metadata for one already-materialized packed SFT store."""

    cache_path: str
    tokenizer: str
    max_length: int
    seed: int
    sources: dict[str, SftSourceCounts]
    packed_sequences: int


def sft_data_config(stores: Mapping[str, SftTokenStore], *, minimum_weight: float) -> LmDataConfig:
    """Weight prepared stores by packed sequences and pool sources below a minimum share."""
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
            format=TextLmDatasetFormat(),
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
