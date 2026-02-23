# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Re-export public API for levanter.data.text
from .cache import build_lm_dataset_cache, cached_token_count, load_lm_dataset_cache
from .datasets import (
    BlockShuffleConfig,
    CausalLmDataset,
    ChatDataset,
    DatasetComponent,
    DatasetComponentBase,
    DirectDatasetComponent,
    HfDatasetSourceConfig,
    LMMixtureDatasetConfig,
    LmDataConfig,
    LmDatasetSourceConfigBase,
    NamedLmDataset,
    TokenSeqDataset,
    UrlDatasetSourceConfig,
    dataset_for_component,
    count_corpus_sizes,
)
from .examples import (
    GrugAttentionMask,
    GrugLmExample,
    grug_attention_mask_from_named,
    grug_lm_example_from_named,
    named_attention_mask_from_grug,
    named_lm_example_from_grug,
)
from .formats import (
    ChatLmDatasetFormat,
    ChatProcessor,
    LmDatasetFormatBase,
    PrebuiltLmDatasetFormat,
    ProcessedChatDict,
    TextLmDatasetFormat,
    preprocessor_for_format,
)
from .preference import (
    DpoExample,
    PreferenceChatLmDatasetFormat,
    PreferenceChatProcessor,
    PreferenceLmDataConfig,
    PreferencePairDataset,
    ProcessedPreferenceChatDict,
    dataset_for_preference_format,
    preprocessor_for_preference_format,
)
from ._batch_tokenizer import BatchTokenizer

__all__ = [
    "BatchTokenizer",
    "ChatProcessor",
    "LmDatasetFormatBase",
    "TextLmDatasetFormat",
    "ChatLmDatasetFormat",
    "PrebuiltLmDatasetFormat",
    "ProcessedChatDict",
    "preprocessor_for_format",
    "TokenSeqDataset",
    "BlockShuffleConfig",
    "NamedLmDataset",
    "CausalLmDataset",
    "ChatDataset",
    "GrugAttentionMask",
    "GrugLmExample",
    "grug_attention_mask_from_named",
    "grug_lm_example_from_named",
    "named_attention_mask_from_grug",
    "named_lm_example_from_grug",
    "DatasetComponent",
    "DatasetComponentBase",
    "DirectDatasetComponent",
    "build_lm_dataset_cache",
    "load_lm_dataset_cache",
    "cached_token_count",
    "dataset_for_component",
    "LmDatasetSourceConfigBase",
    "UrlDatasetSourceConfig",
    "HfDatasetSourceConfig",
    "LmDataConfig",
    "LMMixtureDatasetConfig",
    "count_corpus_sizes",
    # Preference/DPO exports
    "DpoExample",
    "PreferenceChatLmDatasetFormat",
    "PreferenceChatProcessor",
    "PreferenceLmDataConfig",
    "PreferencePairDataset",
    "ProcessedPreferenceChatDict",
    "dataset_for_preference_format",
    "preprocessor_for_preference_format",
]
