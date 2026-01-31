# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Re-export public API for levanter.data.text
from .cache import build_lm_dataset_cache, cached_token_count, load_lm_dataset_cache
from .datasets import (
    CausalLmDataset,
    ChatDataset,
    DatasetComponent,
    DatasetComponentBase,
    DirectDatasetComponent,
    HfDatasetSourceConfig,
    LMMixtureDatasetConfig,
    LmDataConfig,
    LmDatasetSourceConfigBase,
    TokenSeqDataset,
    UrlDatasetSourceConfig,
    dataset_for_component,
    count_corpus_sizes,
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
    "CausalLmDataset",
    "ChatDataset",
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
    # Preference/DPO classes
    "DpoExample",
    "PreferenceChatLmDatasetFormat",
    "PreferenceChatProcessor",
    "PreferencePairDataset",
    "ProcessedPreferenceChatDict",
    "dataset_for_preference_format",
    "preprocessor_for_preference_format",
]
