# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.fineweb2_multilingual import (
    FINEWEB2_PARQUET_REVISION,
    fineweb2_multilingual_parquet_pattern,
    fineweb2_multilingual_tags,
    fineweb2_multilingual_tokenized,
)
from marin.processing.tokenize.tokenize import TokenizeConfig


def test_fineweb2_multilingual_parquet_pattern_matches_all_split_shards():
    path = fineweb2_multilingual_parquet_pattern("rus_Cyrl", "test")

    assert path == ("hf://datasets/HuggingFaceFW/fineweb-2@" f"{FINEWEB2_PARQUET_REVISION}/rus_Cyrl/test/*.parquet")


def test_fineweb2_multilingual_tags_include_language_script_and_subsets():
    assert fineweb2_multilingual_tags("hin_Deva") == [
        "fineweb2_multilingual",
        "fineweb2_multilingual/script/Deva",
        "fineweb2_multilingual/language/hin",
        "fineweb2_multilingual/top_50_by_rows",
        "fineweb2_multilingual/indic",
    ]


def test_fineweb2_multilingual_tags_require_language_script_config():
    with pytest.raises(AssertionError, match="lang_Script"):
        fineweb2_multilingual_tags("hin")


def test_fineweb2_multilingual_eval_tokenized_uses_validation_paths_and_tags():
    steps = fineweb2_multilingual_tokenized(
        split="test",
        configs=("hin_Deva",),
        cache_split="validation",
        name_prefix="fineweb2_test",
        tokenizer="gpt2",
    )

    step = steps["fineweb2_test/hin_Deva"]
    assert isinstance(step.config, TokenizeConfig)
    assert step.config.train_paths == []
    assert step.config.validation_paths == [fineweb2_multilingual_parquet_pattern("hin_Deva", "test")]
    assert step.config.tags == fineweb2_multilingual_tags("hin_Deva")
