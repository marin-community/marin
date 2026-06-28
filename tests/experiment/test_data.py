# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.execution.lazy import materialized_config
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig

_PREFIX = "gs://prefix"
_TOKENIZER = "gpt2"
_V = "2026.06.28"


def test_tokenized_hf_id_source_builds_hf_config():
    cfg = materialized_config(tokenized("c", source="org/corpus", tokenizer=_TOKENIZER, version=_V), _PREFIX)
    assert isinstance(cfg, HfTokenizeConfig)
    assert cfg.id == "org/corpus"


def test_tokenized_path_source_builds_filesystem_config():
    path = "hf://buckets/demo-user/demo-bucket/data/train.jsonl"
    cfg = materialized_config(tokenized("c", source=path, tokenizer=_TOKENIZER, version=_V), _PREFIX)
    assert isinstance(cfg, TokenizeConfig)
    assert cfg.train_paths == [path]
    assert cfg.validation_paths == []


def test_tokenized_validation_routes_to_validation_split():
    path = "gs://bucket/val.jsonl"
    cfg = materialized_config(tokenized("c", source=path, tokenizer=_TOKENIZER, validation=True, version=_V), _PREFIX)
    assert cfg.train_paths == []
    assert cfg.validation_paths == [path]


def test_tokenized_requires_exactly_one_raw_input():
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", tokenizer=_TOKENIZER, version=_V)
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", source="org/corpus", paths=["gs://b/x"], tokenizer=_TOKENIZER, version=_V)
