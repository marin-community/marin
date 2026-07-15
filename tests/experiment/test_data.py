# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.data.text.formats import SupervisedLmDatasetFormat
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, TokenizedCache

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


def test_tokenized_preserves_supervised_format_in_config_and_artifact(tmp_path):
    dataset_format = SupervisedLmDatasetFormat(
        input_key="prompt",
        target_key="answer",
        pack=True,
        slice_strategy="right",
    )
    cfg = materialized_config(
        tokenized(
            "c",
            source="gs://bucket/val.jsonl",
            tokenizer=_TOKENIZER,
            validation=True,
            dataset_format=dataset_format,
            version=_V,
        ),
        _PREFIX,
    )
    assert cfg.format == dataset_format

    write_record(
        ArtifactRecord(
            output_path=str(tmp_path),
            config={
                "tokenizer": _TOKENIZER,
                "format": {
                    "input_key": "prompt",
                    "target_key": "answer",
                    "pack": True,
                    "slice_strategy": "right",
                },
            },
        )
    )
    assert TokenizedCache.raw_load(str(tmp_path)).as_component().format == dataset_format


def test_tokenized_requires_exactly_one_raw_input():
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", tokenizer=_TOKENIZER, version=_V)
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", source="org/corpus", paths=["gs://b/x"], tokenizer=_TOKENIZER, version=_V)
