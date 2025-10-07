# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import fsspec

from haliax.partitioning import ResourceAxis

import levanter.config
from levanter.data.text import HfSingleDatasetLMConfig, LMMixtureDatasetConfig
from levanter.trainer import TrainerConfig


def test_main_wrapper_loads_from_fsspec():
    with fsspec.open("memory://test.yaml", "w") as f:
        f.write(
            """
        project: test
        """
        )

    args = ["--config_path", "memory://test.yaml", "--x", "2"]

    @dataclasses.dataclass
    class Config:
        project: str
        x: int = 1

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.project == "test"
        assert config.x == 2

    main()


def test_new_style_axis_mapping():
    config = TrainerConfig(
        tensor_parallel_axes=["a1", "a2"],
    )

    assert config.tensor_parallel_axes == ["a1", "a2"]
    assert config.compute_axis_mapping == {
        "batch": (ResourceAxis.REPLICA, ResourceAxis.DATA),
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
    }
    assert config.parameter_axis_mapping == {
        "embed": ResourceAxis.DATA,
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
        "batch": (ResourceAxis.REPLICA, ResourceAxis.DATA),
    }


def test_lm_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: HfSingleDatasetLMConfig = dataclasses.field(default_factory=HfSingleDatasetLMConfig)

    yaml_config = """
    data:
        id: dlwh/wikitext_103_detokenized
        cache_dir: "gs://levanter-data/tokenized/wikitext"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data.id == "dlwh/wikitext_103_detokenized"
        assert config.data.cache_dir == "gs://levanter-data/tokenized/wikitext"

    main()


def test_lm_mixture_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LMMixtureDatasetConfig = dataclasses.field(default_factory=LMMixtureDatasetConfig)

    yaml_config = """
    data:
        configs:
            owt:
                train_urls:
                    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
                validation_urls:
                    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
            wikitext:
                id: dlwh/wikitext_103_detokenized
        train_weights:
            owt: 0.6
            wikitext: 0.4
        tokenizer: gpt2
        cache_dir: "gs://levanter-data/tokenized/mixture"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data is not None
        # TODO: assert more things

    main()


def _write_yaml_to_memory(yaml: str, path: str = "memory://test.yaml"):
    with fsspec.open(path, "w") as f:
        f.write(yaml)
    return path
