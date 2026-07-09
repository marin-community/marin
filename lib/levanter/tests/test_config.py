# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

import pytest
from rigging.filesystem import StoragePath

import levanter.config
from levanter.data.text import LmDataConfig, UrlDatasetSourceConfig


def test_main_wrapper_loads_from_fsspec():
    StoragePath("memory://test.yaml").write_text(
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


def test_remote_config_temp_file_handle_is_closed():
    fd_dir = "/proc/self/fd"
    if not os.path.isdir(fd_dir):
        pytest.skip("/proc/self/fd is required to inspect open file descriptors")

    StoragePath("memory://test_fd.yaml").write_text("project: test\n")

    config_path, remaining_args = levanter.config._maybe_get_config_path_and_cmdline_args(
        ["--config_path", "memory://test_fd.yaml"]
    )

    assert remaining_args == []
    assert os.path.exists(config_path)
    assert _open_file_descriptors_for_path(config_path) == []


def test_lm_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LmDataConfig = dataclasses.field(default_factory=LmDataConfig)

    yaml_config = """
    data:
        tokenizer: gpt2
        cache_dir: "gs://levanter-data/tokenized"
        components:
          wikitext:
            source:
              type: url
              validation_urls:
                - "gs://example"
              train_urls:
                - "gs://example"
            cache_dir: "gs://levanter-data/tokenized/wikitext"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert "wikitext" in config.data.components
        comp = config.data.components["wikitext"]
        assert isinstance(comp.source, UrlDatasetSourceConfig)
        assert comp.cache_dir == "gs://levanter-data/tokenized/wikitext"

    main()


def test_lm_mixture_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LmDataConfig = dataclasses.field(default_factory=LmDataConfig)

    yaml_config = """
    data:
        components:
            owt:
                source:
                    type: url
                    train_urls:
                        - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
                    validation_urls:
                        - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
            wikitext:
                source:
                    type: hf
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
    StoragePath(path).write_text(yaml)
    return path


def _open_file_descriptors_for_path(path: str):
    open_fds = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            fd_path = os.readlink(os.path.join("/proc/self/fd", fd))
        except FileNotFoundError:
            continue
        if fd_path == path:
            open_fds.append(fd)
    return open_fds
