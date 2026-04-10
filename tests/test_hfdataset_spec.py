# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.defaults import default_download, default_tokenize
from marin.datakit.download.huggingface import DownloadConfig
from marin.processing.tokenize import HfDatasetSpec
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig


def test_default_tokenize_with_dataset_name():
    step = default_tokenize(
        name="dummy",
        dataset=HfDatasetSpec(id="cnn_dailymail", name="3.0.0"),
        tokenizer="gpt2",
    )
    assert isinstance(step.config, HfTokenizeConfig)
    assert step.config.id == "cnn_dailymail"
    assert step.config.name == "3.0.0"


def test_default_tokenize_with_hf_bucket_path_uses_filesystem_tokenize_config():
    bucket_path = "hf://buckets/demo-user/demo-bucket/data/train.jsonl"
    step = default_tokenize(
        name="dummy",
        dataset=bucket_path,
        tokenizer="gpt2",
    )

    assert isinstance(step.config, TokenizeConfig)
    assert step.config.train_paths == [bucket_path]


def test_default_download_with_hf_bucket_path_uses_bucket_prefix():
    bucket_path = "hf://buckets/demo-user/demo-bucket/data"
    step_input = default_download(
        name="dummy-bucket-download",
        hf_dataset_id=bucket_path,
    )

    assert step_input.step is not None
    assert isinstance(step_input.step.config, DownloadConfig)
    assert step_input.step.config.hf_repo_type_prefix == ""
    assert step_input.step.config.hf_dataset_id == "buckets/demo-user/demo-bucket/data"


def test_default_download_requires_revision_for_hub_dataset_ids():
    try:
        default_download(
            name="dummy-hf-download",
            hf_dataset_id="allenai/c4",
        )
    except ValueError as error:
        assert "revision is required" in str(error)
    else:
        raise AssertionError("expected ValueError when revision is missing for non-bucket dataset IDs")
