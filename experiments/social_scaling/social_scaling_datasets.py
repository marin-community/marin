#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download and tokenize the social scaling datasets as plain text.

This module treats the ``text`` column of the social scaling SFT datasets as a
standard pretraining corpus and exposes tokenization steps that can be used in
mixtures or standalone runs.
"""

import os.path

from levanter.data.text import TextLmDatasetFormat

from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

SCALING_LONGITUDINAL_REVISION = "06219a1"
SCALING_OSF_BINARY_REVISION = "010f234"
SCALING_PSYCH101_REVISION = "fa327f7"
SCALING_WVS30_REVISION = "fc90a92"

PSYCH101_CONFIGS: list[str] = [
    "badham2017deficits",
    "bahrami2020four",
    "feng2021dynamics",
    "flesch2018comparing",
    "frey2017cct",
    "gershman2018deconstructing",
    "gershman2020reward",
    "hebart2023things",
    "hilbig2014generalized",
    "krueger2022identifying",
    "ludwig2023human",
    "peterson2021using",
    "plonsky2018when",
    "ruggeri2022globalizability",
    "sadeghiyeh2020temporal",
    "schulz2020finding",
    "somerville2017charting",
    "speekenbrink2008learning",
    "steingroever2015data",
    "tomov2021multitask",
    "waltz2020differential",
    "wilson2014humans",
    "wu2018generalisation",
    "wulff2018description",
    "wulff2018sampling",
    "xiong2023neural",
]

TEXT_FORMAT = TextLmDatasetFormat(text_key="text")


downloads: dict[str, ExecutorStep[DownloadConfig]] = {
    "scaling_longitudinal": ExecutorStep(
        name="raw/social_scaling/scaling_longitudinal",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="cjziems/scaling_longitudinal",
            revision=SCALING_LONGITUDINAL_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    ),
    "scaling_osf_binary": ExecutorStep(
        name="raw/social_scaling/scaling_osf_binary",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="cjziems/scaling_osf_binary",
            revision=SCALING_OSF_BINARY_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    ),
    "scaling_psych101": ExecutorStep(
        name="raw/social_scaling/scaling_psych101",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="cjziems/scaling_psych101",
            revision=SCALING_PSYCH101_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    ),
    "scaling_wvs30": ExecutorStep(
        name="raw/social_scaling/scaling_wvs30",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id="cjziems/scaling_wvs30",
            revision=SCALING_WVS30_REVISION,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    ),
}


def _get_longitudinal_train_paths() -> list:
    """Return train split paths for the longitudinal dataset."""
    return [output_path_of(downloads["scaling_longitudinal"], "data/train-*")]


def _get_osf_train_paths() -> list:
    """Return train split paths for the OSF dataset."""
    return [output_path_of(downloads["scaling_osf_binary"], "data/train-*")]


def _get_psych101_train_paths() -> list:
    """Return train split paths for all Psych-101 subsets."""
    return [output_path_of(downloads["scaling_psych101"], f"{cfg}/train-*") for cfg in PSYCH101_CONFIGS]


def _get_wvs30_train_paths() -> list:
    """Return train split paths for the WVS Nigeria and Indonesia subsets."""
    return [
        output_path_of(downloads["scaling_wvs30"], "nigeria/train-*"),
        output_path_of(downloads["scaling_wvs30"], "indonesia/train-*"),
    ]


def tokenize_social_scaling(
    *,
    base_path: str = "tokenized/social_scaling",
    tokenizer: str = llama3_tokenizer,
) -> dict[str, TokenizerStep]:
    """Create tokenization steps for the social scaling datasets.

    Args:
        base_path: Base directory (relative to the output prefix) for tokenized data.
        tokenizer: Pretrained tokenizer name or path to use.

    Returns:
        Mapping from dataset key to tokenization ExecutorStep.
    """

    def _make_step(suffix: str, train_paths: list) -> ExecutorStep[TokenizeConfig]:
        output = os.path.join(base_path, suffix)
        return ExecutorStep(
            name=output,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=train_paths,
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
                format=TEXT_FORMAT,
            ),
        )

    steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    steps["social_scaling/longitudinal"] = _make_step("longitudinal", _get_longitudinal_train_paths())
    steps["social_scaling/osf"] = _make_step("osf", _get_osf_train_paths())
    steps["social_scaling/psych101_all"] = _make_step("psych101_all", _get_psych101_train_paths())
    steps["social_scaling/wvs30_combined"] = _make_step("wvs30_combined", _get_wvs30_train_paths())

    return steps


if __name__ == "__main__":
    all_steps = list(tokenize_social_scaling().values())
    executor_main(steps=all_steps, description="Tokenize social scaling text datasets")
