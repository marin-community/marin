# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.sft_sources import all_sft_sources

from experiments.datakit.register_sft import registered_download


@pytest.mark.parametrize("name", ["agenttrove", "gpt-oss-rollouts", "numinamath-1.5"])
def test_registration_resolves_download_behind_processed_chat(name):
    source = all_sft_sources()[name]
    download = source.chat_steps[0].deps[0]
    selected = registered_download(source, download.hash_attrs["hf_dataset_id"], download.hash_attrs["revision"])
    assert selected is download
