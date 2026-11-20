# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from finestore.sft import DatasetVariantInput
from marin.datakit.sft_sources import all_sft_sources

from experiments.datakit.register_sft import registered_download


@pytest.mark.parametrize("name", ["agenttrove", "gpt-oss-rollouts", "numinamath-1.5"])
def test_registration_resolves_download_behind_processed_chat(name):
    source = all_sft_sources()[name]
    pending = list(source.chat_normalized.deps)
    while pending:
        step = pending.pop()
        if "hf_dataset_id" in step.hash_attrs:
            identity = DatasetVariantInput(
                source_id=name,
                variant_id="sft",
                repository=step.hash_attrs["hf_dataset_id"],
                revision=step.hash_attrs["revision"],
                data_sha256="a" * 64,
            )
            assert registered_download(source, identity).output_path == step.output_path
            assert source.chat_steps[0].output_path != step.output_path
            break
        pending.extend(step.deps)
    else:
        pytest.fail("Registered source has no pinned HF download")
