# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the Heretic-abliterated GPT-OSS 20B BF16 vLLM-serving subset."""

from experiments.models import gpt_oss_20b_heretic_vllm
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        steps=[gpt_oss_20b_heretic_vllm],
        description="Download p-e-w/gpt-oss-20b-heretic vLLM serving subset",
    )
