# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the abliterated GPT-OSS 120B BF16 vLLM-serving subset."""

from experiments.models import gpt_oss_120b_abliterated_vllm
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        steps=[gpt_oss_120b_abliterated_vllm],
        description="Download huihui-ai/Huihui-gpt-oss-120b-BF16-abliterated vLLM serving subset",
    )
