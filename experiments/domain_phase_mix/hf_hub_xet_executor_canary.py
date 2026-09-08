# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify spawn-time Hugging Face environment propagation through ExecutorStep."""

import os

from fray.types import ResourceConfig
from huggingface_hub import constants
from marin.execution.context import executor_context
from marin.execution.executor import executor_main
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep
from transformers import AutoTokenizer

HF_HUB_DISABLE_XET_ENV_VAR = "HF_HUB_DISABLE_XET"
QWEN_REFERENCE_MODEL = "Qwen/Qwen3-0.6B"


def verify_hf_hub_http_tokenizer_path(output_path: str) -> dict[str, str | int]:
    """Load the Qwen tokenizer after verifying Xet was disabled before import."""
    assert os.environ.get(HF_HUB_DISABLE_XET_ENV_VAR) == "1"
    assert constants.HF_HUB_DISABLE_XET
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REFERENCE_MODEL, trust_remote_code=True)
    return {
        "output_path": output_path,
        "reference_model": QWEN_REFERENCE_MODEL,
        "vocabulary_size": len(tokenizer),
    }


def build_step() -> ExecutorStep[dict[str, str]]:
    resources = ResourceConfig.with_cpu(
        cpu=1,
        ram="4g",
        disk="10g",
        regions=["us-east5"],
        zone="us-east5-a",
    )
    return ExecutorStep(
        name="domain_phase_mix/canary/hf_hub_http_tokenizer_path",
        fn=remote(
            verify_hf_hub_http_tokenizer_path,
            env_vars={HF_HUB_DISABLE_XET_ENV_VAR: "1"},
        ),
        resources=resources,
        config={"reference_model": QWEN_REFERENCE_MODEL},
    )


def main() -> None:
    with executor_context():
        step = build_step()
    executor_main(steps=[step])


if __name__ == "__main__":
    main()
