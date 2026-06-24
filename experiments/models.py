# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Hugging Face model downloads.

Each model is exposed as a zero-argument factory that builds its download
``ExecutorStep`` on demand, so no step is constructed at import time. Call the
factory inside an executor build phase:

```
from marin.execution import executor_context, executor_main
from experiments.models import llama_3_1_8b

if __name__ == "__main__":
    with executor_context():
        executor_main(steps=[llama_3_1_8b()])
```

To add a model, declare a new factory mirroring the ones below.
"""

from dataclasses import dataclass

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.utils import get_directory_friendly_name


@dataclass(frozen=True)
class ModelConfig:
    hf_repo_id: str
    hf_revision: str


MODEL_OUTPUT_SUBDIR = "models"


def download_model_step(model_config: ModelConfig) -> ExecutorStep:
    model_name = get_directory_friendly_name(model_config.hf_repo_id)
    model_revision = get_directory_friendly_name(model_config.hf_revision)
    download_step = ExecutorStep(
        name=f"{MODEL_OUTPUT_SUBDIR}/{model_name}--{model_revision}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=model_config.hf_repo_id,
            revision=versioned(model_config.hf_revision),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_repo_type_prefix="",
        ),
        # must override because it because if we don't then it will end in a hash
        # if it ends in a hash, then we cannot determine the local path
        override_output_path=f"{MODEL_OUTPUT_SUBDIR}/{model_name}--{model_revision}",
    )

    return download_step


def smollm2_1_7b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            hf_revision="450ff1f",
        )
    )


# Note(Will): I don't think we actually support Qwen models in Levanter?
def qwen2_5_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-7B",
            hf_revision="d149729398750b98c0af14eb82c78cfe92750796",
        )
    )


def qwen2_5_7b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
            hf_revision="a09a354",
        )
    )


def qwen2_5_32b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-32B",
            hf_revision="1818d35",
        )
    )


def qwen2_5_72b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen2.5-72B-Instruct",
            hf_revision="495f393",
        )
    )


def llama_3_3_70b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
            hf_revision="6f6073b",
        )
    )


def llama_3_1_8b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
            hf_revision="0e9e39f",
        )
    )


def llama_3_1_8b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.1-8B",
            hf_revision="d04e592",
        )
    )


def llama_3_1_70b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.1-70B",
            hf_revision="349b2ddb53ce8f2849a6c168a81980ab25258dac",
        )
    )


def tulu_3_1_8b_sft() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
            hf_revision="f2a0b46",
        )
    )


def tulu_3_1_8b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/Llama-3.1-Tulu-3.1-8B",
            hf_revision="46239c2",
        )
    )


def olmo_2_sft_8b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/OLMo-2-1124-7B-SFT",
            hf_revision="1de02c0",
        )
    )


def olmo_2_base_8b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/OLMo-2-1124-7B",
            hf_revision="7df9a82",
        )
    )


def olmo_2_base_32b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/OLMo-2-0325-32B",
            hf_revision="stage2-ingredient2-step9000-tokens76B",
        )
    )


def olmo_3_1025_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/Olmo-3-1025-7B",
            hf_revision="18b40a1e895f829c68a132befa20109c41488e62",
        )
    )


def amber_base_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="LLM360/Amber",
            hf_revision="83c188f",
        )
    )


def map_neo_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="m-a-p/neo_7b",
            hf_revision="81bad32",
        )
    )


def marin_8b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="marin-community/marin-8b-base",
            hf_revision="0f1f658",
        )
    )


def marin_8b_instruct() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="marin-community/marin-8b-instruct",
            hf_revision="0378f9c",
        )
    )


def llama_3_2_1b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.2-1B",
            hf_revision="4e20de3",
        )
    )


def llama_3_2_3b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-3.2-3B",
            hf_revision="13afe5124825b4f3751f836b40dafda64c1ed062",
        )
    )


def llama_2_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-2-7b-hf",
            hf_revision="01c7f73d771dfac7d292323805ebc428287df4f9",
        )
    )


def llama_2_13b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="meta-llama/Llama-2-13b-hf",
            hf_revision="5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1",
        )
    )


def olmo_2_base_13b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="allenai/OLMo-2-1124-13B",
            hf_revision="3fefddc1bf18a30e1d9b91000271630718f2aa8b",
        )
    )


def qwen3_0_6b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-0.6B",
            hf_revision="c1899de",
        )
    )


def qwen3_1_7b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-1.7B",
            hf_revision="70d244c",
        )
    )


def qwen3_4b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-4B",
            hf_revision="1cfa9a7",
        )
    )


def qwen3_8b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-8B",
            hf_revision="b968826",
        )
    )


def qwen3_0_6b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-0.6B-Base",
            hf_revision="da87bfb",
        )
    )


def qwen3_1_7b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-1.7B-Base",
            hf_revision="ea980cb",
        )
    )


def qwen3_4b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-4B-Base",
            hf_revision="906bfd4",
        )
    )


def qwen3_8b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-8B-Base",
            hf_revision="49e3418",
        )
    )


def qwen3_14b_base() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-14B-Base",
            hf_revision="0b0bd3732e2c374d483664439ea334928b65f304",
        )
    )


def qwen3_32b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="Qwen/Qwen3-32B",
            hf_revision="9216db5781bf21249d130ec9da846c4624c16137",
        )
    )


def apertus_8b() -> ExecutorStep:
    return download_model_step(
        ModelConfig(
            hf_repo_id="swiss-ai/Apertus-8B-2509",
            hf_revision="9325d4a",
        )
    )
