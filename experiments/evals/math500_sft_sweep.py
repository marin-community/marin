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

"""
SFT sweep (lr × schedule) then evaluate every model on MATH-500.

Sweeps over multiple SFT configs (learning rate × schedule) for a
selected subset of models and evaluates every resulting checkpoint on MATH-500.
"""

import argparse
import dataclasses
import logging
import os
import sys
import warnings
from dataclasses import dataclass

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)  # Root logger

from datasets import load_dataset, concatenate_datasets
from zephyr import Dataset, ZephyrContext, load_jsonl

from experiments.evals.math500_eval import PROMPT_FORMAT_REGISTRY

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
    ModelConfig,
)
from experiments.evals.exp_isoflop_hf_math500_sft import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_SFT_CONFIG,
    SFTMath500EvalConfig,
    run_math500_eval_after_sft,
)
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import lm_data_config

logger = logging.getLogger(__name__)

DEFAULT_TPU_TYPE = "v5p-8"


@dataclass(frozen=True)
class DownloadMathConfig:
    output_path: str
    split: str = "train"
    num_examples: int | None = None


def download_math(config: DownloadMathConfig):
    cfgs = ["algebra", "counting_and_probability", "geometry",                             
             "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    dataset = concatenate_datasets([
        load_dataset("HuggingFaceH4/MATH", cfg, split=config.split)
        for cfg in cfgs
    ])
    rows = [dict(row) for row in dataset]
    if config.num_examples is not None:
        rows = rows[:config.num_examples]

    pipeline = (
        Dataset.from_list(rows)
        .reshard(1)
        .write_jsonl(f"{config.output_path}/math_data-{{shard:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="download-math")
    ctx.execute(pipeline)


download_math_train_step = ExecutorStep(
    name="raw/math_train",
    fn=download_math,
    config=DownloadMathConfig(
        output_path=this_output_path(),
        split="train",
    ),
    resources=ResourceConfig.with_cpu(),
)

download_math_test_step = ExecutorStep(
    name="raw/math_test",
    fn=download_math,
    config=DownloadMathConfig(
        output_path=this_output_path(),
        split="test",
        num_examples=500,
    ),
    resources=ResourceConfig.with_cpu(),
)


@dataclass(frozen=True)
class ProcessMathConfig:
    raw_data_path: str
    output_path: str
    prompt_format: str = "standard_fewshot"


def process_math(config: ProcessMathConfig):
    prompt_formatter = PROMPT_FORMAT_REGISTRY[config.prompt_format]

    def to_chat(row):
        return {
            "messages": [
                {"role": "user", "content": prompt_formatter(row["problem"])},
                {"role": "assistant", "content": row["solution"]},
            ],
        }

    dataset = (
        Dataset.from_files(os.path.join(config.raw_data_path, "*.jsonl.gz"))
        .flat_map(load_jsonl)
        .map(to_chat)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="process-math")
    ctx.execute(dataset)


def build_steps(
    sft_train_data: ExecutorStep,
    sft_test_data: ExecutorStep,
    sft_configs: dict[str, SimpleSFTConfig],
    base_models: list[ModelConfig],
    prompt_format: str = "standard_fewshot",
    prefix: str = "math_sft_sweep",
    tpu_type: str = DEFAULT_TPU_TYPE,
):
    steps = []
    for model_config in base_models:
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        name = f"{directory_friendly_name}"
        tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
        hf_model_config = HFCheckpointConverter.from_hf(model_config.model_name).config_from_hf_checkpoint(model_config.model_name)

        tokenized_train = default_tokenize(
            name=f"{prefix}/math_train_sft_data/{name}",
            dataset=sft_train_data,
            tokenizer=tokenizer,
            format=ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE),
        )

        tokenized_test = default_tokenize(
            name=f"{prefix}/math_test_sft_data/{name}",
            dataset=sft_test_data,
            tokenizer=tokenizer,
            format=ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE),
            is_validation=True,
        )

        data_config = lm_data_config(
            training_set=tokenized_train,
            validation_sets={"test": tokenized_test},
        )

        for config_name, sft_config in sft_configs.items():
            per_model_sft_config = dataclasses.replace(
                sft_config,
                initialize_from_hf=output_path_of(model_instance),
            )
            sft_step = default_sft(
                name=f"{prefix}/math500_sft/{name}----{config_name}",
                tokenized=data_config,
                model_config=hf_model_config,
                sft_config=per_model_sft_config,
                tags=["sft", "math500", name, config_name],
            )

            steps.append(
                ExecutorStep(
                    name=f"analysis/{prefix}/math500_sft_rollouts/{name}/{config_name}/best",
                    fn=run_math500_eval_after_sft,
                    config=SFTMath500EvalConfig(
                        sft_output_path=output_path_of(sft_step),
                        output_path=this_output_path(),
                        prompt_format=versioned(prompt_format),
                        use_best_checkpoint=True,
                    ),
                    resources=ResourceConfig.with_tpu(tpu_type),
                    pip_dependency_groups=["vllm", "math"],
                )
            )
            steps.append(
                ExecutorStep(
                    name=f"analysis/{prefix}/math500_sft_rollouts/{name}/{config_name}/latest",
                    fn=run_math500_eval_after_sft,
                    config=SFTMath500EvalConfig(
                        sft_output_path=output_path_of(sft_step),
                        output_path=this_output_path(),
                        prompt_format=versioned(prompt_format),
                        use_best_checkpoint=False,
                    ),
                    resources=ResourceConfig.with_tpu(tpu_type),
                    pip_dependency_groups=["vllm", "math"],
                )
            )

    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    parser = argparse.ArgumentParser(description="SFT sweep on MATH train, eval on MATH-500.")
    parser.add_argument(
        "--tpu-type",
        type=str,
        default=DEFAULT_TPU_TYPE,
        help=f"TPU type for ResourceConfig.with_tpu (default {DEFAULT_TPU_TYPE}).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    prompt_format = "standard_fewshot"
    base_models = [models[0], models[6], models[12]]  # marin-8b, qwen-3-0.6b, qwen-3-4b-base

    sft_train_data = ExecutorStep(
        name="documents/math_train_sft",
        fn=process_math,
        config=ProcessMathConfig(
            raw_data_path=output_path_of(download_math_train_step),
            output_path=this_output_path(),
            prompt_format=prompt_format,
        ),
        resources=ResourceConfig.with_cpu(),
        pip_dependency_groups=["math"],
    )
    sft_test_data = ExecutorStep(
        name="documents/math_test_sft",
        fn=process_math,
        config=ProcessMathConfig(
            raw_data_path=output_path_of(download_math_test_step),
            output_path=this_output_path(),
            prompt_format=prompt_format,
        ),
        resources=ResourceConfig.with_cpu(),
        pip_dependency_groups=["math"],
    )
    sft_configs = {
        f"lr_{lr}_warmup_{warmup}": dataclasses.replace(
            DEFAULT_SFT_CONFIG, learning_rate=versioned(lr), warmup=versioned(warmup), num_train_steps=versioned(10000),
        )
        for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
        for warmup in [0.01, 0.05]
    }

    steps = build_steps(
        sft_train_data=sft_train_data,
        sft_test_data=sft_test_data,
        sft_configs=sft_configs,
        base_models=base_models,
        prompt_format=prompt_format,
        prefix="math_sft_sweep",
        tpu_type=args.tpu_type,
    )

    executor_main(
        steps=steps,
        description="SFT sweep (lr × warmup) on MATH train solutions for marin-8b and qwen-3-0.6b, eval on MATH-500.",
    )


if __name__ == "__main__":
    main()
