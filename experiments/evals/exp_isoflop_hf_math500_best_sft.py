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
SFT sweep then evaluate best model on MATH-500.

Builds on exp_isoflop_hf_math500_sft.py by sweeping over multiple SFT configs,
selecting the best checkpoint based on validation loss, and evaluating only the
best model on MATH-500.
"""

import dataclasses
import json
import logging
import os
import warnings
from dataclasses import dataclass

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)  # Root logger

import fsspec

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
)
from experiments.evals.exp_isoflop_hf_math500 import (
    build_hf_steps,
    get_isoflop_hf_model,
)
from experiments.evals.exp_isoflop_hf_math500_sft import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_SFT_CONFIG,
)
from experiments.evals.math500_eval import (
    Math500EvalConfig, 
    Math500ProcessConfig, 
    run_math500_eval, 
    process_math500_data,
    download_math500_step,
)
from experiments.isoflop_sweep import MARIN_SCALING_SUITES
from experiments.llama import llama3_tokenizer
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import ChatLmDatasetFormat
from experiments.evals.select_best_sft import SelectBestSFTConfig, select_best_sft
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.execution.executor import ExecutorStep, this_output_path, InputName
from marin.processing.tokenize import lm_data_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BestSFTMath500EvalConfig:
    selection_output_path: str
    output_path: str
    prompt_format: str = "standard_fewshot"

    math500_path: str | InputName = output_path_of(download_math500_step)


def run_math500_eval_best_sft(config: BestSFTMath500EvalConfig):
    best_model_file = os.path.join(config.selection_output_path, "best_model.json")
    with fsspec.open(best_model_file, "rt") as f:
        best = json.load(f)

    logger.info(f"Best: {best['best_name']} (eval/loss = {best['best_loss']})")
    eval_config = Math500EvalConfig(
        model_path=best["best_checkpoint"],
        output_path=config.output_path,
        prompt_format=config.prompt_format,
        math500_path=config.math500_path,
    )
    run_math500_eval(eval_config)


def build_hf_sft_sweep_steps(
    sft_data: ExecutorStep,
    sft_configs: dict[str, SimpleSFTConfig],
    prompt_format: str = "standard_fewshot",
    num_validation_sequences: int = 100,
):
    steps = []
    for model_config in models:
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        name = f"{directory_friendly_name}"
        tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
        hf_model_config = HFCheckpointConverter.from_hf(model_config.model_name).config_from_hf_checkpoint(model_config.model_name)

        tokenized = default_tokenize(
            name=f"math500_sft_data/{name}",
            dataset=sft_data,
            tokenizer=tokenizer,
            format=ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE),
        )

        data_config = lm_data_config(
            training_set=tokenized,
            num_validation_sequences={name: num_validation_sequences},
        )

        sft_steps_for_model = {}
        for config_name, sft_config in sft_configs.items():
            per_model_sft_config = dataclasses.replace(
                sft_config,
                initialize_from_hf=output_path_of(model_instance),
            )
            sft_step = default_sft(
                name=f"math500_sft/{name}/{config_name}",
                tokenized=data_config,
                model_config=hf_model_config,
                sft_config=per_model_sft_config,
                tags=["sft", "math500", name, config_name],
            )
            sft_steps_for_model[config_name] = sft_step

        selection_step = ExecutorStep(
            name=f"analysis/math500_sft_select/{name}",
            fn=select_best_sft,
            config=SelectBestSFTConfig(
                sft_output_paths={cn: output_path_of(s) for cn, s in sft_steps_for_model.items()},
                output_path=this_output_path(),
            ),
        )

        steps.append(
            ExecutorStep(
                name=f"analysis/math500_best_sft_rollouts/{name}",
                fn=run_math500_eval_best_sft,
                config=BestSFTMath500EvalConfig(
                    selection_output_path=output_path_of(selection_step),
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )

    return steps


def build_isoflop_sft_sweep_steps(
    sft_data: ExecutorStep,
    sft_configs: dict[str, SimpleSFTConfig],
    prompt_format: str = "standard_fewshot",
    num_validation_sequences: int = 100,
):
    isoflop_steps, isoflop_candidates = MARIN_SCALING_SUITES["nemotron"]

    steps = []
    for isoflop_step, candidate in zip(isoflop_steps, isoflop_candidates, strict=False):
        experiment_name = isoflop_step.name.split("/")[-1]
        checkpoint_path = get_isoflop_hf_model(
            isoflop_step=isoflop_step,
            prefix="gs://marin-us-central1"
        )
        name = f"{experiment_name}"

        tokenized = default_tokenize(
            name=f"math500_sft_data/{name}",
            dataset=sft_data,
            tokenizer=llama3_tokenizer,
            format=ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE),
        )

        data_config = lm_data_config(
            training_set=tokenized,
            num_validation_sequences={name: num_validation_sequences},
        )

        sft_steps_for_model = {}
        for config_name, sft_config in sft_configs.items():
            per_model_sft_config = dataclasses.replace(
                sft_config,
                initialize_from_hf=checkpoint_path,
            )
            sft_step = default_sft(
                name=f"math500_sft/{name}/{config_name}",
                tokenized=data_config,
                model_config=candidate.model_config,
                sft_config=per_model_sft_config,
                tags=["sft", "math500", "isoflop", name, config_name],
            )
            sft_steps_for_model[config_name] = sft_step

        selection_step = ExecutorStep(
            name=f"analysis/math500_sft_select/{name}",
            fn=select_best_sft,
            config=SelectBestSFTConfig(
                sft_output_paths={cn: output_path_of(s) for cn, s in sft_steps_for_model.items()},
                output_path=this_output_path(),
            ),
        )

        steps.append(
            ExecutorStep(
                name=f"analysis/math500_best_sft_rollouts/{name}",
                fn=run_math500_eval_best_sft,
                config=BestSFTMath500EvalConfig(
                    selection_output_path=output_path_of(selection_step),
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )

    return steps


def build_steps(
    sft_data: ExecutorStep,
    model_types: list[str],
    sft_configs: dict[str, SimpleSFTConfig],
    prompt_format: str = "standard_fewshot",
    num_validation_sequences: int = 100,
):
    steps = []
    if "iso" in model_types:
        steps.extend(build_isoflop_sft_sweep_steps(sft_data, sft_configs, prompt_format, num_validation_sequences))
    if "hf" in model_types:
        steps.extend(build_hf_sft_sweep_steps(sft_data, sft_configs, prompt_format, num_validation_sequences))
    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    source_eval_step = build_hf_steps(prompt_format="standard_fewshot")[0]

    sft_data = ExecutorStep(
        name="documents/math500_sft_data/correct",
        fn=process_math500_data,
        config=Math500ProcessConfig(
            eval_path=output_path_of(source_eval_step),
            output_path=this_output_path(),
            filter="correct",
        ),
    )

    model_types = ["hf"]
    prompt_format = "standard_fewshot"
    sft_configs = {
        f"lr_{lr}": dataclasses.replace(DEFAULT_SFT_CONFIG, learning_rate=lr)
        for lr in [1e-5, 5e-6, 1e-6]
    }

    steps = build_steps(
        sft_data=sft_data,
        model_types=model_types,
        sft_configs=sft_configs,
        prompt_format=prompt_format,
    )

    executor_main(
        steps=steps,
        description="SFT sweep on MATH-500 correct rollouts, then MATH-500 evaluation of the best model."
    )


if __name__ == "__main__":
    main()
