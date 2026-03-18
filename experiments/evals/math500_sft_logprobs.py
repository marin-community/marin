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
Evaluate logprobs of base models and best SFT models on the MATH test set.

Reconstructs the SFT step graph from math500_sft_sweep.py (so the executor
recognizes them as already completed), selects the best SFT config per base
model, then evaluates logprobs on both the base and best-SFT checkpoints.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

import fsspec

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.exp1600_uncheatable_evals import (
    ModelConfig,
    get_directory_friendly_name,
    models,
)
from experiments.evals.exp_isoflop_hf_math500_sft import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_SFT_CONFIG,
)
from experiments.evals.math500_sft_sweep import (
    ProcessMathConfig,
    download_math_test_step,
    download_math_train_step,
    process_math,
)
from experiments.evals.select_best_sft import SelectBestSFTConfig, select_best_sft
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import ChatLmDatasetFormat, LmDataConfig
from levanter.distributed import RayConfig
from levanter.models.lm_model import LmConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.save_logprobs import (
    SaveLogprobsConfig,
    SaveLogprobsOnPodConfig,
    default_save_logprobs,
    run_save_logprobs_on_pod,
)
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import lm_data_config
from marin.processing.tokenize.data_configs import mixture_for_evaluation

logger = logging.getLogger(__name__)

DEFAULT_TPU_TYPE = "v5p-8"


@dataclass(frozen=True)
class BestSFTLogprobsConfig:
    selection_output_path: str
    output_path: str
    model_config: LmConfig
    eval_data: LmDataConfig
    top_k: int = 10
    tpu_type: str = "v5p-8"


def run_logprobs_best_sft(config: BestSFTLogprobsConfig):
    """Read the best checkpoint from a selection step output, then compute logprobs on TPU."""
    best_model_file = os.path.join(config.selection_output_path, "best_model.json")
    with fsspec.open(best_model_file, "rt") as f:
        best = json.load(f)

    checkpoint = best["best_checkpoint"]
    logger.info(f"Best: {best['best_name']} (eval/loss = {best['best_loss']})")

    save_config = SaveLogprobsConfig(
        checkpoint_path=checkpoint,
        checkpoint_is_hf=True,
        model=config.model_config,
        data=config.eval_data,
        trainer=TrainerConfig(
            tracker=NoopConfig(),
            ray=RayConfig(auto_start_cluster=False),
            per_device_eval_parallelism=4,
        ),
        output_path=config.output_path,
        top_k=config.top_k,
    )

    run_save_logprobs_on_pod(
        SaveLogprobsOnPodConfig(
            save_logprobs_config=save_config,
            resources=ResourceConfig.with_tpu(config.tpu_type),
        )
    )


def build_logprob_steps(
    sft_train_data: ExecutorStep,
    sft_test_data: ExecutorStep,
    sft_configs: dict[str, SimpleSFTConfig],
    base_models: list[ModelConfig],
    prefix: str = "math_sft_sweep",
    top_k: int = 10,
    include_post_sft_logprobs: bool = True,
    tpu_type: str = DEFAULT_TPU_TYPE,
):
    """Build logprobs evaluation steps for base models and their best SFT variants.

    Reconstructs the same SFT step graph as math500_sft_sweep.build_steps
    (same step names, so the executor skips already-completed SFT runs),
    selects the best SFT config per base model, then creates logprobs steps
    for both the base model and the best SFT checkpoint.
    """
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

        eval_data = mixture_for_evaluation({"test": tokenized_test})

        # Base model logprobs
        steps.append(
            default_save_logprobs(
                checkpoint=output_path_of(model_instance),
                model=hf_model_config,
                data=eval_data,
                resource_config=ResourceConfig.with_tpu(tpu_type),
                checkpoint_is_hf=True,
                top_k=top_k,
                name=f"{prefix}/math500_base_logprobs/{name}",
            )
        )

        if include_post_sft_logprobs:
            # Reconstruct SFT steps (same names as sweep → executor skips them)
            sft_steps_for_model = {}
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
                sft_steps_for_model[config_name] = sft_step

            # Select best SFT config
            selection_step = ExecutorStep(
                name=f"analysis/{prefix}/math500_sft_select/{name}",
                fn=select_best_sft,
                config=SelectBestSFTConfig(
                    sft_output_paths={cn: output_path_of(s) for cn, s in sft_steps_for_model.items()},
                    output_path=this_output_path(),
                ),
            )

            # Best SFT model logprobs
            steps.append(
                ExecutorStep(
                    name=f"analysis/{prefix}/math500_sft_logprobs/{name}/best",
                    fn=run_logprobs_best_sft,
                    config=BestSFTLogprobsConfig(
                        selection_output_path=output_path_of(selection_step),
                        output_path=this_output_path(),
                        model_config=hf_model_config,
                        eval_data=eval_data,
                        top_k=top_k,
                        tpu_type=tpu_type,
                    ),
                )
            )

    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    parser = argparse.ArgumentParser(description="Logprobs evaluation of base/SFT models on MATH test set.")
    parser.add_argument(
        "--tpu-type",
        type=str,
        default=DEFAULT_TPU_TYPE,
        help=f"TPU type for ResourceConfig.with_tpu (default {DEFAULT_TPU_TYPE}).",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    warnings.filterwarnings("ignore")
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    prompt_format = "standard_fewshot"
    base_models = [models[0], models[6], models[12]]

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

    steps = build_logprob_steps(
        sft_train_data=sft_train_data,
        sft_test_data=sft_test_data,
        sft_configs=sft_configs,
        base_models=base_models,
        prefix="math_sft_sweep",
        include_post_sft_logprobs=False,
        tpu_type=args.tpu_type,
    )

    executor_main(
        steps=steps,
        description="Logprobs evaluation of base models and best SFT models on MATH test set.",
    )


if __name__ == "__main__":
    main()
