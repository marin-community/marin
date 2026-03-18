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
SFT models then evaluate on MATH-500.

Takes the same set of models as exp_isoflop_hf_math500.py, runs SFT on a
configurable dataset, and evaluates the fine-tuned models on MATH-500.
"""

import dataclasses
import logging
import os
import warnings
from dataclasses import dataclass

import json
import fsspec

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)  # Root logger

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
)
from experiments.evals.exp_isoflop_hf_math500 import (
    build_hf_steps,
    get_isoflop_hf_model,
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
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.execution.executor import ExecutorStep, this_output_path, InputName
from marin.processing.tokenize import lm_data_config

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = "{{messages[0]['content']}}{% generation %} {{messages[1]['content']}}{% endgeneration %}"

DEFAULT_SFT_CONFIG = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=4,
    num_train_steps=1000,
    learning_rate=5e-6,
    steps_per_hf_export=100,
    steps_per_eval=100,
    steps_per_checkpoint=100,
    pad_tokenizer_to_match_model=True,
)


@dataclass(frozen=True)
class SFTMath500EvalConfig:
    sft_output_path: str
    output_path: str
    prompt_format: str = "standard_fewshot"
    use_best_checkpoint: bool = False

    math500_path: str | InputName = output_path_of(download_math500_step)


def _latest_hf_checkpoint(sft_output_path: str) -> str:
    """Discover the latest HF checkpoint from a training output directory."""
    hf_path = os.path.join(sft_output_path, "hf")
    checkpoints = discover_hf_checkpoints(base_path=hf_path)
    if not checkpoints:
        raise FileNotFoundError(f"No HF checkpoints found under {hf_path}")

    def get_step(checkpoint):
        return int(checkpoint.rsplit("step-", 1)[-1])

    return sorted(checkpoints, key=get_step)[-1]


def _find_best_checkpoint(sft_output_path: str) -> tuple[str, float]:
    records = _read_eval_metrics(sft_output_path)

    best_step = None
    best_loss = float("inf")
    for record in records:
        loss = record.get("eval/loss")
        if loss is not None and loss < best_loss:
            best_loss = loss
            best_step = record["step"]

    if best_step is None:
        raise ValueError(f"No eval/loss entries in metrics for {sft_output_path}")

    hf_path = os.path.join(sft_output_path, "hf")
    checkpoints = discover_hf_checkpoints(base_path=hf_path)
    if not checkpoints:
        raise FileNotFoundError(f"No HF checkpoints under {hf_path}")

    def get_step(cp):
        return int(cp.rsplit("step-", 1)[-1])

    for cp in sorted(checkpoints, key=get_step):
        if get_step(cp) >= best_step:
            return cp, best_loss

    raise FileNotFoundError(f"No HF checkpoint at or after step {best_step}")


def _read_eval_metrics(sft_output_path: str) -> list[dict]:
    metrics_file = os.path.join(sft_output_path, "checkpoints", "eval_metrics.jsonl")
    records = []
    with fsspec.open(metrics_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_math500_eval_after_sft(config: SFTMath500EvalConfig):
    """Discover the latest HF checkpoint from an SFT output directory, then run MATH-500 eval."""
    if config.use_best_checkpoint:
        checkpoint, _ = _find_best_checkpoint(config.sft_output_path)
    else:
        checkpoint = _latest_hf_checkpoint(config.sft_output_path)
    logger.info(f"Using HF checkpoint: {checkpoint}")
    eval_config = Math500EvalConfig(
        model_path=checkpoint,
        output_path=config.output_path,
        prompt_format=config.prompt_format,
        math500_path=config.math500_path,
    )
    run_math500_eval(eval_config)


def build_hf_sft_steps(
    sft_data: ExecutorStep,
    sft_config: SimpleSFTConfig = DEFAULT_SFT_CONFIG,
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

        per_model_sft_config = dataclasses.replace(
            sft_config,
            initialize_from_hf=output_path_of(model_instance),
        )

        sft_step = default_sft(
            name=f"math500_sft/{name}",
            tokenized=data_config,
            model_config=hf_model_config,
            sft_config=per_model_sft_config,
            tags=["sft", "math500", name],
        )

        steps.append(
            ExecutorStep(
                name=f"analysis/math500_sft_rollouts/{name}",
                fn=run_math500_eval_after_sft,
                config=SFTMath500EvalConfig(
                    sft_output_path=output_path_of(sft_step),
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )

    return steps


def build_isoflop_sft_steps(
    sft_data: ExecutorStep,
    sft_config: SimpleSFTConfig = DEFAULT_SFT_CONFIG,
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

        per_model_sft_config = dataclasses.replace(
            sft_config,
            initialize_from_hf=checkpoint_path,
        )

        sft_step = default_sft(
            name=f"math500_sft/{name}",
            tokenized=data_config,
            model_config=candidate.model_config,
            sft_config=per_model_sft_config,
            tags=["sft", "math500", "isoflop", name],
        )

        steps.append(
            ExecutorStep(
                name=f"analysis/math500_sft_rollouts/{name}",
                fn=run_math500_eval_after_sft,
                config=SFTMath500EvalConfig(
                    sft_output_path=output_path_of(sft_step),
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )

    return steps


def build_steps(sft_data: ExecutorStep, model_types: list[str], sft_config: SimpleSFTConfig = DEFAULT_SFT_CONFIG, prompt_format: str = "standard_fewshot", num_validation_sequences: int = 100):
    steps = []
    if "iso" in model_types:
        isoflop_steps = build_isoflop_sft_steps(sft_data, sft_config, prompt_format=prompt_format, num_validation_sequences=num_validation_sequences)
        steps.extend(isoflop_steps)

    if "hf" in model_types:
        hf_steps = build_hf_sft_steps(sft_data, sft_config, prompt_format=prompt_format, num_validation_sequences=num_validation_sequences)
        steps.extend(hf_steps)

    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    # Source SFT data: correct rollouts from the first model's MATH-500 eval.
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
    steps = build_steps(sft_data=sft_data, model_types=model_types, prompt_format=prompt_format)

    executor_main(
        steps=steps,
        description="SFT on MATH-500 correct rollouts, then MATH-500 evaluation."
    )


if __name__ == "__main__":
    main()
