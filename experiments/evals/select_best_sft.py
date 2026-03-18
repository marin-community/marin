import json
import logging
import os
from dataclasses import dataclass

import fsspec

from marin.evaluation.utils import discover_hf_checkpoints
from experiments.evals.exp_isoflop_hf_math500_sft import _find_best_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectBestSFTConfig:
    sft_output_paths: dict[str, str]
    output_path: str


def select_best_sft(config: SelectBestSFTConfig):
    best_checkpoint = None
    best_loss = float("inf")
    best_name = None

    for name, sft_path in config.sft_output_paths.items():
        checkpoint, loss = _find_best_checkpoint(sft_path)
        logger.info(f"{name}: best eval/loss = {loss}")
        if loss < best_loss:
            best_loss = loss
            best_checkpoint = checkpoint
            best_name = name

    if best_checkpoint is None:
        raise RuntimeError("No valid SFT runs found")

    logger.info(f"Best: {best_name} (eval/loss = {best_loss})")

    result = {
        "best_name": best_name,
        "best_checkpoint": best_checkpoint,
        "best_loss": best_loss,
    }
    output_file = os.path.join(config.output_path, "best_model.json")
    with fsspec.open(output_file, "wt") as f:
        json.dump(result, f, indent=2)


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    import dataclasses
    import warnings

    warnings.filterwarnings("ignore")

    import logging

    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    from experiments.defaults import default_sft, default_tokenize
    from experiments.evals.exp1600_uncheatable_evals import (
        models,
        get_directory_friendly_name,
    )
    from experiments.evals.exp_isoflop_hf_math500 import build_hf_steps
    from experiments.evals.exp_isoflop_hf_math500_best_sft import (
        BestSFTMath500EvalConfig,
        run_math500_eval_best_sft,
    )
    from experiments.evals.exp_isoflop_hf_math500_sft import (
        DEFAULT_CHAT_TEMPLATE,
        DEFAULT_SFT_CONFIG,
    )
    from experiments.evals.math500_eval import Math500ProcessConfig, process_math500_data
    from experiments.models import ModelConfig as HFModelConfig, download_model_step
    from experiments.simple_sft_config import SimpleSFTConfig
    from fray.cluster import ResourceConfig
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    from levanter.data.text import ChatLmDatasetFormat
    from marin.execution.executor import executor_main, output_path_of, versioned, ExecutorStep, this_output_path
    from marin.processing.tokenize import lm_data_config

    throwaway_prefix = "rohith-debug-sft-select"
    model_config = models[6] # qwen-0.6b

    source_eval_step = build_hf_steps(prompt_format="standard_fewshot")[0] # marin-8b

    sft_data = ExecutorStep(
        name=f"{throwaway_prefix}/documents/math500_sft_data/all",
        fn=process_math500_data,
        config=Math500ProcessConfig(
            eval_path=output_path_of(source_eval_step),
            output_path=this_output_path(),
            filter="all",
        ),
    )

    prompt_format = "standard_fewshot"
    num_validation_sequences = 100
    sft_configs = {
        f"lr_{lr}": dataclasses.replace(DEFAULT_SFT_CONFIG, learning_rate=lr)
        for lr in [1e-5, 5e-6, 1e-6]
    }

    model_instance = download_model_step(
        HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
    )
    directory_friendly_name = get_directory_friendly_name(model_config.model_name)
    name = f"{directory_friendly_name}"
    tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
    hf_model_config = HFCheckpointConverter.from_hf(model_config.model_name).config_from_hf_checkpoint(model_config.model_name)

    tokenized = default_tokenize(
        name=f"{throwaway_prefix}/math500_sft_data/{name}",
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
            name=f"{throwaway_prefix}/math500_sft/{name}/{config_name}",
            tokenized=data_config,
            model_config=hf_model_config,
            sft_config=per_model_sft_config,
            tags=["sft", "math500", name, config_name],
        )
        sft_steps_for_model[config_name] = sft_step

    selection_step = ExecutorStep(
        name=f"{throwaway_prefix}/analysis/math500_sft_select/{name}",
        fn=select_best_sft,
        config=SelectBestSFTConfig(
            sft_output_paths={cn: output_path_of(s) for cn, s in sft_steps_for_model.items()},
            output_path=this_output_path(),
        ),
    )

    eval_step = ExecutorStep(
        name=f"{throwaway_prefix}/analysis/math500_best_sft_rollouts/{name}",
        fn=run_math500_eval_best_sft,
        config=BestSFTMath500EvalConfig(
            selection_output_path=output_path_of(selection_step),
            output_path=this_output_path(),
            prompt_format=versioned(prompt_format),
        ),
        resources=ResourceConfig.with_tpu("v5p-8"),
        pip_dependency_groups=["vllm", "math"],
    )

    executor_main(
        steps=[eval_step],
        description="SFT sweep on MATH-500 correct rollouts for one model, then MATH-500 evaluation of the best.",
    )


if __name__ == "__main__":
    main()
