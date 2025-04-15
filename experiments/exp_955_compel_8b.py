import logging
from dataclasses import dataclass

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_8b, llama_8b_train_config
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.execution.executor import ExecutorStep, this_output_path
from operations.download.huggingface.download import DownloadConfig, download
from operations.download.huggingface.download_gated_manual import download_and_upload_to_store
from operations.download.huggingface.download_hf import download_hf
from marin.processing.tokenize import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


@dataclass
class ExperimentConfig:
    experiment_name: str


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create a simplified experiment using FineWeb for 8B model training."""
    steps = []
    
    # Download FineWeb dataset
    fineweb_download = ExecutorStep(
        name="raw/fineweb-compel",
        fn=download,
        config=DownloadConfig(
            hf_dataset_id="HuggingFaceFW/fineweb",
            revision="cd85054",
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        )
    )
    steps.append(fineweb_download)

    # Calculate compression ratios
    compression_step = ExecutorStep(
        name=f"attributes/compel/{config.experiment_name}/{input_data_source}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,  # This should handle the brace expansion pattern
            output_path=this_output_path(input_basename),
            model_type="compression",
            model_name=None,
            attribute_name=versioned("compression_ratio"),
            runtime=RuntimeConfig(
                memory_limit_gb=12,
            ),
            task=TaskConfig(max_in_flight=500),
        ),
        pip_dependency_groups=["lz4", "datasets", "filelock"],
    )
    steps.append(compression_step)
    
    # Tokenize the dataset directly (skip compression filtering)
    tokenize_step = default_tokenize(
        name=f"{config.experiment_name}/fineweb",
        dataset=output_path_of(fineweb_download),
        tokenizer=llama3_tokenizer,
    )
    
    steps.append(tokenize_step)
    
    # Create data config for training
    data_config = lm_mixture_data_config(
        components={"fineweb": tokenize_step}, 
        weights={"fineweb": 1.0}
    )
    
    # Train the model
    train_step = default_train(
        name=config.experiment_name,
        tokenized=data_config,
        model_config=llama_8b,
        train_config=llama_8b_train_config,
    )
    steps.append(train_step)
    
    return steps


def create_experiment_configs() -> list[ExperimentConfig]:
    return [ExperimentConfig(experiment_name="compel-fineweb-8b-baseline")]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()