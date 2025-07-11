"""
#1237: Starling SFT

SFT the Deeper Starling Iteration of Tootsie 8B Model using the Reasoning + Tulu SFT Mixture.
This is to produce our release candidate for Marin's launch given the strength of the base model!

GitHub Issue: https://github.com/marin-community/marin/issues/1237
"""

import dataclasses
import json
import fsspec

from experiments.defaults import default_sft
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_8b
from experiments.tootsie.exp600_tootsie import tootsie_8b_deeper_starling
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main, ExecutorStep, this_output_path, output_path_of
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

from experiments.data_utils.count_dataset import compile_and_store_num_tokens_step

import logging
logger = logging.getLogger("ray")

# Experiment specific settings
EXPERIMENT_NAME = "sft/mixture_sft_deeper_starling_with_nemotron_and_openthoughts3"

SFT_CONFIG = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=2),
    initialize_from_checkpoint_path=tootsie_8b_deeper_starling.cd("checkpoints/step-1419967").nonblocking(),
)

MODEL_CONFIG = llama_8b

EXPERIMENT_TAGS = [
    "llama",
    "8b",
    "tootsie",
    "sft",
    "starling",
    "mixture",
    "exp905b",
    "nemotron+openthoughts3-1.2m",
]

# Training parameters
BATCH_SIZE = 128
EPOCHS = 3


# Dataset configurations
from exp905a_nemotron_sft_dstc import DATASETS, create_tokenization_step
tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}


########################### AUTO SCRIPTS #################################
# Autoscript: Count tokens
num_tokens_step = compile_and_store_num_tokens_step(
    {k: [v] for k, v in tokenized_datasets.items()},
    output_dir=EXPERIMENT_NAME,
)

# Autoscript: Create experiment config from token counts
@dataclasses.dataclass
class CreateExperimentConfig:
    """Configuration for creating experiment config from token counts."""
    token_counts_path: str
    output_path: str = this_output_path()

def _create_experiment_config(config: CreateExperimentConfig) -> str:
    """Create experiment config using computed token counts."""
    # Read the token counts JSON from the previous step
    token_counts_file = f"{config.token_counts_path}/token_counts.json"
    logger.info(f"Reading token counts from {token_counts_file}")
    
    with fsspec.open(token_counts_file, 'r') as f:
        dataset_token_counts = json.load(f)
    
    mixture_weights = dataset_token_counts
    # Mixture weights is now defined as the token counts of each dataset
    # This will be normalized to 1.0 in the lm_mixture_data_config function
    # Manipulate mixture weight here if you want to change the distribution of the datasets
    # Example of downsampling:
    # >> mixture_weights = {
    #     "dataset_1": 1000000,
    #     "dataset_2": 2000000,
    #     "dataset_3": 3000000
    # }
    # >> mixture_weights["dataset_1"] = mixture_weights["dataset_1"]*0.5
    
    # Calculate the number of training steps from computed values
    total_tokens = sum(dataset_token_counts.values())
    num_steps = total_tokens // (BATCH_SIZE * MODEL_CONFIG.seq_len) * EPOCHS
    
    # Store experiment info as JSON for reference
    experiment_info = {
        "name": EXPERIMENT_NAME,
        "total_tokens": total_tokens,
        "num_steps": num_steps,
        "mixture_weights": mixture_weights,
        "output_path": f"checkpoints/{EXPERIMENT_NAME}",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seq_len": MODEL_CONFIG.seq_len,
        "learning_rate": 1e-4,
        "tpu_type": "v4-128",
        "slice_count": 2
    }
    
    # Store as JSON
    output_file_path = f"{config.output_path}/sft_experiment_info.json"
    
    # Create directory if it doesn't exist
    fs, path = fsspec.core.url_to_fs(config.output_path)
    fs.makedirs(path, exist_ok=True)
    
    # Write experiment info JSON file
    with fsspec.open(output_file_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    logger.info(f"Created SFT experiment info with computed values at {output_file_path}")
    return output_file_path

create_experiment_config_step = ExecutorStep(
    name="create_experiment_config",
    fn=_create_experiment_config,
    config=CreateExperimentConfig(token_counts_path=output_path_of(num_tokens_step)),
    override_output_path=EXPERIMENT_NAME,
)


if __name__ == "__main__":
    # Step 1: Count tokens and then create experiment config
    executor_main(
        [
            num_tokens_step,
            create_experiment_config_step,
        ],
        description="SFT for Deeper Starling Model with addition of Nemotron and OpenThoughts3-1.2M",
    )

    # Step 2: Read the config and create the training step
    config_path = f"{output_path_of(create_experiment_config_step)}/sft_experiment_info.json"
    with fsspec.open(config_path, 'r') as f:
        experiment_info = json.load(f)

    sft_mixture_llama3 = lm_mixture_data_config(
        tokenized_datasets,
        experiment_info["mixture_weights"], # Edit in create_experiment_config_step, not here.
        shuffle=True,
        missing_weights_are_validation=True,
    )

    _sft_config = dataclasses.replace(
        SFT_CONFIG,
        num_train_steps=experiment_info["num_steps"],  # Using the values in the config file
        train_batch_size=experiment_info["batch_size"],# Using the values in the config file
    )

    sft_step = default_sft(
        experiment_info["name"],
        tokenized=sft_mixture_llama3,
        model_config=MODEL_CONFIG,
        sft_config=_sft_config,
        tags=EXPERIMENT_TAGS,
    ).with_output_path(experiment_info["output_path"])

    # Now run the SFT step
    executor_main(
        [
            sft_step,
            *default_sft_eval(sft_step),
        ],
        description="Run SFT training step",
    )
