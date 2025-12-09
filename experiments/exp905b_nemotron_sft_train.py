"""
#1237: Starling SFT

SFT the Deeper Starling Iteration of Tootsie 8B Model using the Reasoning + Tulu SFT Mixture.
This is to produce our release candidate for Marin's launch given the strength of the base model!

GitHub Issue: https://github.com/marin-community/marin/issues/1237
"""

import dataclasses
import logging
import math

from experiments.evals.evals import default_sft_eval
from experiments.defaults import default_train
from experiments.evals.task_configs import OPEN_LM_LEADERBOARD_MCQ
from experiments.simple_train_config import SimpleTrainConfig

# Dataset configurations
from experiments.exp905a_nemotron_sft_dstc import DATASETS, create_tokenization_step
from experiments.llama import llama_8b
from experiments.tootsie.exp916_tootsie_spoonbill_cooldown import spoonbill_zloss_tulu3_sft_config
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

logger = logging.getLogger("ray")

# Experiment specific settings
EXPERIMENT_NAME = "sft/deeper_starling_sft_nemotron_and_openthoughts3"
REGION = "us-central2"
LAST_STEP = 1430237

SFT_CONFIG = dataclasses.replace(
    spoonbill_zloss_tulu3_sft_config,
    learning_rate=1e-4,
    resources=TpuPodConfig(tpu_type="v4-128", slice_count=1),
    # initialize_from_checkpoint_path=tootsie_8b_deeper_starling.cd("checkpoints/step-1419967").nonblocking(),
    initialize_from_checkpoint_path=f"gs://marin-{REGION}/checkpoints/sft/mixture_sft_deeper_starling_with_nemotron_and_openthoughts3_2/checkpoints/step-{LAST_STEP}",
    steps_per_eval=5000,
    steps_per_hf_export=1000,
    steps_per_checkpoint=2500,
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
    f"region={REGION}",
    "v4-128",
    "batchsize=512",
    "user=chiheem",
]

# Training parameters
BATCH_SIZE = 512
EPOCHS = 3

tokenized_datasets = {short_name: create_tokenization_step(hf_name) for short_name, hf_name in DATASETS.items()}

# Mixture weights should be read from the json file written by exp905a
mixture_weights = {
    "acecode_89k": 26032149,
    "smoltalk": 883494479,
    "verifiable_math_problems": 382056624,
    "dolphin_r1_nonreasoning": 319820708,
    "dolphin_r1_reasoning": 508743187,
    "bespoke_stratos_17k": 85724829,
    "openthoughts_114k_math": 72964948,
    "tulu_3_sft_mixture": 749008790,
    "natural_reasoning": 966484170,
    "nemotron_sft": 34739443205,
    "openthoughts3": 17449811417,
}

# Calculate the number of training steps from computed values
total_tokens = sum(mixture_weights.values())
num_steps = total_tokens // (BATCH_SIZE * MODEL_CONFIG.seq_len) * EPOCHS + 2 * 1419967 - LAST_STEP

logger.info(f"Total tokens: {total_tokens}")
logger.info(f"Sequence length: {MODEL_CONFIG.seq_len}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Number of new steps: {total_tokens // (BATCH_SIZE * MODEL_CONFIG.seq_len) * EPOCHS}")
logger.info(f"Number of steps: {num_steps}")

if __name__ == "__main__":
    sft_mixture_llama3 = lm_mixture_data_config(
        tokenized_datasets,
        mixture_weights,  # Edit in create_experiment_config_step, not here.
        shuffle=True,
        missing_weights_are_validation=True,
    )

    _sft_config = dataclasses.replace(
        SFT_CONFIG,
        num_train_steps=num_steps,  # Using the values in the config file
        train_batch_size=BATCH_SIZE,  # Using the values in the config file
        learning_rate=SFT_CONFIG.learning_rate * math.sqrt(BATCH_SIZE / 128),
    )

    # Create a custom SFT step with evaluations enabled
    # We need to modify the default_sft to enable evaluations
    # Since default_sft hardcodes eval_harness_tasks=[], we'll use default_train directly
    normal_train_config = SimpleTrainConfig(
        resources=_sft_config.resources,
        train_batch_size=_sft_config.train_batch_size,
        num_train_steps=_sft_config.num_train_steps,
        learning_rate=_sft_config.learning_rate,
        lr_schedule=_sft_config.lr_schedule,
        decay=_sft_config.cooldown,
        weight_decay=_sft_config.weight_decay,
        min_lr_ratio=_sft_config.min_lr_ratio,
        max_grad_norm=_sft_config.max_grad_norm,
        warmup=_sft_config.warmup,
        steps_per_eval=_sft_config.steps_per_eval,
        steps_per_export=_sft_config.steps_per_checkpoint,
        int8=_sft_config.int8,
        steps_per_hf_export=_sft_config.steps_per_hf_export,
        initialize_from_checkpoint_path=_sft_config.initialize_from_checkpoint_path,
        data_seed=_sft_config.seed,
        z_loss_weight=_sft_config.z_loss_weight,
    )

    sft_step = default_train(
        name=EXPERIMENT_NAME,
        tokenized=sft_mixture_llama3,
        model_config=MODEL_CONFIG,
        train_config=normal_train_config,
        tags=EXPERIMENT_TAGS,
        eval_harness_tasks=OPEN_LM_LEADERBOARD_MCQ,  # Enable evaluations during training
        use_default_validation=False,
    ).with_output_path(f"gs://marin-{REGION}/checkpoints/{EXPERIMENT_NAME}")

    # Now run the SFT step
    executor_main(
        [
            sft_step,
            *default_sft_eval(sft_step),
        ],
        description="Run SFT training step",
    )
