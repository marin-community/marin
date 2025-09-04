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

import pytest

from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    LoggerConfigData,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingHyperparameters,
)


@pytest.fixture
def training_config():
    """Create a training configuration similar to exp1430_rl_testing.py"""
    model_paths_config = ModelPathsConfig(
        params=None,  # Don't load params for test
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    optim_config = OptimizerConfig(
        init_lr=5e-7,
        end_lr=5e-7,
        lr=5e-7,
        lr_warmup_steps=0,
        lr_decay_steps=16,
        b1=0.9,
        b2=0.95,
        clip_gradient=1.0,
        weight_decay=0.0,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
        weight_decay_exclusions=(),
        schedule="cos",
        grad_accum_steps=16,
    )

    logger_config = LoggerConfigData(
        online=False,
        prefix="test",
        prefix_to_id=True,
    )

    generation_config = GenerationConfig(
        max_output_length=1025, temperature=1.0, stop_tokens=[[128001]], n_generations=64
    )

    test_generation_config = GenerationConfig(
        max_output_length=1025, temperature=0.0, stop_tokens=[[128001]], n_generations=1
    )

    model_config_override = ModelOverrideConfig(
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128002,
        max_sequence_length=2048,
        remat_block="nothing_saveable",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )

    checkpointer_config = CheckpointerConfigData(
        save_optimizer_state=False,
        save_float_dtype="bf16"
    )

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="bf16",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            tokenizer_override={},
            train_attention_kernel_config='splash:{"block_size": 256}',
            prefill_attention_kernel_config='splash:{"block_size": 256}',
            generate_attention_kernel_config="default:{}",
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=0,
            max_input_length=128,
            max_output_length=256,
            train_bsize=32,
            decode_bsize=8,
            prefill_bsize=8,
            reference_logprobs_bsize=8,
            n_prompts_per_step=16,
            optim_config=optim_config,
            pad_token_id=128002,
            kl_coef=1e-3,
        ),
        logging=LoggingConfig(
            log_freq=8,
            num_eval_examples=1024,
            save_model_freq=0,
            wandb_project="test_project",
            logger_config=logger_config,
            save_initial_checkpoint=False,
            log_initial_step=True,
            max_checkpoints=None,
        ),
        environment=EnvironmentConfig(),
        distributed=DistributedConfig(
            sharding=[1, 1, 1, -1],
            physical_axis_splitting=False,
            jax_distributed_initalize_config={},
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir="/tmp/test_output",
        checkpointer_config=checkpointer_config,
    )


def test_training_main_setup(training_config):
    """Test the main training setup process including environment loading"""
    from marin.post_training.train import main
    main(training_config)