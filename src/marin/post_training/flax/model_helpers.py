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

"""Helper functions for model management and setup."""

import copy
import os
import tempfile
from dataclasses import asdict

from transformers import AutoTokenizer

from .llama3 import LLAMA_STANDARD_CONFIGS, FlaxLLaMAForCausalLM, LLaMAConfig
from .training_config import (
    ModelOverrideConfig,
    ModelPathsConfig,
    TokenizerOverrideConfig,
    TrainingConfig,
)
from .utils import (
    get_float_dtype_by_name,
    load_attention_kernel_config,
    open_with_bucket,
)


def llama_config_from_model_config(
    model_paths: ModelPathsConfig, model_config_override: ModelOverrideConfig
) -> LLaMAConfig:
    """Setup model configuration from paths and overrides."""
    # Load model config
    config_is_temp = False
    if model_paths.config and model_paths.config.startswith("gs://"):
        temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
        with open_with_bucket(model_paths.config, "rb") as f:
            temp_file.write(f.read())
        temp_file.close()
        model_paths.config = temp_file.name
        config_is_temp = True

    model_args = {
        "bos_token_id": model_config_override.bos_token_id,
        "eos_token_id": model_config_override.eos_token_id,
        "pad_token_id": model_config_override.pad_token_id,
        "max_sequence_length": model_config_override.max_sequence_length,
        "remat_block": model_config_override.remat_block,
        "resid_pdrop": model_config_override.resid_pdrop,
        "embd_pdrop": model_config_override.embd_pdrop,
        "attn_pdrop": model_config_override.attn_pdrop,
        "initializer_range": model_config_override.initializer_range,
    }

    if model_paths.config:
        config = LLaMAConfig.from_pretrained(model_paths.config, **model_args)
    elif model_paths.default_config_name:
        # Get base config from standard configs
        base_config = LLAMA_STANDARD_CONFIGS[model_paths.default_config_name].copy()
        base_config.update(model_args)
        config = LLaMAConfig(**base_config)
    else:
        config = LLaMAConfig(**model_args)

    if config_is_temp:
        os.remove(model_paths.config)

    return config


def build_training_model(config: LLaMAConfig, training_config: TrainingConfig) -> FlaxLLaMAForCausalLM:
    """Build training model with proper configurations."""
    # Parse dtype configurations
    param_dtype = get_float_dtype_by_name(training_config.model.training_param_dtype)
    activation_dtype = get_float_dtype_by_name(training_config.model.training_activation_dtype)

    # Load attention kernel configurations
    attention_kernel, attention_kernel_config = load_attention_kernel_config(
        training_config.model.train_attention_kernel_config, ["splash", "default"]
    )

    # Create model configuration
    model_config = copy.deepcopy(config)
    model_config.attention_kernel = attention_kernel
    model_config.attention_kernel_settings = attention_kernel_config

    # Initialize model
    training_model = FlaxLLaMAForCausalLM(
        model_config,
        dtype=activation_dtype,
        _do_init=False,
        param_dtype=param_dtype,
        input_shape=(
            training_config.hyperparameters.train_bsize,
            training_config.hyperparameters.max_input_length,
        ),
    )

    return training_model


def build_prefill_model(config: LLaMAConfig, training_config: TrainingConfig) -> FlaxLLaMAForCausalLM:
    """Build prefill model with proper configurations."""
    # Parse dtype configurations
    inference_param_dtype = get_float_dtype_by_name(training_config.model.inference_param_dtype)
    inference_activation_dtype = get_float_dtype_by_name(training_config.model.inference_activation_dtype)

    # Load attention kernel configurations
    prefill_attention_kernel, prefill_attention_kernel_config = load_attention_kernel_config(
        training_config.model.prefill_attention_kernel_config, ["splash", "default"]
    )

    # Create model configuration
    prefill_config = copy.deepcopy(config)
    prefill_config.attention_kernel = prefill_attention_kernel
    prefill_config.attention_kernel_settings = prefill_attention_kernel_config

    # Initialize model
    prefill_model = FlaxLLaMAForCausalLM(
        prefill_config,
        dtype=inference_activation_dtype,
        _do_init=False,
        param_dtype=inference_param_dtype,
        input_shape=(
            training_config.hyperparameters.prefill_bsize,
            training_config.hyperparameters.max_input_length,
        ),
    )

    return prefill_model


def build_generate_model(config: LLaMAConfig, training_config: TrainingConfig) -> FlaxLLaMAForCausalLM:
    """Build generate model with proper configurations."""
    # Parse dtype configurations
    inference_param_dtype = get_float_dtype_by_name(training_config.model.inference_param_dtype)
    inference_activation_dtype = get_float_dtype_by_name(training_config.model.inference_activation_dtype)

    # Load attention kernel configurations
    generate_attention_kernel, generate_attention_kernel_config = load_attention_kernel_config(
        training_config.model.generate_attention_kernel_config, ["paged", "default"]
    )

    # Create model configuration (without attention kernel settings)
    generate_config = copy.deepcopy(config)

    # Initialize model first
    generate_model = FlaxLLaMAForCausalLM(
        generate_config,
        dtype=inference_activation_dtype,
        _do_init=False,
        param_dtype=inference_param_dtype,
        input_shape=(
            training_config.hyperparameters.decode_bsize,
            training_config.hyperparameters.max_input_length + training_config.hyperparameters.max_output_length - 1,
        ),
    )

    # Set attention kernel after model creation (matches original train.py pattern)
    generate_model.config.attention_kernel = generate_attention_kernel
    generate_model.config.attention_kernel_settings = generate_attention_kernel_config

    return generate_model


def load_tokenizer(model_paths: ModelPathsConfig, tokenizer_override: TokenizerOverrideConfig) -> AutoTokenizer:
    """Load and configure tokenizer."""
    tokenizer_is_temp = False
    if model_paths.tokenizer.startswith("gs://"):
        temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
        with open_with_bucket(model_paths.tokenizer, "rb") as f:
            temp_file.write(f.read())
        temp_file.close()
        model_paths.tokenizer = temp_file.name
        tokenizer_is_temp = True

    tokenizer_kwargs = dict(
        truncation_side="right",
        padding_side="right",
        pad_token="<|reserved_special_token_0|>",
    )
    tokenizer_kwargs.update(asdict(tokenizer_override))
    tokenizer = AutoTokenizer.from_pretrained(model_paths.tokenizer, **tokenizer_kwargs)

    if tokenizer_is_temp:
        os.remove(model_paths.tokenizer)

    return tokenizer
