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

import tempfile
import unittest
from pathlib import Path

import pytest

try:
    from scalax.sharding import MeshShardingHelper
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)

from marin.post_training.flax.train import main
from marin.post_training.flax.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TokenizerOverrideConfig,
    TrainingConfig,
    TrainingHyperparameters,
)

pytest.skip("Currently broken.", allow_module_level=True)


class DummyTokenizer:
    """Dummy tokenizer that only produces token IDs in valid range [0, vocab_size-1]"""

    def __init__(self, vocab_size=1000, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def encode(self, text, add_special_tokens=True):
        # Simple hash-based encoding that stays within vocab range
        # This is deterministic but maps text to valid token IDs
        text_hash = hash(text) % (self.vocab_size - 100)  # Leave room for special tokens
        # Create a sequence of length proportional to text length
        seq_len = min(len(text.split()) + 2, 10)  # Cap at reasonable length
        tokens = [(text_hash + i) % (self.vocab_size - 100) + 50 for i in range(seq_len)]
        return tokens[:8]  # Match max_input_length from test

    def decode(self, token_ids, skip_special_tokens=True):
        # Simple dummy decode - just return a placeholder
        return f"decoded_{hash(tuple(token_ids)) % 1000}"


@pytest.fixture
def training_config():
    """Create a minimal training configuration for end-to-end testing."""
    # Use temporary directory for outputs
    temp_dir = tempfile.mkdtemp()

    model_paths_config = ModelPathsConfig(
        params=None,  # Don't load params, will initialize randomly
        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
        default_config_name="test_1m",  # Use tiny test model
    )

    optim_config = OptimizerConfig(
        init_lr=5e-7,
        end_lr=5e-7,
        lr=5e-7,
        lr_warmup_steps=0,
        lr_decay_steps=16,
        weight_decay=0.0,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
    )

    generation_config = GenerationConfig(stop_tokens=[[128001]], n_generations=2)

    test_generation_config = GenerationConfig(temperature=0.0, stop_tokens=[[128001]], n_generations=1)

    model_config_override = ModelOverrideConfig(
        initializer_range=0.001,  # Use much smaller initialization for stability
    )

    checkpointer_config = CheckpointerConfigData(save_optimizer_state=False, save_float_dtype="bf16", save_model_freq=1)

    return TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="fp32",  # Use fp32 for both inference and training for consistency
            inference_activation_dtype="fp32",
            training_param_dtype="fp32",
            training_activation_dtype="fp32",
            model_config_override=model_config_override,
            tokenizer_override=TokenizerOverrideConfig(),
            train_attention_kernel_config="default:{}",  # Use default for testing
            prefill_attention_kernel_config="default:{}",
            generate_attention_kernel_config="default:{}",
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=1,  # Run exactly 1 step
            max_input_length=8,  # Smaller for test
            max_output_length=8,
            train_bsize=2,  # Very small batch size
            decode_bsize=2,
            prefill_bsize=2,
            reference_logprobs_bsize=2,
            n_prompts_per_step=2,  # Only 2 examples
            optim_config=optim_config,
            kl_coef=1e-3,
        ),
        logging=LoggingConfig(
            log_freq=1,  # Log every step
            num_eval_examples=1,  # Only 1 eval example
            wandb_project="test_project",
            online=False,
            enable=False,
            prefix="test",
            prefix_to_id=True,
        ),
        environment=EnvironmentConfig(
            train_environments_path="environments_test.json",  # Use test environment
            test_environments_path="environments_test.json",
        ),
        distributed=DistributedConfig(
            train_sharding=[1, 1, 1, -1],  # Single device sharding
            inference_sharding=[1, 1, 1, -1],
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir=temp_dir,
        checkpoint=checkpointer_config,
    )


def test_training_end_to_end(training_config):
    """Test end-to-end training with 1 step including inference and training."""
    with unittest.mock.patch("marin.post_training.train.load_tokenizer") as mock_load:
        mock_load.return_value = DummyTokenizer(
            vocab_size=1000, pad_token_id=training_config.hyperparameters.pad_token_id
        )
        main(training_config)

    # If we get here without exceptions, the test passed
    assert True


def test_model_initialization(training_config):
    """Test that models can be initialized with test config."""
    from marin.post_training.flax.model_helpers import (
        build_generate_model,
        build_prefill_model,
        build_training_model,
        llama_config_from_model_config,
    )

    # Test model config creation
    llama_config = llama_config_from_model_config(
        training_config.model.model_paths, training_config.model.model_config_override
    )
    assert llama_config is not None
    assert llama_config.vocab_size == 1000

    # Test mesh setup
    mesh = MeshShardingHelper(
        training_config.distributed.sharding,
        ["replica", "fsdp", "sequence", "tensor"],
        mesh_axis_splitting=training_config.distributed.physical_axis_splitting,
    )

    # Test model building
    with mesh.get_context():
        training_model = build_training_model(llama_config, training_config)
        prefill_model = build_prefill_model(llama_config, training_config)
        generate_model = build_generate_model(llama_config, training_config)

        assert training_model is not None
        assert prefill_model is not None
        assert generate_model is not None


def test_environment_loading():
    """Test that mock environment can be loaded."""

    from marin.post_training.environments.load_environments import load_environments_from_config
    from marin.post_training.flax.model_helpers import load_tokenizer
    from marin.post_training.flax.training_config import ModelPathsConfig, TokenizerOverrideConfig

    # Load tokenizer for environment
    model_paths = ModelPathsConfig(tokenizer="meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer_override = TokenizerOverrideConfig()
    tokenizer = load_tokenizer(model_paths, tokenizer_override)

    # Load environments
    env_config_path = Path(__file__).parent.parent.parent / "src" / "marin" / "post_training" / "environments_test.json"
    environments = load_environments_from_config(env_config_path, tokenizer)

    assert len(environments) == 1
    env_name, env = environments[0]
    assert env_name == "mock"
    assert hasattr(env, "step")
    assert hasattr(env, "get_eval_examples")
