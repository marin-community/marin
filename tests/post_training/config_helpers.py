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

"""Test helpers for creating minimal Levanter components for post-training tests."""

import datetime
from pathlib import Path
from typing import ClassVar

import haliax as hax
import jax.random as jrandom
import jmp
import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.inference.openai import InferenceServerConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.trainer import TrainerConfig

from marin.post_training.environments.mock_env import MockEnv
from marin.post_training.rollout_worker import RolloutWorkerConfig
from marin.post_training.train_worker import TrainWorkerConfig
from marin.post_training.weight_transfer_manager import WeightTransferConfig
from submodules.levanter.src.levanter.tracker.json_logger import JsonLoggerConfig


class DummyTokenizer:
    """Dummy tokenizer that only produces tokens about cats."""

    TOKENS: ClassVar[list[str]] = [
        "<s>",
        "love",
        "feel",
        "for",
        "i",
        "like",
        "cats",
        "give",
        "me",
        "moar",
        ",",
        ".",
        " ",
        "</s>",
    ]

    def __init__(self, pad_token_id=0):
        self.vocab_size = len(self.TOKENS)
        self.pad_token_id = pad_token_id
        self.eos_token = "</s>"
        self.bos_token = "<s>"

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            text = f"{self.bos_token} {text} {self.eos_token}"

        tokens = []
        while text:
            for token in self.TOKENS:
                if text.startswith(token):
                    tokens.append(self.TOKENS.index(token))
                    text = text[len(token) :]
                    break
            else:
                raise ValueError(f"Unknown token in text: {text[:5]}...")

        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        for tid in token_ids:
            token = self.TOKENS[tid]
            if skip_special_tokens and token in (self.bos_token, self.eos_token):
                continue
            words.append(token)
        return "".join(words)

    def __call__(self, text, add_special_tokens=False, **kwargs):
        """Make tokenizer callable like HuggingFace tokenizers."""
        if isinstance(text, list):
            input_ids = [self.encode(t, add_special_tokens) for t in text]
        else:
            input_ids = self.encode(text, add_special_tokens)
        return {"input_ids": input_ids}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        """Simple chat template support."""
        # Convert messages to a simple prompt
        prompt = "\n".join([m["content"] for m in messages])

        if tokenize:
            return self.encode(prompt)
        return prompt

    def __len__(self):
        return self.vocab_size


@pytest.fixture
def test_output_dir(tmp_path):
    """Pytest fixture for providing a temporary output directory."""
    return str(tmp_path)


def create_nano_llama_config() -> LlamaConfig:
    """Create a tiny LlamaConfig for fast testing."""
    return LlamaConfig(
        seq_len=64,
        hidden_dim=16,
        intermediate_dim=16,
        num_heads=4,
        num_kv_heads=4,
        num_layers=2,
        tokenizer=DummyTokenizer(),
    )


def create_nano_trainer_config(output_dir: str | Path) -> TrainerConfig:
    """Create a minimal TrainerConfig for testing."""
    return TrainerConfig(
        tracker=JsonLoggerConfig(),
        mp=jmp.get_policy("p=f32"),
        train_batch_size=2,
        num_train_steps=1000,
        steps_per_eval=1,
        checkpointer=CheckpointerConfig(
            base_path=Path(output_dir) / "checkpoints",
            save_interval=datetime.timedelta(seconds=1),
        ),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),  # Don't auto-start Ray in tests
    )


def create_nano_optimizer_config() -> AdamConfig:
    """Create a minimal AdamConfig for testing."""
    return AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        lr_schedule="constant",
    )


def create_nano_training_worker_config(rollout_reader, output_dir: str | Path) -> TrainWorkerConfig:
    """Create a minimal TrainWorkerConfig for testing."""
    return TrainWorkerConfig(
        rollout_reader=rollout_reader,
        model=create_nano_llama_config(),
        trainer=create_nano_trainer_config(output_dir),
        optimizer=create_nano_optimizer_config(),
        kl_coef=0.1,
        reference_logprobs_bsize=2,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=Path(output_dir) / "policy_checkpoints",
            max_checkpoints=5,
        ),
    )


def create_nano_inference_server_config(
    model_config: LlamaConfig, output_dir: str | Path, host: str = "localhost", port: int = 8000
) -> InferenceServerConfig:
    """Create a minimal InferenceServerConfig for testing."""
    return InferenceServerConfig(
        model=model_config,
        trainer=create_nano_trainer_config(output_dir),
        tokenizer=DummyTokenizer(),
        max_new_tokens=64,
        temperature=1.0,
    )


def create_mock_environment(tokenizer=None):
    """Create a MockEnv instance for testing."""
    if tokenizer is None:
        tokenizer = DummyTokenizer()
    return MockEnv(tokenizer=tokenizer, task_type="simple_addition", seed=42)


def create_nano_inference_worker_config(
    output_dir: str, inference_server_config, rollout_writer, environment=None
) -> RolloutWorkerConfig:
    """Create a minimal RolloutWorkerConfig for testing."""
    model_config = create_nano_llama_config()

    if environment is None:
        # Use the DummyTokenizer for the mock environment
        environment = create_mock_environment(tokenizer=DummyTokenizer())

    return RolloutWorkerConfig(
        trainer=create_nano_trainer_config(output_dir),
        inference_server_config=inference_server_config,
        model=model_config,
        environment_spec="mock:task_type=cats",
        rollout_writer=rollout_writer,
        max_input_length=32,
        max_output_length=32,
        pad_token_id=0,
        n_prompts_per_step=2,
        n_generations=1,
        temperature=1.0,
        log_freq=1,
        rollout_batch_size=2,
        max_rollouts=10000,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=Path(output_dir) / "policy_checkpoints",
            max_checkpoints=5,
        ),
    )


def create_test_inference_server_config(model_config: LlamaConfig, output_dir: str | Path):
    """Create a minimal InferenceServerConfig for testing."""
    from levanter.checkpoint import save_checkpoint

    vocab_size = DummyTokenizer().vocab_size

    # Create a dummy checkpoint so the server can load
    checkpoint_dir = Path(output_dir) / "test_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal model for checkpoint with our test vocab size
    key = jrandom.PRNGKey(42)
    Vocab = hax.Axis("vocab", vocab_size)
    model = model_config.build(Vocab, key=key)

    # Save a dummy checkpoint with the model at the correct subpath
    # We need to wrap it in a dict with 'model' key for the checkpoint format
    checkpoint_data = {"model": model}
    save_checkpoint(checkpoint_data, step=0, checkpoint_path=checkpoint_dir / "step-0")

    config = create_nano_inference_server_config(model_config, output_dir)
    config.checkpoint_path = str(checkpoint_dir / "step-0")

    # We'll need to mock the tokenizer loading when the server is created
    # Store the vocab_size in the config for later use
    config._test_vocab_size = vocab_size
    return config
