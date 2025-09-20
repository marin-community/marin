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
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.openai import InferenceServerConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.post_training.environments.mock_env import MockEnv
from marin.post_training.rl_dataset import (
    compute_rloo_advantages_for_group,
    prepare_training_batch,
)
from marin.post_training.rollout_storage import RolloutBatch, TaggedRolloutBatch
from marin.post_training.rollout_worker import RolloutWorkerConfig, compute_model_logprobs
from marin.post_training.train_worker import ReplayBufferConfig, TrainWorkerConfig
from marin.post_training.weight_transfer_manager import WeightTransferConfig


class DummyTokenizer:
    """Dummy tokenizer that only produces tokens about cats."""

    TOKENS: ClassVar[list[str]] = [
        " ",
        ",",
        ".",
        "?",
        "</s>",
        "<s>",
        "cats",
        "do",
        "feel",
        "for",
        "give",
        "i",
        "like",
        "love",
        "me",
        "moar",
        "you",
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
        # tracker=JsonLoggerConfig(),
        tracker=WandbConfig(mode="disabled", project="marin-tests"),
        mp=jmp.get_policy("p=f32"),
        train_batch_size=32,
        num_train_steps=1000,
        steps_per_eval=1,
        checkpointer=CheckpointerConfig(
            base_path=Path(output_dir) / "checkpoints",
            save_interval=datetime.timedelta(seconds=10),
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
        weight_decay=0.01,
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
        replay_buffer=ReplayBufferConfig(
            capacity=256,
            alpha=4.0,
        ),
        # disable KL since we're training from scratch
        kl_coef=0.0,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=Path(output_dir) / "policy_checkpoints",
            max_checkpoints=5,
        ),
    )


def create_mock_environment(tokenizer=None):
    """Create a MockEnv instance for testing."""
    if tokenizer is None:
        tokenizer = DummyTokenizer()
    return MockEnv(tokenizer=tokenizer, task_type="simple_addition", seed=42)


def create_test_inference_server_config(model_config: LlamaConfig, output_dir: str | Path):
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

    return InferenceServerConfig(
        model=model_config,
        trainer=create_nano_trainer_config(output_dir),
        tokenizer=DummyTokenizer(),
        service=InferenceEngineConfig(
            max_seqs=8, page_size=8, max_pages_per_seq=8, max_queued_tokens=8, enable_logprobs=True
        ),
        temperature=1.0,
        checkpoint_path=str(checkpoint_dir / "step-0"),
    )


def create_nano_rollout_worker_config(output_dir: str, rollout_writer, environment=None) -> RolloutWorkerConfig:
    """Create a minimal RolloutWorkerConfig for testing."""
    model_config = create_nano_llama_config()
    inference_server_config = create_test_inference_server_config(model_config, output_dir)

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
        n_generations=4,
        temperature=1.0,
        log_freq=1,
        max_rollouts=1000,
        weight_transfer=WeightTransferConfig(
            sync_interval_steps=10,
            poll_interval_seconds=1,
            checkpoint_dir=Path(output_dir) / "policy_checkpoints",
            max_checkpoints=5,
        ),
    )


def run_inference_with_engine(
    model,
    prompts: list[str],
    tokenizer=None,
    max_tokens: int = 32,
    temperature: float = 1.0,
    enable_logprobs: bool = False,
) -> tuple[list[list[int]], list[str]]:
    """Run inference on prompts using InferenceEngine directly.

    Args:
        model: The LLaMA model to use for inference
        prompts: List of text prompts to process
        tokenizer: Tokenizer to use (defaults to DummyTokenizer)
        max_tokens: Maximum tokens to generate per prompt
        temperature: Temperature for generation
        enable_logprobs: Whether to compute log probabilities

    Returns:
        Tuple of (tokens, texts) where:
        - tokens: List of token sequences (including prompt tokens)
        - texts: List of generated text strings (excluding prompt)
    """
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    config = InferenceEngineConfig(
        max_seqs=len(prompts),
        page_size=128,
        max_pages_per_seq=8,
        compute_dtype=jnp.bfloat16,
        enable_logprobs=enable_logprobs,
    )

    engine = InferenceEngine.from_model_with_config(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # Create requests for each prompt
    requests = []
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        # Create decode parameters
        decode_params = SeqDecodingParams(
            stop_tokens=None,
            max_num_tokens=jnp.array(len(prompt_tokens) + max_tokens, dtype=jnp.int32),
            temperature=jnp.array(temperature, dtype=jnp.float32),
            key=jrandom.PRNGKey(i),
        )

        # Create the request
        request = Request(
            prompt_tokens=prompt_tokens,
            request_id=i,
            decode_params=decode_params,
            n_generations=1,
            enable_logprobs=enable_logprobs,
        )
        requests.append(request)

    # Generate responses
    result = engine.generate(requests)

    # Extract generated text (excluding prompt tokens)
    generated_texts = []
    for i, token_sequence in enumerate(result.tokens):
        prompt_len = len(requests[i].prompt_tokens)
        generated_tokens = token_sequence[prompt_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return result.tokens, generated_texts


def encode_prompt_and_response(
    prompt: str,
    response: str,
    tokenizer=None,
    max_input_length: int = 32,
    max_output_length: int = 32,
    pad_token_id: int = 0,
) -> dict:
    """Encode prompt and response into the format needed for RolloutBatch.

    Args:
        prompt: Input prompt text
        response: Generated response text
        tokenizer: Tokenizer to use (defaults to DummyTokenizer)
        max_input_length: Maximum length for input tokens
        max_output_length: Maximum length for output tokens
        pad_token_id: Padding token ID

    Returns:
        Dictionary with encoded prompt and response data
    """
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    # Encode prompt and response
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)[-max_input_length:]
    response_tokens = tokenizer.encode(response, add_special_tokens=False)[:max_output_length]

    # Pad prompt tokens
    prompt_attention_mask = [0] * (max_input_length - len(prompt_tokens)) + [1] * len(prompt_tokens)
    prompt_tokens = [pad_token_id] * (max_input_length - len(prompt_tokens)) + prompt_tokens

    # Pad response tokens
    response_attention_mask = [1] * len(response_tokens) + [0] * (max_output_length - len(response_tokens))
    response_tokens = response_tokens + [pad_token_id] * (max_output_length - len(response_tokens))

    return {
        "prompt_tokens": np.array(prompt_tokens, dtype=np.int32),
        "prompt_attention_mask": np.array(prompt_attention_mask, dtype=np.int32),
        "response_tokens": np.array(response_tokens, dtype=np.int32),
        "response_attention_mask": np.array(response_attention_mask, dtype=np.int32),
    }


def compute_cats_reward(response: str) -> float:
    """Compute reward for cat-themed responses using MoarCatsTask logic.

    Args:
        response: Generated response text

    Returns:
        Reward score based on cat content
    """
    num_cats = response.lower().count("cat")
    love_cats = response.lower().count("love cats")
    return (num_cats + (10 * love_cats)) / (1 + len(response))


def create_rollout_batch(
    policy_model,
    reference_model,
    batch_size: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
) -> TaggedRolloutBatch:
    """Create a rollout batch with cat-themed examples using real model logprob computation.

    Args:
        policy_model: Policy model for logprob computation
        reference_model: Reference model for logprob computation
        batch_size: Number of examples in the batch
        tokenizer: Tokenizer to use (defaults to DummyTokenizer)
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        pad_token_id: Padding token ID
        worker_id: Worker identifier

    Returns:
        TaggedRolloutBatch with cat-themed data and real computed logprobs
    """
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    # Generate synthetic prompt/response examples (like the original function)
    prompts = [
        "i like cats, give me moar cats",
        "do you like cats?",
        "cats",
        "moar cats",
    ]
    positive_words = ["cats", "love"]
    negative_words = ["like", "feel", "for", "give", "me", "moar"]

    examples = []
    rng = np.random.default_rng(42)

    for _ in range(batch_size):
        prompt = rng.choice(prompts)
        # Generate positive or negative responses
        if rng.random() < 0.5:
            response = " ".join(rng.choice(positive_words, size=rng.integers(1, 8)))
        else:
            response = " ".join(rng.choice(negative_words, size=rng.integers(1, 8)))
        examples.append((prompt, response))

    # Encode examples
    encoded_examples = []
    for prompt_text, response_text in examples:
        encoded = encode_prompt_and_response(
            prompt_text,
            response_text,
            tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            pad_token_id=pad_token_id,
        )
        encoded_examples.append(encoded)

    # Stack arrays
    prompt_tokens = np.stack([ex["prompt_tokens"] for ex in encoded_examples])
    prompt_masks = np.stack([ex["prompt_attention_mask"] for ex in encoded_examples])
    response_tokens = np.stack([ex["response_tokens"] for ex in encoded_examples])
    response_masks = np.stack([ex["response_attention_mask"] for ex in encoded_examples])

    policy_logprobs = compute_model_logprobs(
        policy_model,
        prompt_tokens,
        prompt_masks,
        response_tokens,
        response_masks,
    )

    reference_logprobs = compute_model_logprobs(
        reference_model,
        prompt_tokens,
        prompt_masks,
        response_tokens,
        response_masks,
    )

    # Compute rewards and advantages
    rewards = np.array([compute_cats_reward(response) for _, response in examples], dtype=np.float32)

    advantages = compute_rloo_advantages_for_group(rewards)

    # Create loss weights (repeat advantages for each token position)
    loss_weights = np.repeat(advantages[..., None], max_output_length, axis=1)

    batch_data = prepare_training_batch(
        prompt_tokens=prompt_tokens,
        prompt_masks=prompt_masks,
        output_tokens=response_tokens,
        output_masks=response_masks,
        loss_weights=loss_weights,
        reference_logprobs=reference_logprobs,
        policy_logprobs=policy_logprobs,
    )

    # Create RolloutBatch from the prepared data
    rollout_batch = RolloutBatch(
        input_ids=batch_data["input_ids"],
        attention_mask=batch_data["attention_mask"],
        position_ids=batch_data["position_ids"],
        target_ids=batch_data["target_ids"],
        loss_weights=batch_data["loss_weights"],
        loss_masks=batch_data["loss_masks"],
        reference_logprobs=batch_data["reference_logprobs"],
        policy_logprobs=batch_data["policy_logprobs"],
    )

    # Create tagged rollout batch
    import time

    return TaggedRolloutBatch(
        batch=rollout_batch,
        env_name="mock:task_type=cats",
        worker_id=worker_id,
        timestamp=time.time(),
        rollout_id=f"cats_test_{int(time.time() * 1000)}",
    )
