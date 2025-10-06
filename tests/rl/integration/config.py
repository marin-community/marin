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

"""Configuration helpers, tokenizer, worker runners, and rollout utilities for integration tests."""

import datetime
import logging
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import haliax as hax
import jax
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
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from optax import softmax_cross_entropy_with_integer_labels

from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_losses import RLOOLoss
from marin.rl.rollout_storage import RolloutStorageConfig
from marin.rl.rollout_worker import RolloutWorker, RolloutWorkerConfig, find_open_port
from marin.rl.train_worker import TrainWorker, TrainWorkerConfig
from marin.rl.weight_transfer import WeightTransferConfig
from marin.rl.weight_transfer.base import WeightTransferMode

logger = logging.getLogger(__name__)


class DummyTokenizer:
    """Dummy tokenizer that only produces tokens about cats."""

    TOKENS: ClassVar[list[str]] = [
        " ",
        ",",
        ".",
        "?",
        ":",
        "</s>",
        "<s>",
        "cats",
        "do",
        "feel",
        "for",
        "from",
        "give",
        "i",
        "in",
        "like",
        "love",
        "me",
        "moar",
        "you",
        "to",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    def __init__(self, pad_token_id=0):
        self.vocab_size = len(self.TOKENS)
        self.TOKENS.sort(key=len, reverse=True)  # Sort by length for greedy matching
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
                raise ValueError(f"Unknown token in text: '{text}'")

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
        prompt = "\n".join([m["content"] for m in messages])

        if tokenize:
            return self.encode(prompt)
        return prompt

    def __len__(self):
        return self.vocab_size


def create_nano_llama_config() -> LlamaConfig:
    """Create a tiny LlamaConfig for fast testing."""
    return LlamaConfig(
        seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_heads=8,
        num_kv_heads=8,
        num_layers=4,
        tokenizer=DummyTokenizer(),
    )


def create_nano_trainer_config(output_dir: str | Path) -> TrainerConfig:
    """Create a minimal TrainerConfig for testing."""
    return TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32"),
        train_batch_size=32,
        num_train_steps=1000,
        steps_per_eval=1,
        checkpointer=CheckpointerConfig(
            base_path=Path(output_dir) / "checkpoints",
            save_interval=datetime.timedelta(seconds=10),
        ),
        tensor_parallel_axes=["mlp", "kv_heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )


def create_nano_optimizer_config() -> AdamConfig:
    """Create a minimal AdamConfig for testing."""
    return AdamConfig(
        learning_rate=1e-2,
        weight_decay=0.00,
        warmup=0.0,
        lr_schedule="constant",
    )


def create_weight_transfer_config():
    return WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
    )


def create_test_curriculum_config(actor_name: str = "test_curriculum"):
    """Create a minimal CurriculumConfig for testing."""
    from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
    from marin.rl.environments import EnvConfig

    return CurriculumConfig(
        lessons={
            "cats": LessonConfig(
                lesson_id="cats",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
                sampling_params=SamplingParams(temperature=1.0, n_prompts=8, n_generations_per_prompt=4, max_tokens=64),
            )
        },
        eval_frequency=100,
        actor_name=actor_name,
    )


def create_nano_train_worker_config(rollout_storage: RolloutStorageConfig, output_dir: str | Path) -> TrainWorkerConfig:
    """Create a minimal TrainWorkerConfig for testing."""
    return TrainWorkerConfig(
        run_id="test-0",
        rollout_storage=rollout_storage,
        model=create_nano_llama_config(),
        trainer=create_nano_trainer_config(output_dir),
        optimizer=create_nano_optimizer_config(),
        weight_transfer=create_weight_transfer_config(),
        curriculum_config=create_test_curriculum_config(),
        tokenizer=DummyTokenizer(),
        replay_buffer=ReplayBufferConfig(
            capacity=2048,
            alpha=3.0,
            max_samples=1,
            max_rollout_delay=1,
        ),
        loss=RLOOLoss(kl_coef=0.0, clip_epsilon=5.0),
        initial_checkpoint=None,
    )


def create_test_inference_server_config(model_config: LlamaConfig, output_dir: str | Path):
    return InferenceServerConfig(
        trainer=create_nano_trainer_config(output_dir),
        tokenizer=DummyTokenizer(),
        service=InferenceEngineConfig(
            max_seqs=8, page_size=8, max_pages_per_seq=32, max_queued_tokens=8, enable_logprobs=True
        ),
        temperature=1.0,
        port=find_open_port(),
    )


def create_nano_rollout_worker_config(output_dir: str, rollout_storage: RolloutStorageConfig) -> RolloutWorkerConfig:
    """Create a minimal RolloutWorkerConfig for testing."""
    model_config = create_nano_llama_config()
    inference_server_config = create_test_inference_server_config(model_config, output_dir)

    return RolloutWorkerConfig(
        run_id="test-0",
        trainer=create_nano_trainer_config(output_dir),
        inference_server_config=inference_server_config,
        model=model_config,
        curriculum_config=create_test_curriculum_config(),
        rollout_storage=rollout_storage,
        tokenizer=DummyTokenizer(),
        log_freq=1,
        max_rollouts=1000,
        weight_transfer=create_weight_transfer_config(),
        initial_checkpoint=None,
    )


def create_test_rollout_storage_config() -> RolloutStorageConfig:
    """Create in-memory storage config for testing."""
    from marin.rl.rollout_storage import StorageType

    test_id = uuid.uuid4().hex[:8]
    return RolloutStorageConfig(storage_type=StorageType.IN_MEMORY, queue_name=f"test_{test_id}")


@jax.jit
def compute_model_logprobs(
    model,
    input_tokens: np.ndarray,
    input_attention_mask: np.ndarray,
    target_tokens: np.ndarray,
    target_attention_mask: np.ndarray,
) -> np.ndarray:
    """Compute log probabilities for target tokens.

    N.B. This does not compute full log-probs, just the log-probs of the target
    tokens themselves.
    """
    # Concatenate input and target tokens
    full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
    full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)
    full_position_ids = jnp.maximum(jnp.cumsum(full_attention_mask, axis=1) - 1, 0)

    # Convert to Haliax named arrays
    Batch = hax.Axis("batch", full_tokens.shape[0])
    SeqLen = hax.Axis("position", full_tokens.shape[1])

    tokens_named = hax.named(full_tokens, (Batch, SeqLen))
    attn_mask_named = hax.named(full_attention_mask.astype(jnp.bool_), (Batch, SeqLen))
    position_ids_named = hax.named(full_position_ids, (Batch, SeqLen))

    # Compute logits using the model
    input_tokens_named = tokens_named[SeqLen, hax.ds(0, SeqLen.size - 1)]
    input_mask_named = attn_mask_named[SeqLen, hax.ds(0, SeqLen.size - 1)]
    input_position_ids_named = position_ids_named[SeqLen, hax.ds(0, SeqLen.size - 1)]

    logits = model(input_tokens_named, attn_mask=input_mask_named, pos_ids=input_position_ids_named)

    # Extract logits corresponding to target positions
    logits_array = logits.array[:, input_tokens.shape[1] - 1 :]

    logprobs = -softmax_cross_entropy_with_integer_labels(
        logits_array.astype(jnp.float32), target_tokens.astype(jnp.int32)
    )
    return logprobs


def encode_prompt_and_response(
    prompt: str,
    response: str,
    tokenizer=None,
    max_input_length: int = 32,
    max_output_length: int = 32,
    pad_token_id: int = 0,
) -> dict:
    """Encode prompt and response into the format needed for RolloutBatch."""
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


def run_inference_with_engine(
    model,
    prompts: list[str],
    tokenizer=None,
    max_tokens: int = 64,
    temperature: float = 1.0,
    enable_logprobs: bool = False,
) -> tuple[list[list[int]], list[str]]:
    """Run inference on prompts using InferenceEngine directly."""
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

        decode_params = SeqDecodingParams(
            stop_tokens=None,
            max_num_tokens=jnp.array(len(prompt_tokens) + max_tokens, dtype=jnp.int32),
            temperature=jnp.array(temperature, dtype=jnp.float32),
            key=jrandom.PRNGKey(i),
        )

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


def disable_noisy_loggers():
    """Disable verbose INFO logging from inference engine and HTTP clients."""
    noisy_loggers = [
        "levanter.inference.engine",
        "levanter.inference.openai",
        "httpx",
        "uvicorn.access",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class ThreadedWorkerRunner(ABC):
    """Base class for managing workers in separate threads with error handling."""

    def __init__(self, config):
        self.config = config

        # State tracking
        self.worker = None
        self.thread = None
        self.error = None
        self.done = threading.Event()

    @abstractmethod
    def _create_and_run_worker(self):
        """Create and run the worker. Must be implemented by subclasses."""
        pass

    def _run(self):
        """Thread target - runs the worker with error handling."""
        try:
            self._create_and_run_worker()
        except Exception as e:
            print(f"{self.__class__.__name__} encountered exception:", e, file=sys.stderr)
            logger.error(f"{self.__class__.__name__} failed", exc_info=True)
            self.error = e
        finally:
            self.done.set()

    def start(self):
        """Start worker in background thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop worker if running."""
        if self.worker:
            self.worker.stop()

    def alive(self):
        return self.thread.is_alive() if self.thread else False

    def join(self, timeout=5):
        """Wait for thread completion."""
        if self.thread:
            self.thread.join(timeout)

    def __enter__(self):
        """Context manager entry - start the worker."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop worker and check for errors."""
        self.stop()
        self.join(timeout=5)

        if self.error:
            import traceback

            print(f"{self.__class__.__name__} error: {self.error}")
            print(
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(self.error),
                        self.error,
                        self.error.__traceback__,
                    )
                ),
            )
            pytest.fail(f"{self.__class__.__name__} failed: {self.error}")

        return False


class RolloutWorkerRunner(ThreadedWorkerRunner):
    """Manages running an inference worker in a separate thread with metric tracking."""

    def __init__(self, rollout_worker_config):
        super().__init__(rollout_worker_config)
        self.rollout_worker_config = rollout_worker_config

        # Metrics
        self.rollouts_generated = 0
        self.weight_transfers = 0

    @classmethod
    def from_job(cls, job):
        """Create runner from RLJob."""

        _, rollout_config = job.to_worker_configs()
        return cls(rollout_config)

    def _track_rollout_generation(self):
        """Called when rollout is generated."""
        self.rollouts_generated += 1

    def _create_and_run_worker(self):
        """Create and run the rollout worker with tracking hooks."""
        disable_noisy_loggers()
        self.worker = RolloutWorker(
            config=self.rollout_worker_config,
        )

        _sync_weights_original = self.worker._sync_weights

        def sync_and_track():
            result = _sync_weights_original()
            if result:
                self.weight_transfers += 1
            return result

        self.worker._sync_weights = sync_and_track

        original_sample_batch = self.worker._sample_batch

        def counting_sample_batch(lesson_id, mode, rng):
            batch_data, metrics = original_sample_batch(lesson_id, mode=mode, rng=rng)
            if batch_data is None or metrics is None:
                return None, None
            self._track_rollout_generation()
            # Add metadata about rollout
            metrics["rollout_number"] = self.rollouts_generated
            return batch_data, metrics

        self.worker._sample_batch = counting_sample_batch

        # Run the worker normally
        self.worker.run()


class TrainWorkerRunner(ThreadedWorkerRunner):
    """Manages running a training worker in a separate thread with metric tracking."""

    def __init__(self, training_worker_config):
        super().__init__(training_worker_config)
        self.training_worker_config = training_worker_config

        self.steps_completed = 0
        self.losses = []
        self.trained_model = None
        self.reference_model = None
        self.all_steps_seen = []

    @classmethod
    def from_job(cls, job):
        """Create runner from RLJob."""

        train_config, _ = job.to_worker_configs()
        return cls(train_config)

    def _track_training_step(self):
        """Called after each training step."""
        self.steps_completed += 1

    def _create_and_run_worker(self):
        """Create and run the training worker with tracking hooks."""
        disable_noisy_loggers()
        self.worker = TrainWorker(config=self.training_worker_config)

        self.reference_model = self.trained_model = jax.device_get(self.worker.reference_model)

        # Override _configure_training_hooks to inject our tracking hooks
        original_configure_hooks = self.worker._configure_training_hooks

        def patched_configure_hooks(trainer):
            original_configure_hooks(trainer)

            def step_tracking_hook(info):
                current_step = int(info.step)
                self.all_steps_seen.append(current_step)
                self._track_training_step()
                current_loss = float(info.loss)
                self.losses.append(current_loss)

            def model_capture_hook(info):
                # Make a copy of the model on the CPU.
                self.trained_model = jax.device_get(info.state.model)

            trainer.add_hook(step_tracking_hook, every=1)
            trainer.add_hook(model_capture_hook, every=1)

        self.worker._configure_training_hooks = patched_configure_hooks
        self.worker.train()


@dataclass
class RolloutBatchFeeder:
    """Continuously generates and writes rollout batches in background thread.

    Automatically uses the runner's trained model (or reference model as fallback)
    to generate batches. Stops when runner completes.
    """

    runner: TrainWorkerRunner
    batch_generator: Callable
    queue_writer: Any
    tokenizer: Any = None

    def __post_init__(self):
        self.thread = None
        self.stop_flag = threading.Event()

    def _run(self):
        """Thread target - continuously generate batches until runner completes."""
        try:
            while not self.runner.worker and not self.runner.done.is_set():
                time.sleep(0.1)

            if not self.runner.worker:
                return

            # Generate initial batch with reference model
            model = self.runner.reference_model
            batch_size = self.runner.training_worker_config.trainer.train_batch_size

            # Continuously generate batches
            while not self.runner.done.is_set() and not self.stop_flag.is_set():
                # Use trained model if available, otherwise reference
                if self.runner.trained_model:
                    model = self.runner.trained_model

                batch = self.batch_generator(policy_model=model, batch_size=batch_size, tokenizer=self.tokenizer)
                self.queue_writer.write_batch(batch)
        except Exception:
            logger.error("RolloutBatchFeeder failed", exc_info=True)
        finally:
            logger.info("RolloutBatchFeeder exiting")

    def __enter__(self):
        """Start batch generation thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop batch generation and wait for thread."""
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=2)
        return False
