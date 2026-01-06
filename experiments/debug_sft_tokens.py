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

"""
Direct SFT debugging script - bypasses Ray to allow ipdb debugging.

Run with:
    uv pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --python /opt/marin/marin/.venv/bin/python --prerelease=allow
    LD_LIBRARY_PATH=/opt/marin/marin/.venv/lib/python3.11/site-packages/libtpu:$LD_LIBRARY_PATH JAX_PLATFORMS=tpu,cpu uv run python experiments/debug_sft_tokens.py

This script lets you set breakpoints to inspect tokens during training.
Output is saved to debug_sft_logs.txt
"""
import dataclasses
import gc
import logging
import sys
from dataclasses import dataclass, field

import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter.data.text import UrlDatasetSourceConfig, LMMixtureDatasetConfig, ChatLmDatasetFormat
from levanter.models.lm_model import LmExample, compute_next_token_loss
from levanter.optim import AdamConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.checkpoint import CheckpointerConfig
from datetime import timedelta

# Import model config from your experiment
from experiments.qwen3 import qwen2_5_7b_instruct, qwen2_5_7b_instruct_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Edit these values as needed
# =============================================================================
TOKENIZED_DATA_PATH = "gs://marin-us-central2/tokenized/openthoughts3_qwen2_5_7b_instruct_tokenizer-0905ba"
# Checkpoint from exp2199b_redo3_sft_pt2 (step 11718)
HF_CHECKPOINT = "gs://marin-us-central2/checkpoints/exp2199b_redo3_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-c05011/hf/step-11718/"
DEBUG_BATCH_SIZE = 4  # Small batch for debugging (training used 512 on v4-512)
DEBUG_NUM_STEPS = 20   # Just a few steps to inspect
MAX_SEQ_LEN = 16384
LOG_FILE = "debug_sft_logs.txt"
# NOTE: This script uses pack=True to match actual training behavior.
# =============================================================================


class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def inspect_batch(example: LmExample, tokenizer, step: int):
    """
    Inspect tokens BEFORE they enter the JIT-compiled training step.
    This runs in Python, not inside JAX tracing.
    """
    import numpy as np

    # Convert JAX arrays to numpy for inspection
    # Use np.asarray to get concrete values outside of JIT
    tokens_array = np.asarray(example.tokens.array)
    loss_weight_array = np.asarray(example.loss_weight.array)

    # Get first example in batch for debugging
    if tokens_array.ndim > 1:
        token_ids = tokens_array[0].tolist()
        loss_weights = loss_weight_array[0].tolist()
    else:
        token_ids = tokens_array.tolist()
        loss_weights = loss_weight_array.tolist()

    decoded_text = tokenizer.decode(token_ids)

    # Find where loss_weight transitions (shows input vs target split)
    transition_idx = None
    for i, w in enumerate(loss_weights):
        if w > 0:
            transition_idx = i
            break

    num_with_loss = sum(1 for w in loss_weights if w > 0)

    print("\n" + "=" * 100)
    print(f"*** TOKEN INSPECTION (Step {step})")
    print("=" * 100)
    print(f"Batch shape: {tokens_array.shape}")
    print(f"Sequence length: {len(token_ids)}")
    print(f"Tokens with loss_weight > 0: {num_with_loss} / {len(loss_weights)}")
    print(f"First token with loss_weight > 0: index {transition_idx}")

    if transition_idx == 0:
        print("\n*** WARNING: ALL tokens have loss computed! No prompt masking! ***")
    elif transition_idx is None:
        print("\n*** WARNING: NO tokens have loss computed! ***")

    print(f"\n*** FULL DECODED TEXT")
    print(decoded_text)

    print(f"\n*** TOKEN-BY-TOKEN (all {len(token_ids)} tokens)")
    for i in range(len(token_ids)):
        tok_id = token_ids[i]
        tok_text = tokenizer.decode([tok_id])
        loss_marker = "LOSS" if loss_weights[i] > 0 else "----"
        transition_marker = " <-- TRANSITION" if i == transition_idx else ""
        print(f"[{i:5d}] {loss_marker} | {tok_id:6d} | {repr(tok_text)}{transition_marker}")

    print("=" * 100)

    # ==============================================
    # UNCOMMENT THIS LINE TO SET A BREAKPOINT:
    # import ipdb; ipdb.set_trace()
    # ==============================================


@dataclass
class DebugTrainConfig:
    """Minimal config for debugging."""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data_path: str = TOKENIZED_DATA_PATH
    batch_size: int = DEBUG_BATCH_SIZE
    num_steps: int = DEBUG_NUM_STEPS
    max_seq_len: int = MAX_SEQ_LEN
    seed: int = 42


def main():
    """
    Main function to run SFT training directly without Ray.
    """
    # Set up logging to both stdout and file
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    try:
        config = DebugTrainConfig()

        print("=" * 80)
        print("DEBUG SFT TRAINING")
        print("=" * 80)
        print(f"Log file: {LOG_FILE}")

        # Get the tokenizer - load the actual tokenizer object from the name
        from transformers import AutoTokenizer
        tokenizer_name = qwen2_5_7b_instruct_tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Model config
        model_config = dataclasses.replace(
            qwen2_5_7b_instruct,
            max_seq_len=config.max_seq_len,
        )

        # Trainer config
        trainer_config = TrainerConfig(
            num_train_steps=config.num_steps,
            train_batch_size=config.batch_size,
            seed=config.seed,
            checkpointer=CheckpointerConfig(
                base_path="/tmp/debug_checkpoints",
                save_interval=timedelta(days=365),  # Very long checkpoint interval to effectively disable saving checkpoints
            ),
        )

        # Dataset config pointing to existing pre-built cache
        # The data at TOKENIZED_DATA_PATH is already a Levanter cache
        # Cache structure: {cache_dir}/{split}/ -> so cache_dir should be the base path
        data_config = LMMixtureDatasetConfig(
            tokenizer=tokenizer_name,
            configs={
                "openthoughts3": UrlDatasetSourceConfig(
                    # Set cache_dir on the source config to load from existing cache
                    # This makes cache_path = {cache_dir}/{split} = gs://.../train
                    cache_dir=config.data_path,
                    # Use ChatLmDatasetFormat to properly use assistant_masks for loss masking
                    # pack=True matches actual training behavior (multiple conversations per sequence)
                    format=ChatLmDatasetFormat(pack=True),
                )
            },
            train_weights={"openthoughts3": 1.0},
            auto_build_caches=False,  # Don't try to build, just load existing
        )

        # Optimizer
        optimizer = AdamConfig(
            learning_rate=1e-25,
            weight_decay=0.0,
        ).build(config.num_steps)

        # Use standard loss function
        import functools
        loss_function = functools.partial(compute_next_token_loss, logsumexp_weight=0.0)

        # Initialize Levanter
        levanter.initialize(trainer_config)

        print("=" * 80)
        print("*** STARTING DEBUG SFT TRAINING")
        print("=" * 80)
        print(f"Data path: {config.data_path}")
        print(f"Batch size: {config.batch_size}")
        print(f"Num steps: {config.num_steps}")
        print(f"Max seq len: {config.max_seq_len}")
        print("=" * 80)

        with Trainer(trainer_config, optimizer, loss_function) as trainer:
            seed = trainer_config.seed
            data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

            compute_axis_mapping = trainer.compute_axis_mapping
            parameter_axis_mapping = trainer.parameter_axis_mapping

            # Get position axis
            Pos = model_config.max_Pos.resize(config.max_seq_len)

            # Round vocab for partitioning
            vocab_size = len(tokenizer)
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)

            print(f"Training with seq_len={config.max_seq_len}, vocab_size={Vocab.size}")

            # Get training dataset
            train_dataset = data_config.train_set(
                Pos,
                trainer.config.batch_schedule,
                key=data_key,
            )

            # Initialize model state
            state = trainer.initial_state(
                training_key,
                model_init=lambda: model_config.build(Vocab, key=model_key)
            )

            # Load from HF checkpoint
            print(f"*** Loading weights from HuggingFace: {HF_CHECKPOINT}")

            converter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(tokenizer=tokenizer)

            # Free memory from randomly initialized model
            state = dataclasses.replace(state, model=None)
            gc.collect()

            model = converter.load_pretrained(
                model_config.model_type,
                config=model_config,
                axis_mapping=parameter_axis_mapping,
                dtype=trainer.mp.compute_dtype,
            )
            model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
            state = dataclasses.replace(state, model=model)

            print("Model loaded successfully!")
            print("=" * 80)
            print("*** Beginning training loop - token inspection will appear below")
            print("=" * 80)

            # Create data loader
            train_loader = trainer.data_loader(train_dataset)
            train_loader = train_loader.iter_from_step(0)

            # Manual training loop with token inspection BEFORE each step
            iter_data = iter(train_loader)
            step = 0

            while step < config.num_steps:
                # Get next batch
                example = next(iter_data)

                # ==============================================
                # INSPECT TOKENS HERE (outside JIT)
                # ==============================================
                if step < 3:  # Only inspect first 3 steps
                    inspect_batch(example, tokenizer, step)

                # Run training step
                step_info = trainer.train_step(state, example)
                state = step_info.state
                step = int(step_info.step)

                print(f"Step {step}: loss = {step_info.loss:.4f}")

            print("=" * 80)
            print("*** Debug training complete!")
            print("=" * 80)
            print(f"\nOutput saved to: {LOG_FILE}")

    finally:
        # Restore stdout and close log file
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
