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

import dataclasses

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from experiments.models import get_model_local_path, llama_30b

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
import jmp

from experiments.probextraction.utils import list_books, make_run_eval_sliding_fn

# Create original Llama 30B config based on HF config parameters
llama_30b_config = LlamaConfig(
    seq_len=101,  # Override for evaluation sliding window
    hidden_dim=6656,  # From HF config
    intermediate_dim=17920,  # From HF config
    num_heads=52,  # From HF config
    num_kv_heads=52,  # MHA for original Llama 1
    num_layers=60,  # From HF config
    activation_function="silu",
    initializer_range=0.02,
    layer_norm_epsilon=1e-6,  # Original Llama uses 1e-6 (vs 1e-5 for Llama 2)
    tie_word_embeddings=False,
    use_bias=False,
    use_layer_norm_weight=True,
    reference_checkpoint="huggyllama/llama-30b",
)

# -----------------------------------------------------------------------------
# Parallel single-book sliding-window evaluation (Original Llama 30B)
# -----------------------------------------------------------------------------

STEP_PREFIX = "probextraction/llama_30b_50_books_eval"

base_config = EvalSlidingTotalConfig(
    tokenizer_name="huggyllama/llama-30b",
    model=llama_30b_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="marin",
            name="llama_30b_50_books_eval",
        ),
        mp=jmp.get_policy("p=f32,c=f32"),
        per_device_eval_parallelism=-1,
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
    ),
    initialize_from_hf=get_model_local_path(llama_30b),
    use_hf_model_config=False,
    # if you change the below, make sure to update seq_len above!
    chunk_size=100,
    slice_length=2000,
    prompt_tokens=50,
    cursor_inc_chars=10,
    token_mode=True,
    cursor_inc_tokens=5,
    eval_batch_size=256,  # Estimated batch size for 30B model with 2 TPU slices
    output_base_path=this_output_path(),
    gcp_log=True,  # Save plots and data to GCP instead of WandB artifacts
)

BOOKS_PATH = "gs://marin-us-central2/documents/books/50_books/"
# BOOKS_PATH = "gs://marin-us-central2/books_evals/2_books/"  # debug
# BOOKS_PATH = "gs://marin-us-central2/books_evals/1_books/harry_potter_1.txt"  # debug single file

book_steps: list[ExecutorStep] = []
for book_title, txt_path in list_books(BOOKS_PATH):
    per_book_cfg = dataclasses.replace(
        base_config,
        books={book_title.lower(): BookConfig(txt_path=txt_path, book_title=book_title)},
    )
    step = ExecutorStep(
        name=f"{STEP_PREFIX}/{book_title}",
        fn=make_run_eval_sliding_fn(slice_count=2),
        config=per_book_cfg,
    )
    book_steps.append(step)


if __name__ == "__main__":
    executor_main(steps=book_steps)
