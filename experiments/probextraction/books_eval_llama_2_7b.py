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

from dataclasses import replace

import jmp
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from experiments.models import get_model_local_path, llama2_7b
from levanter.distributed import RayConfig
from levanter.main.eval_pz import PzEvalConfig
from levanter.models.llama import LlamaConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.probextraction.utils import list_books, make_run_eval_pz_fn, choose_hw_and_batch

"""P(z) evaluation over many books (Llama-2-7B), one step per book for parallelism.

Uses a hardware preset helper (in utils) that maps approximate model parameter
sizes to TPU hardware and evaluation batch sizes. You can override the hardware
via the TPU_TYPE_OVERRIDE environment variable (e.g., "v4-64").
"""


# Create Llama 2 7B config based on HF config parameters
llama2_7b_config = LlamaConfig(
    seq_len=101,  # Override for evaluation sliding window
    hidden_dim=4096,
    intermediate_dim=11008,
    num_heads=32,
    num_kv_heads=32,  # MHA for Llama 2
    num_layers=32,
    activation_function="silu",
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=False,
    use_bias=False,
    use_layer_norm_weight=True,
    reference_checkpoint="meta-llama/Llama-2-7b-hf",
)


# Base config for single-book P(z) evaluation
# Select hardware + batch size based on model size (~7B for Llama-2-7B)
_TPU_TYPE, _EVAL_BATCH = choose_hw_and_batch(7.0)

_TPU_TYPE = "v4-128"  # override for now, since v4-64 seems unavailable
base_pz_config = PzEvalConfig(
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    model=llama2_7b_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(project="marin", name="llama2_7b_pz_eval"),
        mp=jmp.get_policy("p=f32,c=f32"),
        per_device_eval_parallelism=-1,
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
    ),
    initialize_from_hf=get_model_local_path(llama2_7b),
    use_hf_model_config=False,
    # if you change the below, make sure to update seq_len above!
    chunk_size=100,
    slice_length=2000,
    prompt_tokens=50,
    cursor_inc_chars=10,
    token_mode=True,
    cursor_inc_tokens=5,
    eval_batch_size=256,  # e.g., 512 on v4-128; 256 on v4-64
    output_base_path=this_output_path(),
    gcp_log=True,  # Save plots and data via fsspec
)


# -----------------------------------------------------------------------------
# Parallel single-book P(z) evaluation over many books (Llama-2-7B)
# -----------------------------------------------------------------------------
# Build one ExecutorStep per book to let Executor manage concurrency and waiting.
# BOOKS_PATH = "gs://marin-us-central2/documents/books/50_books/"
# BOOKS_PATH = "gs://marin-us-central2/books_evals/2_books/"  # debug
BOOKS_PATH = (
    "gs://marin-us-central2/documents/books/50_books/harry_potter_and_the_sorcerer_s_stone.txt"  # debug single file
)

book_steps: list[ExecutorStep[PzEvalConfig]] = []
for book_title, txt_path in list_books(BOOKS_PATH):
    # Make W&B run metadata depend on the book. This creates a distinct run per book
    # with a descriptive name and group, and helpful tags for filtering.
    base_tracker = base_pz_config.trainer.tracker
    book_tracker = replace(
        base_tracker,
        name=f"llama2_7b_pz_{book_title}",
        group="llama2_7b_pz_books",
        tags=(*getattr(base_tracker, "tags", []), "pz", "llama2_7b", f"book:{book_title}"),
    )
    book_trainer = replace(base_pz_config.trainer, tracker=book_tracker)

    per_book_cfg = replace(
        base_pz_config,
        book_title=book_title,
        txt_path=txt_path,
        trainer=book_trainer,
    )
    step = ExecutorStep(
        name=f"probextraction_llama2_7b_pz_{book_title}_v2",
        fn=make_run_eval_pz_fn(tpu_type=_TPU_TYPE),
        config=per_book_cfg,
    )
    book_steps.append(step)


if __name__ == "__main__":
    executor_main(steps=book_steps)
