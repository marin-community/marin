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
from experiments.models import get_model_local_path, llama_3_1_70b

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from experiments.llama import llama_8b
import jmp

from experiments.probextraction.utils import list_books, make_run_eval_sliding_fn

# -----------------------------------------------------------------------------
# Parallel single-book sliding-window evaluation (Llama-3.1-70B)
# -----------------------------------------------------------------------------

STEP_PREFIX = "probextraction/llama3.1_70b_50_books"

base_config = EvalSlidingTotalConfig(
    tokenizer_name="meta-llama/Llama-3.1-8B",
    model=dataclasses.replace(llama_8b, seq_len=101),  # Use standard llama_8b config, override seq_len
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="marin",
            name="llama_3.1_70b_50_books",
        ),
        mp=jmp.get_policy("p=f32,c=f32"),
        per_device_eval_parallelism=-1,
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
    ),
    initialize_from_hf=get_model_local_path(llama_3_1_70b),
    use_hf_model_config=False,
    # if you change the below, make sure to update seq_len above!
    chunk_size=100,
    slice_length=2000,
    prompt_tokens=50,
    cursor_inc_chars=10,
    token_mode=True,
    cursor_inc_tokens=5,
    eval_batch_size=128,  # max batch size is 128 for TPU v4-128 / v6e-64
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
        fn=make_run_eval_sliding_fn(),
        config=per_book_cfg,
    )
    book_steps.append(step)


if __name__ == "__main__":
    executor_main(steps=book_steps)
