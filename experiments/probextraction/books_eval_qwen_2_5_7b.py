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
from experiments.models import get_model_local_path, qwen2_5_7b

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.models.qwen import QwenConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
import jmp

from experiments.probextraction.utils import list_books, make_run_eval_sliding_fn

# Qwen-2.5-7B configuration based on the official HF config
qwen2_5_7b_config = QwenConfig(
    seq_len=101,  # Override for extraction eval (HF config has max_position_embeddings: 131072)
    hidden_dim=3584,  # hidden_size from HF config
    intermediate_dim=18944,  # intermediate_size from HF config
    num_heads=28,  # num_attention_heads from HF config
    num_kv_heads=4,  # num_key_value_heads from HF config (GQA)
    num_layers=28,  # num_hidden_layers from HF config
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0),  # rope_theta from HF config
    use_bias=False,  # Qwen2 mixed bias: q/k/v=True, o_proj=False, mlp=False. Majority is False
    use_sliding_window=False,  # use_sliding_window from HF config
    sliding_window=131072,  # sliding_window from HF config
    max_window_layers=28,  # max_window_layers from HF config
)


"""Qwen-2.5-7B sliding-window eval: one ExecutorStep per book for parallelism."""

STEP_PREFIX = "probextraction/qwen2.5_7b_50_book"

base_config = EvalSlidingTotalConfig(
    tokenizer_name="Qwen/Qwen2.5-7B",  # Use Qwen tokenizer
    model=qwen2_5_7b_config,  # Use Qwen-2.5-7B config
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="marin",
            name="qwen2.5_7b_50_book",
        ),
        mp=jmp.get_policy("p=f32,c=f32"),
        per_device_eval_parallelism=-1,
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
    ),
    initialize_from_hf=get_model_local_path(qwen2_5_7b),
    use_hf_model_config=True,
    # if you change the below, make sure to update seq_len above!
    chunk_size=100,
    slice_length=2000,
    prompt_tokens=50,
    cursor_inc_chars=10,
    token_mode=True,
    cursor_inc_tokens=5,
    eval_batch_size=256,  # max batch size is 256 for TPU v4-128
    output_base_path=this_output_path(),
    gcp_log=True,  # Save plots and data to GCP instead of WandB artifacts
)

BOOKS_PATH = "gs://marin-us-central2/documents/books/50_books/"

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
