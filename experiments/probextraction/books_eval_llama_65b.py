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

from pathlib import Path

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.utils import fsspec_glob
from experiments.models import get_model_local_path, llama_65b
from levanter.infra.ray_tpu import run_on_pod_resumable

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig, main as eval_sliding_main
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
import jmp
import ray


def run_levanter_eval_sliding(config: EvalSlidingTotalConfig) -> None:
    """Run Levanter's eval_sliding_total with proper TPU infrastructure like training."""
    hw_config = TpuPodConfig(
        tpu_type="v4-128", slice_count=4, runtime_env={"env_vars": {}}
    )  # Larger model needs more resources

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def eval_lm_task():
        eval_sliding_main(config)

    return run_on_pod_resumable(eval_lm_task, hw_config.accelerator_descriptor(), max_retries_failure=10)


def create_books_from_gcp_directory(gcp_path: str) -> dict[str, BookConfig]:
    """Create BookConfig instances for all .txt files in a GCP directory or a single book file.

    Args:
        gcp_path: GCP directory path like 'gs://marin-us-central2/books_evals/50_books/'
                 or single file path like 'gs://marin-us-central2/books_evals/50_books/twilight.txt'

    Returns:
        Dict mapping book IDs to BookConfig instances
    """
    # Check if it's a single file or directory
    txt_files = [gcp_path] if gcp_path.endswith(".txt") else fsspec_glob(f"{gcp_path.rstrip('/')}/*.txt")

    books = {}
    for txt_path in txt_files:
        # Extract book name from path and create a clean book ID
        filename = Path(txt_path).stem  # removes .txt extension
        book_id = filename.lower()  # Use as-is (already has underscores)
        book_title = filename  # Keep underscores for WandB artifact compatibility

        books[book_id] = BookConfig(txt_path=txt_path, book_title=book_title)

    return books


# Create original Llama 65B config based on HF config parameters
llama_65b_config = LlamaConfig(
    seq_len=101,  # Override for evaluation sliding window
    hidden_dim=8192,  # Much larger than 7B models
    intermediate_dim=22016,  # Much larger than 7B models
    num_heads=64,  # Double the heads of 7B models
    num_kv_heads=64,  # MHA for original Llama 1
    num_layers=80,  # Much deeper than 7B models (80 vs 32)
    activation_function="silu",
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,  # 65B uses 1e-5 (same as Llama 2)
    tie_word_embeddings=False,
    use_bias=False,
    use_layer_norm_weight=True,
    reference_checkpoint="huggyllama/llama-65b",
)

# -----------------------------------------------------------------------------
# Multi-book sliding-window likelihood evaluation (Original Llama 65B)
# -----------------------------------------------------------------------------
eval_sliding_step = ExecutorStep(
    name="probextraction/llama_65b_50_books_eval",
    fn=run_levanter_eval_sliding,
    config=EvalSlidingTotalConfig(
        tokenizer_name="huggyllama/llama-65b",
        model=llama_65b_config,
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="marin",
                name="llama_65b_50_books_eval",
            ),
            mp=jmp.get_policy("p=f32,c=f32"),
            per_device_eval_parallelism=-1,
            tensor_parallel_axes=["mlp", "heads"],
            fsdp_axis="embed",
            batch_axis="batch",
            ray=RayConfig(auto_start_cluster=False, start_workers=False),
        ),
        initialize_from_hf=get_model_local_path(llama_65b),
        use_hf_model_config=False,
        # if you change the below, make sure to update seq_len above!
        chunk_size=100,
        slice_length=2000,
        prompt_tokens=50,
        cursor_inc_chars=10,
        token_mode=True,
        cursor_inc_tokens=5,
        eval_batch_size=128,  # Smaller batch size for larger model
        output_base_path=this_output_path(),
        gcp_log=True,  # Save plots and data to GCP instead of WandB artifacts
        # run with 50 books from open-weight copyright memorization paper
        books=create_books_from_gcp_directory("gs://marin-us-central2/books_evals/50_books/"),
        # run with 2 books (debugging)
        # books=create_books_from_gcp_directory("gs://marin-us-central2/books_evals/2_books/"),
        # run with 1 book (debugging)
        # books=create_books_from_gcp_directory("gs://marin-us-central2/books_evals/1_books/"),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[eval_sliding_step])
