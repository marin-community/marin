import dataclasses
from pathlib import Path

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.utils import fsspec_glob
from experiments.models import get_model_local_path, llama_3_1_8b
from levanter.infra.ray_tpu import run_on_pod_resumable

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig, main as eval_sliding_main
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from experiments.llama import llama_8b
import jmp
import ray


def run_levanter_eval_sliding(config: EvalSlidingTotalConfig) -> None:
    """Run Levanter's eval_sliding_total with proper TPU infrastructure like training."""
    hw_config = TpuPodConfig(tpu_type="v4-128", slice_count=1, runtime_env={"env_vars": {}})

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def eval_lm_task():
        eval_sliding_main(config)

    return run_on_pod_resumable(eval_lm_task, hw_config.accelerator_descriptor(), max_retries_failure=10)


def create_books_from_gcp_directory(gcp_dir: str) -> dict[str, BookConfig]:
    """Create BookConfig instances for all .txt files in a GCP directory.

    Args:
        gcp_dir: GCP directory path like 'gs://marin-us-central2/books_evals/50_books/'

    Returns:
        Dict mapping book IDs to BookConfig instances
    """
    # Get all .txt files from the GCP directory
    txt_files = fsspec_glob(f"{gcp_dir.rstrip('/')}/*.txt")

    books = {}
    for txt_path in txt_files:
        # Extract book name from path and create a clean book ID
        filename = Path(txt_path).stem  # removes .txt extension
        book_id = filename.lower()  # Use as-is (already has underscores)
        book_title = filename  # Keep underscores for WandB artifact compatibility

        books[book_id] = BookConfig(txt_path=txt_path, book_title=book_title)

    return books


# -----------------------------------------------------------------------------
# Multi-book sliding-window likelihood evaluation (Llama-3.1-8B)
# -----------------------------------------------------------------------------
eval_sliding_step = ExecutorStep(
    name="probextraction/llama3.1_8b_2_books",
    fn=run_levanter_eval_sliding,
    config=EvalSlidingTotalConfig(
        tokenizer_name="meta-llama/Llama-3.1-8B",
        model=dataclasses.replace(llama_8b, seq_len=101),  # Use standard llama_8b config, override seq_len
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="marin",
                name="llama_3.1_8b_2_books",
            ),
            mp=jmp.get_policy("p=f32,c=f32"),
            per_device_eval_parallelism=-1,
            tensor_parallel_axes=["mlp", "heads"],
            fsdp_axis="embed",
            batch_axis="batch",
            ray=RayConfig(auto_start_cluster=False, start_workers=False),
        ),
        initialize_from_hf=get_model_local_path(llama_3_1_8b),
        use_hf_model_config=False,
        # if you change the below, make sure to update seq_len above!
        chunk_size=100,
        slice_length=2000,
        prompt_tokens=50,
        cursor_inc_chars=10,
        token_mode=True,
        cursor_inc_tokens=5,
        eval_batch_size=512,  # max batch size is 512 for TPU v4-128
        output_base_path=this_output_path(),
        gcp_log=True,  # Save plots and data to GCP instead of WandB artifacts
        books=create_books_from_gcp_directory("gs://marin-us-central2/books_evals/2_books/"),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[eval_sliding_step])
