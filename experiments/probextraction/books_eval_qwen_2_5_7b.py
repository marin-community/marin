from pathlib import Path

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.utils import fsspec_glob
from experiments.models import get_model_local_path, qwen2_5_7b
from levanter.infra.ray_tpu import run_on_pod_resumable

from levanter.main.marin_eval_sliding_total import EvalSlidingTotalConfig, BookConfig, main as eval_sliding_main
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.models.qwen import QwenConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
import jmp
import ray

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


def run_levanter_eval_sliding(config: EvalSlidingTotalConfig) -> None:
    """Run Levanter's eval_sliding_total with proper TPU infrastructure like training."""
    hw_config = TpuPodConfig(tpu_type="v4-128", slice_count=1)

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
# Multi-book sliding-window likelihood evaluation (Qwen-2.5-72B)
# -----------------------------------------------------------------------------
# run with 1 book
eval_sliding_step = ExecutorStep(
    name="probextraction/qwen2.5_7b_50_book",
    fn=run_levanter_eval_sliding,
    config=EvalSlidingTotalConfig(
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
