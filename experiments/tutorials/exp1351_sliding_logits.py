"""
python marin/run/ray_run.py \
    --env_vars HF_TOKEN $HF_TOKEN \
    --env_vars WANDB_API_KEY $WANDB_API_KEY \
    --pip_deps '--find-links https://storage.googleapis.com/libtpu-releases/index.html,\
                --find-links https://storage.googleapis.com/libtpu-wheels/index.html,\
                torch~=2.6.0,torch_xla[tpu]~=2.6.0,transformers~=4.53.0,matplotlib' \
    -- \
    python experiments/tutorials/exp1351_sliding_logits.py --force_run_failed True
"""
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.sliding_logits import Precision, SlidingLogitsConfig, compute_sliding_logits_remote

# -----------------------------------------------------------------------------
# Single-step experiment: sliding-window forward pass + plot generation
# -----------------------------------------------------------------------------

sliding_logits_step = ExecutorStep(
    name="extraction/sliding-forward-logits_v2_batch32_compress_batchesper100",
    description="Run sliding-window LM forward pass over a text file, store logits + generate heat-map.",
    fn=compute_sliding_logits_remote,
    config=SlidingLogitsConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir=this_output_path(),  # executor will create a hashed directory
        batch_size=32,  # smaller batch to reduce host-transfer memory with full-vocab logits
        memory_gb=32,
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT16,
        num_devices=4,
        uncompress=False,
        batches_per_save=50,
    ),
)

if __name__ == "__main__":
    executor_main([sliding_logits_step]) 