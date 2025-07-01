"""
python marin/run/ray_run.py \
    --env_vars HF_TOKEN $HF_TOKEN \
    --env_vars WANDB_API_KEY $WANDB_API_KEY \
    --pip_deps '--find-links https://storage.googleapis.com/libtpu-releases/index.html,\
                --find-links https://storage.googleapis.com/libtpu-wheels/index.html,\
                torch~=2.6.0,torch_xla[tpu]~=2.6.0,transformers~=4.53.0' \
    -- \
    python experiments/tutorials/exp1350_forward_logits.py --force_run_failed True
"""
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.logits import TextLogitsConfig, compute_logits

logits_step = ExecutorStep(
    name="tutorials/forward-logits",
    fn=compute_logits,
    config=TextLogitsConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_path=this_output_path(),
        batch_size=8,
        memory_gb=16,
        span_chars=4096,
    ),
)

if __name__ == "__main__":
    executor_main([logits_step])
