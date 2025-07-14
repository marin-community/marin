"""
python marin/run/ray_run.py \
    --env_vars XLA_USE_F16 0 \
    --env_vars XLA_USE_BF16 0 \
    --env_vars XLA_DOWNCAST_BF16 0 \
    --env_vars HF_TOKEN $HF_TOKEN \
    --env_vars WANDB_API_KEY $WANDB_API_KEY \
    --pip_deps '--find-links https://storage.googleapis.com/libtpu-releases/index.html,\
                --find-links https://storage.googleapis.com/libtpu-wheels/index.html,\
                torch~=2.6.0,torch_xla[tpu]~=2.6.0,transformers~=4.53.0,matplotlib' \
    -- \
    python experiments/tutorials/exp1354_sliding_logits_tp_fp32_multi.py --force_run_failed True
"""

from experiments.models import get_model_local_path, llama_3_1_70b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.plot_sliding_logits import PlotSlidingLogitsConfig, create_sliding_logits_plot
from marin.generation.sliding_logits_tp_fp32 import (
    Precision,
    SlidingLogitsTPFP32Config,
    compute_sliding_logits_tp_fp32_remote,
)

# -----------------------------------------------------------------------------
# Step 1: Tensor-parallel sliding-window forward pass + logits extraction
# -----------------------------------------------------------------------------

sliding_logits_tp_fp32_step = ExecutorStep(
    name="extraction/sliding-forward-logits-tp_70b_fp32_multi",
    description="Run tensor-parallel sliding-window LM forward pass on a v6e-16 slice.",
    fn=compute_sliding_logits_tp_fp32_remote,
    config=SlidingLogitsTPFP32Config(
        model_name=get_model_local_path(llama_3_1_70b),
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir=this_output_path(),
        batch_size=1,
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT32,
        num_devices=16,
        mesh_shape=(1, 16),
        uncompress=True,
        batches_per_save=1,
        background_queue=True,
        num_background_writers=4,
        debug=False,
    ),
)

# -----------------------------------------------------------------------------
# Step 2: Plot generation from sliding logits results
# -----------------------------------------------------------------------------

plot_step = ExecutorStep(
    name="visualization/sliding-logits-plot-tp_70b_fp32_multi",
    description="Create heatmap from tensor-parallel FP32 sliding logits results.",
    fn=create_sliding_logits_plot,
    config=PlotSlidingLogitsConfig(
        input_path=sliding_logits_tp_fp32_step,
        original_text_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_path=this_output_path(),
        plot_title="TP FP32 Sliding Logits: Great Gatsby (70B)",
        colormap="Blues",
        figsize=(20, 3),
        dpi=300,
        save_combined_arrays=True,
        compute_extraction_stats=True,
    ),
)

if __name__ == "__main__":
    executor_main([sliding_logits_tp_fp32_step, plot_step])
