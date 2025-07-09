"""
python marin/run/ray_run.py \
    --env_vars HF_TOKEN $HF_TOKEN \
    --env_vars WANDB_API_KEY $WANDB_API_KEY \
    --pip_deps '--find-links https://storage.googleapis.com/libtpu-releases/index.html,\
                --find-links https://storage.googleapis.com/libtpu-wheels/index.html,\
                torch~=2.6.0,torch_xla[tpu]~=2.6.0,transformers~=4.53.0,matplotlib' \
    -- \
    python experiments/tutorials/exp1353_sliding_logits_tp_70b.py --force_run_failed True
"""
from experiments.models import get_model_local_path, llama_3_1_70b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.sliding_logits_tp import Precision, SlidingLogitsTPConfig, compute_sliding_logits_tp_remote
from marin.generation.plot_sliding_logits import PlotSlidingLogitsConfig, create_sliding_logits_plot

# -----------------------------------------------------------------------------
# Step 1: Tensor-parallel sliding-window forward pass + logits extraction
# -----------------------------------------------------------------------------

sliding_logits_tp_step = ExecutorStep(
    name="extraction/sliding-forward-logits-tp_70b",
    description="Run tensor-parallel sliding-window LM forward pass over a text file, store logits + generate heat-map.",
    fn=compute_sliding_logits_tp_remote,
    config=SlidingLogitsTPConfig(
        model_name=get_model_local_path(llama_3_1_70b),
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir=this_output_path(),  # executor will create a hashed directory
        batch_size=1,  # Fixed to 1 for tensor parallel
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT16,
        num_devices=8,  # All 8 TPU cores for tensor parallel
        mesh_shape=(1, 8),  # 1 data parallel, 8 model parallel
        uncompress=True,  # Use uncompressed writes for speed
        batches_per_save=1, 
        background_queue=True,
        num_background_writers=4,
        debug=False,  # Set to True for detailed logging
    ),
)

# -----------------------------------------------------------------------------
# Step 2: Plot generation from sliding logits results
# -----------------------------------------------------------------------------

plot_step = ExecutorStep(
    name="visualization/sliding-logits-plot-tp_70b",
    description="Create character-level heatmap visualization from tensor-parallel sliding logits results.",
    fn=create_sliding_logits_plot,
    config=PlotSlidingLogitsConfig(
        input_path=sliding_logits_tp_step,  # Automatically resolves to sliding_logits_tp_step's output_path
        original_text_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_path=this_output_path(),  # This step's output directory
        plot_title="Tensor-Parallel Sliding Logits: Great Gatsby Character Analysis (70B)",
        colormap="Blues",
        figsize=(20, 3),
        dpi=300,
        save_combined_arrays=True,
        compute_extraction_stats=True,
    ),
)

if __name__ == "__main__":
    executor_main([sliding_logits_tp_step, plot_step]) 