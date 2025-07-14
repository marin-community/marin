# Sliding Logits Evaluation

This guide explains how to perform sliding logits evaluation on language models using Marin's executor framework. Sliding logits analysis helps understand how a model's predictions change as it processes text sequentially, revealing patterns in the model's internal representations and decision-making processes.

## What is Sliding Logits Evaluation?

Sliding logits evaluation involves:
1. **Sequential Processing**: The model processes text in overlapping windows
2. **Logit Extraction**: At each position, we extract the model's predicted logits
3. **Pattern Analysis**: We analyze how predictions evolve as the model sees more context
4. **Visualization**: Create heatmaps showing prediction confidence across the text

This analysis is particularly useful for:
- Understanding model attention patterns
- Identifying where models are most/least confident
- Analyzing how context affects predictions
- Debugging model behavior on specific text passages

## Prerequisites

- Ray cluster setup (run `ray dashboard infra/us-east1-dev.yaml`)
- Access to TPU/GPU resources
- Text file to analyze (uploaded to GCS)

## Step-by-Step Setup

### Step 1: Upload Your Text File to GCS

**This is the only manual step required.** You need to upload a `.txt` file to Google Cloud Storage.

```bash
# Example: Upload a text file to GCS
gsutil cp your_text_file.txt gs://marin-us-central2/documents/books_txt/
```

**Important**: 
- Use a descriptive filename (e.g., `gatsby.txt`, `shakespeare.txt`)
- Place it in a logical directory structure
- Note the full GCS path for use in the experiment

### Step 2: Add Your Model to models.py

You need to add your model configuration to `experiments/models.py`. Look at the existing examples:

```python
# Example: Adding a new model
your_model_name = download_model_step(
    ModelConfig(
        hf_repo_id="your-org/your-model-name",  # HuggingFace repository ID
        hf_revision="commit-hash",               # Specific commit hash
    )
)
```

**To find the correct model name and revision:**
1. Go to the model's HuggingFace page
2. Copy the repository ID (e.g., `meta-llama/Llama-3.1-70B`)
3. Find the specific commit hash you want to use
4. Add the configuration following the pattern above

**Example configurations:**
```python
# 7B model example
llama_3_1_8b = download_model_step(
    ModelConfig(
        hf_repo_id="meta-llama/Llama-3.1-8B",
        hf_revision="d04e592",
    )
)

# 70B model example  
llama_3_1_70b = download_model_step(
    ModelConfig(
        hf_repo_id="meta-llama/Llama-3.1-70B",
        hf_revision="d4cd2f9",
    )
)
```

### Step 3: Configure Your Experiment

Import your model and modify the experiment configuration. The setup differs based on model size:

#### For Large Models (70B+): Use Tensor Parallel

```python
from experiments.models import get_model_local_path, your_model_name

sliding_logits_tp_step = ExecutorStep(
    name="extraction/sliding-forward-logits-tp_70b",
    description="Run tensor-parallel sliding-window LM forward pass",
    fn=compute_sliding_logits_tp_remote,
    config=SlidingLogitsTPConfig(
        model_name=get_model_local_path(your_model_name),  # Import your model
        input_path="gs://marin-us-central2/documents/books_txt/your_file.txt",  # Your text file
        output_dir=this_output_path(),
        batch_size=1,  # Fixed for tensor parallel
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT16,
        num_devices=8,  # Use all 8 TPU cores
        mesh_shape=(1, 8),  # 1 data parallel, 8 model parallel
        # ... other config
    ),
)
```

#### For Smaller Models (7B and below): Use Standard Processing

```python
from experiments.models import get_model_local_path, your_model_name

sliding_logits_step = ExecutorStep(
    name="extraction/sliding-forward-logits",
    description="Run sliding-window LM forward pass",
    fn=compute_sliding_logits_remote,
    config=SlidingLogitsConfig(
        model_name=get_model_local_path(your_model_name),  # Import your model
        input_path="gs://marin-us-central2/documents/books_txt/your_file.txt",  # Your text file
        output_dir=this_output_path(),
        batch_size=4,  # Can be larger for smaller models
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        # ... other config
    ),
)
```

### Step 4: Run the Experiment

Execute your experiment using the Ray runner:

```bash
python marin/run/ray_run.py \
    --env_vars HF_TOKEN $HF_TOKEN \
    --env_vars WANDB_API_KEY $WANDB_API_KEY \
    -- \
    python your_experiment.py --force_run_failed True
```

## Key Configuration Parameters

### Model-Specific Settings

| Parameter | Small Models (≤7B) | Large Models (70B+) |
|-----------|-------------------|-------------------|
| `batch_size` | 4-8 | 1 (fixed) |
| `num_devices` | 1-4 | 8 (all TPU cores) |
| `mesh_shape` | (1, 1) | (1, 8) |
| `precision` | FLOAT16/FLOAT32 | FLOAT16 |

### Processing Parameters

- **`chunk_size`**: Number of tokens processed in each window
- **`slice_length`**: Total length of text slice to analyze
- **`cursor_inc`**: How many tokens to advance between windows
- **`max_length`**: Maximum sequence length for processing
- **`prompt_tokens`**: Number of tokens to use as context

## Output and Analysis

The experiment produces:

1. **Logit Files**: Raw model predictions at each position
2. **Heatmap Visualization**: Character-level confidence visualization
3. **Extraction Statistics**: Summary statistics about the analysis

You can view results through:
- The experiment URL provided after execution
- Direct inspection of output files in GCS
- Generated visualization plots

## Troubleshooting

### Common Issues

1. **Model too large for memory**: Use tensor parallel configuration
2. **Text file not found**: Verify GCS path is correct
3. **Model download fails**: Check HuggingFace token and model access
4. **TPU allocation issues**: Ensure proper TPU configuration

### Performance Tips

- For large models, use tensor parallel (TP) configuration
- Adjust `batch_size` based on available memory
- Use `FLOAT16` precision for better memory efficiency
- Consider smaller `slice_length` for very long texts

## Example: Complete Experiment

Here's a complete example for analyzing a 70B model:

```python
from experiments.models import get_model_local_path, llama_3_1_70b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.sliding_logits_tp import Precision, SlidingLogitsTPConfig, compute_sliding_logits_tp_remote
from marin.generation.plot_sliding_logits import PlotSlidingLogitsConfig, create_sliding_logits_plot

# Step 1: Extract sliding logits
sliding_logits_tp_step = ExecutorStep(
    name="extraction/sliding-forward-logits-tp_70b",
    fn=compute_sliding_logits_tp_remote,
    config=SlidingLogitsTPConfig(
        model_name=get_model_local_path(llama_3_1_70b),
        input_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_dir=this_output_path(),
        batch_size=1,
        chunk_size=100,
        slice_length=2000,
        cursor_inc=10,
        max_length=100,
        prompt_tokens=50,
        precision=Precision.FLOAT16,
        num_devices=8,
        mesh_shape=(1, 8),
    ),
)

# Step 2: Generate visualization
plot_step = ExecutorStep(
    name="visualization/sliding-logits-plot-tp_70b",
    fn=create_sliding_logits_plot,
    config=PlotSlidingLogitsConfig(
        input_path=sliding_logits_tp_step,
        original_text_path="gs://marin-us-central2/documents/books_txt/gatsby.txt",
        output_path=this_output_path(),
        plot_title="Sliding Logits Analysis: Great Gatsby (70B)",
        colormap="Blues",
        figsize=(20, 3),
        dpi=300,
    ),
)

if __name__ == "__main__":
    executor_main([sliding_logits_tp_step, plot_step])
```

This setup provides a complete pipeline for analyzing how large language models process and predict text, revealing insights into their internal representations and decision-making processes. 

 python marin/run/ray_run.py  --env_vars XLA_USE_F16 1 --env_vars XLA_USE_BF16 0 --env_vars XLA_DOWNCAST_BF16 0  --env_vars HF_TOKEN $HF_TOKEN   --env_vars WANDB_API_KEY $WANDB_API_KEY  -- python experiments/tutorials/exp1353_sliding_logits_tp_70b.py --force_run_failed True 
