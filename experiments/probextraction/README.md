# Probability Extraction Experiments

This directory contains experiments for evaluating memorization in large language models using sliding-window probability extraction and related data processing tools.

## Table of Contents
- [Setup](#setup)
- [Experiment Categories](#experiment-categories)
  - [Book Evaluation Scripts](#book-evaluation-scripts)
  - [Data Preparation Scripts](#data-preparation-scripts)
  - [Training Scripts](#training-scripts)
- [Running Experiments](#running-experiments)
- [Common Parameters](#common-parameters)

---

## Setup

### Prerequisites
1. **Installation**: Follow the [installation guide](../../docs/tutorials/installation.md) to set up Marin
2. **Co-develop with Levanter**: You must co-develop with Levanter to use these experiments:
   - Follow the [co-development guide](../../docs/tutorials/co-develop.md)
   - **Important**: Check out the `memorize` branch of Levanter:
     ```bash
     cd marin
     mkdir -p submodules
     cd ../levanter
     git worktree add ../marin/submodules/levanter memorize
     cd ../marin
     uv pip install -e submodules/levanter
     ```
3. **Ray cluster**: Must be running on `infra/marin-us-central2-memorize.yaml`
4. **GCP credentials**: Ensure you have access to `gs://marin-us-central2/` bucket

### Access Ray Dashboard
```bash
# View Ray dashboard (monitoring, logs, resource usage)
uv run scripts/ray/cluster.py dashboard
```

### Running on the Cluster
Run experiments directly from your local machine (no need to attach):
```bash
# General pattern
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/<script_name>.py

# Example: Evaluate Qwen 2.5 7B on 50 books
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/books_eval_qwen_2_5_7b.py

# Force re-run failed steps
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/books_eval_qwen_2_5_7b.py --force_run_failed True
```

---

## Experiment Categories

### Book Evaluation Scripts

These scripts evaluate pre-trained models on a corpus of 50 books to measure memorization using sliding-window probability extraction. Each script evaluates a different model architecture.

**Scripts:**
- `books_eval_llama_7b.py` - Original LLaMA 7B
- `books_eval_llama_13b.py` - Original LLaMA 13B
- `books_eval_llama_2_7b.py` - LLaMA 2 7B
- `books_eval_llama_30b.py` - Original LLaMA 30B
- `books_eval_llama_65b.py` - Original LLaMA 65B
- `books_eval_llama_3_70b.py` - LLaMA 3 70B
- `books_eval_llama_3_1_8b.py` - LLaMA 3.1 8B
- `books_eval_llama_3_1_70b.py` - LLaMA 3.1 70B
- `books_eval_qwen_2_5_7b.py` - Qwen 2.5 7B
- `books_eval_qwen_2_5_72b.py` - Qwen 2.5 72B
- `books_eval_qwen3_32b.py` - Qwen 3 32B

#### How They Work

These scripts use Levanter's `marin_eval_sliding_total` to:
1. Load a pre-trained model from HuggingFace format
2. Tokenize each book using a sliding window approach
3. Compute sequence log probabilities for each window
4. Generate visualizations and metrics for memorization detection

**Key Configuration Parameters:**
- `chunk_size`: Size of sliding window (default: 100 tokens)
- `prompt_tokens`: Number of tokens used as prompt context (default: 50)
- `cursor_inc_tokens`: How many tokens to slide the window (default: 5)
- `eval_batch_size`: Batch size for evaluation (model-dependent, typically 256-512)
- `token_mode`: Use token-based sliding (True) vs character-based (False)

**Output:**
- Per-book probability histograms saved to GCS
- WandB logging of evaluation metrics
- Success markers for completed books (enables resume/skip)

#### Example: Running LLaMA 7B Evaluation

```bash
# Run evaluation
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/books_eval_llama_7b.py
```

**Expected Output Location:**
```
gs://marin-us-central2/probextraction/llama_7b_50_books_eval-<hash>/
├── <book_name_1>/
│   ├── bar_plot_max_pz_<book_name_1>.png
│   ├── pz_distribution_histogram_<book_name_1>.png
│   └── pz_data_<book_name_1>.npz
├── <book_name_2>/
│   └── ...
└── <book_name_1>.success  # Completion marker
```

#### Two Execution Patterns

**Pattern 1: Single ExecutorStep (e.g., `books_eval_llama_7b.py`)**
- Creates one ExecutorStep that evaluates all books sequentially
- Simpler but slower (books processed one-by-one)
- Uses `run_levanter_eval_sliding()` function directly

**Pattern 2: Parallel ExecutorSteps (e.g., `books_eval_qwen_2_5_7b.py`)**
- Creates one ExecutorStep per book for parallelism
- Faster (Ray can schedule multiple books concurrently)
- Uses `make_run_eval_sliding_fn()` from `utils.py`

---

### Data Preparation Scripts

These scripts prepare book data for training or analysis.

#### `filter_books_by_text.py`
Filters a single book shard by searching for a substring.

**Configuration:**
```python
SUBSTRING = "Path-relinking is introduced"  # Search phrase
CASE_SENSITIVE = False
```

**Usage:**
```bash
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/filter_books_by_text.py
```

#### `filter_books_by_text_parallel.py`
Filters multiple book shards in parallel from a GCS directory.

**Configuration:**
```python
SUBSTRING = "They were careless people..."  # Search phrase
input_path = "gs://marin-us-central2/raw/books3/"  # Directory with shards
```

**Usage:**
```bash
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/filter_books_by_text_parallel.py
```

**Output:** Filtered books saved to `<experiment>/filtered.jsonl.gz`

---

### Training Scripts

#### `exp1081_book_to_sft.py`
Converts a single book into SFT (Supervised Fine-Tuning) format using character-based sliding windows.

**Configuration:**
```python
INPUT_BOOK_FILE = "gs://marin-us-central2/documents/books/great_gatsby-b71c3c/matches.jsonl.gz"
ROW_INDEX = 0  # Which row in the JSONL contains the book
WINDOW_SIZE = 500  # Characters per window
STEP_SIZE = 10  # Sliding step size
SPLIT_RATIO = 0.4  # 40% prompt, 60% response
```

**Usage:**
```bash
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/exp1081_book_to_sft.py
```

#### `exp1090_sliding_window_sft.py`
Converts a book into SFT format using token-based sliding windows.

**Configuration:**
```python
tokenizer_name = "meta-llama/Llama-3.1-8B"
prompt_tokens = 50
response_tokens = 50
slice_length = 2000  # Max context for tokenization
cursor_inc = 10  # Token increment for sliding
```

**Usage:**
```bash
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/exp1090_sliding_window_sft.py
```

---

## Running Experiments

### Step 1: Run Experiment
```bash
# General pattern
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/<script_name>.py

# Example: Evaluate Qwen 2.5 7B on 50 books
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/books_eval_qwen_2_5_7b.py

# Force re-run failed steps
uv run python src/marin/run/ray_run.py --cluster infra/marin-us-central2-memorize.yaml -- python experiments/probextraction/books_eval_qwen_2_5_7b.py --force_run_failed True
```

### Step 2: Monitor Progress
- **Ray Dashboard**: `uv run scripts/ray/cluster.py dashboard` - Monitor TPU utilization and task progress
- **WandB**: Check the project `marin` for real-time metrics
- **GCS**: View output files at `gs://marin-us-central2/probextraction/<experiment_name>/`

### Step 3: Resume Interrupted Runs
All book evaluation scripts support automatic resume via `.success` files. If a run is interrupted, simply re-run the same command - completed books will be skipped automatically.

---

## Common Parameters

### Evaluation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 100 | Size of sliding window in tokens |
| `prompt_tokens` | 50 | Number of prompt tokens (rest are evaluated) |
| `cursor_inc_tokens` | 5 | Tokens to slide window per step |
| `eval_batch_size` | 256-512 | Batch size (depends on model size and TPU) |
| `token_mode` | True | Use token-based (True) or char-based (False) sliding |
| `gcp_log` | True | Save artifacts to GCS instead of WandB |

### TPU Configuration
- **Default TPU type**: `v4-64` (configurable in `utils.py:make_run_eval_sliding_fn()`)
- **Slice count**: 1 (single TPU pod) or 2+ for larger models
- **Batch size scaling**:
  - v4-64: 256 (default for most models)
  - v4-128: 512 (larger models like 13B+)

### Hardware Selection Helper
The `utils.py:choose_hw_and_batch()` function automatically selects TPU type based on model size:
- ≤8B params → v4-64, batch_size=256
- ≤15B params → v4-128, batch_size=512
- ≤35B params → v4-128, batch_size=512
- >35B params → v4-256, batch_size=256

Override via environment variable:
```bash
export TPU_TYPE_OVERRIDE=v4-128
```

---

## Utilities (`utils.py`)

### Functions
- `list_books(gcp_path)` - List all `.txt` files in GCS directory or single file
- `run_eval_sliding_on_tpu(config, tpu_type, slice_count)` - Run Levanter eval on TPU
- `make_run_eval_sliding_fn(tpu_type, slice_count)` - Curry TPU params for ExecutorStep
- `run_eval_pz_on_tpu(config, tpu_type, slice_count)` - Run P(z) evaluation on TPU
- `make_run_eval_pz_fn(tpu_type, slice_count)` - Curry TPU params for P(z) eval
- `choose_hw_and_batch(params_b)` - Auto-select TPU type and batch size

---

## Troubleshooting

### "No chunks generated" Warning
- **Cause**: Book is too short for sliding window parameters
- **Fix**: Reduce `chunk_size`, `slice_length`, or `cursor_inc_tokens`

### Out of Memory (OOM)
- **Cause**: Batch size too large for TPU memory
- **Fix**: Reduce `eval_batch_size` in the script

### Ray Connection Timeout
- **Cause**: Ray cluster not running or network issues
- **Fix**: Check cluster status with `uv run scripts/ray/cluster.py dashboard`

### Model Not Found
- **Cause**: Model checkpoint not downloaded to cluster
- **Fix**: Check `experiments/models.py` for local model paths. Ensure models are synced to cluster.

---

## Notes

- All scripts use the **Marin Executor framework** for versioning, idempotence, and reproducibility
- Book evaluations are **resumable** - completed books are tracked via `.success` files
- **WandB logging** is configured for project `marin` - ensure API key is set up on cluster
- **GCP artifacts** are saved to `gs://marin-us-central2/` by default when `gcp_log=True`
- Scripts use **Levanter's TPU infrastructure** via `run_on_pod_resumable()` for automatic retry on preemption
