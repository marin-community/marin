# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marin is an open-source framework for research and development of foundation models, primarily for training language models like Llama, DeepSeek, and Qwen. The framework emphasizes reproducibility by recording every step from raw data to final model, including failed experiments.

## Core Architecture

### Executor Framework & Ray Integration
The heart of Marin is the `Executor` framework (`marin/execution/executor.py`) which provides a DAG-based system for specifying and running `ExecutorStep`s in topological order using Ray for distributed execution.

See detailed documentation in:
- [docs/explanations/executor.md](docs/explanations/executor.md) - Core mechanics and Ray integration
- [docs/tutorials/executor-101.md](docs/tutorials/executor-101.md) - Step-by-step tutorial
- [docs/explanations/experiments.md](docs/explanations/experiments.md) - Experiment organization

#### Key Concepts:

**Steps & Dependencies**: Each `ExecutorStep` consists of:
- **name**: Identifier for the function (includes versioning)
- **function**: Either regular Python function or Ray remote function (for parallelism)
- **config**: Dataclass argument; fields can reference previous steps via `output_path_of(step)`
- Data flows between steps via filesystem (not memory) for transparency and robustness

**Versioning & Idempotence**:
- Each step version = hash(name + versioned config fields + dependency versions)
- Steps that have already run are automatically skipped (idempotent execution)
- Output paths include version hash (e.g., `documents/fineweb-resiliparse-8c2f3a`)

**Ray Integration Under the Hood**:
- Step functions can be decorated with `@ray.remote` for distributed execution
- When you pass a Ray remote function to an `ExecutorStep`, it returns a Ray `ObjectRef`
- The `Executor.run()` method collects all ObjectRefs and calls `ray.get(list(self.refs.values()))` at line 584
- Ray packages local code and ships to appropriate cluster machines
- Each step can specify `pip_dependency_groups` for isolated environments
- Launch via `python marin/run/ray_run.py -- python experiments/script.py`
- **Important**: Never call `ray.get()` in your step functions - let the executor framework handle it!

### Key Components

- **`marin/core/`**: Core runtime and data handling
- **`marin/processing/`**: Data processing pipelines (tokenization, classification, deduplication)
- **`marin/training/`**: Model training infrastructure (adapters around Levanter’s `train_lm.py` via `run_levanter_train_lm` for CPU/GPU/TPU/Ray and TPU‑pod training, with checkpointing and W&B integration)
- **`marin/evaluation/`**: Model evaluation using various harnesses (lm-eval, HELM, Alpaca)
- **`experiments/`**: Experiment definitions and configurations
- **`infra/`**: Infrastructure configuration files for GCP/TPU clusters

### Experiment Structure
Experiments are Python scripts that define `ExecutorStep`s and their dependencies. Two main patterns:

**Simple Pattern** (using defaults):
```python
# 1. Tokenize dataset
tokenized_data = default_tokenize(name="dataset", dataset="hf_id", tokenizer=tokenizer)

# 2. Train model
model = default_train(name="model", tokenized=tokenized_data, model_config=config)

# 3. Execute
executor_main(steps=[model])
```

**Explicit ExecutorStep Pattern** (see [docs/tutorials/executor-101.md](docs/tutorials/executor-101.md)):
```python
# Define steps explicitly
data_step = ExecutorStep(
    name="hello_world/data",
    fn=generate_data,  # Can be @ray.remote function
    config=GenerateDataConfig(n=100, output_path=this_output_path()),
)

stats_step = ExecutorStep(
    name="hello_world/stats",
    fn=compute_stats,
    config=ComputeStatsConfig(
        input_path=output_path_of(data_step),  # Reference previous step
        output_path=this_output_path()
    ),
)

executor_main(steps=[stats_step])
```

## Development Commands

### Environment Setup
```bash
make init                    # Initialize development environment
uv sync                      # Install all dependencies
```

### Code Quality
```bash
make check                   # Run linting without changing files (ruff, black, mypy)
make autoformat             # Auto-format code with ruff and black
make lint                   # Run pre-commit hooks on all files
```

### Testing
```bash
make test                   # Run all tests with pytest (parallel execution)
pytest tests/path/to/test.py # Run specific test file
RAY_ADDRESS= PYTHONPATH=tests:. pytest tests/specific_test.py -v # Run single test with verbose output
```

### Build
```bash
make cluster_docker_build   # Build Docker image for cluster deployment
make cluster_docker_push    # Push Docker images to GCP Artifact Registry
```

## Project Structure

- **`experiments/`**: Contains all experiment definitions, organized by theme (e.g., `tootsie/`, `dclm/`, `speedrun/`, `probextraction/`)
- **`marin/`**: Core framework code
- **`docs/`**: Documentation (tutorials, explanations, reports)
- **`tests/`**: Test suite including integration tests and snapshots
- **`infra/`**: Infrastructure configurations for GCP clusters
- **`data_browser/`**: React app for browsing experimental data

## Configuration Files

- **`pyproject.toml`**: Python project configuration with dependency groups (test, lint, docs, gcp, etc.)
- **`Makefile`**: Development commands and Docker build automation
- **`uv.lock`**: Dependency lock file for reproducible environments

## Testing Approach

The project uses pytest with:
- Parallel test execution (`-n 4`)
- Snapshot testing for data transformations
- Integration tests that require actual model/data processing
- Coverage reporting with `pytest-cov`

## Key Patterns

1. **Experiment Scripts**: Located in `experiments/`, define step dependencies and execute with `executor_main()`
2. **Default Configurations**: Use patterns like `default_tokenize()`, `default_train()` for common operations
3. **Resource Management**: Different resource configs for CPU, GPU, TPU deployments
4. **Data Processing**: Modular pipeline for crawling, filtering, tokenizing, and training

## Submodule Editing Guidelines
- **submodules/levanter/** code should only be edited for debugging purposes (adding debug prints, etc.)
- **Production fixes** should be implemented in the main Marin codebase, not in submodules
- **Ray configuration fixes** for Levanter integration should use helper functions from `marin.training.training` in the main codebase
- Submodule changes are temporary and won't exist in production - handle compatibility issues in Marin code

## Levanter Integration Patterns

### CRITICAL: TPU Infrastructure for Levanter Scripts
When creating new experiments that call Levanter functions (especially those that use `trainer_config.initialize()`), you **MUST** use proper TPU infrastructure. **Never use CPU-only Ray workers** for TPU-dependent operations.

#### ✅ Correct Pattern (follow exp808):
```python
from marin.resources import TpuPodConfig
from levanter.infra.ray_tpu import run_on_pod_resumable

def run_levanter_function(config: SomeLevanterConfig) -> None:
    """Run Levanter function with proper TPU infrastructure."""
    # Use TPU configuration matching your cluster
    hw_config = TpuPodConfig(tpu_type="v4-128", slice_count=1, runtime_env={"env_vars": {}})

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)
    def levanter_task():
        levanter_main_function(config)

    return run_on_pod_resumable(levanter_task, hw_config.accelerator_descriptor(), max_retries_failure=10)

# In your ExecutorStep:
step = ExecutorStep(
    name="my_experiment",
    fn=run_levanter_function,  # Uses TPU infrastructure
    config=SomeLevanterConfig(
        trainer=TrainerConfig(
            ray=RayConfig(auto_start_cluster=False, start_workers=False),  # Use RayConfig object, not dict
            # ... other trainer config
        )
    )
)
```

#### ❌ Wrong Pattern (causes stalling):
```python
@ray.remote(num_cpus=0.1)  # CPU worker, no TPU access!
def run_levanter_function(config: SomeLevanterConfig) -> None:
    levanter_main_function(config)  # Will stall at trainer_config.initialize()
```

#### Key Requirements:
1. **TPU Resource Allocation**: Use `TpuPodConfig(tpu_type="v4-128")` matching your cluster
2. **Ray Remote with TPU**: `@ray.remote(**hw_config.as_remote_kwargs(), max_calls=1)`
3. **Pod Management**: Wrap with `run_on_pod_resumable()` for TPU slice coordination
4. **Ray Config Objects**: Always use `RayConfig(auto_start_cluster=False, start_workers=False)`, never dicts
5. **Correct Imports**: `from levanter.infra.ray_tpu import run_on_pod_resumable`

#### Common Levanter Operations Requiring TPU Infrastructure:
- `trainer_config.initialize()` (used in training and evaluation)
- Model loading and initialization
- Distributed JAX operations
- Any function that sets up device meshes

#### Reference Examples:
- `experiments/exp808_sft_mixture.py` - Shows proper TPU configuration pattern
- `marin/training/training.py:192-199` - Shows how training uses TPU infrastructure
- `experiments/probextraction/exp_eval_sliding_total_8b.py` - Fixed evaluation example
