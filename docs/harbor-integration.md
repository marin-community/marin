# Harbor Framework Integration

This document describes Marin's integration with [Harbor](https://harborframework.com), a framework for evaluating and optimizing agents in containerized environments.

## Overview

The Harbor integration enables running **any Harbor dataset** from the [Harbor registry](https://harborframework.com/registry) without custom adapters. Harbor provides 45+ benchmarks including:

- **AIME** (60 math problems from AIME 2024, 2025-I, 2025-II)
- **Terminal-Bench** (89 terminal tasks)
- **SWE-bench Verified** (500 software engineering tasks)
- **And 40+ more benchmarks** for code, reasoning, data science, and more

## Key Features

✅ **Generic Integration** - No custom adapters needed for each benchmark
✅ **45+ Datasets** - Access entire Harbor registry with one evaluator
✅ **Containerized Execution** - Tasks run in isolated Docker containers
✅ **Multi-Environment Support** - Local Docker, Daytona, E2B, Modal
✅ **Agent Flexibility** - Use Claude Code, custom agents, or build your own

## Quick Start

### Prerequisites

Harbor is an optional dependency. The `evaluate_harbor()` function automatically installs it via the `harbor` extra.

Alternatively, install manually:
```bash
cd lib/marin
uv add --optional harbor "harbor>=0.1.42"
```

### Running AIME Evaluation

```python
from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig

# Evaluate AIME@1.0 (60 math problems)
step = evaluate_harbor(
    model_name="anthropic/claude-opus-4-1",
    model_path=None,  # API model
    dataset="aime",
    version="1.0",
    max_eval_instances=5,  # Start with 5 tasks
    agent="claude-code",
    n_concurrent=2,
    env="local",
)
```

Or use the provided sanity check script:
```bash
export ANTHROPIC_API_KEY=your_key_here
python experiments/exp_harbor_aime_sanity_check.py
```

## Available Datasets

See the full list at [harborframework.com/registry](https://harborframework.com/registry).

Popular datasets include:

| Dataset | Version | Tasks | Description |
|---------|---------|-------|-------------|
| `aime` | 1.0 | 60 | Competition math problems |
| `terminal-bench` | 2.0 | 89 | Terminal/bash tasks |
| `swebench-verified` | 1.0 | 500 | Software engineering bugs |
| `ds-1000` | 6.0 | 1000 | Data science problems |
| `gpqa-diamond` | 1.0 | 198 | Graduate-level science Q&A |
| `usaco` | 2.0 | 304 | Programming competition problems |

## API Reference

### `evaluate_harbor()`

```python
def evaluate_harbor(
    model_name: str,
    model_path: str | None,
    dataset: str,
    version: str = "1.0",
    max_eval_instances: int | None = None,
    resource_config: ResourceConfig | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    generation_params: dict | None = None,
    agent: str = "claude-code",
    n_concurrent: int = 4,
    env: str = "local",
) -> ExecutorStep
```

**Parameters:**
- `model_name`: Model identifier (e.g., "anthropic/claude-opus-4-1", "qwen2.5-7b-instruct")
- `model_path`: Path to model (None for API models, GCS path for custom models)
- `dataset`: Harbor dataset name from registry
- `version`: Dataset version (default: "1.0")
- `max_eval_instances`: Limit number of tasks (None = all tasks)
- `agent`: Harbor agent type:
  - `"claude-code"` - Anthropic's Claude Code agent (default)
  - `"terminus-2"` - Harbor's reference agent
  - `"custom-vllm"` - Custom agent for vLLM models (TODO)
- `n_concurrent`: Number of parallel trials (default: 4)
- `env`: Container environment:
  - `"local"` - Local Docker (default, good for testing)
  - `"daytona"` - Daytona cloud containers (requires API key)
  - `"e2b"` - E2B containers (requires API key)
  - `"modal"` - Modal containers (requires API key)
- `wandb_tags`: Additional W&B tags
- `resource_config`: Fray resource configuration for Ray

## Examples

### Terminal-Bench (Terminal Tasks)

```python
step = evaluate_harbor(
    model_name="anthropic/claude-opus-4-1",
    model_path=None,
    dataset="terminal-bench",
    version="2.0",
    max_eval_instances=10,
    agent="claude-code",
    n_concurrent=4,
    env="local",
)
```

### SWE-bench Verified (Software Engineering)

```python
step = evaluate_harbor(
    model_name="anthropic/claude-opus-4-1",
    model_path=None,
    dataset="swebench-verified",
    version="1.0",
    max_eval_instances=50,
    agent="claude-code",
    n_concurrent=8,
    env="daytona",  # Use cloud for better performance
)
```

### Custom Model (Qwen 2.5 Instruct)

```python
# TODO: Requires implementing custom-vllm agent
step = evaluate_harbor(
    model_name="qwen2.5-7b-instruct",
    model_path="gs://marin-us-central2/models/qwen2.5-7b-instruct",
    dataset="aime",
    version="1.0",
    agent="custom-vllm",  # Not yet implemented
    n_concurrent=4,
    env="local",
)
```

## Architecture

### HarborEvaluator

The `HarborEvaluator` class in `lib/marin/src/marin/evaluation/evaluators/harbor_evaluator.py` provides the integration.

**Key design decisions:**
1. **No adapters needed** - Uses Harbor's registry system to load any dataset
2. **CLI-based execution** - Wraps `harbor run` command for reliability
3. **Standardized results** - Parses Harbor's JSON output into Marin format
4. **Optional dependency** - Harbor installed via `--extra harbor` only when needed

### Result Format

Harbor returns results with per-task rewards (0.0 to 1.0):

```json
{
  "trials": {
    "aime_60": {
      "reward": 1.0,
      "correct": true,
      "status": "success",
      "trajectory_length": 15
    },
    "aime_61": {
      "reward": 0.0,
      "correct": false,
      "status": "failed",
      "trajectory_length": 42
    }
  },
  "aggregate": {
    "total_trials": 60,
    "successful_trials": 42,
    "mean_reward": 0.70,
    "accuracy": 0.70
  }
}
```

## Limitations & TODOs

### Current Limitations

1. **API Models Only** - Currently only supports API-based models (Claude, GPT-4)
2. **Single-turn Evaluation** - Harbor executes full agent trajectories, not single completions
3. **Different from lm-eval** - Harbor tasks are agentic (multi-turn, tool use) vs lm-eval (single-turn completions)

### Planned Improvements

1. **Custom vLLM Agent** - Support for vLLM-hosted models (Qwen, Llama, etc.)
2. **Trajectory Analysis** - Rich insights from Harbor's trajectory logs
3. **RL Integration** - Use Harbor trajectories for RL training (following ARES pattern)
4. **Multi-seed Evaluation** - Run same tasks with different seeds for variance analysis

## Environment Variables

Required for different agents and environments:

- `ANTHROPIC_API_KEY` - For Claude Code agent
- `OPENAI_API_KEY` - For OpenAI-based agents
- `DAYTONA_API_KEY` - For Daytona environment
- `E2B_API_KEY` - For E2B environment
- `MODAL_API_KEY` - For Modal environment

## Troubleshooting

### "harbor: command not found"

Harbor should be auto-installed via `pip_dependency_groups=["harbor"]`. If not:
```bash
cd lib/marin
uv sync --extra harbor
```

### Docker errors in local environment

Ensure Docker daemon is running:
```bash
docker ps
```

### Slow execution with local Docker

Consider using cloud environment for better performance:
```python
env="daytona"  # Or "e2b", "modal"
```

### API key errors

Check environment variables:
```bash
echo $ANTHROPIC_API_KEY
```

## References

- [Harbor Documentation](https://harborframework.com/docs)
- [Harbor Registry](https://harborframework.com/registry)
- [Harbor GitHub](https://github.com/laude-institute/harbor)
- [ARES (Harbor RL Integration)](https://github.com/withmartian/ares)
- [Terminal-Bench 2.0 Announcement](https://www.tbench.ai/news/announcement-2-0)
