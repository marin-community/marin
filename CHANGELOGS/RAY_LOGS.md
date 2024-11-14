# Changelog

## 2024-11-14

Ray Cluster Update: We are restarting the cluster with only the core ray and vllm packages installed.
We are still keeping vllm as it requires significant time to build.
From now on, expect missing dependencies when running experiments.
You can now install dependencies in a fine-grained way using [ray_run.py](../marin/run/ray_run.py) or
using pip_dependency_groups argument in `ExecutorStep`

Dependency Management:

1. Use [ray_run.py](../marin/run/ray_run.py) to handle dependencies across an entire experiment.
Just add any extra packages you need with `--pip_deps`. Core dependencies (levanter, draccus, fspsec, etc.)
are automatically installed from [pyproject.toml](../pyproject.toml).
2. For step-specific dependencies, use `pip_dependency_groups` in `ExecutorStep`.
This takes a list where each item is either (1) A key from `project.optional-dependencies` dictionary in pyproject.toml
or (2) A specific pip package. Check out [quickstart.py](../experiments/quickstart.py) for an example.
