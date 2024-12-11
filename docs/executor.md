# Executor framework

Marin's executor framework manages the execution of experiments.
This document is more about the mechanics, read [this](docs/experiments.md) to
learn more about the conventions.

An **experiment** is a sequence (really, a DAG) of steps, where each **step** is
specified by the following:
- **name**: an identifier describing the function (and its version)
- **function**: a normal Python function or a [Ray](https://docs.ray.io) remote function (which enables massive parallelism)
- **config**: the single argument to the function, which is a dataclass; fields of the config can refer to previous steps.

A key decision in Marin is that data gets passed between steps by reading and writing to the filesystem.
The rationale is two-fold:
- For very large datasets, where efficiency and robustness is a concern, we give
  the steps full control over serialization and deserialization.
- It makes the intermediate state completely transparent, and one can do things
  like monitor the state while it's being created (e.g., a jsonl file).

In particular, each step associated with an **output path** where that step
writes its output (in any format).
When a step A references another step B in its config, that step simply resolves to step B's output path,
and step A is responsible for reading the data from that output path.
The name of the output path includes the step name and a hash of the
config (at least the part of it that's explicitly versioned) and all its dependencies.

In the [hello world example](experiments/hello_world.py), we have two steps,
generating data and compute statistics.

See the documentation in [executor.py](marin/execution/executor.py) for more details.

## Ray

Recall that a step's function can either be a normal Python function or in most realistic cases,
a [Ray](https://docs.ray.io/) remote function.
Ray allows us to run a large computation distributed over a cluster and is
generally used for large data processing tasks.  For example, we can break a large
dataset into shards and create a Ray function to process each shard.

Ray packages up the code from the local directory and ships it off to the appropriate machine.
The environment will have the following packages installed:
- **Default packages**: installed on the Ray cluster (`dependencies` in
  [pyproject.toml](pyproject.toml)), which include fsspec, draccus, etc.
- **Step-specific packages**: each `ExecutorStep` can specify
  `pip_dependency_groups`, a list of either (i) a key from
  `project.optional-dependencies` dictionary (e.g., `tokenize_train`), or (2) a
  specific pip package.  This allows each step to have its own environment and
  not interfere with other steps.

For example, to install the dependencies specified in the
`quality_dedup_consolidate` groups and also pip install `google-cloud-logging`,
one can do:

```python
number_of_restarts = ExecutorStep(
    name=...,
    fn=...,
    config=...,
    pip_dependency_groups=["quality_dedup_consolidate", "google-cloud-logging"],
)
```

Finally, to launch an experiment, use [ray_run.py](marin/run/ray_run.py), which
launches jobs to the Ray cluster:

```bash
python marin/run/ray_run.py -- python experiments/hello_world.py
```

This script ensure that:
- All the relevant libraries (specified above) are installed.
- The working directory is set appropriately.
- Any subpaths under submodules are appended to PYTHONPATH, which is useful
  when co-developing with another submodule (e.g., levanter).

Check out [quickstart.py](experiments/quickstart.py) for a full example.
