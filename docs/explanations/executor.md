# Executor framework

Marin's executor framework manages the execution of experiments.
This document is more about the mechanics, read [this](../explanations/experiments.md) to
learn more about the conventions.

## Steps

An **experiment** is a sequence (really, a DAG) of steps, where each **step** is
specified by the following:
- **name**: an identifier describing the function (and its version)
- **function**: a normal Python function, or a function wrapped with `remote()` to run as a distributed [Fray](https://github.com/marin-community/marin/tree/main/lib/fray) sub-job (which enables massive parallelism across the cluster)
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

In the [hello world example](../tutorials/executor-101.md), we have two steps,
generating data and compute statistics.

See the documentation in [`executor.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/execution/executor.py) for more details.

Coordination between multiple pipelines is handled via lease files. This
prevents duplicate execution if, for example, 2 Executor pipelines share common
ancestor steps.

## Mirrored inputs

Some datasets live in a specific regional bucket (e.g.
`gs://marin-us-central2/documents/stackexchange/...`) but experiments may run
from any region.  The `mirrored()` wrapper marks an input path for
**cross-region mirroring** so that the executor copies the data to the local
marin prefix before the step runs.

```python
from marin.execution.executor import mirrored, versioned

step = ExecutorStep(
    name="train",
    fn=my_training_fn,
    config=TrainConfig(
        dataset=mirrored(versioned("documents/stackexchange/v1"), budget_gb=50),
    ),
)
```

At config instantiation time, `mirrored()` rewrites the path to use the
`mirror://` protocol.  When the step's function opens the path via `fsspec`,
the `MirrorFileSystem` transparently copies data from whichever regional bucket
has it into the local marin prefix, respecting the per-path transfer budget.

**Key details:**

- `budget_gb` (default 10) caps how much data (in GB) a single step may copy
  cross-region.  The budget is enforced via the `mirror_budget` context manager
  from `rigging.filesystem`.
- Paths that already exist in the local prefix are not re-copied.
- `mirrored()` can wrap plain strings or `VersionedValue` / `InputName`
  references.
- To adjust the global mirror budget default, set the `MARIN_MIRROR_BUDGET_GB`
  environment variable before the process starts.

## Distributed execution (Fray + Iris)

Each step's function runs either directly in the executor driver process or,
when the step requests accelerators or a special environment, as a separate
Fray sub-job. Fray is Marin's abstraction for job/task scheduling; on shared
infrastructure it dispatches to [Iris](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md).

Steps can declare environment extras via `remote()`:
- **Default packages**: installed into the driver job from
  [`pyproject.toml`](https://github.com/marin-community/marin/blob/main/pyproject.toml) (fsspec, draccus, etc.).
- **Step-specific packages**: `remote()` accepts `pip_dependency_groups` — a list
  of either (1) a key from `project.optional-dependencies` in
  [`lib/marin/pyproject.toml`](https://github.com/marin-community/marin/blob/main/lib/marin/pyproject.toml) (e.g., `rl`), or (2) a specific pip
  package. Each step runs in its own environment without interfering with others.

For example, to install the dependencies specified in the
`rl` extra and also uv pip install `google-cloud-logging`,
one can do:

```python
number_of_restarts = ExecutorStep(
    name=...,
    fn=remote(my_fn, pip_dependency_groups=["rl", "google-cloud-logging"]),
    config=...,
)
```

To launch an experiment on the shared Iris cluster, submit the executor script
as the entrypoint of a CPU-only Iris job. The script then uses `executor_main`
to spawn the accelerated sub-jobs via Fray:

```bash
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -- python -m experiments.tutorials.hello_world
```

See [`lib/iris/OPS.md`](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md)
for the full Iris CLI reference, including `--no-wait`, log streaming, and
job lifecycle commands.

> Agents can use the `add-dataset` skill for a guide to dataset schema inspection and addition.
