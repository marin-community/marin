# Executor 101: Creating a Marin Experiment

This tutorial will guide you through creating an experiment using Marin's executor framework. We'll build a simple experiment that:
1. Generates a sequence of numbers
2. Computes basic statistics on those numbers

## Goals

In our [First Experiment](first-experiment.md), we trained a tiny model on TinyStories.
That tutorial used the executor framework to run a sequence of steps, but didn't really cover how it works.

In this tutorial, you will learn:

- How to define steps in Marin
- How to connect steps together
- How to run an experiment
- How to inspect the output of an experiment

## Prerequisites

Before starting this tutorial, make sure you have:

- Completed the [installation](installation.md).

## Understanding the Components

A Marin experiment consists of one or more `ExecutorStep`s that can be chained together. Each step:

- Has a unique name and description
- Takes a configuration object
- Processes data and produces output
- Can depend on outputs from previous steps

## Required Imports

Let's start by importing the necessary modules:

```python
import json
import logging
import os
from dataclasses import dataclass
import fsspec

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path
)
```

Key imports:

- `dataclass`: For creating configuration classes
- `fsspec`: For file system operations (local or cloud)
- `marin.execution.executor`: Core components for building experiments

## Step 1: Generating Data

First, we'll create a step that generates numbers from 0 to n-1:

```python
@dataclass(frozen=True)
class GenerateDataConfig:
    n: int  # Number of data points to generate
    output_path: str  # Where to write the numbers

def generate_data(config: GenerateDataConfig):
    """Generate numbers from 0 to `n` - 1 and write them to `output_path`."""
    numbers = list(range(config.n))

    # Write to file
    numbers_path = os.path.join(config.output_path, "numbers.json")
    with fsspec.open(numbers_path, "w") as f:
        json.dump(numbers, f)
```

## Step 2: Computing Statistics

Next, we'll create a second step that reads the generated numbers and computes statistics:

```python
@dataclass(frozen=True)
class ComputeStatsConfig:
    input_path: str  # Path to the file with numbers
    output_path: str  # Where to write the stats

def compute_stats(config: ComputeStatsConfig):
    """Compute statistics on the input numbers and write results."""
    # Read from file
    numbers_path = os.path.join(config.input_path, "numbers.json")
    with fsspec.open(numbers_path) as f:
        numbers = json.load(f)

    # Compute statistics
    stats = {
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    stats_path = os.path.join(config.output_path, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)
```

## Putting It All Together

Now we'll create the experiment pipeline by connecting our steps:

```python
n = 100  # Number of data points to generate

# Step 1: Generate data
data = ExecutorStep(
    name="hello_world/data",
    description=f"Generate data from 0 to {n}-1.",
    fn=generate_data,
    config=GenerateDataConfig(
        n=n,
        output_path=this_output_path(),
    ),
)

# Step 2: Compute statistics
stats = ExecutorStep(
    name="hello_world/stats",
    description="Compute stats of the generated data.",
    fn=compute_stats,
    config=ComputeStatsConfig(
        input_path=output_path_of(data),  # Use output from previous step
        output_path=this_output_path(),
    ),
)

# Run the experiment
if __name__ == "__main__":
    executor_main(
        steps=[data, stats],
        description="Simple experiment to compute stats of some numbers.",
    )
```

## Running the Experiment

To run this experiment:

```bash
python experiments/tutorials/hello_world.py --prefix local_store
```

This command will create several output files:

1. `local_store/experiments/hello_world-7063e5.json`: Stores a record of all the steps in this experiment
2. `local_store/hello_world/data-d50b06`: Contains the output of step 1 (numbers.json with generated data)
3. `local_store/hello_world/stats-b5daf3`: Contains the output of step 2 (stats.json with computed statistics)

!!! note

    If you run the same command again, it will detect that both steps have already been run and return automatically. This saves computation time when rerunning experiments.

## Complete Code

The complete code for this tutorial is available at: [experiments/tutorials/hello_world.py](https://github.com/marin-community/marin/blob/main/experiments/tutorials/hello_world.py)

## Next Steps

- Train a [tiny language model](first-experiment.md) using Marin.
- Learn about the [Executor framework](../explanations/executor.md): how to manage Python libraries, run big parallel jobs using Ray, how versioning works, etc.