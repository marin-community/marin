# Marin Architecture

Marin is an open-source framework for reproducible foundation model research, covering the full pipeline from raw data to trained and evaluated models. This document describes how the system is structured and how its components fit together.

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Repository Structure](#repository-structure)
3. [Subprojects](#subprojects)
  - [Rigging — Shared Utilities](#rigging)
  - [Haliax — Named Tensors](#haliax)
  - [Iris — Job Orchestration](#iris) · [deep dive →](iris.md)
  - [Fray — Distributed Execution Abstraction](#fray)
  - [Zephyr — Dataset Processing](#zephyr) · [deep dive →](zephyr.md)
  - [Levanter — LM Training](#levanter)
  - [Marin — Pipeline Framework](#marin)
4. [The Executor System](#the-executor-system)
5. [End-to-End Data Flow](#end-to-end-data-flow)
6. [Configuration System](#configuration-system)
7. [Infrastructure](#infrastructure)
8. [Experiments](#experiments)

---

## High-Level Overview

Marin solves two core problems in ML research: **reproducibility** and **scale**.

Every pipeline step is content-addressed: its output path is a hash of its name, configuration, and dependency outputs. Re-running with the same config is a no-op; changed configs create new outputs. This makes it safe to iterate freely without corrupting previous results.

Scale is handled by a layered execution stack. Python dataclass configs describe *what* to run; Fray submits them as distributed jobs; Iris schedules those jobs on TPU/GPU/CPU workers via a central controller. Researchers write experiment scripts without thinking about cluster management.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Experiment Scripts                            │
│        (experiments/tootsie/, tutorials/, evals/, …)                 │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ define ExecutorSteps / StepSpecs
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Marin Executor (DAG runner)                      │
│  content-addressed versioning · caching · dependency resolution      │
└───┬──────────────────────────┬───────────────────────────────────────┘
    │ data steps                │ training steps
    ▼                           ▼
┌──────────┐             ┌──────────────────────────────────────────────┐
│  Zephyr  │             │                Levanter                      │
│ (dataset │             │  JAX LM training, checkpointing, evaluation  │
│ pipeline)│             └──────────┬───────────────────────────────────┘
└────┬─────┘                        │
     │                              │ both submit jobs via
     └──────────────────────────────┤
                                    ▼
                         ┌──────────────────────┐
                         │         Fray         │
                         │ (execution abstraction│
                         │  Iris / Ray / Local) │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │         Iris         │
                         │ (job orchestration:  │
                         │  controller, workers,│
                         │  autoscaling, Docker)│
                         └──────────────────────┘
```

---

## Repository Structure

Marin is a Python monorepo managed with **uv workspaces**. Each subproject under `lib/` is an independent installable package.

```
marin/
├── ARCHITECTURE.md          ← this file
├── CLAUDE.md / AGENTS.md    ← agent guidelines
├── pyproject.toml           ← uv workspace root
│
├── lib/
│   ├── rigging/             ← shared utilities (filesystem, timing, locks)
│   ├── haliax/              ← named tensor library for JAX
│   ├── iris/                ← job orchestration (controller + workers)
│   ├── fray/                ← distributed execution abstraction layer
│   ├── zephyr/              ← lazy dataset processing library
│   ├── levanter/            ← JAX LM training library
│   └── marin/               ← top-level pipeline framework
│
├── experiments/             ← experiment scripts (define the actual runs)
│   ├── defaults.py          ← default_download/tokenize/train/eval helpers
│   ├── simple_train_config.py
│   ├── tootsie/             ← flagship 8B/32B/70B model runs
│   ├── pretraining_datasets/
│   ├── evals/
│   ├── posttrain/
│   └── …
│
├── docs/                    ← MkDocs documentation site
├── infra/                   ← Iris cluster YAML configs, pre-commit, CI
├── scripts/                 ← operational scripts (iris, ray, storage, …)
├── docker/                  ← container build configs
└── rust/                    ← Rust packages (dupekit, kitoken)
```

### Dependency Direction

This invariant is enforced throughout the codebase — violations are bugs:

```
rigging  ←──  iris  ──┐
rigging  ←── haliax    │
                        ├──  fray  ──┐
rigging  ←── zephyr ───┘             │
                                      ├──  marin
rigging  ←── levanter ───────────────┘
haliax   ←── levanter
```

In English: `marin` depends on everything; `levanter` depends on `haliax`, `fray`, `zephyr`, `rigging`; `zephyr` depends on `fray`, `rigging`; `fray` depends on `iris`, `rigging`; `iris` and `haliax` depend only on `rigging`. **No reverse imports.**

---

## Subprojects

### Rigging

**Location**: `lib/rigging/src/rigging/`  
**Purpose**: Low-level shared utilities used by every other subproject.


| Module                | Contents                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `filesystem.py`       | `marin_prefix()` resolves the base GCS path via `MARIN_PREFIX` env var or GCS metadata. Region-to-bucket mappings, cross-region transfer budget guards, `open_url()`, `url_to_fs()`. |
| `timing.py`           | `Timestamp`, `Duration`, `Deadline`, `Timer`, `ExponentialBackoff` — used instead of raw `datetime`/`time` throughout.                                                               |
| `distributed_lock.py` | Distributed locking primitives (used by the executor to prevent duplicate step execution).                                                                                           |
| `log_setup.py`        | Logging configuration.                                                                                                                                                               |
| `config_discovery.py` | Configuration file discovery.                                                                                                                                                        |


The most important export is `marin_prefix()`, which resolves where all pipeline outputs land. In production this is a regional GCS bucket; in testing it falls back to `/tmp/marin`.

---

### Haliax

**Location**: `lib/haliax/src/haliax/`  
**Purpose**: Named tensor library for JAX. Instead of indexing arrays by position, all operations reference axes by name.

```python
# Instead of: x.mean(axis=0)
Embed = hax.Axis("embed", 4096)
x.mean(Embed)  # name-safe, reshape-safe
```


| Module            | Contents                                                                      |
| ----------------- | ----------------------------------------------------------------------------- |
| `core.py`         | `NamedArray` — wraps a JAX array with a tuple of `Axis` objects               |
| `axis.py`         | `Axis(name, size)`, `AxisSelector`, `AxisSpec`                                |
| `partitioning.py` | Device mesh partitioning, `ResourceMapping`, `named_jit` for TPU/GPU sharding |
| `nn/`             | Neural network layers (attention, MLP, embedding, norm) using named axes      |
| `quantization.py` | Quantization support                                                          |
| `state_dict.py`   | HuggingFace checkpoint compatibility                                          |


Named axes make **tensor parallelism** explicit and safe: `ResourceMapping` maps axis names to mesh dimensions, so sharding decisions live in config rather than scattered through model code.

---

### Iris

**Location**: `lib/iris/src/iris/`  
**Deep dive**: [iris.md](iris.md)

Iris is Marin's job orchestration system, replacing Ray for most production workloads. It uses a central controller with a SQLite database, gRPC-based communication, and per-VM worker agents that run jobs in Docker containers.

```
┌─────────────────────────────────────────────────┐
│              Iris Controller                     │
│  gRPC service · Scheduler · Autoscaler           │
│  SQLite (jobs, tasks, workers, endpoints)        │
└───────────────────┬─────────────────────────────┘
                    │ heartbeat RPCs (controller-initiated)
           ┌────────┴────────┐
           ▼                 ▼
     Worker VM A       Worker VM B
     Docker containers  Docker containers
     (iris.managed=true)
```

**Key concepts**:

- **Scale Groups**: hardware pools (e.g., `v5litepod-256`) with min/max slices, autoscaling, and preemptibility
- **Jobs / Tasks**: a job is submitted to the controller; tasks are per-slice subdivisions
- **Priority bands**: PRODUCTION > INTERACTIVE > BATCH, with preemption across bands
- **Actors**: long-running services (e.g., Zephyr coordinators) with gRPC endpoint discovery

See the [Iris deep dive](iris.md) for the full state machine, scheduling algorithm, autoscaler design, actor system, and configuration reference.

---

### Fray

**Location**: `lib/fray/src/fray/`  
**Purpose**: Unified execution abstraction. All Marin code submits work through Fray, which delegates to Iris, Ray, or a local thread backend depending on the environment.

#### The `Client` Protocol

```python
class Client(Protocol):
    def submit(self, request: JobRequest, adopt_existing: bool = True) -> JobHandle: ...
    def host_actor(self, actor_class, *args, name, actor_config, **kwargs) -> HostedActor: ...
    def create_actor(self, actor_class, *args, name, resources, actor_config, **kwargs) -> ActorHandle: ...
    def create_actor_group(self, actor_class, *args, name, count, resources, **kwargs) -> ActorGroup: ...
    def shutdown(self, wait: bool = True) -> None: ...
```

`current_client()` auto-detects the environment: Iris → Ray → `LocalClient` (for tests).

#### Resource Configuration

```python
# TPU job
config = ResourceConfig().with_tpu("v5litepod-256", slice_count=2)

# GPU job
config = ResourceConfig().with_gpu("H100", count=8)

# CPU job (default)
config = ResourceConfig().with_cpu()
```

`ResourceConfig` also carries `region`, `preemptible`, `docker_image`, `env_vars`, `min_replicas`.

#### Backends


| Backend          | When used                                          |
| ---------------- | -------------------------------------------------- |
| `FrayIrisClient` | Production — wraps `iris.client.IrisClient`        |
| Ray backend      | Legacy Ray clusters                                |
| `LocalClient`    | Tests and local development — runs jobs in threads |


---

### Zephyr

**Location**: `lib/zephyr/src/zephyr/`  
**Deep dive**: [zephyr.md](zephyr.md)

Zephyr is Marin's lazy dataset processing library. It provides a declarative `Dataset` API with chainable transformations, operation fusion, and a pull-based distributed execution model.

```python
ctx = ZephyrContext(max_workers=200)
pipeline = (
    Dataset.from_files("gs://bucket/input/", "**/*.jsonl.gz")
    .load_jsonl()
    .filter(col("score") > 0.5)   # Parquet predicate pushdown
    .map(transform_record)
    .reshard(num_shards=500)
    .write_jsonl("gs://bucket/output/{shard:05d}-of-{total:05d}.jsonl.gz")
)
result = ctx.execute(pipeline)
```

```
ZephyrContext.execute(pipeline)
      │  compute_plan() → operation fusion
      ▼
ZephyrCoordinator (Fray actor)
      │  pull-based task assignment
      ▼
ZephyrWorkers × N (Fray actor_group)
      │  each runs one shard at a time in a subprocess
      ▼
GCS output (atomic per-shard writes)
```

Key operations: `.map()`, `.flat_map()`, `.filter()`, `.reshard()`, `.group_by()`, `.deduplicate()`, `.sorted_merge_join()`. Outputs: `.write_jsonl()`, `.write_parquet()`, `.write_vortex()`.

See the [Zephyr deep dive](zephyr.md) for operation fusion details, the pull-based execution model, shuffle/scatter design, deduplication implementation, and fault tolerance.

---

### Levanter

**Location**: `lib/levanter/src/levanter/`  
**Purpose**: JAX-based language model training library. Handles the full training loop, checkpointing, evaluation, and model export.

#### Key Abstractions

```
TrainLmConfig ──────────────────────────────────┐
  ├── TrainerConfig                              │
  │     ├── LearningRateSchedule                │
  │     ├── OptimizerConfig                     │  draccus YAML
  │     ├── CheckpointerConfig                  │  config parsing
  │     └── DistributedConfig                   │
  ├── LmConfig (plugin registry)                │
  │     ├── LlamaConfig   (registered "llama")  │
  │     ├── GemmaConfig   (registered "gemma")  │
  │     ├── QwenConfig    (registered "qwen")   │
  │     └── …                                   │
  └── LMDatasetConfig / LMMixtureDatasetConfig  │
        └── tokenized GCS paths ────────────────┘
```

`LmConfig` uses a **plugin registry** (`draccus.PluginRegistry`) so model types are auto-discovered: any registered subclass is selectable from YAML with `type: llama`.

#### Training Loop

```python
# Simplified — levanter/trainer.py
trainer = Trainer(config, optimizer, loss_fn)
state = trainer.initial_training_state(model, dataset)

for batch in data_loader:
    state, metrics = trainer.train_step(state, batch)
    trainer.run_hooks(state, metrics)
```

Hooks handle: W&B logging, checkpointing, evaluation, profiling, watchdog.

**Supported training modes**: standard LM pre-training, DPO, SFT. Export/import between Levanter checkpoints and HuggingFace format.

**Supported models**: Llama, Gemma, Qwen, Mistral, Mixtral, OLMo, GPT-2, Apertus, Whisper, Grug/Grugformer.

---

### Marin

**Location**: `lib/marin/src/marin/`  
**Purpose**: Top-level pipeline framework. Provides the executor DAG system, data processing modules, training orchestration, and evaluation.

#### Key Packages


| Package       | Contents                                                             |
| ------------- | -------------------------------------------------------------------- |
| `execution/`  | Executor, StepSpec, StepRunner — the DAG engine                      |
| `download/`   | Dataset download (HuggingFace Hub, etc.)                             |
| `transform/`  | Raw → text: ar5iv, StackExchange, Wikipedia, conversation formats, … |
| `processing/` | Tokenization, quality classification, deduplication                  |
| `training/`   | Bridge from executor to Levanter training                            |
| `evaluation/` | Evaluation runner, task configs                                      |
| `rl/`         | PPO workers, replay buffer, curriculum, environments                 |
| `inference/`  | Inference serving                                                    |
| `export/`     | Checkpoint export/conversion                                         |
| `datakit/`    | Data toolkit utilities                                               |
| `markdown/`   | HTML → Markdown conversion                                           |


---

## The Executor System

The executor is Marin's central nervous system. It implements a content-addressed DAG where each node is a `StepSpec` (newer) or `ExecutorStep` (legacy).

### Versioning and Caching

Every step gets a **hash ID** computed from:

- Step name
- All `hash_attrs` (config values that affect outputs)  
- Sorted output paths of all dependency steps

```
hash_id = sha256(name + str(hash_attrs) + sorted(dep_output_paths))[:8]
output_path = f"{prefix}/{name}_{hash_id}"
```

If `{output_path}/_SUCCESS` exists, the step is skipped. Otherwise it runs, and on success writes the status file. This makes re-runs safe and incremental.

### Step Types

**StepSpec** (preferred for new code):

```python
@dataclass(frozen=True)
class StepSpec:
    name: str
    deps: list[StepSpec]
    hash_attrs: dict[str, Any]   # what to hash for versioning
    fn: Callable[[str], Any]     # receives output_path, writes results there
    output_path_prefix: str | None = None
    override_output_path: str | None = None
```

**ExecutorStep** (legacy, being migrated away from):

```python
@dataclass(frozen=True)
class ExecutorStep:
    name: str
    fn: Callable           # or RemoteCallable
    config: ConfigT        # dataclass with InputName / VersionedValue fields
    description: str | None = None
    override_output_path: str | None = None
```

### Remote Execution

The `@remote` decorator wraps a function into a `RemoteCallable` that submits itself as a Fray job:

```python
@remote(resources=ResourceConfig().with_tpu("v5litepod-256"))
def run_training(config: TrainLmConfig) -> None:
    levanter.main.train_lm.main(config)
```

When the executor runs this step, it calls `fray.v2.current_client().submit(JobRequest(...))` instead of executing locally.

### Execution Flow

```
experiment.py
    │  defines steps with deps
    ▼
executor_main() / StepRunner.run()
    │  topological sort of DAG
    │  check _SUCCESS for each step → skip if cached
    │  acquire distributed lock (rigging.distributed_lock)
    │
    ├── local step: call fn(output_path) directly
    │
    └── remote step: fray.submit(JobRequest)
              │
              ▼
         Iris controller
              │ schedules on worker VM
              ▼
         Docker container
              │ runs step fn
              ▼
         GCS output (writes _SUCCESS on completion)
```

---

## End-to-End Data Flow

```
Raw Data (HuggingFace Hub / S3 / Web)
         │
         ▼  ExecutorStep: marin.download
gs://marin-{region}/downloads/{dataset}_{hash}/
         │
         ▼  ExecutorStep: marin.transform
gs://marin-{region}/documents/{dataset}_{hash}/   ← JSONL, one doc per line
         │
         ▼  ExecutorStep: marin.processing.classification (optional)
         │  (train quality classifier, score documents)
         │
         ▼  ExecutorStep: marin.processing.tokenize
gs://marin-{region}/tokenized/{dataset}_{hash}/   ← Levanter cache format
         │
         ▼  ExecutorStep: marin.training (runs Levanter on TPU/GPU)
gs://marin-{region}/train/{run_name}_{hash}/
         │    ├── checkpoints/                     ← JAX checkpoints
         │    └── hf_export/                       ← HuggingFace format
         │
         ▼  ExecutorStep: marin.evaluation
gs://marin-{region}/eval/{run_name}_{hash}/       ← lm-eval results
```

Each arrow is one `ExecutorStep`. The hash suffix on every path ensures that any config change produces a new path, leaving old results intact.

### Zephyr's Role in Data Steps

Download, transform, filtering, and deduplication steps typically use Zephyr internally:

```
marin.transform.wikipedia.process()
    └── ZephyrContext.execute(
            Dataset.from_files(raw_download_path, "**/*.html.gz")
            .flat_map(parse_html_to_text)
            .filter(quality_filter)
            .write_jsonl(output_path)
        )
```

Zephyr handles distributed fan-out across hundreds of workers, progress tracking, and fault-tolerant shard writing.

---

## Configuration System

Marin uses **Python dataclasses** for all configuration, parsed from YAML or CLI flags via the `draccus` library.

### Config Hierarchy for a Training Run

```
experiments/tootsie/exp600_tootsie.py
    └── SimpleTrainConfig           ← user-facing shorthand
          ├── resources: ResourceConfig   (fray — what hardware)
          ├── train_batch_size, num_train_steps, learning_rate
          ├── model: LmConfig subclass    (levanter — what architecture)
          │         (LlamaConfig, GemmaConfig, …)
          └── data: LMDatasetConfig       (levanter — what data)
                    └── train_urls: list of tokenized GCS paths
```

`SimpleTrainConfig` in `experiments/simple_train_config.py` is a convenience wrapper that `defaults.default_train()` expands into a full `TrainLmConfig` before submitting to Levanter.

### draccus Plugin Registry

Model types are registered so they can be selected from YAML:

```yaml
# config.yaml
model:
  type: llama
  hidden_dim: 4096
  num_layers: 32
  num_heads: 32
```

```python
# levanter/models/llama.py
@LmConfig.register_subclass("llama")
@dataclass
class LlamaConfig(LmConfig):
    hidden_dim: int = 4096
    …
```

---

## Infrastructure

### Iris Cluster Configs

`infra/*.yaml` files define production clusters:

```yaml
# infra/marin-us-central2.yaml (illustrative)
platform: gcp
region: us-central2
controller:
  machine_type: n2-standard-8
scale_groups:
  - name: v5litepod-256
    accelerator: v5litepod-256
    min_slices: 0
    max_slices: 16
    preemptible: true
  - name: v4-8
    accelerator: v4-8
    min_slices: 0
    max_slices: 32
```

Multiple clusters exist for different regions and purposes (training, vLLM serving, staging).

### CI / Canary System

```
scripts/ferries/   ← integration test pipeline runs ("ferry runs")
scripts/canary/    ← canary tests that run before merges
infra/tpu-ci/      ← TPU CI infrastructure
```

Ferry runs are integration tests that run a small end-to-end pipeline on real hardware. Canary runs are lighter checks. Both use the standard executor/Iris stack.

### Pre-commit and Linting

```bash
./infra/pre-commit.py --all-files --fix
```

This is the **only** supported lint entry point. It runs ruff, pyrefly (type checking), and other checks. `uv run pyrefly` for type checking alone.

---

## Experiments

Experiment scripts are the entry point for actual research work. They live in `experiments/` and are named `exp{issue_number}_{description}.py`.

### Typical Experiment Script

```python
# experiments/tutorials/tinyllama.py
from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, ExecutorStep

tokenize_step = default_tokenize(
    name="tokenized/tinyweb",
    input=raw_data_step,
)

train_step = default_train(
    SimpleTrainConfig(
        name="train/tinyllama",
        resources=ResourceConfig().with_tpu("v4-8"),
        data=tokenize_step.output_path,
        model=LlamaConfig(hidden_dim=512, num_layers=8, num_heads=8),
        num_train_steps=10_000,
        learning_rate=3e-4,
    ),
    deps=[tokenize_step],
)

if __name__ == "__main__":
    executor_main([tokenize_step, train_step])
```

### `defaults.py` Helpers

`experiments/defaults.py` provides ready-made step builders:


| Helper                               | What it does                                                       |
| ------------------------------------ | ------------------------------------------------------------------ |
| `default_download(dataset)`          | Creates an `ExecutorStep` to download a HF dataset                 |
| `default_tokenize(name, input, ...)` | Creates an `ExecutorStep` to tokenize documents                    |
| `default_train(config, deps)`        | Creates a remote `ExecutorStep` submitting a Levanter training job |
| `default_eval(model_path, tasks)`    | Creates eval steps against standard benchmarks                     |
| `default_validation_sets(...)`       | Creates validation set evaluation steps                            |


### Major Experiment Groups


| Directory               | Contents                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tootsie/`              | Flagship 8B/32B/70B model training runs. "Tootsie" refers to folding previous data batches into new training runs, like the candy-making process. |
| `pretraining_datasets/` | Data pipeline configs for FineWeb, DCLM, Nemotron, Dolma, etc.                                                                                    |
| `evals/`                | Evaluation configurations and benchmark runners                                                                                                   |
| `posttrain/`            | Post-training: SFT, DPO, preference learning                                                                                                      |
| `grug/`                 | Grugformer model variant experiments                                                                                                              |
| `scaling_law_sweeps/`   | Scaling law analysis across model sizes                                                                                                           |
| `tutorials/`            | Tutorial experiments (tiny model training for onboarding)                                                                                         |
| `ferries/`              | CI integration run pipelines                                                                                                                      |


---

## Key Design Decisions

**Content-addressed outputs**: Every step output is at a path derived from its config hash. This makes caching automatic, prevents accidental overwriting, and lets researchers branch freely without coordination.

**No global mutable state**: Configs are frozen dataclasses. `dataclasses.replace()` is used to derive variants, not in-place mutation.

**Fray as the only execution boundary**: All distributed work goes through `fray.v2.current_client()`. This means the entire pipeline can run locally (with `LocalClient`) for testing, on Ray for legacy workloads, or on Iris in production — with no code changes.

**Zephyr's pull-based model**: Workers pull chunks from the coordinator rather than having work pushed to them. This makes straggler handling automatic: a slow worker just takes fewer chunks. It also bounds memory regardless of dataset size.

**Named tensors in Haliax**: Axes are named objects, not integer indices. This eliminates an entire class of shape bugs and makes sharding/partitioning decisions explicit in config rather than hardcoded in model forward passes.

**Plugin registry for models**: `LmConfig.register_subclass("llama")` lets model types be selected by string in YAML. New model architectures can be added without modifying the training loop.