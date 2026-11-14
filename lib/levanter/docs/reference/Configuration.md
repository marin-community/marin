# In-Depth Configuration

This page gives an overview of the various settings you can use to customize a training run.

We use [Draccus](https://github.com/dlwh/draccus) for configuration. Draccus is yet-another yaml-to-dataclass library
that uses both dataclasses to generate yaml and argparse to parse command line arguments.

Typically, your config data class will look something like this:

```python
@dataclass
class TrainLmConfig:
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
```

Your training run will typically be associated with a single config file. For instance, you might have a file
`my-run.yaml` that looks like this:

```yaml
data:
    train_urls:
      - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
    validation_urls:
      - "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
    cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
model:
  type: gpt2
  hidden_dim: 768
  num_heads: 12
  num_layers: 12
  seq_len: 1024
  gradient_checkpointing: true
  scale_attn_by_inverse_layer_idx: true
trainer:
  tracker:
    type: wandb
    project: "levanter"
    tags: [ "openwebtext", "gpt2"]

  mp: p=f32,c=bfloat16
  per_device_parallelism: 4

  train_batch_size: 512
optimizer:
  learning_rate: 6E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
```

### Including Other Config Files

Draccus supports inclusion of config files via the `!include` special syntax. For instance, this:

```yaml
# my-run.yaml
data: !include data.yaml
trainer:
    num_train_steps: 1000000

# data.yaml
train_urls:
    - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
validation_urls:
    -  "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
```

will expand to:

```yaml
data:
    train_urls:
        - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
    validation_urls:
        - "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
    cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
trainer:
    num_train_steps: 1000000
```

The inclusion path is always relative to the config file. Unfortunately, we don't (can't) support inclusion
at the top level.

## Trainer and TrainerConfig

The [levanter.trainer.Trainer][] class is governed by the [levanter.trainer.TrainerConfig][] dataclass.

Trainer has a lot of stuff in it. We highlight some of them in the following sections.

The following table lists some of the parameters that you might want to change.

### Core Training Loop Configuration

| Parameter                      | Description                                                         | Default                                                   |
|--------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| `seed`                         | The random seed                                                     | 0                                                         |
| `num_train_steps`              | The number of training steps to run                                 | 400,000                                                   |
| `train_batch_size`             | The batch size                                                      | 32                                                        |
| `per_device_train_parallelism` | Number of examples to process on each device during training        | `train_batch_size / data_axis_size` |
| `per_device_eval_parallelism`  | Number of examples to process on each device during eval            | `per_device_train_parallelism`                            |
| `steps_per_eval`               | How often to evaluate the model during training                     | 1,000                                                     |
| `max_eval_batches`             | How many batches to evaluate during each evaluation                 | `None` (meaning all)                                      |
| `mp`                           | Mixed Precision policy using [jmp](https://github.com/deepmind/jmp) | `f32` (full precision)                                    |

### Logging and Reporting

| Parameter      | Description                                                                   | Default |
|----------------|-------------------------------------------------------------------------------|---------|
| `log_dir`      | Where to save logs (python logger). `$run_id` will be appended                | `logs/` |



### Partitioning / Mesh

Levanter sharding is controlled by mapping logical axes (e.g. `batch`, `embed`, `heads`) onto a named JAX device mesh
(e.g. `data`, `model`, `replica`, `replica_dcn`).

The full explanation (including the difference between `compute_mapping` and `param_mapping`, and how to add context
parallelism) lives in [Mesh Parallelism](../explanation/Mesh-Parallelism.md).

### Checkpointing and Initialization

See also [Checkpointer](#checkpointer).

| Parameter                    | Description                                                                                   | Default                                    |
|------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------|
| `load_checkpoint`            | Whether to load checkpoint from `base_path`                                                   | `None`: load if possible, but don't error. |
| `initialize_from`            | Initialize training state from this path. May be a parent dir. Useful for continued training. | `None`                                     |
| `checkpointer.base_path`     | Base path to save checkpoints to                                                              | `checkpoints/${run_id}`                    |
| `checkpointer.save_interval` | How often to save checkpoints (time)                                                          | 15 minutes                                 |
| `checkpointer.keep`          | How often to keep checkpoints (steps). See below.                                             | 10000 steps                                |

#### Checkpointer Save Policy

The checkpointer logic has two kinds of checkpoints:

* **time-based checkpoints**: **temporary** checkpoints that are saved every `save_interval` minutes. The previous time-based checkpoint is deleted when a new one is saved.
* **step-based checkpoints**: **permanent** checkpoints that are saved according to a policy. These checkpoints are never deleted.

Step-based checkpoint configuration looks like this:
```yaml
checkpointer:
  keep:
    - every: 1000  # steps
      until: 10000 # step
    - every: 5000  # steps
      until: 40000 # step
    - every: 10000
```

This policy will save permanent checkpoints every 1,000 steps until 10,000 steps, then every 5,000 steps until 40,000 steps, then every 10,000 steps.
The default step-based checkpoint policy is to save a checkpoint every 10,000 steps.

#### Checkpoint Format

A checkpoint directory contains three kinds of file:

* `manifest.json`: the format declaration and the inventory of arrays, each with its shape,
  dtype and chunk grid. Written when the save starts. A reader that meets a `format_version`
  above its own fails. Older checkpoints have no manifest and fall back to listing.
* `metadata.json`: step, timestamp, and whether the checkpoint is temporary. Written after
  every process has committed, so a directory containing it is complete. Checkpoint
  discovery keys off this file.
* the array data itself, as a single OCDBT database of zarr3 arrays.

#### Checkpoint Write Parallelism

Every process writes the part of the state it holds, and arrays replicated across processes
are split across their replicas. A save also bounds the host memory it stages: one whose
share fits under `max_staged_host_bytes` returns while the commits drain, and a larger save
rolls through it.

| Parameter                                  | Description                                                                            | Default |
|--------------------------------------------|----------------------------------------------------------------------------------------|---------|
| `checkpointer.write.max_write_replicas`    | Cap on how many replicas of an array write part of it. `1` disables replica splitting. | 1024    |
| `checkpointer.write.min_replica_slice_bytes` | Do not split an array when doing so would give each replica less than this.          | 16 MiB  |
| `checkpointer.write.max_chunk_bytes`       | Upper bound on one zarr3 chunk, which bounds a single object store write.              | 512 MiB |
| `checkpointer.write.max_staged_host_bytes` | Cap on host memory one process stages at once during a save.                           | 16 GiB  |
| `checkpointer.write.cache_pool_bytes`      | Soft limit for each TensorStore write cache.                                            | 1 GiB   |
| `checkpointer.write.data_copy_concurrency` | Maximum CPU concurrency TensorStore uses to copy and encode checkpoint data.           | 16      |

#### Checkpoint Telemetry

Checkpoint debug mode publishes numeric data from each JAX process to Finelog. This configuration disables allocation tracing and forced garbage collection:

```yaml
checkpointer:
  debug:
    enabled: true
    tracemalloc_frames: null
    force_gc_before_serialize: false
    top_allocations: 0
    flush_logs: false
```

This configuration does not start `tracemalloc` or force Python garbage collection. It publishes these gauges:

| Metric | Value |
|--------|-------|
| `checkpoint_phase_duration_seconds` | Time in one checkpoint phase. |
| `checkpoint_total_duration_seconds` | Time from checkpoint start through metadata commit. Process zero publishes this metric. |
| `checkpoint_staged_host_bytes` | Peak host bytes staged by one process. |
| `checkpoint_process_peak_rss_bytes` | Peak resident memory for one process. |

The phase metric uses the `phase` attribute. All metrics use `checkpoint_step`. Finelog also records the run, job, process, node, and cluster fields.

Use a short run that saves at least one checkpoint for a cluster smoke test. Then query the regional Finelog deployment:

```bash
uv run finelog query <cluster> --format table <<'SQL'
SELECT to_timestamp_millis(timestamp_ms) AS ts,
       process_index,
       name,
       value,
       json_get(attributes_json, 'checkpoint_step') AS checkpoint_step,
       json_get(attributes_json, 'phase') AS phase
FROM telemetry_v1
WHERE run_id = '<run-id>'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '2 hours') * 1000 AS BIGINT)
  AND name IN (
    'checkpoint_phase_duration_seconds',
    'checkpoint_total_duration_seconds',
    'checkpoint_staged_host_bytes',
    'checkpoint_process_peak_rss_bytes'
  )
ORDER BY timestamp_ms, process_index, name
SQL
```

The checkpoint is complete when process zero reports `checkpoint_total_duration_seconds`. Each process reports its local asynchronous commit phase.

### JAX Compilation Cache Configuration

Levanter allows you to configure JAX's persistent compilation cache. This can significantly speed up startup times by caching compiled JAX functions.
The primary way to specify the cache directory is via the `jax_compilation_cache_dir` field in the `TrainerConfig`.

| Parameter                   | Description                                                                                               | Type            | Default                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------|
| `jax_compilation_cache_dir` | Path to a directory to store the persistent compilation cache. Can be a local path or a GCS path.         | `Optional[str]` | `None` (JAX default, usually `~/.cache/jax` or platform specific)         |

Other JAX compilation cache settings (like `jax_persistent_cache_min_compile_time_secs`, `jax_persistent_cache_min_entry_size_bytes`, `jax_persistent_cache_enable_xla_caches`, etc.)
can be configured by including them in the `trainer.jax_config` dictionary. This dictionary allows you to pass arbitrary JAX configuration options.
For more details on all available JAX compilation cache options and how JAX's compilation cache works, please refer to the [official JAX documentation](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

Here's an example of how to configure these options in your YAML file:

```yaml
trainer:
  # ... other trainer configs
  jax_compilation_cache_dir: "/path/to/your/jax_cache"  # Or "gs://your-bucket/jax_cache"

  # To set other JAX compilation cache options or any other JAX global flag:
  jax_config:
    jax_persistent_cache_min_compile_time_secs: 5.0
    jax_persistent_cache_min_entry_size_bytes: 1024
    jax_persistent_cache_enable_xla_caches: "all" # or "xla_gpu_kernel_cache_file", etc.
    # ... other jax settings like jax_threefry_partitionable
```

Alternatively, JAX's compilation cache directory can be set using the `JAX_COMPILATION_CACHE_DIR` environment variable.
This method is particularly useful for workflows involving `launch.py` on TPUs, as environment variables can be specified in the `~/.config/marin/config.yaml` configuration file used by `launch.py`.
Checkout-local `.levanter.yaml` files are also read and override matching values from `~/.config/marin/config.yaml`.
For more details on using `launch.py`, see the [Using launch.py](../Getting-Started-TPU-VM.md#using-launchpy) section in the TPU VM guide.

Example `~/.config/marin/config.yaml` snippet:
```yaml
env:
  JAX_COMPILATION_CACHE_DIR: "gs://your-compile-cache-bucket/path"
  # ... other environment variables
```


## Trackers and Logging


We mostly use [W&B](https://wandb.ai/site) for tracking values and other metadata about a run. However, we also support
Tensorboard and a few other trackers. You can also use multiple trackers at once, or even write your own.
See  [Trackers](../reference/Trackers.md) for more information.

### W&B

Wandb is the default tracker and is installed by default. To use it, you can configure it in your config file:

```yaml
trainer:
    tracker:
        type: wandb
        project: my-project
        entity: my-entity
```

Because wandb is the default, you can also just do:

```yaml
trainer:
    tracker:
      project: my-project
      entity: my-entity
```



| Parameter      | Description                                    | Default                    |
|----------------|------------------------------------------------|----------------------------|
| entity         | The wandb entity to use.                       | your default entity        |
| project        | The wandb project to use.                      | wandb's default            |
| tags           | Tags to add to the run.                        | `[]`                       |
| id             | Unique run id                                  | wandb's autogenerated id   |
| name           | The name of the run.                           | wandb's autogenerated name |
| resume         | Resume policy for an existing run ID.          | `"allow"`                 |
| fork_from      | Optional `<source-run-id>?_step=<step>` lineage point for a new child run. | `None` |
| save_code      | Whether to save the code to wandb.             | `True`                     |


Notes:

* WandB's code saving logic isn't very good for our use case, so we have our own. We automatically sniff out the git repo
of your main script.
* `fork_from` creates a new W&B child run with inherited history through the named step. It is mutually exclusive with
  W&B resume: use it only for the initial cutover, then omit it and use the child run ID with `resume: allow` for recovery.

### Tensorboard

Tensorboard is also supported. To use it, you can configure it in your config file:

```yaml
trainer:
    tracker:
        type: tensorboard
        logdir: logs
```

Install the optional dependencies for TensorBoard support with one of:

- `pip install "levanter[profiling]"`
- `uv sync --extra profiling`

Viewing profiles: when profiling is enabled, JAX writes traces under `<logdir>/plugins/profile/<timestamp>`.
Launch the UI with `tensorboard --logdir <logdir>` and open http://localhost:6006/#profile.
If running remotely, forward the port: `ssh -L 6006:localhost:6006 <host>`.

### Multiple Trackers

In some cases, you may want to use multiple trackers at once.
For example, you may want to use both W&B and Tensorboard.

To do this, you can use the [levanter.tracker.tracker.CompositeTracker][] class, or, if using a config file, you
can specify multiple trackers:

```yaml
trainer:
  tracker:
    - type: wandb
      project: my-project
      entity: my-entity
    - type: tensorboard
      logdir: logs
```

## Distributed Config

JAX can automatically sniff out clusters in SLURM and TPU environments.
If you're not using SLURM or TPUs, you can specify the cluster manually using this config.

**Don't use this on TPU, and possibly not on SLURM either.**

| Parameter             | Description                                                               | Default                 |
|-----------------------|---------------------------------------------------------------------------|-------------------------|
| `coordinator_address` | The address of the coordinator. If `None`, we'll use the default address. | `None`                  |
| `num_processes`       | The number of processes in the cluster.                                   | `None`                  |
| `process_id`          | The process id of this process.                                           | `None`                  |
| `local_device_ids`    | The local device ids of this process.                                     | ${CUDA_VISIBLE_DEVICES} |

## Optimizer

### RandRAND

`optimizer.type: randrand` selects paired Gram-EMA RandRAND for body matrices
and AdamW for token embeddings, the output head, and normalization/bias vectors.
It requires an untied LM head and FP32 master parameters. Computation may use
BF16. Each Linear is treated as its full logical output-by-input matrix;
attention heads are not optimized as separate matrices. Scanned layers have
independent moments and Gram factors.

```yaml
optimizer:
  type: randrand
  learning_rate: 0.0045
  matrix_betas: [0.95, 0.99]
  weight_decay: 0.0303
  embed_lr: 0.2
  head_lr: 0.006
  vector_lr: 0.03
  warmup: 1000
  lr_schedule: cosine
  min_lr_ratio: 0.0
  rank: 32
  rank_sketch: 64
  pair_k: 25
  randrand_warmup: 100
  basis_every: 10
  basis_ema_beta: 0.95
  boost_factor: 5.0
  correction_scale: 0.5
```

These are the hybrid-normalized Qwen3 experiment defaults. Other architectures
need their own tuning. Embedding, head, and vector routes each expose their own
`*_lr`, `*_betas`, and `*_weight_decay`. Their default betas are `(0.9, 0.98)`
and weight decay is zero. `matrix_eps` defaults to `1e-8`; auxiliary `adam_eps`
defaults to `1e-15`. All vector parameters share `vector_lr`.

RandRAND uses one Adam moment pair on the raw gradient. For its Adam direction
`J`, the matrix direction is `boost_factor * (J - selected(J)) +
correction_scale * selected(J)`. The selected pairs come from randomized bases
of persistent left/right gradient Gram EMAs. With `randrand_warmup=100`, the
first refresh precedes update 101 and uses gradients through update 100.
Before the first refresh the matrix update is standard AdamW. After activation,
matrix weight decay uses the boosted LR, matching the reference implementation.

The optimizer performs clipping internally in three buckets: body matrices,
embedding plus vectors, and head. `max_grad_norm` defaults to 1.0. Do not add
another clipping or scheduling transform around it. Adam buffers retain named
axes for parameter sharding; Gram factors and bases are replicated. Checkpoints
include the explicit basis RNG state (`basis_seed=42`).

#### Qwen3 130M evaluation recipes

From the Marin repository root, `experiments.benchmarks.randrand_qwen3_130m`
binds a model and optimizer recipe to existing benchmark caches. `--recipe` is
required; selecting it changes the normalization and the corresponding tuned LRs
together. The standard recipe follows the model geometry used in the completed
[PR #6301 comparison](https://github.com/marin-community/marin/pull/6301);
the hybrid recipe adds the attention/MLP output RMSNorms used by MuonH.
Qwen3's query/key normalization is present in both recipes.

| Recipe | Hybrid norm | Matrix LR | Embedding LR | Vector LR | Head LR | Local val at 4800 | Local val at 4959 |
|---|---|---|---|---|---|---|---|
| `pr6301` | Off | 0.006 | 0.6 | 0.006 | 0.006 | 3.29637961 | 3.29576360 |
| `hybrid` | On | 0.0045 | 0.2 | 0.03 | 0.006 | 3.29596433 | 3.29544320 |

The LRs are peak base rates, before the matrix complement's factor of 5.
Local validation scores are cross-entropy in nats per token.
The source local run IDs are `20261102_054449_558993` and
`20261107_190126_612406`, respectively. All other optimizer settings are shared:
matrix betas `(0.95, 0.99)` and WD `0.0303`; auxiliary betas `(0.9, 0.98)`
and WD zero; the basis, boost, correction and schedule settings shown above.
The additional hybrid-norm vectors use the same vector LR as all other vectors.
These are the best observed single-seed local notebook results, not official
Paloma BPB measurements or claims of an optimum.

```bash
uv run --package marin-levanter python -m experiments.benchmarks.randrand_qwen3_130m \
  --recipe hybrid \
  --train-cache /path/to/fineweb-edu-10B-cache \
  --c4-cache /path/to/paloma-c4-en-cache \
  --output-path /path/to/randrand-checkpoints \
  --run-id randrand-hybrid-130m
```

This prints the resolved `TrainLmConfig` only. Add `--run` inside the allocated
accelerator job to train. `--per-device-batch` controls microbatching while
preserving the global batch of 128. The cache roots must contain `train/` and
`validation/`, respectively: use the original FineWeb-Edu 10B training cache
and full Paloma C4 English validation cache, tokenized with
`marin-community/marin-tokenizer` (Paloma raw revision `65cd6fc`). No data is
downloaded or retokenized by this entrypoint. `--optimizer-config PATH` accepts
an explicit Levanter optimizer YAML to run a baseline on the same architecture.

The `pr6301` and `hybrid` recipes train for 4,959 updates of 128 sequences of length 4,096,
with FP32 parameters and BF16 computation, model seed 0, data seed 42,
linear permutation shuffling, and EOS-separated attention. Evaluation uses
all validation batches every 1,000 updates and at the end; `--eval-every 200`
changes the interval. The final metric is `eval/paloma/c4_en/bpb`. Native
checkpoints and `eval_metrics.jsonl` are written under the supplied output
path/run ID. HF export is disabled because the hybrid model is not representable
as standard Hugging Face Qwen3.

The local notebooks use a different initialization, training sampler, seed and
fixed 32-batch validation subset. Their loss numbers do not predict the official
BPB. The recipes preserve benchmark geometry and evaluation intent on current
Levanter, not bitwise equivalence to an old trainer revision. Evaluate a fresh
baseline with the same recipe, caches and software revision, and report quality,
throughput and peak memory. Full accelerator validation remains necessary.

#### RandRAND/MuonH adaptive blend

`--recipe randrand_muon_hybrid` selects `optimizer.type: randrand_muon_hybrid`
and the hybrid-normalized Qwen3 model. The existing `pr6301` and `hybrid`
recipes retain their settings. This third recipe reproduces the optimizer
computation and settings supplied in `w_v06_slr0.015`:

| Route | Algorithm | Peak LR | Betas | WD |
|---|---|---|---|---|
| Body matrices | RandRAND candidate | 0.006 | (0.95, 0.95) | 0.0303 |
| Body matrices | MuonH candidate | 0.015 relative step | Momentum 0.95, Nesterov | 0 |
| Embedding | AdamW | 0.1 | (0.9, 0.98) | 0 |
| Vectors, including output RMSNorm gains | AdamW | 0.06 | (0.9, 0.98) | 0 |
| Head | AdamW | 0.006 | (0.9, 0.98) | 0 |

RandRAND uses paired projection with rank 32, sketch 64 plus 8 oversampling,
K=25 selected left/right eigenvector pairs, Gram EMA beta 0.95, a first refresh before update 101, and subsequent
refreshes every 10 updates. The complement boost is 5 and low-rank correction
is disabled: remove those pairs from the Adam direction and multiply the
remaining direction by 5. Decoupled decay also uses 5 times the scheduled
base matrix LR after update 100. Matrix Adam epsilon is 1e-8; auxiliary epsilon is 1e-15. Clip each of three gradient buckets (matrices, embedding plus vectors, head)
at norm 1. All LRs share a 1,000-step linear warmup from zero, followed by
cosine decay to zero over the 4,959-update horizon.
The new recipe uses model seed 42 to match the supplied notebook; the two
existing recipes retain model seed 0. Data seed remains 42 for all recipes.
The notebook's PyTorch initialization and sampling still differ from Levanter.

For each body matrix, both candidate weights start from the current weights
W. RandRAND produces W_RR, including its boosted decoupled decay. MuonH
produces W_MH using five Newton–Schulz iterations (NS5) with FP32 coefficients (3.4445, -4.7750, 2.0315),
normalization epsilon 1e-5, and no shape scaling. Its Hyperball retraction
preserves the Frobenius norm of W at that update, rather than the initialization
norm. If D is the NS5 direction, r is the current Frobenius norm of W, and
s is the shared LR multiplier, form `P = W - 0.015*s*r*D/max(norm(D),1e-10)`
and `W_MH = r*P/max(norm(P),1e-10)`. Muon momentum accumulates from update 1; it is not reset at the switch.
The applied weights are `(1-a) * W_RR + a * W_MH`. Embedding, vectors and head
remain AdamW throughout; the head does not use AdamH.

The global cosine c compares the current clipped gradient with the previous
actual applied update `W_before - W_after`, pooled over all body matrices.
The blend a is zero through update 1000. Its cosine EMA e has beta 0.98 and initializes on the first valid cosine. Beginning
at update 1001, `target = clip((0-e)/0.05, 0, 1)` and
`a = min(max(a_previous, target), a_previous + 1/400)`. The blend never
decreases. A zero previous update leaves the EMA unchanged. RandRAND state,
basis refreshes and updates continue after blending begins, even if a reaches
one; this recipe does not implement a hard handoff.

The supplied notebook records local FineWeb-Edu holdout cross-entropy
(nats/token, 32 fixed batches) of 3.276339 at step 4800,
3.275633 at step 4959, and Paloma BPB 1.165051. It reports a blend near 0.9811
from approximately step 1400 onward; this value is an observed outcome, not a
fixed setting. These are collaborator-reported single-run PyTorch results,
not completed Marin measurements of this new recipe. The source file is `w_v06_slr0.015_randrand_muonh_switch_qwen3_130m.ipynb`,
supplied privately for this evaluation and not bundled in Marin. Its
SHA-256 is `86ed722e4fa22824ff4d11f0553e7790f6f49ff0650eddd8dd1af5df14d1ed5c`.
The notebook was run with PyTorch 2.11.0; the independent CPU numerical fixture
was generated from its unchanged optimizer cells using PyTorch 2.9.1; see
`lib/levanter/tests/data/randrand/switch_reference.json` for provenance and
`lib/levanter/tests/test_randrand_muon_hybrid.py` for the replay.

The runtime needs no Torch. Checkpoints include the Muon momentum, previous
applied updates, cosine EMA, blend weight and the complete RandRAND state.
Large momentum/update buffers retain named axes for sharding. Trainer logs
`optim/switch/blend`, `cosine`, `cosine_ema`, `initialized`, `matrix_norm`, and
`matrix_norm_over_init` under the same `optim/switch/` prefix. The norm is
pooled across body matrices, not the notebook's median per-matrix norm ratio.
A cosine with no valid denominator is logged as zero; `initialized` indicates
whether the cosine EMA has received any valid observation.

Run with `--recipe randrand_muon_hybrid --eval-every 200` and the same cache,
output and `--run` arguments shown above. Use a fresh MuonH comparison on this
same model, seed, cache and budget, using the MuonH 130M sweep configuration
linked below as the baseline reference. An `--optimizer-config` baseline override
preserves the selected recipe's model and seed. Report final C4 BPB, throughput
and memory; the blend keeps both algorithms' work and adds momentum/update
buffers, so it carries no claim of lower cost than MuonH.

The baseline override is provided as
`experiments/benchmarks/randrand_muonh_baseline.yaml`, using current Levanter's
`muonH` optimizer with the historical sweep's settings: body MuonH and head
AdamH at LR 0.02; embedding/vectors Adam at LR 0.008; WD zero throughout;
Adam betas (0.9, 0.98), epsilon 1e-15; Muon momentum 0.95, Nesterov, five
quintic NS steps and epsilon 1e-5; clipping 1; warmup 1000 and cosine floor zero.
These auxiliary settings intentionally differ from the blend recipe. This is
a comparison of complete recipes, not an ablation changing only the matrices.
The blend's internal MuonH candidate LR 0.015 does not set the baseline LR.
Use the same invocation as the blend, changing the run ID/output path and adding:

```bash
--optimizer-config experiments/benchmarks/randrand_muonh_baseline.yaml
```

#### Spectral RandRAND/MuonH blend

`--recipe spectral_randrand_muon_hybrid` adds a fourth recipe. It uses the
same hybrid-normalized model, model/data seed 42, [adaptive blend controller](#randrandmuonh-adaptive-blend),
MuonH candidate, clipping, schedule, training budget and evaluation path as
`randrand_muon_hybrid`. The three earlier recipes keep their optimizer behavior.
The new recipe selects `optimizer.type: randrand_muon_hybrid` with
`variant: spectral`; `type: randrand` with `variant: spectral` also supports
the spectral optimizer without blending.

| Route | Algorithm | Peak LR | Betas | WD |
|---|---|---|---|---|
| Body matrices | Spectral RandRAND candidate | 0.0045 | (0.95, 0.99) | 0.0303 |
| Body matrices | MuonH candidate | 0.015 relative step | Momentum 0.95, Nesterov | 0 |
| Embedding | AdamW | 0.2 | (0.9, 0.98) | 0 |
| Vectors, including output RMSNorm gains | AdamW | 0.03 | (0.9, 0.98) | 0 |
| Head | AdamW | 0.006 | (0.9, 0.98) | 0 |

The spectral settings are rank 64, sketch width 256 (capped at the smaller
matrix dimension, without oversampling), EMA beta 0.98, first basis refresh
before update 201, and refresh interval 200. The active and standby banks each contain independent left/right orthonormal
probes and matching sketches. Both banks accumulate raw clipped-gradient
covariance sketches every step:
`Y_L = beta*Y_L + (1-beta)*G*(G.T*Omega_L)` and
`Y_R = beta*Y_R + (1-beta)*G.T*(G*Omega_R)`. These sketches have neither
bias correction nor dimension normalization. No full Gram matrix is stored.

Each sketch SVD retains `r = min(64, sketch_width)` left singular vectors and singular
values scaled by `sqrt(side_dimension / sketch_width)`. These are sketch-based
estimates, not exact covariance eigenpairs. For each side, let those vectors be
v_i and decreasing estimates be lambda_i, with r retained values. Define
`mu = max(1e-6 * lambda_1, float32.tiny)` and
`s_i = ((lambda_r + mu)/(lambda_i + mu))**0.25` for i < r. The filter is
`P = I + sum_i (s_i - 1)*v_i*v_i.T`; the last retained vector sets the cutoff
and is not attenuated. Apply `P_L*G*P_R` before updating Adam's moments, then
apply the same filters to Adam's direction. Once a nonzero basis is available, use `effective_lr = 5 * scheduled_matrix_lr`
and `W_new = (1 - effective_lr * 0.0303) * W - effective_lr * filtered_direction`.
The WD coefficient itself is unchanged; the decay subtraction grows by 5, not 25. There is one Adam
moment pair; paired selection and low-rank correction are not used.

Every fifth basis refresh, use the warm standby sketch for the new bases,
promote that bank to active, and redraw and clear only the retired bank. The
first such handover precedes update 1001. Both banks then receive that update's
raw gradient. Adam moments persist across refreshes and handovers. Checkpoints
include both banks, bases, spectra, refresh counters and the explicit RNG key.
Optimizer contractions request highest FP32 precision. Framework RNG streams
and accelerator SVD arithmetic can differ; equal seeds do not imply equal probes.

The variant is explicit in optimizer YAML:

```yaml
optimizer:
  type: randrand_muon_hybrid
  variant: spectral
  learning_rate: 0.0045
  matrix_betas: [0.95, 0.99]
  weight_decay: 0.0303
  embed_lr: 0.2
  vector_lr: 0.03
  head_lr: 0.006
  rank: 64
  rank_sketch: 256
  randrand_warmup: 200
  basis_every: 200
  basis_ema_beta: 0.98
  spectral:
    probe_every: 5
    regularization: 1e-6
```

The recipe supplies these settings automatically. `pair_k`,
`low_rank_correction`, `correction_scale`, and `max_gram_dim` are paired-variant
controls and do not affect the spectral variant. Other shared settings and the
switch defaults remain as documented in [the adaptive blend](#randrandmuonh-adaptive-blend); embedding, head and vectors stay
AdamW throughout.

Run with `--recipe spectral_randrand_muon_hybrid --eval-every 200` and the
cache, output and `--run` arguments in [the evaluation instructions](#qwen3-130m-evaluation-recipes). The local reference run
`20261112_065847_624170` recorded cross-entropy 3.27194103 at update 4800 and
3.27137539 at update 4959 on the notebook's fixed 32-batch validation subset.
Its final logged blend coefficient was 0.7706; this is an observed result,
not a configured constant: weights are `(1-a)*W_RandRAND + a*W_MuonH`, so
0.7706 weights the MuonH candidate. This coefficient is not an update-norm fraction. These are single-run PyTorch results, not Marin
Paloma BPB. Compare with MuonH using the same caches and
`randrand_muonh_baseline.yaml`, reusing the completed reference if its conditions
match, and report final BPB, timing and memory.

The independent numerical fixture executes `randrand_spectral_muonh_best_config_qwen3_130m_training.ipynb`'s optimizer
cells without changing their arithmetic, records their probes, and replays
2,002 updates across two bank handovers and the blend transition. Provenance
is in `lib/levanter/tests/data/randrand/spectral_switch_reference.json`, including
source and fixture SHA-256 hashes, cell indices, reference PyTorch version and
the reduced matrix shapes used for replay.
The private source notebook is not bundled; the runtime and replay tests have
no Torch dependency. Full accelerator training remains the evaluation target.

#### Hybrid-normalization model reference

The reference is Marin's historical MuonH Qwen3 scaling configuration at
commit `2d6290a99cd2e51ea4fda7108731b203a54dad54`:
[`_to_qwen3_from_llama` explicitly sets `hybrid_norm=True`](https://github.com/marin-community/marin/blob/2d6290a99cd2e51ea4fda7108731b203a54dad54/experiments/speedrun/muonh_qwen3_scaling/muonh_sweep.py#L55-L80).
Its 130M entry selects the `llama_150m` dimensions and converts them to Qwen3.
This historical experiment was removed from current main; the model option
remains implemented in Levanter.

At implementation reference commit `e8eeba00da7aefd98b31620ad8d938e1d442d52d`,
[`Qwen3LMHeadModel` constructs `LlamaTransformer`](https://github.com/marin-community/marin/blob/e8eeba00da7aefd98b31620ad8d938e1d442d52d/lib/levanter/src/levanter/models/qwen.py#L384-L399),
and [`LlamaDecoderLayer`](https://github.com/marin-community/marin/blob/e8eeba00da7aefd98b31620ad8d938e1d442d52d/lib/levanter/src/levanter/models/llama.py#L297-L347)
adds a learned RMSNorm to each attention and MLP output before residual addition.
The existing pre-attention/pre-MLP norms remain present. QK normalization remains
present in both standard and hybrid-normalized Qwen3.

The concrete model is six layers, width 512, MLP width 1792, eight query and
KV heads of dimension 64, context 4096, untied vocabulary 128256, RoPE theta
10000, and RMSNorm epsilon 1e-5. Hybrid normalization adds 12 vectors of length
512 (6,144 parameters): total 154,154,240 versus standard 154,148,096. The
historical benchmark calls this size "130M". Here "hybrid normalization"
identifies this architecture option; "RandRAND/MuonH blend" identifies the
optimizer and does not define a new model architecture.

### Standard Options

All optimizers in Levanter are based on the [levanter.optim.OptimizerConfig][] dataclass. This class has the following fields,
which are common to all optimizers (and most have to do with learning rate scheduling):


| Parameter       | Description                                                                   | Default  |
|-----------------|-------------------------------------------------------------------------------|----------|
| `weight_decay`  | The weight decay.                                                             | `0.0`    |
| `learning_rate` | The learning rate.                                                            | `1e-4`   |
| `lr_schedule`   | The type of learning rate schedule for decay. See below.                      | `cosine` |
| `min_lr_ratio`  | The minimum learning rate ratio.                                              | `0.1`    |
| `warmup`        | Warmup fraction or number of steps                                            | `0.01`   |
| `decay`         | Decay fraction or number of steps                                             | `None`    |
| `rewarmup`      | The learning rate re-warmup, if using cycles.                                 | `0.0`    |
| `cycles`        | The number of cycles for the learning rate, or steps where cycles end         | `None`   |
| `cycle_length`  | How long the cycles should be (as an int, fraction), or list of cycle lengths | `None`   |

By default, Levanter uses a cosine learning rate decay with warmup. The learning rate is decayed to
`min_lr_ratio * learning_rate` over the course of the training run. This is a fairly standard default for LLM training.

#### Learning Rate Schedules

The `lr_schedule` parameter specifies the learning rate schedule. The following schedules are supported:

* `constant`: Constant learning rate.
* `linear`: Linear decay.
* `cosine`: Cosine decay.
* `inv_sqrt`: Inverse square root decay.
* `inv`: Inverse decay.

#### Cycles

By default, there is only one cycle, and Levanter's LR schedule looks like this:

```
[warmup] -> [stable] -> [decay]
```

But you can specify more with either the `cycles` or `cycle_length` parameters.
If you want to use a learning rate schedule with cycles, you can specify the number of cycles with the `cycles`
or `cycle_length` parameters. The LR will be decayed to `min_lr_ratio * learning_rate` at the end of each cycle.
With cycles, Levanter's LR schedule looks like this:


```
[warmup] -> [stable] -> [decay] -> {[rewarmup] -> [stable] -> [decay]} x (cycles - 1)
```

or more compactly:

```
{[(re)?warmup] -> [stable] -> [decay]} x cycle
```

Here's what the phases mean:

* `warmup`: The first warmup in training, which is part of the first cycle. The LR will start at 0 and linearly increase to the learning rate over this period.
* `stable`: The stable period. The LR will stay at the learning rate for this period.
* `decay`: The decay period. The LR will decay to `min_lr_ratio * learning_rate` over this period.
* `rewarmup`: The re-warmup period. If using cycles, the LR will be re-warmed from the final value of the previous cycle back to the peak value of the next cycle.

Also note that if *rewarmup* is 0, there will be no rewarmup period, meaning the LR will jump
back to the max LR. This is the default, and works surprisingly well. In addition, the stable
and decay phase of the first cycle will generally be different from the stable and decay phase of the other cycles,
since rewarmup and warmup are typically different.

`stable` cannot be specified directly. It is the period between `warmup` and `decay` in the first cycle, and the period
between `rewarmup` and `decay` in subsequent cycles. By default, there is no `stable` period.

All of these parameters can be specified in terms of a fraction of the total number of steps of a cycle or as an absolute number of
steps.

Here are what the `cycles` and `cycle_length` parameters mean:

* `cycle_length`: If you specify an int or float for `cycle_length`, the learning rate will cycle through the
schedule with the specified length. This is equivalent to specifying `cycles` as `num_train_steps / cycle_length`.
If `cycle_length` is a float < 1.0, it is interpreted as a fraction of the total number of steps.
If you specify a list of ints, the learning rate will cycle through the schedule with the specified cycle lengths.
* `cycles`: If you specify an int for `cycles`, the learning rate will cycle through the schedule `cycles` times.
If you specify a list of ints, the learning rate will cycle through the schedule with the specified steps as the minima
of the cycles.

It is an error to specify both `cycles` and `cycle_length`.

You can also specify `cycles` as a list, e.g. `[10000, 25000, 50000]`. In this case,
`cycles` is interpreted as the minima for the cycles, with the first and final steps being cycle minima as well.
`cycles` as an int is equivalent to list `cycles` with the low points evenly spaced at
`[num_train_steps / (c + 1)]`.

See [our paper on WSD-S](https://arxiv.org/pdf/2410.05192) for more information on cyclic LR schedules for training LLMs
with short or no rewarmup.

### AdamConfig

Additionally, [levanter.optim.AdamConfig][] has the following fields:

| Parameter       | Description                                  | Default |
|-----------------|----------------------------------------------|---------|
| `beta1`         | The beta1 parameter for Adam.                | `0.9`   |
| `beta2`         | The beta2 parameter for Adam.                | `0.95`  |
| `epsilon`       | The epsilon parameter for Adam.              | `1e-8`  |
| `max_grad_norm` | The maximum gradient norm (for clipping).    | `1.0`   |


## LM Model Config

[levanter.models.lm_model.LmConfig][] is a Draccus "choice class" that acts as a base class for all autoregressive
language models in Levanter. You typically will specify a kind of model by using the `type` field, which is a string
that specifies the kind of model. For instance, `type: gpt2` will use the [levanter.models.gpt2.Gpt2Config][] class,
while `type: llama` will use the [levanter.models.llama.LlamaConfig][] class.

We won't go into detail here. You can see the auto-generated docs below.


## Auto-generated Documentation

### Trainer

::: levanter.trainer.TrainerConfig

::: levanter.trainer.Trainer


### Mesh

::: levanter.utils.mesh.MeshConfig

### Checkpointer

::: levanter.checkpoint.CheckpointerConfig

::: levanter.checkpoint.Checkpointer

### Trackers and Metrics

See also [Trackers](../reference/Trackers.md) for more information. Basic configuration is shown below.

#### Single Tracker

```yaml
trainer:
  tracker:
    type: wandb
    project: my-project
    entity: my-entity
```



### Distributed

::: levanter.distributed.DistributedConfig

### Model Averaging

Levanter can average model weights during training. Specify one of the
registered strategies in `trainer.model_averaging`:

```yaml
trainer:
  model_averaging:
    type: ema          # or 'ema_decay_sqrt'
```

* `ema` – classic exponential moving average with parameter `beta`.
* `ema_decay_sqrt` – EMA until `switch_step`, then decays with
  :math:`1 - \sqrt{x}` over `decay_steps`.

::: levanter.optim.model_averaging.EmaModelAveragingConfig

::: levanter.optim.model_averaging.EmaDecaySqrtConfig

### Optimizer

::: levanter.optim.OptimizerConfig

::: levanter.optim.AdamConfig

::: levanter.optim.SkipStepConfig


### LM Model

::: levanter.models.lm_model.LmConfig

::: levanter.models.gpt2.Gpt2Config

::: levanter.models.llama.LlamaConfig
