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
  model_axis_size: 1
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
| `per_device_train_parallelism` | Number of examples to process on each device during training        | `train_batch_size / (num_accelerators * model_axis_size)` |
| `per_device_eval_parallelism`  | Number of examples to process on each device during eval            | `per_device_train_parallelism`                            |
| `steps_per_eval`               | How often to evaluate the model during training                     | 1,000                                                     |
| `max_eval_batches`             | How many batches to evaluate during each evaluation                 | `None` (meaning all)                                      |
| `mp`                           | Mixed Precision policy using [jmp](https://github.com/deepmind/jmp) | `f32` (full precision)                                    |

### Logging and Reporting

| Parameter      | Description                                                                   | Default |
|----------------|-------------------------------------------------------------------------------|---------|
| `log_dir`      | Where to save logs (python logger). `$run_id` will be appended                | `logs/` |



### Partitioning / FSDP

Sharding in Levanter is done with axis mappings, which specify how to map logical axes (e.g. "batch") to physical axes in the JAX device mesh.
(See the [Haliax Scaling Tutorial](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?authuser=1#scrollTo=lFZOnJD7QtZm&uniqifier=2)
for a more detailed explanation of axis mappings.) Levanter's Trainer uses two axis mappings:  `parameter_axis_resources` and `compute_axis_resources`.
`parameter_axis_resources` specifies how to shard the model parameters and optimizer state: basically how the model is sharded "at rest",
while the `compute_axis_resources` specifies how to shard the model during computation.

`TrainerConfig` allows you to specify these axis mappings in two ways, with a "basic" mode that has
reasonable defaults and an "advanced" mode that gives you more control.

#### Basic Mode

| Parameter              | Description                                                                  | Default   |
|------------------------|------------------------------------------------------------------------------|-----------|
| `batch_axis`           | The axis to shard the batch over, for distributed data parallelism           | `"batch"` |
| `fsdp_axis`            | The axis or axes to shard the model over, for Fully Sharded Data Parallelism | `"embed"` |
| `tensor_parallel_axes` | The axis or axes to shard the model over, for Tensor Parallelism             | `None`    |
| `model_axis_size`      | How many devices for tensor parallelism                                      | `1`       |

#### Advanced Mode

| Parameter                  | Description                                                          | Default |
|----------------------------|----------------------------------------------------------------------|---------|
| `axis_resources`           | Mapping from logical axis to physical axis shared by both mappings   | --      |
| `parameter_axis_resources` | Mapping from logical axis to physical axis for the parameter mapping | --      |
| `compute_axis_resources`   | Mapping from logical axis to physical axis for the compute mapping   | --      |
| `model_axis_size`          | How many devices for tensor parallelism                              | `1`     |

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
This method is particularly useful for workflows involving `launch.py` on TPUs, as environment variables can be specified in the `.levanter.yaml` configuration file used by `launch.py`.
For more details on using `launch.py`, see the [Using launch.py](../Getting-Started-TPU-VM.md#using-launchpy) section in the TPU VM guide.

Example `.levanter.yaml` snippet:
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
| save_code      | Whether to save the code to wandb.             | `True`                     |
| save_xla_dumps | Whether to save XLA compiler outputs to wandb. | `False`                    |


Notes:

* WandB's code saving logic isn't very good for our use case, so we have our own. We automatically sniff out the git repo
of your main script.
* `save_xla_dumps` is useful for debugging XLA compilation issues. It tends to dump a lot of stuff, so we don't save it by default.
To use it, you must also set the right environment variables. Something like `XLA_FLAGS="--xla_dump_to=/tmp/output_folder/xla_dumps --xla_dump_hlo_pass_re=.*`.
We will automatically parse out the env variable.

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

## Ray Config

Levanter will by default automatically start a Ray cluster with all
the machines being used for training. This is useful for distributed
preprocessing. You can disable this behavior using `auto_start_cluster: false`.


| Parameter            | Description                                                             | Default |
|----------------------|-------------------------------------------------------------------------|---------|
| `address`            | The address of the Ray cluster to connect to.                           | `None`  |
| `start_workers`      | Whether to start Ray workers. If `False`, you must start them yourself. | `True`  |
| `auto_start_cluster` | Whether to start a Ray cluster automatically.                           | `True`  |


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



### Distributed and Ray

::: levanter.distributed.DistributedConfig

::: levanter.distributed.RayConfig

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
