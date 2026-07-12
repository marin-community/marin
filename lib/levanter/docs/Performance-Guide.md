# Performance Guide

## Introduction

This is the very beginnings of a performance guide for Levanter. It's currently mostly a collection of notes and ideas,
but it will eventually be a comprehensive guide to optimizing Levanter (and potentially other JAX programs).

See also the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)

## Profiling

### Enabling the Profiler

Levanter uses JAX's built-in profiler. You can enable it by adding the `--trainer.profiler.enabled true` flag
to the command line. This will generate a trace file in the `./logs` directory, under `./logs/<run_id>/profiler/plugins/profile/<datetime>`.
(Yeah, it's a mess, but it's what JAX wants to do.)
The trace stays on disk in the run directory, so you can inspect it from whatever durable storage backs `log_dir`.

Install profiling dependencies (TensorBoard) with one of:

- `pip install "levanter[profiling]"`
- `uv sync --extra profiling`

Here are the full list of profiling related options:

| Argument                                  | Description | Default |
|-------------------------------------------|-------------|---------|
| `--trainer.profiler.enabled`              | Enable the profiler | `false` |
| `--trainer.profiler.start_step`           | The step to start profiling | `5`     |
| `--trainer.profiler.num_steps`            | The number of steps to profile | `25`    |
| `--trainer.profiler.perfetto_link`        | Whether to generate a Perfetto URL | `false` |

As usual, these can be specified in the yaml configuration file as well.

In a multi-process setup, each node saves its own profile under that node's `log_dir` tree. JAX does not merge or align
those traces automatically, so compare the per-host `plugins/profile/<datetime>/` directories directly if you need a
cross-host view.


### Examining a Profile

See the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html) for more information on how to examine a profile.

JAX offers two main ways to examine a profile: Perfetto and TensorBoard.

#### Perfetto

[Perfetto](https://ui.perfetto.dev/) is a web-based tool for examining profiles.

Open the trace from the run directory, then go to https://ui.perfetto.dev/ and upload `perfetto_trace.json.gz`.
The file lives under `plugins/profile/<datetime>/` inside the profiler output directory.

If you enabled host profiling, the companion `host_profile.pstats` and `host_profile.txt` files are written alongside the
JAX trace files in that same profiler directory.

Alternatively, you can enable the `--trainer.profiler.perfetto_link` flag.
This will generate a link that will automatically upload the `perfetto_trace.json.gz` file in the same directory as the TensorBoard profile.
This link is a little tricky to use on TPU. The JAX guide has [some instructions](https://docs.jax.dev/en/latest/profiling.html#remote-profiling)
on how to use it. (Basically, set up SSH port forwarding and then use the link in your local browser.)

#### TensorBoard

TensorBoard is a locally-run tool for examining profiles.
You want to download the trace files (e.g. `plugins/profile/2024_03_16_07_26_24`)
and run `tensorboard --logdir <dir>` where `<dir>` is the *directory containing plugins* (not the plugins directory itself).
Then you can navigate to http://localhost:6006/#profile in your browser and see the profile.

#### Fetching traces

If your run directory is on durable remote storage, download or sync the profiler output directory locally and point
TensorBoard at the directory containing `plugins/`.

```bash
# Example: launch TensorBoard from a local copy of a profiler output directory
tensorboard --logdir /path/to/run/profiler
```

TensorBoard install tips:

- Avoid installing both stable and nightly variants together (e.g., `tensorboard` and `tb-nightly`).
  If you see “Duplicate plugins” errors, uninstall all TB/TF variants and reinstall a single choice.
- If the Profile plugin fails to load with a Protobuf version error, align major versions:
  - Upgrade Protobuf runtime to 6.x: `pip install -U 'protobuf>=6,<7'` (or `uv pip install -U 'protobuf>=6,<7'`).
  - Ensure `xprof` matches your TensorBoard (stable TB → `xprof`, nightly TB → `xprof-nightly`).
  - Restart TensorBoard after upgrading.

There are three sections I find particularly useful:

1. The overview page tells you MMU utilization and the top 10 operations.
2. **op_profile** shows you the time spent in each operation (by type). You end up with annoying names like `fusion.1772`,
but with some patience and work you can back those out by looking at the next section (under XLA Ops).
3. **trace_viewer** shows you the actual trace of operations as a big timeline. It takes a long time to load.

## Interpreting JAX terms in profiles

* `jvp(OP)` means the forward pass. (JVP stands for Jacobian-vector product.)
* `transpose(jvp(OP))` means the backward pass.
* `remat` (short for rematerialization) means that the operation is recomputed in the backward pass, i.e. gradient checkpointing.
