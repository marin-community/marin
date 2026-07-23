# Performance Guide

## Introduction

This is the very beginnings of a performance guide for Levanter. It's currently mostly a collection of notes and ideas,
but it will eventually be a comprehensive guide to optimizing Levanter (and potentially other JAX programs).

See also the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)

## Profiling

### Enabling the Profiler

Levanter captures JAX XPlane data, uploads it to `MARIN_PREFIX` TTL storage, and
logs an authenticated XProf URL:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 10
```

Profiles default to a 30-day lifetime. Set
`--trainer.profiler.upload.ttl_days 3` to request another lifetime or
`--trainer.profiler.upload.enabled false` for local-only capture. A local
`MARIN_PREFIX` also disables upload.

Install local viewers with one of:

- `pip install "levanter[profiling]"`
- `uv sync --extra profiling`

| Argument | Description | Default |
|---|---|---|
| `--trainer.profiler.enabled` | Capture, upload, and log an XProf link | `false` |
| `--trainer.profiler.start_step` | First profiled training step | `5` |
| `--trainer.profiler.num_steps` | Number of profiled steps | `25` |
| `--trainer.profiler.upload.enabled` | Upload to TTL storage and print a hosted link | `true` |
| `--trainer.profiler.upload.ttl_days` | Remote profile lifetime | `30` |
| `--trainer.profiler.process_index` | Capture one JAX process; unset captures all hosts | unset |
| `--trainer.profiler.create_perfetto_trace` | Also export Perfetto JSON | `false` |
| `--trainer.profiler.perfetto_link` | Generate the interactive Perfetto URL | `false` |

The same fields are available in YAML. All JAX processes capture into one remote
XProf run unless `process_index` selects one process.

### Adding HLO graphs

HLO metadata enables XProf graph and memory views and increases artifact size:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.num_steps 5 \
  --trainer.profiler.profile_options.enable_hlo_proto true
```

`profile_options` also exposes host, Python, and device tracer levels,
`include_dataset_ops`, and `advanced_configuration`.

### Examining a Profile

Open the logged `XProf profile:` URL. The service stages the GCS or CoreWeave S3
tree and opens the XProf interface. See the
[JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html) for
profiler details.

#### Perfetto

[Perfetto](https://ui.perfetto.dev/) displays standalone timelines. Set
`--trainer.profiler.create_perfetto_trace true`, then upload
`plugins/profile/<datetime>/perfetto_trace.json.gz`.

If you enabled host profiling, the companion `host_profile.pstats` and `host_profile.txt` files are written alongside the
JAX trace files in that same profiler directory.

`--trainer.profiler.perfetto_link true` prints an interactive link. TPU runs need
the [JAX remote profiling setup](https://docs.jax.dev/en/latest/profiling.html#remote-profiling).

#### Local XProf or TensorBoard

For offline inspection, download the profiler directory and point XProf or
TensorBoard at the directory containing `plugins/`:

```bash
uv run --with xprof xprof --logdir /path/to/run/profiler
```

TensorBoard install tips:

- Avoid installing both stable and nightly variants together (e.g., `tensorboard` and `tb-nightly`).
  If you see “Duplicate plugins” errors, uninstall all TB/TF variants and reinstall a single choice.
- If the Profile plugin fails to load with a Protobuf version error, align major versions:
  - Upgrade Protobuf runtime to 6.x: `pip install -U 'protobuf>=6,<7'` (or `uv pip install -U 'protobuf>=6,<7'`).
  - Ensure `xprof` matches your TensorBoard (stable TB → `xprof`, nightly TB → `xprof-nightly`).
  - Restart TensorBoard after upgrading.

Useful XProf views:

1. The overview page tells you MMU utilization and the top 10 operations.
2. `op_profile` groups time by operation type.
3. `trace_viewer` displays the operation timeline and can be slow for large traces.

## Interpreting JAX terms in profiles

* `jvp(OP)` means the forward pass. (JVP stands for Jacobian-vector product.)
* `transpose(jvp(OP))` means the backward pass.
* `remat` (short for rematerialization) means that the operation is recomputed in the backward pass, i.e. gradient checkpointing.
