# Performance Guide

## Introduction

This is the very beginnings of a performance guide for Levanter. It's currently mostly a collection of notes and ideas,
but it will eventually be a comprehensive guide to optimizing Levanter (and potentially other JAX programs).

See also the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)

## Profiling

### Enabling the Profiler

Levanter uses JAX's built-in profiler. You can enable it by adding the `--trainer.profiler.enabled true` flag
to the training command. The callback captures XPlane data under
`./logs/<run_id>/profiler/plugins/profile/<datetime>`, uploads it to the region-local
`MARIN_PREFIX` TTL bucket, and logs an authenticated link to the hosted XProf service.

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.start_step 5 \
  --trainer.profiler.num_steps 10
```

The default remote lifetime is seven days. Override it inline with
`--trainer.profiler.upload.ttl_days 3`. Values are rounded up to a lifecycle
duration configured for the active Marin storage backend. If no remote Marin store
is configured, the callback keeps the original local trace and does not print a
hosted link.

Install local XProf and TensorBoard dependencies with one of:

- `pip install "levanter[profiling]"`
- `uv sync --extra profiling`

Here are the full list of profiling related options:

| Argument | Description | Default |
|---|---|---|
| `--trainer.profiler.enabled` | Capture, upload, and log an XProf link | `false` |
| `--trainer.profiler.start_step` | First profiled training step | `5` |
| `--trainer.profiler.num_steps` | Number of profiled steps | `25` |
| `--trainer.profiler.upload.enabled` | Upload to TTL storage and print a hosted link | `true` |
| `--trainer.profiler.upload.ttl_days` | Remote profile lifetime | `7` |
| `--trainer.profiler.process_index` | Capture one JAX process; unset captures all hosts | unset |
| `--trainer.profiler.create_perfetto_trace` | Also export Perfetto JSON | `false` |
| `--trainer.profiler.perfetto_link` | Generate the interactive Perfetto URL | `false` |

As usual, these can be specified in the yaml configuration file as well.

All JAX processes capture by default. Their host-specific files are uploaded under one
remote run root so XProf can present the distributed profile together. Set
`process_index` only when a single-host trace is intentional.


### Adding HLO graphs

HLO protobufs enable XProf's graph and memory views, but enlarge the artifact. Keep
the window short and enable them only when needed:

```bash
uv run ... \
  --trainer.profiler.enabled true \
  --trainer.profiler.num_steps 5 \
  --trainer.profiler.profile_options.enable_hlo_proto true
```

`profile_options` also exposes host, Python, and device tracer levels, dataset-op
capture, and JAX's `advanced_configuration` map.

### Examining a Profile

Open the `XProf profile:` URL printed after the upload barrier. Iris authenticates
the request, the service stages the approved GCS or CoreWeave S3 tree locally, and
then the complete XProf interface is available, including overview, trace, memory,
graph/HLO, operation, kernel, roofline, and utilization views.

See the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html) for more information on how to examine a profile.

Use hosted or local XProf for the full profile, and Perfetto for a standalone timeline.

#### Perfetto

[Perfetto](https://ui.perfetto.dev/) is a web-based tool for examining profiles.

Enable `--trainer.profiler.create_perfetto_trace true` in the training command. After
the run, open https://ui.perfetto.dev/ and upload `perfetto_trace.json.gz` from the
profile directory.
The file lives under `plugins/profile/<datetime>/` inside the profiler output directory.

If you enabled host profiling, the companion `host_profile.pstats` and `host_profile.txt` files are written alongside the
JAX trace files in that same profiler directory.

Alternatively, you can enable the `--trainer.profiler.perfetto_link` flag.
This will generate a link that will automatically upload the `perfetto_trace.json.gz` file in the same directory as the TensorBoard profile.
This link is a little tricky to use on TPU. The JAX guide has [some instructions](https://docs.jax.dev/en/latest/profiling.html#remote-profiling)
on how to use it. (Basically, set up SSH port forwarding and then use the link in your local browser.)

#### Local XProf or TensorBoard

The hosted service is the normal path. For offline inspection, download the trace
tree and run XProf or TensorBoard locally.
You want to download the trace files (e.g. `plugins/profile/2024_03_16_07_26_24`)
and run `xprof --logdir <dir>` or `tensorboard --logdir <dir>` where `<dir>` is the *directory containing plugins* (not the plugins directory itself).
Then you can navigate to http://localhost:6006/#profile in your browser and see the profile.

#### Fetching traces

If your run directory is on durable remote storage, download or sync the profiler output directory locally and point
TensorBoard at the directory containing `plugins/`.

```bash
# Example: launch XProf from a local copy of a profiler output directory
uv run --with xprof xprof --logdir /path/to/run/profiler
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
