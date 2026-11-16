# Reserve a dev accelerator

Use these only when the user explicitly requests an interactive dev GPU or TPU. Holder jobs keep expensive hardware allocated until release.

## GPU

Choose the cluster from checked-in configs after confirming current capacity with `list-backends`.

```bash
uv run scripts/iris/dev_gpu.py --config lib/iris/config/<cluster>.yaml \
  --name <unique-name> allocate
uv run scripts/iris/dev_gpu.py --name <unique-name> connect
uv run scripts/iris/dev_gpu.py --name <unique-name> status
uv run scripts/iris/dev_gpu.py --name <unique-name> release
```

The default holds one H100 node. Use `--gpu-variant GB200` for GB200 and `--nodes N` only for a distributed test. `allocate` blocks; release from another shell after interruption. CoreWeave pods use regional object storage; do not read GCS or print pod environment values.

## TPU

Run at most one TPU job at a time on a dev TPU VM. `--tpu-type` is required.

```bash
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name <unique-name> allocate --tpu-type v5p-8
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name <unique-name> execute -- <command>
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name <unique-name> connect
uv run scripts/iris/dev_tpu.py --config lib/iris/config/marin.yaml \
  --tpu-name <unique-name> release
```

`execute` syncs local files; add `--no-sync` for an inner loop. Use `--worker <index>` for a multi-host TPU. Never start or restart a shared controller to obtain a session. If allocation cleanup fails, confirm the holder job is gone.
