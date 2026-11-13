# Reserve a dev accelerator

Use these only when the user explicitly requests an interactive dev GPU, TPU or CPU box. Holder jobs keep hardware allocated until release.

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

## CPU

Use a privileged CPU dev box to debug Linux-only behavior interactively: bubblewrap sandboxes, Docker inside a pod, and grader or verifier liveness during dataset ingestion. The box runs under `CONTAINER_PROFILE_PRIVILEGED` on a non-preemptible CoreWeave node, as curation workers do. Name, size, priority and lifetime have no defaults.

```bash
uv run python -m scripts.iris.dev_cpu --config lib/iris/config/cw-rno2a.yaml --name <unique-name> \
  allocate --cpu 16 --memory 64GB --disk 200GB --priority interactive --max-lifetime 259200
uv run python -m scripts.iris.dev_cpu --name <unique-name> exec -- bash -lc 'bwrap --version; docker info'
uv run python -m scripts.iris.dev_cpu --name <unique-name> shell
uv run python -m scripts.iris.dev_cpu --name <unique-name> cp-to ./grader.py /tmp/grader.py
uv run python -m scripts.iris.dev_cpu --name <unique-name> cp-from /tmp/out ./out
uv run python -m scripts.iris.dev_cpu --name <unique-name> status
uv run python -m scripts.iris.dev_cpu --name <unique-name> release
```

Run these from the repository root. `allocate` returns once the pod runs and records the session under `~/.cache/marin/dev_cpu_iris/`. Iris ends the holder job after `--max-lifetime` seconds; `release` ends it earlier. Release the box when you finish. If Iris moves the task to a new pod, `exec` fails; release and allocate again.

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
