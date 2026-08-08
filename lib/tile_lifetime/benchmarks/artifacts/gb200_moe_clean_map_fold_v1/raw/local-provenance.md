# Local replay provenance

- Replay date: 2026-08-08 (America/Los_Angeles)
- Shuttle target: `31f600f22837ec4a3c4c3eaf07c2e4a9a5ddc268`
- Controller client: `86f20400aec5c174d756de441060632b6bd664ca`
- Iris config: `lib/iris/config/cw-us-east-08a.yaml`
- Iris config SHA256: `a1eee433200ebaafe70f95d4aecad114b4894589adc23cbf788124898a1b6d30`
- Cluster: `cw-us-east-08a`
- Reservation: 1 node, 4 GB200, batch priority, 128 CPU, 850 GB memory, 500 GB disk
- Clock policy: cluster default, unpinned
- Preemptions: one; the full environment and extensions were rebuilt on the replacement pod
- Release verified: no active `dlwh-shuttle-map-fold-replay` dev-GPU session and no matching pod remained

Allocation command:

```text
uv run ../../scripts/iris/dev_gpu.py \
  --config config/cw-us-east-08a.yaml \
  --name dlwh-shuttle-map-fold-replay \
  allocate --gpu-variant gb200 --gpus-per-node 4 --nodes 1 \
  --priority batch --cpu 128 --memory 850GB --disk 500GB \
  --timeout 1800 --pod-timeout 300
```

The benchmark command is recorded verbatim in each JSON under
`environment.command`. The two final captures use 10 warmups, 30 measured
iterations, and opposite `acceptance_order` values.
