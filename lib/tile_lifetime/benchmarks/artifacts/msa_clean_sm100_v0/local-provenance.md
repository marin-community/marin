# MSA SM100 oracle replay provenance

- Date: 2026-08-08 (America/Los_Angeles)
- Iris holder job: `/dlwh/dev-gpu-dlwh-msa-sm100-oracle`
- Iris task: `dlwh.dev-gpu-dlwh-msa-sm100-oracle.0`
- Successful pod: `iris-dlwh-dev-gpu-dlwh-msa-sm100-or-d0b3ca2a-1-4206335f74da90bb`
- Initial pod: `iris-dlwh-dev-gpu-dlwh-msa-sm100-or-d0b3ca2a-0-cf5977ac59d84c18` (preempted before first command)
- Controller client revision: `86f20400aec5c174d756de441060632b6bd664ca`
- Cluster config: `cw-us-east-08a`
- Config SHA256: `a1eee433200ebaafe70f95d4aecad114b4894589adc23cbf788124898a1b6d30`
- Reservation: one node, one GB200, batch priority, 64 CPU, 400 GB memory, 300 GB disk
- Clock policy: cluster default, unpinned

Allocation command:

```text
uv run ../../scripts/iris/dev_gpu.py \
  --config config/cw-us-east-08a.yaml \
  --name dlwh-msa-sm100-oracle \
  allocate --gpu-variant gb200 --gpus-per-node 1 --nodes 1 \
  --priority batch --cpu 64 --memory 400GB --disk 300GB \
  --timeout 1800 --pod-timeout 300
```

The official MSA implementation is used only as a correctness and performance
oracle. The natural capture boundary includes the dense proxy pass,
`sparse_topk_select`, and sparse attention payload. Plans, tensors, and memory
allocations are outside the timed region.
