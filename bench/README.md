# Policy-transfer topology screen

Harness for the bulk HBM-to-HBM transfer screen reported in
[marin-community/marin#7695](https://github.com/marin-community/marin/issues/7695).
This branch exists to pin the exact code behind those numbers. It is not meant
to merge and touches no production code.

It measures how long it takes to move one BF16 policy copy
(`S = 2 x 359.6e9 = 719.2e9` bytes) between GPUs over an already-initialised
NCCL communicator. It does not export a checkpoint, does not reproduce the
MarinSkyRL per-parameter broadcast cadence, and does not load weights into a
model.

## Files

| Path | Role |
|---|---|
| `policy_transfer_bench.py` | The benchmark. One process per local GPU; `p2p`, `broadcast`, and `striped` modes. |
| `run_matrix.sh` | One side of the screen. Both Iris jobs run it with the same `ROUND`; only `ROLE` differs. |
| `diagnostics/gpu_netcheck.py` | GPU-node reachability probe: dumps interfaces and routes, then listens and dials a peer node. |
| `diagnostics/netprobe.py` | Same idea for CPU `hostNetwork` pods, with the pod manifests below. |
| `diagnostics/netinfo.py` | Dumps routes, interfaces, IB devices, and HTTP/TCP egress probes from a node. |
| `diagnostics/netprobe-*.yaml` | Pod manifests, one per cluster, for the CPU reachability sweep. |
| `diagnostics/netinfo-*.yaml` | Pod manifests for the egress dump. |

## Running the screen

Both sides need a torch build with CUDA. `run_matrix.sh` installs one at task
start because the Iris task image has none.

```sh
ROLE=source ROUND=1 bash bench/run_matrix.sh   # ranks [0, 8)
ROLE=dest   ROUND=1 bash bench/run_matrix.sh   # ranks [8, 16)
```

Each side blocks until an operator writes the rendezvous address into
`/tmp/master_addr`. That gate is deliberate: it is what lets you read
`ds.coreweave.com/nvlink.domain` on every participating node and confirm the
source and destination label sets are disjoint before any bytes move. Two
independently submitted GB200 gangs land in the same NVLink domain by default,
and a same-rack pair reports a rate about 25x higher than a cross-rack pair.

Exact per-round commands are in the issue.
