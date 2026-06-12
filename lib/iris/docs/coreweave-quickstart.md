# CoreWeave Quickstart — `marin-gpu` (US-EAST-02A)

Get from zero to a running job on the `marin-gpu` H100 cluster. For the full
operator runbook (RBAC, NodePools, troubleshooting, other regions) see
[`coreweave.md`](coreweave.md).

**Cluster:** `marin-gpu`, region US-EAST-02A — 32× H100 (256 GPUs) + 4× CPU
Genoa, all pinned warm. Iris config: `lib/iris/config/cw-us-east-02a.yaml`
(cluster name `cw-us-east-02a`).

Console links:
- Tokens (kubeconfig): https://console.coreweave.com/tokens
- Cluster details: https://console.coreweave.com/zones/US-EAST-02A/clusters/marin-gpu#details
- Health dashboard: https://cks-grafana.coreweave.com/d/cluster-health/cluster-health?var-cluster-org=208261&var-cluster=marin-gpu&var-region=US-EAST-02

## 1. Make a token / kubeconfig

In the [Tokens console](https://console.coreweave.com/tokens), create a token
for `marin-gpu` and download its kubeconfig.

## 2. Install the kubeconfig

The config expects it at `~/.kube/coreweave-iris-gpu` (context
`marin-gpu_US-EAST-02A`):

```bash
mkdir -p ~/.kube
mv ~/Downloads/kubeconfig.yaml ~/.kube/coreweave-iris-gpu
export KUBECONFIG=~/.kube/coreweave-iris-gpu
kubectl cluster-info   # sanity check
```

You also need the controller extras and (for `s3://` state) R2 credentials:

```bash
uv pip install 'marin-iris[controller]'
export R2_ACCESS_KEY_ID=<your-r2-access-key-id>
export R2_SECRET_ACCESS_KEY=<your-r2-secret-access-key>
```

## 3. Check cluster status

`--cluster=cw-us-east-02a` resolves the in-tree config and opens a
`kubectl port-forward` to the controller for you:

```bash
uv run iris --cluster=cw-us-east-02a cluster status
```

If the controller isn't up yet, start it (idempotent):
`uv run iris --cluster=cw-us-east-02a cluster start`.

## 4. Hello world

CPU job:

```bash
uv run iris --cluster=cw-us-east-02a job run \
  --cpu 1 --memory 2GB --extra cpu \
  -- python -c "print('Hello from CoreWeave!')"
```

One GPU, proving JAX sees the H100:

```bash
uv run iris --cluster=cw-us-east-02a job run \
  --cpu 8 --memory 64GB --gpu H100x1 --extra gpu \
  -- python -c "import jax; print(jax.devices())"
```

Follow logs of a detached job with
`uv run iris --cluster=cw-us-east-02a job logs <job-id> -f`.
