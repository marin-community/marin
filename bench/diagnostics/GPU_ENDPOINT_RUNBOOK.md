# GPU endpoint reachability runbook

These are the commands used to test direct TCP connectivity from one GB200
node in `cw-us-east-08a` to one H100 node in each candidate cluster. The H100
clusters were tested separately. Each pair ran
[`gpu_netcheck.py`](gpu_netcheck.py) concurrently at both endpoints, so both
nodes listened on TCP port 29700 and dialled the peer three times.

The allocations used Marin commit
`2ebc65b091eaf8a87d3ce7f2ca3a7e3b1658fe32`. The probe is pinned at
[`13fb15592eb5e4a79e9255b4e5666358d2f41b98`](https://github.com/marin-community/marin/blob/13fb15592eb5e4a79e9255b4e5666358d2f41b98/bench/diagnostics/gpu_netcheck.py),
SHA-256
`6c3eda3806a1c723fa979367c42e6251d88578e4c0c6875bf66d6c8eb79ac6bc`.

`dev_gpu.py` generated the Iris jobs dynamically; there was no static GPU
manifest to retain. The commands below reconstruct the allocations and preserve
the pod names and node addresses from the reported run.

## Allocate

Run each allocation command in its own terminal. `allocate` holds the session
until `release` stops it.

```sh
cd /home/romain/dev/marin

uv run python scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-08a.yaml \
  --name ptb08a allocate --gpu-variant GB200 --priority interactive

uv run python scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name ptb02a allocate --gpu-variant H100 --priority interactive
```

Release `ptb02a` after the New Jersey test, then allocate the Reno endpoint:

```sh
cd /home/romain/dev/marin

uv run python scripts/iris/dev_gpu.py --name ptb02a release

uv run python scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-rno2a.yaml \
  --name ptbrno allocate --gpu-variant H100 --priority interactive
```

## Record placement

The reported run used these pods and nodes:

| Cluster | Pod | Node | Host and pod IP |
|---|---|---|---|
| `cw-us-east-08a` | `iris-romain-dev-gpu-ptb08a-0-e7f2bee0-0-3183010a92b7a6b2` | `s62pys64` | `10.186.213.79` |
| `cw-us-east-02a` | `iris-romain-dev-gpu-ptb02a-0-347f1b1a-0-78eda07ba84e843d` | `g856b8a` | `10.184.201.5` |
| `cw-rno2a` | `iris-romain-dev-gpu-ptbrno-0-85da3e1e-0-39378ec4fc2d0aa2` | `gb95f50` | `10.168.194.123` |

For a new run, read the current pod name from
`~/.cache/marin/dev_gpu_iris/<session>.json`, then record its placement:

```sh
KC=/home/romain/.kube/coreweave-iris
POD=$(jq -r .pod.pod_name \
  /home/romain/.cache/marin/dev_gpu_iris/ptb08a.json)

kubectl --kubeconfig "$KC" --context marin-us-east-08a_US-EAST-08A \
  -n iris get pod "$POD" \
  -o jsonpath='{.spec.nodeName}{"\t"}{.status.hostIP}{"\t"}{.status.podIP}{"\t"}{.spec.hostNetwork}{"\t"}{.status.phase}{"\n"}'
```

Repeat with contexts `marin-gpu_US-EAST-02A` and `marin-rn02a_RNO2A`.

## Copy the probe

Set `PROBE` to the pinned checkout of `gpu_netcheck.py`.

```sh
KC=/home/romain/.kube/coreweave-iris
PROBE=/path/to/bench/diagnostics/gpu_netcheck.py

kubectl --kubeconfig "$KC" --context marin-us-east-08a_US-EAST-08A \
  -n iris cp "$PROBE" \
  iris/iris-romain-dev-gpu-ptb08a-0-e7f2bee0-0-3183010a92b7a6b2:/tmp/gpu_netcheck.py \
  -c task

kubectl --kubeconfig "$KC" --context marin-gpu_US-EAST-02A \
  -n iris cp "$PROBE" \
  iris/iris-romain-dev-gpu-ptb02a-0-347f1b1a-0-78eda07ba84e843d:/tmp/gpu_netcheck.py \
  -c task

kubectl --kubeconfig "$KC" --context marin-rn02a_RNO2A \
  -n iris cp "$PROBE" \
  iris/iris-romain-dev-gpu-ptbrno-0-85da3e1e-0-39378ec4fc2d0aa2:/tmp/gpu_netcheck.py \
  -c task
```

## Probe `cw-us-east-08a` ↔ `cw-us-east-02a`

Start these commands concurrently:

```sh
KC=/home/romain/.kube/coreweave-iris

kubectl --kubeconfig "$KC" --context marin-us-east-08a_US-EAST-08A \
  -n iris exec iris-romain-dev-gpu-ptb08a-0-e7f2bee0-0-3183010a92b7a6b2 \
  -c task -- sh -c \
  'PROBE_SELF=cw-us-east-08a/s62pys64 PROBE_PEERS=10.184.201.5 PROBE_ROUNDS=3 python3 /tmp/gpu_netcheck.py'

kubectl --kubeconfig "$KC" --context marin-gpu_US-EAST-02A \
  -n iris exec iris-romain-dev-gpu-ptb02a-0-347f1b1a-0-78eda07ba84e843d \
  -c task -- sh -c \
  'PROBE_SELF=cw-us-east-02a/g856b8a PROBE_PEERS=10.186.213.79 PROBE_ROUNDS=3 python3 /tmp/gpu_netcheck.py'
```

## Probe `cw-us-east-08a` ↔ `cw-rno2a`

Start these commands concurrently:

```sh
KC=/home/romain/.kube/coreweave-iris

kubectl --kubeconfig "$KC" --context marin-us-east-08a_US-EAST-08A \
  -n iris exec iris-romain-dev-gpu-ptb08a-0-e7f2bee0-0-3183010a92b7a6b2 \
  -c task -- sh -c \
  'PROBE_SELF=cw-us-east-08a/s62pys64 PROBE_PEERS=10.168.194.123 PROBE_ROUNDS=3 PROBE_HOLD=30 python3 /tmp/gpu_netcheck.py'

kubectl --kubeconfig "$KC" --context marin-rn02a_RNO2A \
  -n iris exec iris-romain-dev-gpu-ptbrno-0-85da3e1e-0-39378ec4fc2d0aa2 \
  -c task -- sh -c \
  'PROBE_SELF=cw-rno2a/gb95f50 PROBE_PEERS=10.186.213.79 PROBE_ROUNDS=3 PROBE_HOLD=30 python3 /tmp/gpu_netcheck.py'
```

## Release

```sh
cd /home/romain/dev/marin

uv run python scripts/iris/dev_gpu.py --name ptb08a release
uv run python scripts/iris/dev_gpu.py --name ptbrno release
```
