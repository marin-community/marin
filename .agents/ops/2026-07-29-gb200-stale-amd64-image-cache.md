# GB200 nodes with a stale amd64 `iris-task` image cache

Cluster: `cw-us-east-08a`. Observed 2026-07-28 through 2026-07-29.

## Symptom

Gang-scheduled GB200 jobs failed before reaching model code. The terminal reason is
`Init:Error stage-workdir`, and the init container exits with:

```
exec /usr/local/bin/python: exec format error
```

Because Grug training gangs run with `max_task_failures=0`, a single bad member
atomically bounces all 16 tasks. One derisking leg (D-1a) lost 7 placements across two
rounds on 2026-07-28 and 12 more on 2026-07-29 before drawing a clean 16-node set on the
thirteenth attempt.

## Root cause

Three nodes have a stale `iris-task` image in their containerd cache whose payload is
built for the wrong architecture. The nodes are arm64:

| field | value |
| --- | --- |
| architecture | `arm64` |
| OS image | Ubuntu 24.04.4 LTS |
| kernel | `6.17.13-155-gdec5393c-coreweave-arm64` |
| container runtime | `containerd://2.1.4` |

The cached image is
`iris-task@sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.
Its `/usr/local/bin/python` is not an arm64 binary, so any pod that resolves to it dies
at exec. The failure is deterministic per node, not intermittent.

Of 208 nodes in the cluster, exactly three cache that image:

- `s4bk6j84`
- `s5kvxs64`
- `s6xvdgb4`

Healthy nodes carry different `iris-task` digests (`b4e966e1…`, `4f261437…`) and
typically several images; the three bad nodes cache only the failing one.

## Two wrong conclusions reached first

Both were recorded before the node-level evidence was gathered. They are listed so they
are not re-derived.

1. **"It is not the image — Running and Failed pods have overlapping digests."** That
   comparison was made at the *pod* level. At the *node* level the split is exact and
   the digest is precisely the discriminator. Compare `.status.images` on the node, not
   digests on the pod.
2. **"The bad nodes are a registration cohort from 2026-07-28 ~22:53 UTC."** `s1zsxs64`
   was implicated at one point and registered a day earlier. It does **not** cache the
   failing image — it holds a single unrelated iris image — so it either failed for a
   different reason or was misattributed. The bad set is three, not four.

## Diagnosis

Read-only. Lists every node holding the failing image:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris get nodes -o json \
  | jq -r '.items[]
           | select(.status.images[]?.names[]? | contains("cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425"))
           | .metadata.name'
```

To confirm a node's architecture and runtime:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris get nodes <node> \
  -o custom-columns='NODE:.metadata.name,ARCH:.status.nodeInfo.architecture,\
OS:.status.nodeInfo.osImage,RUNTIME:.status.nodeInfo.containerRuntimeVersion'
```

## Reproduction

Creates one CPU-only pod pinned to a suspect node. Not run during this investigation;
recorded for whoever holds the fix.

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris run archprobe --restart=Never \
  --image=<registry>/iris-task@sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425 \
  --overrides='{"spec":{"nodeName":"s4bk6j84"}}' \
  --command -- /usr/local/bin/python -c "import platform;print(platform.machine())"
```

Expected: `exec format error` on the three listed nodes, `aarch64` anywhere else.

## Working around it

`--max-retries 50` on the Iris submission. Retries land the gang elsewhere; this is what
carried the d67 family's 350-step legs through the same three nodes overnight. Levanter
now also fails fast on malformed placement (`bb1c8e76c`, uneven accelerator topology
domains) rather than hanging, which is what makes a retry cheap.

The cost is real: 12 wasted placements on one leg, roughly 12 minutes of scheduling
churn, and it scales with gang size because any one member can bounce the set.

## Root cause upstream, and resolution

Confirmed by Rafal Wojdyla and Russell Power on 2026-07-29 (Slack `C0AHF5KV11Q`, thread
`1785343376.364349`), and consistent with the node-level evidence above.

The k8s deployment pulls `ghcr.io/marin-community/iris-task:latest` — a mutable tag —
under an `IfNotPresent` image-pull policy. A node that had cached `latest` while it
briefly pointed at an amd64 build keeps serving that cached layer indefinitely, because
`IfNotPresent` never re-resolves the tag. GCP deployments lock to a hash; that locking
was lost for the k8s deployments through a duplicated config. Nodes swapped in during
the window picked up whatever `latest` resolved to at the time.

Resolution applied the same day:

1. Images rebuilt (Actions run `30475812427`).
2. The amd64 images deleted from the affected nodes.
3. arm64 images refetched and pinned on those nodes.
4. An alert planned for the arch-mismatch case, and a PR to lock the k8s deployments to
   a hash rather than `latest`.

Verified after the fix: the diagnosis command above returns **no** nodes holding
`cfe4e8dd…`.

The durable fix is (4) — pinning the digest. Until that lands, the same failure can
recur on any node that caches a `latest` that briefly pointed at the wrong
architecture.

## What was not done

Cordoning was considered and explicitly rejected: it changes shared cluster state and
affects other users, and the correct fix turned out to be a per-node cache eviction that
costs no capacity. No node was cordoned, tainted, drained, or patched during this
investigation, and no other user's job was touched.
