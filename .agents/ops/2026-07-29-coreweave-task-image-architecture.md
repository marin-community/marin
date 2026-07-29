---
date: 2026-07-29
system: coreweave
severity: degraded
resolution: investigating
pr: none
issue: none
---

## TL;DR

- Iris task pods on `s4bk6j84`, `s6xvdgb4`, `s1zsxs64`, and `s5kvxs64` in `cw-us-east-08a` failed in `stage-workdir` with `exec /usr/local/bin/python: exec format error`.
- All four nodes reported `arm64`. Their cached `iris-task:latest` images resolved to single-platform `amd64` manifests.
- Healthy arm64 nodes kept running because `imagePullPolicy: IfNotPresent` reused older multi-platform images cached under the same mutable tag.
- The live controller config pinned `defaults.worker.default_task_image` to `:a88653b976` but left `kubernetes_provider.default_image` at `:latest`. `stage-workdir` used the latter.
- The code change pins the Kubernetes task image to the deploy tree SHA and stops Iris image publishers from writing remote `:latest` or date tags. Single-platform builds use `<sha>-<architecture>`.
- The live controller has not been restarted. Existing tasks can still resolve `iris-task:latest` until the fix is deployed.

## Original problem report

Four GB200 nodes in `cw-us-east-08a` failed every pod scheduled onto them. The
failure occurred in the `stage-workdir` init container:

```text
exec /usr/local/bin/python: exec format error
```

The same jobs ran on other nodes. The initial report suspected a corrupt
containerd snapshot or image-layer cache because placement determined the
outcome.

## Investigation path

1. `kubectl get nodes` showed that all four nodes were Ready, ran Ubuntu
   24.04 with containerd 2.1.4, and reported `arm64`. CoreWeave and NVIDIA
   DaemonSets were healthy, so the nodes could execute arm64 containers.

2. Preserved Iris pods showed exit code 255 in `stage-workdir`. Container logs
   contained the exact `exec /usr/local/bin/python: exec format error` message.
   Python never started, which excluded bundle download and `/app` workdir
   staging.

3. The failed pods used `ghcr.io/marin-community/iris-task:latest` with
   `imagePullPolicy: IfNotPresent`. `s1zsxs64` resolved it to
   `sha256:e403bc10c60a791b02536d024ab98bd3193bb3947847a29cfff17c3acc0b35f6`.
   `s4bk6j84`, `s6xvdgb4`, and `s5kvxs64` cached
   `sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`.

4. `docker buildx imagetools inspect --format '{{.Image.Architecture}}'`
   reported `amd64` for both failing digests. The two digests were valid images,
   so a corrupt local snapshot was not required to explain the failures.

5. Healthy nodes resolved the same mutable tag to older multi-platform indexes,
   including `sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`
   and `sha256:623e377c00ce92404465b63b534c532a8942ac030fc9d236929a5beb802ea3a7`.
   Both indexes contained `linux/arm64` and `linux/amd64` children.

6. The registry's `iris-task:latest` tag also pointed to a single `amd64`
   manifest during the investigation. This reproduced the architecture
   selection without reading containerd state from the host.

7. The live `iris-cluster-config` ConfigMap contained:

   ```text
   kubernetes_provider.default_image        ghcr.io/marin-community/iris-task:latest
   defaults.worker.default_task_image       ghcr.io/marin-community/iris-task:a88653b976
   controller.image                         ghcr.io/marin-community/iris-controller:a88653b976
   ```

   `lib/iris/src/iris/cli/cluster.py` pinned the worker default but omitted the
   Kubernetes provider default used by `stage-workdir`.

## User course corrections

- The investigation was moving toward a pinned-image repull as a cache test.
  The user stated that CoreWeave nodes require hard SHA pins because `latest`
  selects arbitrary cached versions. The registry and live ConfigMap evidence
  confirmed this mechanism without downloading the 1 GB task image onto four
  nodes.
- The initial prevention change would only have pinned the missed Kubernetes
  config field. The user asked to stop publishing `latest` as well. The image
  build paths and scheduled manifest job were expanded to publish SHA tags
  only.

## Root cause

`lib/iris/src/iris/cli/cluster.py` pinned
`defaults.worker.default_task_image` from `:latest` to the deploy tree SHA but
did not pin `kubernetes_provider.default_image`. Kubernetes task pod manifests
use the latter for `stage-workdir`. New arm64 GB200 nodes pulled whichever
single-platform manifest the mutable registry tag referenced; older nodes used
different cached content because the pod set `IfNotPresent`.

The publishers preserved the footgun. `lib/iris/src/iris/cli/build.py` added a
remote `:latest` tag to every pushed image, and
`.github/workflows/ops-docker-images.yaml` published merged Iris manifests
under SHA, date, and `latest` tags.

## Fix

`lib/iris/src/iris/cli/cluster.py` now pins
`kubernetes_provider.default_image` with the other deploy images:

```python
"kubernetes_task": config.kubernetes_provider.default_image
...
config.kubernetes_provider.default_image = pinned["kubernetes_task"]
```

`lib/iris/src/iris/cli/build.py` rejects remote `:latest` pushes. Multi-platform
images use the tree SHA; single-platform images use
`<tree-sha>-<architecture>` so a later amd64 build cannot overwrite a
multi-platform tag. Local non-push builds retain `latest` as a developer
convenience. `.github/workflows/ops-docker-images.yaml` now publishes merged
Iris manifests only under the commit SHA.

No live data repair was applied. Deploying the fix requires an explicitly
approved controller restart; the repository runbook prohibits an unapproved
restart.

## How OPS.md could have shortened this

The CoreWeave image troubleshooting section needed an architecture check for
pre-start `exec format error` failures. `lib/iris/docs/coreweave.md` now shows
how to compare `.status.nodeInfo.architecture` with each container's resolved
`imageID`, then inspect the registry object with
`docker buildx imagetools inspect`. This separates an architecture mismatch
from a damaged local snapshot before host access or cache deletion.

## Artifacts

- `lib/iris/src/iris/cli/cluster.py`
- `lib/iris/src/iris/cli/build.py`
- `.github/workflows/ops-docker-images.yaml`
- `lib/iris/docs/coreweave.md`
