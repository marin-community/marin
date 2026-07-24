# Debugging log for finelog multi-architecture deploys

Publish and pin a finelog image that can run on either architecture in mixed
CoreWeave control-node pools.

## Initial status

The first `cw-us-east-08a` finelog rollout built and pushed a healthy
`linux/amd64` image, but Kubernetes scheduled the replacement pod on an
`arm64` node. The container exited immediately with `exec format error`.

The deployment was rolled back to its previous image. Its replacement pod
became Ready with zero restarts, a live query succeeded, the Iris controller
remained healthy, and all 40 running tasks remained running.

## Hypothesis 1

The finelog deploy builder defaulted to `linux/amd64` from its introduction in
May 2026. The scheduled image workflow also built only its amd64 runner's
native architecture. Previous deployments worked because their pods landed on
amd64 nodes rather than because the image was multi-architecture.

Publishing a manifest list alone is insufficient: finelog's tag-to-digest
resolver selected the amd64 child manifest from multi-architecture images,
which would still force every Kubernetes node to pull amd64 layers.

## Changes to make

- Build and push both `linux/amd64` and `linux/arm64` images from the deploy
  helper and scheduled workflow.
- Pin the top-level image-index digest so Kubernetes selects the child manifest
  matching the scheduled node.
- Cover both publishing and digest pinning boundaries with regression tests.

## Results

Pending.

## Future work

- [ ] Consider constraining finelog to a dedicated control-node pool once the
  CoreWeave clusters expose a stable scheduling label for that pool.
