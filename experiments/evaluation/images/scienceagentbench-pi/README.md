# ScienceAgentBench Pi actor

Build this directory for linux/amd64. The image contains Pi 0.87.0, Node 22 and the Python scientific packages used in Verified tasks 66 and 92. Other tasks may need different packages.

From the repository root, a builder with write access to the public rollout-images package can publish the image and record its registry digest:

```sh
docker buildx build --platform linux/amd64 --provenance=false --push \
  --tag "ghcr.io/marin-community/rollout-images:scienceagentbench-pi-$(git rev-parse --short=16 HEAD)" \
  --metadata-file /tmp/scienceagentbench-pi-build.json experiments/evaluation/images/scienceagentbench-pi
jq -er '."containerimage.digest"' /tmp/scienceagentbench-pi-build.json
```

Use the resulting digest in the Harbor actor configuration and its matching `environment/shellbox-images.json` declaration. Harbor uploads task inputs under `/setup_files/datasets`; the image exposes that directory at `/testbed/benchmark/datasets`. The build context admits only the Dockerfile and the Python lock. Regenerate the lock from this directory using the command in its header.
