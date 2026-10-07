# SkyRL IFEval grader image

The [Dockerfile](Dockerfile) acquires the unchanged `utils.py` scorer from
MarinSkyRL revision `544d5d6f14116a06bde0209352585903133bd618` and checks
its SHA-256 digest (`3194b7a44ada0a4cd185ab3c85d883f85da8641970fef3dfc4b66d13780079fc`).
It installs `langdetect==1.0.9` for language constraints. The image contains no
TaskCompendium code; converted tasks supply the small native-command runner and
private, normalized constraints. Grading runs with network access denied.

Build the image:

```bash
docker build --platform linux/amd64 \
  -t ghcr.io/marin-community/marinskyrl:ifeval-native-544d5d6-langdetect109 \
  experiments/post_training/task_curation/images/ifeval
```

Use the published image by digest in the source runtime manifest. The runner
hash-checks the installed scorer before loading it and calls the upstream
`compute_score` default, including its fractional reward and unknown-function
behavior. Nemotron instruction IDs are mapped during normalization; the scorer
receives the same function names and arguments as the source's original grader.
