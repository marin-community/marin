# GrugMoE inference preflight

This directory contains the exact serving preflight requested for the frozen
GrugMoE reference. It includes the final findings, compact proposed edits for
the architecture and protocol drafts, and unchanged issue drafts.

The result is **GO for the exact serving baseline and a later, separately
authorized architecture experiment**. Exact tensor parity, every P0
implementation family, prefix behavior, live 65,536-token KV accounting,
unattended EP8, and unattended EP16 acceptance all pass. See
[findings.md](findings.md) for the evidence and limitations.

The accepted throughput comes from deterministic dummy weights. The one
bounded trained Snowball attempt failed before load on object-store access, so
it is not trained-model performance evidence.

## Pinned implementation

- Marin branch:
  `https://github.com/marin-community/marin/tree/grugmoe-inference-preflight`
- Marin live acceptance commit:
  `a3320a3043018ee923bc98bf2e6e6eef3f03a6fe`
- vLLM branch:
  `https://github.com/marin-community/vllm/tree/grugmoe-inference-preflight`
- vLLM evidence commit:
  `2c2bef33dfbd7aef3c9d4433a7e4110f77d56a4a`
- vLLM reviewed branch head:
  `cdfde7e24d8aa3339b4f22444db7b45d43e018fa` (post-evidence fail-closed
  tensor-shape guard only)
- Training oracle:
  `fd3e9bc5b428633027f944be7fdf1136567db028`
- Immutable task image:
  `ghcr.io/marin-community/iris-task@sha256:5e2a69af91a000cb999e6ff0d92933874bd3142eb45469fc64fc7a3f5db64fbb`

## Local verification

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run pytest -q \
  experiments/grug/moe/test_inference_preflight.py \
  tests/cluster/vllm/backend_parity.py \
  tests/cluster/vllm/test_grug_exact_reference_check.py \
  scripts/iris/tests/test_grugmoe_inference_preflight.py \
  tests/inference/test_serve.py
```

Prepare the compact 18-root/144-branch workload without allocating GPUs:

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run scripts/iris/grugmoe_inference_preflight.py prepare \
  --case exact-reference-ep16 \
  --run-id inspect \
  --output /tmp/grugmoe-inspect
```

## Unattended Iris path

`submit` is the path qualified at two and four nodes. It creates one
zero-retry Iris task per whole four-GPU GB200 node, hard-coschedules the tasks
in one NVLink domain, starts every vLLM rank, runs assertions, collects logs,
uploads the bundle, and reads it back.

Example smoke:

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run scripts/iris/grugmoe_inference_preflight.py submit \
  --case reference-ep8 \
  --model-source dummy \
  --mode smoke \
  --run-id <UTC-run-id> \
  --task-image \
  ghcr.io/marin-community/iris-task@sha256:5e2a69af91a000cb999e6ff0d92933874bd3142eb45469fc64fc7a3f5db64fbb \
  --config lib/iris/config/cw-us-east-08a.yaml \
  --wait
```

The acceptance mode is frozen to `exact-reference-ep16`. Do not rerun the
completed acceptance or substitute `granular-ep16`.

## Artifact layout

Every live run writes under:

```text
s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/<case>/<run-id>/
```

Bundles contain configuration, workload, commands, dependency lock, exact
commits, image digest, placement, full response JSON, routed-expert arrays,
Prometheus snapshots, per-rank receipts and logs, aggregate result, and a
byte-hash manifest. Every bundle claimed in [findings.md](findings.md) was
verified by a separate authorized reader job.
