# GrugMoE inference preflight

This directory contains the immutable inputs, harness, live GB200 evidence
index, and proposed issue text for the preflight requested in issue
[#7201](https://github.com/marin-community/marin/issues/7201#issuecomment-5093392733).
It does not edit either source draft.

The result is a no-go for the architecture matrix. The exact model is not
representable by the frozen serving fork, and the required unattended
four-node launcher does not exist. See [findings.md](findings.md) for the
evidence and assumption-by-assumption status.

## Frozen inputs

- Marin base: `75bf2437035cf731d1a4bd71266229dfcdda9478`.
- Live EP8 evidence commit:
  `d043e51266650ee3db2ff041e1c2095fe443f55f`.
- vLLM: `afb26719464d5957e695bde478ae93a160b11d14`.
- Training reference: `fd3e9bc5b428633027f944be7fdf1136567db028`.
- Cluster: `cw-us-east-08a`.
- GPU: whole four-GPU GB200 nodes at Iris `interactive` priority.
- Runtime: BF16 weights and KV cache, seed 1234, prefix caching, chunked
  prefill, PP1, TP1, DP=EP.
- Image digest:
  `sha256:d90bc25fc778b9d4f5b9395cba4ac2457a12e106c4c2bcb4c0b9c7d70dd57dca`.

Future runs enrich `manifest.json` with the Iris job, pod placement, immutable
image IDs, exact commands, dependency-lock hash, and config/workload hashes.
The already-recorded live bundles keep these fields between `result.json`,
their pod records, and this run index.

## Local verification

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run --with pytest --with pytest-timeout --with numpy \
  pytest -q \
  experiments/grug/moe/test_inference_preflight.py \
  scripts/iris/tests/test_grugmoe_inference_preflight.py
```

Prepare and inspect the compact 18-root/144-branch fixture:

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run scripts/iris/grugmoe_inference_preflight.py prepare \
  --case reference-ep8 --run-id inspect --output /tmp/grugmoe-inspect
```

## Reproducing an interactive EP8 smoke

The allocation command stays attached in its own terminal:

```sh
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-08a.yaml \
  --name grugmoe-preflight-ep8 \
  allocate --gpu-variant GB200 --nodes 2 --priority interactive
```

Run every vLLM rank, assertion, log collection, upload, and readback with one
driver command:

```sh
PYTHONPATH=lib/iris/src:lib/marin/src \
  uv run scripts/iris/grugmoe_inference_preflight.py run \
  --session grugmoe-preflight-ep8 \
  --case reference-ep8 \
  --mode smoke \
  --run-id <UTC-run-id>
```

Release promptly:

```sh
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-08a.yaml \
  --name grugmoe-preflight-ep8 \
  release
```

The interactive command is not the required final acceptance entrypoint. It
uses workstation `kubectl` against replicated holder pods. Do not use
`--mode acceptance` as a substitute for the missing unattended Iris path.

## Artifact layout

Every live run writes under:

```text
s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture/<case>/<run-id>/
```

The driver uploads config, manifest, compact workload, full response JSON,
routed-expert arrays, Prometheus snapshots, result, and complete node logs.
It reads each object back and requires byte identity.

The exact live prefixes, holder jobs, outcomes, and failure reasons are in the
run index in [findings.md](findings.md).
