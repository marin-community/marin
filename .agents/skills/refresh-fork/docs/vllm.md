# The vLLM source and device-specific releases

Fork-specific guidance for the `vllm`, `tpu-inference`, and `vllm-gpu` pins.
The generic workflow lives in `../SKILL.md`.

## Source topology

`marin-community/vllm/main` is the only maintained vLLM source lineage. The
`vllm-gpu` unit rebases that lineage and publishes GPU wheels. The `tpu-vllm`
unit selects an exact commit already on that lineage, pairs it with a refreshed
tpu-inference commit, and publishes separate TPU wheels. Never create, refresh,
or promote `vllm/tpu` or `vllm/tpu-next`.

The two release lanes have independent dependency environments, manifests,
qualification gates, and promotion inputs. Advancing tpu-inference does not
rebuild a GPU wheel. Promoting a GPU artifact does not rebuild the TPU pair.

## Refresh the TPU pair

Refresh `vllm` and `tpu-inference` as one `tpu-vllm` unit and one Marin PR. The
launcher installs both exact pins together.

1. Select the newest stable tpu-inference release and replay the Marin overlay
   onto `main-next`. Preserve the upstream `.buildkite/vllm_lkg.version`; it is
   evidence about upstream's tested pair, not a place for a Marin-only vLLM SHA.
2. Resolve the vLLM source from protected `marin-community/vllm/main`. A
   package source named by an immutable Marin release tag is also valid when a
   bounded diff proves that later `main` changes do not alter wheel contents.
   Record its full SHA, tag, upstream base, and package-equivalence command.
3. Audit the retired `vllm/tpu` overlay against that source. Classify every
   still-needed behavior as present, ported, or obsolete. Do not replay it onto
   another vLLM branch.
4. Try the exact pair. Run dependency resolution, both wheel builds, a clean
   install, the native/import boundary, and focused compatibility tests before
   using a TPU. Repair ordinary vLLM API drift in tpu-inference, preferably by
   rebasing onto a newer compatible upstream release or porting its fix.
5. If a correct repair requires changing vLLM source, stop the TPU-only path.
   Route the patch through the `vllm-gpu` source refresh below and require its
   GPU qualification as well as the TPU gate. Do not move vLLM back to an older
   TPU LKG.
6. Push the reviewed tpu-inference tip and dispatch the vLLM fork's candidate
   workflow from its reviewed workflow ref:

```sh
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm \
  --ref <reviewed-workflow-ref> \
  -f lane=tpu \
  -f vllm_commit=<full-main-line-vllm-sha> \
  -f tpu_inference_commit=<full-tpu-inference-sha> \
  -f exclude_newer=<whole-second-utc-cutoff>
```

The candidate tag binds both source SHAs, the workflow SHA, dependency cutoff,
and wheel hashes. Download the manifest and run `verify-candidate` before the
physical gate.

7. Qualify those exact public bytes without finalizing the release:

```sh
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref <same-reviewed-workflow-ref> \
  -f lane=tpu \
  -f candidate_tag=<exact-candidate-tag> \
  -f promote=false
```

This runs the Qwen3-0.6B TP8 serve-and-probe on one production-priority
`v6e-8` in `us-east5`. Monitor the exact workflow and Iris job. The workflow
owns cleanup for its job.

8. Set `config/external/vllm/tpu.toml` to the tested source SHAs. For vLLM,
   record the immutable source tag when one exists. Regenerate
   `external_dependencies.py`; no workspace lock changes are expected because
   TPU serving uses an isolated `uvx` environment.
9. Create rollback and date tags only for the rebased tpu-inference branch.
   The selected vLLM source is already reachable from `main` or its immutable
   release tag and needs no TPU-specific staging or promotion.

Keep the candidate as a prerelease until the reviewed source and Marin pin
changes land. The final `promote=true` dispatch must use the same workflow ref
and candidate tag so it reuses the qualified bytes.

## Refresh the GPU source and artifact

`vllm-gpu` alone rebases the vLLM overlay onto upstream head. Stage the result
on `main-next`, audit the CUDA, Torch, and stable-extension boundaries, then
build and qualify the exact candidate:

```sh
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm --ref main-next -f lane=gpu

gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm --ref main-next \
  -f lane=gpu -f candidate_tag=<exact-gpu-candidate-tag>
```

The release job validates the wheel on the configured GPU hardware and
publishes `marin-vllm-gpu-manifest.json`. Download that manifest and re-pin
without hand-editing `gpu.toml`:

```sh
gh release download <release-tag> \
  --repo marin-community/vllm \
  --pattern marin-vllm-gpu-manifest.json
uv run config/update-external.py \
  --promote-gpu-release marin-vllm-gpu-manifest.json
```

Promote `main-next` to protected `main` only after the GPU gate and review,
using the rollback tags and lease in `promotion-protocol.md`. A later TPU
refresh selects that main-line source independently.

## Fork suite caveat

Without its compiled CUDA or TPU stack, `py_compile` and a conflict-marker
sweep are structural checks only for vLLM. Do not call them behavioral
validation.

## Marin end-to-end gates

- The TPU release workflow above is the required exact-pair gate. Its manifest
  and qualification record must name the same vLLM SHA, tpu-inference SHA,
  workflow SHA, and wheel hashes.
- `tests/cluster/vllm/test_snowball_backend_parity.py` is the GPU model parity
  gate. Run it with `-m cluster -o addopts= --import-mode=importlib`; pair it
  with the fork release workflow's H100 serve smoke.

```sh
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```
