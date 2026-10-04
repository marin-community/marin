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

Prefer one exact vLLM source commit for the GPU and TPU lanes when it passes
both device gates, but never make one lane wait solely to preserve alignment.
If compatibility or qualification requires different source pins, record the
reason and prefer a later common commit once it passes both gates. The
artifacts, dependency environments, gates, promotion inputs, and release
cadences remain independent even while the source commits match.

## Refresh the TPU pair

Refresh `vllm` and `tpu-inference` as one `tpu-vllm` unit and one Marin PR. The
launcher installs both exact pins together.

1. Select the newest stable tpu-inference release and replay the Marin overlay
   onto `main-next`. Preserve the upstream `.buildkite/vllm_lkg.version`; it is
   evidence about upstream's tested pair, not a place for a Marin-only vLLM SHA.
2. Resolve the vLLM source from protected `marin-community/vllm/main`. A
   package source named by an immutable Marin release tag is also valid when a
   bounded diff proves that later `main` changes do not alter wheel contents.
   Record its full SHA and upstream base in the descriptor. Cite the tag and
   package-equivalence command in the PR.
3. Try the exact pair. Run dependency resolution, both wheel builds, a clean
   install, the native/import boundary, and focused compatibility tests before
   using a TPU. Repair ordinary vLLM API drift in tpu-inference, preferably by
   rebasing onto a newer compatible upstream release or porting its fix.
4. If a correct repair requires changing vLLM source, stop the TPU-only path.
   Route the patch through the `vllm-gpu` source refresh below and require its
   GPU qualification as well as the TPU gate. Do not move vLLM back to an older
   TPU LKG.
5. Push the reviewed tpu-inference tip and dispatch the vLLM fork's candidate
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

6. Qualify those exact public bytes without finalizing the release:

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

7. Set `config/external/vllm/tpu.toml` to the tested source SHAs and upstream
   bases. Cite an immutable vLLM source tag or `main` ancestry proof in the PR.
   Regenerate `external_dependencies.py`; no workspace lock changes are expected
   because TPU serving uses an isolated `uvx` environment.
8. Create rollback and date tags only for the rebased tpu-inference branch.
   The selected vLLM source is already reachable from `main` or its immutable
   release tag and needs no TPU-specific staging or promotion.

Keep the candidate as a prerelease until the reviewed source and Marin pin
changes land. Then dispatch the merged release workflow with the same candidate
tag, the accepted qualification run ID, and `promote=true`. The promotion path
verifies that successful run and its artifact against the candidate before
publishing the same bytes; it does not allocate another TPU.

```sh
gh workflow run marin-gpu-release.yaml \
  --repo marin-community/vllm \
  --ref main \
  -f lane=tpu \
  -f candidate_tag=<same-exact-candidate-tag> \
  -f qualification_run_id=<successful-qualification-run-id> \
  -f promote=true
```

## Refresh the GPU source and artifact

GPU wheels build only from `main`. The same process applies after an upstream
refresh or a patch authored in the fork.

1. Prepare and review the vLLM source change. For an upstream rebase, replay
   the overlay on `main-next` and audit the CUDA, Torch, and stable-extension
   boundaries. An admin reviews the source and completes the backed-up swap to
   `main` described in `promotion-protocol.md` before wheel builds. For an
   ordinary patch, merge its reviewed vLLM PR to `main`. Keep the build and
   release automation in the selected source.
2. The build workflow builds both wheels, runs the existing H100 and GB200
   qualification, and publishes one release after both jobs pass. Source
   changes on `main` trigger it automatically. To start it manually:

```sh
gh workflow run marin-gpu-candidate.yaml \
  --repo marin-community/vllm --ref main -f lane=gpu
```

Builds and qualification use the same commit and temporary Actions artifacts.
The published tag is `marin-vllm-gpu-<source-UTC-date>-<12-character-sha>`.
Monitor the exact workflow and Iris jobs. The workflow owns cleanup for its jobs.
A failed qualification publishes no GPU release; fix the source or rerun the
failed jobs in that workflow.

3. In an isolated Marin worktree, download the published manifest, update the
   exact wheel pins, and run Snowball parity:

```sh
gh release download <release-tag> \
  --repo marin-community/vllm \
  --pattern marin-vllm-gpu-manifest.json --dir release
uv run config/update-external.py \
  --promote-gpu-release release/marin-vllm-gpu-manifest.json
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```

Open one draft Marin PR with `gpu.toml`, the generated pins, the full source
SHA, both wheel hashes, and links to the fork qualification and Snowball results.
Review and merge that PR to adopt the wheels. There is no temporary candidate
pin or later GPU publication step.

Source approval precedes GPU qualification and Marin validation. Marin keeps
its previous wheel until the adoption PR merges. If Snowball fails, leave the
Marin pins in production unchanged and fix the failure before adoption. A later
TPU refresh selects a main-line source independently.

## Fork suite caveat

Without its compiled CUDA or TPU stack, `py_compile` and a conflict-marker
sweep are structural checks only for vLLM. Do not call them behavioral
validation.

## Marin end-to-end gates

- The TPU release workflow above is the required exact-pair gate. Its candidate
  manifest must name both source SHAs, the workflow SHA, and both wheel hashes;
  its qualification record must name that exact candidate tag.
- `tests/cluster/vllm/test_snowball_backend_parity.py` is the GPU model parity
  gate. Run it with `-m cluster -o addopts= --import-mode=importlib`; require
  both H100 and GB200 qualification results from the fork release workflow.

```sh
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```
