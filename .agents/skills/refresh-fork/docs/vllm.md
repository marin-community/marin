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
   install, compiled extension imports, and focused compatibility tests before
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

### Triggers and tests

| Event | Who starts the work | Checks and result |
| --- | --- | --- |
| vLLM source PR opened or updated | Automatic PR CI | CPU import, release-metadata, and lint checks. |
| Relevant push after a reviewed patch or refresh reaches vLLM `main` | Automatic fork build workflow | Build both wheels, run the existing H100 and GB200 wheel tests and serving smoke, then publish once. |
| Manual GPU build on vLLM `main` | Developer or refresh agent | Run the same build, qualification, and publication workflow. Use this for release-automation changes excluded from the automatic trigger. |
| Published wheel selected for Marin | Adoption agent in an isolated Marin worktree | Update exact pins, run the full Snowball cluster parity test, and open a Marin adoption PR with the results. |
| Marin adoption PR opened or updated | Automatic PR CI, then review | Run ordinary affected CI. Merge the reviewed pin PR to use those wheels in Marin. |

The fork workflow's `paths-ignore` lists documentation, release-automation, and
cache-maintenance files excluded from automatic builds. The full Snowball
cluster test needs an explicit agent or developer launch and is excluded from
ordinary PR CI. It compares Snowball token probabilities with stored reference
values through Marin's Levanter and vLLM backends. This exercises the model and
application paths beyond the fork's small Qwen serving smoke.

vLLM's scheduled nightly smoke checks fork health separately from the
exact-wheel release and adoption gates. A MarinSkyRL dependency refresh also
qualifies Snowball and Hero on H100 and GB200 in its combined training and
serving profile. The trainer owner's adoption PR records those results.

Publication makes the wheels available after the fork's wheel tests pass.
Marin adoption waits for its application tests and the reviewed pin PR. A failed
Marin test leaves production pins unchanged.

### Source, release, and adoption

1. Prepare and review the vLLM source change. For an upstream rebase, replay
   the overlay on `main-next` and audit the CUDA/Torch versions and compiled
   Torch extension interface. An admin reviews the source and completes the
   backed-up swap to `main` in `promotion-protocol.md` before wheel builds. For an
   ordinary patch, merge its reviewed vLLM PR to `main`. Keep the build and
   release automation in the selected source.

   For an unattended rebase, give the admin the `main-next` compare link, old
   and new source SHAs, replay audit, and rollback and date tags. Wait for the
   admin's review and source promotion before building.

2. The build workflow builds both wheels, runs the existing H100 and GB200
   qualification, and publishes one release after both jobs pass. Relevant
   pushes to `main` trigger it automatically. To start it manually:

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

Old staged GPU candidates remain test evidence. Build a new release through
the main-only workflow before adoption; its GPU jobs test that run's wheels.
For Marin adoption, reuse earlier Marin results only when the exact wheel,
dependency profile, and tested behavior match. State what was reused in the PR.

## Fork suite caveat

Without its compiled CUDA or TPU stack, `py_compile` and a conflict-marker
sweep are structural checks only for vLLM. Do not call them behavioral
validation.

## Marin end-to-end gates

- The TPU release workflow above is the required exact-pair gate. Its candidate
  manifest must name both source SHAs, the workflow SHA, and both wheel hashes;
  its qualification record must name that exact candidate tag.
- The GPU adoption step above runs Snowball parity after both H100 and GB200
  wheel tests pass.
