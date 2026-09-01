# The vLLM forks: TPU release, GPU release, and e2e

Fork-specific guidance for the `vllm` (TPU), `tpu-inference`, and `vllm-gpu`
pins. The generic workflow lives in `../SKILL.md`; this file holds what is
particular to these forks.

## The TPU vLLM release

Refresh the `vllm`/`tpu-inference` group as one unit and one PR. This group is
release-based: select its two upstream bases, replay the retained Marin overlays
to produce two reviewed source tips, build one public vLLM release whose metadata
installs the companion tpu-inference wheel, and put that one vLLM requirement in
Marin. The independent `vllm-gpu` pin keeps its existing release workflow.

Use **Read the Descriptor** and **Scratch setup** from `../SKILL.md`, then follow
this section instead of its generic **Select the base**, **Rebase the overlay**,
**Pin at the staged tip**, and **Prepare the protected-branch promotion** sections.
Resume at **Check the fork's own suite**, **Validate**, and **Review and Open the
PR**. Do not stage a temporary source descriptor, add a second tpu-inference
requirement, or promote a TPU stable branch.

The repositories have distinct roles:

| Source | `origin` (Marin fork) | `upstream` (canonical) |
|---|---|---|
| tpu-inference | `https://github.com/marin-community/tpu-inference.git` | `https://github.com/vllm-project/tpu-inference.git` |
| vLLM | `https://github.com/marin-community/vllm.git` | `https://github.com/vllm-project/vllm.git` |

Use these names throughout:

```text
tpu_base       = selected stable upstream tpu-inference release
tpu_source_tip = reviewed tpu-inference overlay replayed onto tpu_base

vllm_base       = LKG selected by that tpu-inference release
vllm_source_tip = reviewed vLLM overlay replayed onto vllm_base
```

The bases are inputs to the replay. The source tips are the producer inputs.

1. Read `config/external/vllm/tpu.toml`, then read the currently selected GitHub
   release notes and producer receipt. Record the exact current
   `tpu_source_tip` and `vllm_source_tip`, and verify each resolves in its Marin
   fork. Identify and verify the exact upstream base beneath each current source
   tip. Stop with a blocker if the current source tips or their bases cannot be
   established exactly; do not reconstruct an overlay from a net diff.
2. Select the newest canonical upstream tpu-inference release whose tag is
   exactly `vMAJOR.MINOR.PATCH` and whose GitHub release is neither draft nor
   prerelease as `tpu_base`. Read `.buildkite/vllm_lkg.version` at that base and
   resolve the named commit in canonical upstream vLLM as `vllm_base`. Inspect
   its TPU build metadata (`requirements/tpu.txt`, `pyproject.toml`, and
   `setup.py`) for dependency implications.
3. Replay each current Marin overlay onto its corresponding new base:
   - Inventory `current_base..current_source_tip` in logical order with
     `git log --reverse --no-merges`, retaining the original SHAs.
   - Classify every meaningful delta as `carry`, `drop`, or `fix`. Compare it
     with the new base first: drop changes upstream absorbed or made obsolete;
     carry changes still needed; rewrite a `fix` when the intent remains but the
     upstream API or layout moved. Merge commits are not replayed.
   - Replay only `carry` and `fix` in logical order. Audit every touched API and
     dependency against the new base, resolve incompatibilities, and run that
     fork's ordinary checks before proceeding.
   - Run `git range-diff` from the old overlay range to the new one. Record every
     original commit and its carry/drop/fix result, with the reason for each drop
     or rewrite. If an independent fixed-base overlay change is needed, land its
     review through `overlay-only-pr.md` before freezing either source tip; that
     workflow is not a substitute for this replay.
4. Review both replayed ranges and preserve their exact commits on reviewable
   branches in the Marin forks. Record the resulting full SHAs as
   `tpu_source_tip` and `vllm_source_tip`. Do not substitute the bare
   `tpu_base` or `vllm_base` commits.
5. Inspect the two source tips' dependency files together, including the
   tpu-inference torch constraints and vLLM TPU requirements. These commits are
   release evidence, not separate Marin runtime inputs.
6. Select one unused full `marin-vllm-tpu-...` release tag. Freeze that tag, the
   vLLM workflow producer commit, both source tips, and the dependency cutoff.
   Dispatch the TPU lane of `marin-gpu-candidate.yaml` at that producer commit
   with `vllm_source_tip`, `tpu_source_tip`, the tag, and the cutoff as explicit
   inputs. Dispatch it once.
7. Read back the public prerelease. It must contain exactly the vLLM wheel and its
   tpu-inference companion. Inspect the built vLLM wheel's `METADATA` and confirm
   its direct requirement names the companion's public release URL. Record the
   workflow run, producer commit, source tips, upstream bases, cutoff, tag, asset
   names, sizes, and digests as evidence.
8. Before using hardware, resolve the public vLLM requirement in a fresh uv tool
   environment. Confirm it installs both selected wheel versions. Repeat with an
   explicit `tpu-inference @ git+https://github.com/marin-community/tpu-inference@<head>`
   override and confirm uv selects that HEAD instead of the transitive release.
9. Edit only `config/external/vllm/tpu.toml`: the public release tag, dependency
   cutoff, and one vLLM requirement. Regenerate and check the typed object:

   ```sh
   uv run config/update-external.py vllm
   uv run config/update-external.py vllm --check
   ```

10. Run focused Marin checks, commit the final consumer diff, and freeze its full
    SHA as the Marin consumer head. Run the sole physical gate below against that
    exact SHA. Resume at **Review and Open the PR** in `../SKILL.md` with the TPU
    release and validation receipt. Do not add a second physical qualification or
    exact-byte protocol.
11. After validation and the producer change merges, mark the same GitHub release
    final without rebuilding or replacing either asset. Read back the unchanged
    asset IDs and digests before landing the Marin consumer.

Rebuild and rerun the physical gate only after a change that can affect the wheel
bytes or metadata, selected assets, producer path, Marin requirement, or launcher.
Documentation, tests, and PR-body edits do not invalidate the receipt.

For the draft PR body, record the upstream bases, reviewed source tips, producer
and release tag, dependency cutoff, immutable asset evidence, e2e outcome, and
unresolved risks. Include the carry/drop/fix record without branch-promotion,
rollback-tag, or staged-tip fields.

## The vLLM GPU release pipeline

The `vllm-gpu` pin tracks upstream vLLM head independently on the fork's `main`.
It is a prebuilt wheel, so the refresh stages its overlay on `main-next`, builds
and promotes one release through the fork's existing pipeline, and re-pins from
the promoted manifest. Dispatching fork workflows needs `actions:write` on
`marin-community/vllm`, which the fork-ferry profile grants.

1. Push `main-next` to the fork.
2. Build the candidate on that ref:
   `gh workflow run marin-gpu-candidate.yaml --repo marin-community/vllm --ref main-next`.
   The workflow otherwise builds on `push: main`; dispatching on `main-next`
   compiles the staged tip into both arches under a
   `marin-vllm-gpu-candidate-<sha>` prerelease.
3. Once the candidate prerelease exists, promote it:
   `gh workflow run marin-gpu-release.yaml --repo marin-community/vllm --ref main-next -f candidate_tag=<tag>`.
   The release job validates the exact wheel bytes on real GPUs and publishes an
   immutable release carrying `marin-vllm-gpu-manifest.json`.
4. Download that manifest and re-pin without hand-editing `gpu.toml`:
   `gh release download <release_tag> --repo marin-community/vllm --pattern marin-vllm-gpu-manifest.json`,
   then `uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json`.
   The helper writes `gpu.toml` (release tag, source commit, version, torch backend,
   per-arch URL and SHA-256) and regenerates `external_dependencies.py`; it
   re-encodes the wheel URLs the way the pin loader validates.

A base that crosses a CUDA/torch or vLLM stable-ABI boundary is a migration.
Re-audit the wheel verifier and the fork's release gate for the extension name
(`vllm._C_stable_libtorch` on CUDA 13) when the base moves across such a boundary.

## Fork suite caveat

Without its compiled CUDA/TPU stack, `py_compile` and a conflict-marker sweep are
structural checks only for vLLM; do not call them behavioral validation.

## End-to-end validation

- **`experiments/evals/served_qwen3.py::QWEN3_TPU_INFERENCE`** (`vllm`,
  `tpu-inference`) — the one physical gate for the selected public release. Run
  Qwen3-0.6B at tensor parallel size 8 on one `v6e-8` through Iris `marin`, in
  `us-east5`, at production priority on both the launcher and worker. Use a fresh
  run version so the worker cold-installs through `uvx`. Confirm the public
  requirement installs, the server becomes healthy, and a real request succeeds.
  Record the Iris job and child task IDs, release assets, all three frozen commits,
  probe result, and successful resource release:

```sh
uv run iris --config lib/iris/config/marin.yaml job run \
  --job-name served-qwen3-<run-id> --cpu 1 --memory 2G --extra cpu \
  --priority production --no-wait -- python -c \
  "from dataclasses import replace; from fray.types import ResourceConfig; from iris.rpc import job_pb2; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.lm_eval_suite import lm_eval_suite; from experiments.evals.served_qwen3 import QWEN3_TPU_INFERENCE; inference = replace(QWEN3_TPU_INFERENCE, model=replace(QWEN3_TPU_INFERENCE.model, tensor_parallel_size=8), iris=replace(QWEN3_TPU_INFERENCE.iris, worker_resources=ResourceConfig.with_tpu('v6e-8', ram='96g', regions=['us-east5']), priority=job_pb2.PRIORITY_BAND_PRODUCTION)); StepRunner().run([lower(lm_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>', limit=8))])"
```

Preserve this single physical receipt under the invalidation rule above.

- **`tests/cluster/vllm/test_snowball_backend_parity.py`** (`vllm-gpu`) — the
  Snowball-67B next-token logprob parity gate on H100s. It is `-m cluster` marked,
  so run it with `-o addopts= --import-mode=importlib`; the H100s live on CoreWeave
  and are reached through the Marin federation hub (`target_cluster`), not a
  direct controller. Confirm the Levanter reference and `vllm-gpu-pp1`
  (single-node 8xH100) match the goldens within `max_probability_error` (`pp2` is
  a 16xH100 multi-node variant). Pair it with a bounded qwen3-0.6B GPU serve+eval
  to exercise the brokered serving path:

```sh
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```
