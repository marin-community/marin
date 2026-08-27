---
name: refresh-fork
description: Refresh a named Marin external fork pin onto a newer upstream base using its configured descriptor and required end-to-end test.
---

# Refresh a fork

Read `AGENTS.md`. Refresh a `group` as one unit and one PR.

The `vllm`/`tpu-inference` group is release-based. Select its two exact source
commits, build one public vLLM release whose metadata installs the companion
tpu-inference wheel, and put that one vLLM requirement in Marin. Follow
**Refresh the TPU vLLM release** below and then stop; the generic fork-promotion
procedure does not apply to this group.

Other overlay forks stage on `<branch>-next`, validate that tip, re-pin Marin,
and prepare protected-branch promotion. The independent `vllm-gpu` pin keeps
its existing release workflow.

In local mode, ask before pushing fork branches, opening the PR, or filing an
issue. The required end-to-end test needs no extra confirmation.

## Read the Descriptor

Read the target fork's section in `config/external/migration.toml`. It gives:

- `upstream` — the repo we rebase onto. Every fork has one.
- `group` — if present, refresh every section in the group together in one PR
  (read them all now); if absent, this pin refreshes alone.
- `base_select` (+ `derived_from`) — how to choose the new upstream base.
- `pin` — where the resolved pin is recorded (`isolated_project` uv.lock or
  `release:<path>` public wheel); drives the re-pin step. Two grouped source
  sections may name the same release descriptor when one artifact selects the
  complete environment.
- `branch` — for a branch-based refresh, the stable fork branch. The refresh
  stages on `<branch>-next`; an admin promotes that validated tip after reviewing
  the draft Marin PR.
- `e2e` — the Marin end-to-end that validates the refresh.
- `blocker_assignee` — who owns the "can't migrate" issue.
- `nuances` — constraints a human must respect (torch pins, known-good ceilings).

The descriptor records *how* to migrate; the pin source it names holds the actual
revision.

## Outcome

- If no newer base is selected and no pin metadata needs repair, exit successfully
  with a no-op summary.
- On a successful TPU vLLM refresh, open one draft Marin PR with the selected
  public vLLM requirement after the e2e passes. The release notes carry the
  producer and both source commits; Marin does not pin those commits separately.
- On another successful refresh, create the rollback and date tags described in
  `docs/promotion-protocol.md`, then open exactly one draft Marin PR after the e2e
  passes. State the exact `<branch>-next` to `<branch>` admin promotion still
  required and request the descriptor's `blocker_assignee` as reviewer.
- On an unresolved blocker, do not open a PR. Create or update one
  `marin-community/marin` issue assigned to `blocker_assignee`, titled
  `Fork refresh blocked: <fork> — <short reason>`, with current pins, the selected
  base, branch names/SHAs if created, attempted fixes, the remaining failure, and
  artifacts.

## Scratch setup

- Scratch dir: `/tmp/marin-fork-refresh/<run-id>` (run id:
  `${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}` in Actions, else a UTC timestamp plus a
  short label).
- Clone the fork and add its `upstream` remote. The fork URL is the
  `[tool.uv.sources]` git entry for isolated projects or the release-asset host
  for release pins; `<upstream>` is this section's `upstream`.

```sh
git clone <repository> <fork>
git -C <fork> remote add upstream <upstream>
git -C <fork> fetch --tags --multiple origin upstream   # --multiple fetches both remotes; without it "upstream" is read as a refspec
git -C <fork> remote set-head upstream -a                # so upstream/HEAD resolves
```

- Record selected bases, branch SHAs, carry/drop/fix decisions, validation, and
  unresolved risks for the PR.

## Refresh the TPU vLLM release

This is the complete procedure for the `tpu-vllm` group. Do not stage a
temporary source descriptor or add a second tpu-inference requirement.

1. Select the newest stable tpu-inference release. Resolve its exact fork commit
   and read its `.buildkite/vllm_lkg.version`. Resolve that exact vLLM commit in
   `marin-community/vllm`. If either fork needs an overlay change, review and land
   that change through `docs/overlay-only-pr.md` before selecting the commit.
2. Record the full vLLM and tpu-inference commits. Inspect their dependency files
   together, including the tpu-inference torch constraints and vLLM TPU
   requirements. These commits are release evidence, not Marin inputs.
3. Select one unused full `marin-vllm-tpu-...` release tag. Freeze that tag, the
   vLLM workflow producer commit, both source commits, the dependency cutoff, and
   the Marin consumer head. Dispatch the TPU lane of `marin-gpu-candidate.yaml`
   at that producer commit with the tag, two source commits, and cutoff as
   explicit inputs. Dispatch it once.
4. Read back the public prerelease. It must contain exactly the vLLM wheel and its
   tpu-inference companion. Inspect the built vLLM wheel's `METADATA` and confirm
   its direct requirement names the companion's public release URL. Record the
   workflow run, producer commit, source commits, cutoff, tag, asset names, sizes,
   and digests as evidence.
5. Before using hardware, resolve the public vLLM requirement in a fresh uv tool
   environment. Confirm it installs both selected wheel versions. Repeat with an
   explicit `tpu-inference @ git+https://github.com/marin-community/tpu-inference@<head>`
   override and confirm uv selects that HEAD instead of the transitive release.
6. Edit only `config/external/vllm/tpu.toml`: the public release tag, dependency
   cutoff, and one vLLM requirement. Regenerate and check the typed object:

   ```sh
   uv run config/update-external.py vllm
   uv run config/update-external.py vllm --check
   ```

7. Run focused Marin checks, then run the sole physical gate in **Validate**.
   Open one draft Marin PR with the release and validation receipt. Do not add a
   second physical qualification or exact-byte protocol.
8. After validation and the producer change merges, mark the same GitHub release
   final without rebuilding or replacing either asset. Read back the unchanged
   asset IDs and digests before landing the Marin consumer.

Rebuild and rerun the physical gate only after a change that can affect the
wheel bytes or metadata, selected assets, producer path, Marin requirement, or
launcher. Documentation, tests, and PR-body edits do not invalidate the receipt.

## Select the base

- `base_select = upstream_main` (`evalchemy`, `harbor`, `MarinSkyRL`, `vllm-gpu`): the base is the
  tip of the `upstream` default branch. These pins rebase onto upstream `main`; there is no
  release to gate on. `vllm-gpu` tracks vLLM head this way, distinct from the
  tpu-inference-blessed source selected for the TPU release.
- `base_select = latest_release` (`tpu-inference`): use GitHub Releases of the
  fork's `upstream`; do not use raw tags or branches. Select the newest release
  where `draft == false`, `prerelease == false`, and the tag is exactly
  `vMAJOR.MINOR.PATCH`. Resolve it to a commit SHA.
- `base_select = derived` (`vllm`): read the SHA at `derived_from`
  (`tpu-inference:.buildkite/vllm_lkg.version`) from the selected `tpu-inference`
  release. That exact SHA is the base; verify it resolves in the fork's `upstream`.
  Inspect its TPU build metadata (`requirements/tpu.txt`, `pyproject.toml`,
  `setup.py`) for dependency implications.

If the selected base matches the current one and no pin metadata needs repair, exit
no-op. Do not walk back to older releases when the latest eligible one fails; fix
the refresh or file a blocker issue.

For an isolated fork whose `upstream` base has not moved, there is nothing to rebase
and the refresh is a no-op — even if Marin's pin lags the fork's own `main`. Adopting
patches pushed to the fork since Marin last locked belongs to the daily
external-dependency bump (`ops-external-dependencies`); refresh-fork runs only when
there is a newer upstream base to rebase onto.

## Rebase the overlay

This section applies only to non-TPU refreshes.

Branch from the selected base as `<branch>-next` (the pin's `branch` with a `-next`
suffix). Single-pin forks and the vLLM GPU pin use `main-next`. This staging branch is disposable — a re-run
force-updates it — and is distinct from the protected stable `<branch>`, which the
unattended refresh leaves unchanged.

Find the base our commits currently sit on. For an isolated pin, `old_tip` is the
fork's `main` because Marin's recorded lock may lag the fork. For the `vllm-gpu`
release pin, `old_tip` is `gpu.toml`'s `source_commit`. Resolve `old_base` with
`git merge-base <old_tip> upstream/HEAD`. Then, onto `new_base`:

1. Inventory our commits in order: `git log --reverse --no-merges old_base..old_tip`.
   Merge commits (especially merges of `upstream` into a feature branch) are not
   replayed — their content comes from the new base; drop them.
2. Classify each meaningful delta: `carry` (still needed, not upstreamed), `drop`
   (upstream absorbed it, obsolete, or temporary), `fix` (intent needed,
   implementation must change — re-author against the current layout when upstream
   moved or refactored the files it touches). Before carrying anything, check whether
   the new base already did it: grep the base for the symbols, APIs, or dependency
   pins the patch introduces. Drop an absorbed backport or stale version ceiling;
   do not re-add duplicate or obsolete code.
3. Replay only `carry` and `fix` onto `new_base` in the old logical order: clean
   cherry-picks for carries; rewrite fixes as new commits referencing the original
   SHA(s). Separate genuine conflicts from cascade artifacts: a file the overlay
   created (absent from the new base) conflicts only because an earlier commit that
   added it was skipped. Classify each touched path as upstream-shared (exists at both
   bases) or fork-new, resolve fork-new cascades mechanically, and count conflicts only
   on shared files — a single abort-on-conflict pass badly overcounts and can read a
   tractable rebase as intractable.
4. In every retained commit body, state why it is still needed and its future drop
   condition. For non-obvious patches, leave a short code-adjacent rationale.
5. Run `git range-diff old_base..old_tip new_base..<new_tip>` as the replay audit
   and explain every dropped or rewritten delta in the notes and PR.
6. Audit the overlay's call-sites against the new base's API before the build. A
   clean cherry-pick applies textually but does not prove the upstream symbols the
   overlay imports or calls still exist: on a fast-moving fork a class becomes a
   factory function, a helper is deleted, a signature gains a required argument.
   Cross-check every touched constructor, signature, attribute, helper, and test
   against `new_base` before a multi-hour build. A prior vLLM refresh caught
   `FusedMoE` becoming `FusedMoEFactory` and removal of `is_interleaved` this way.
7. Keep history reviewable — no conflict artifacts, unrelated refactors, or
   preserved commits whose behavior is now `drop`. Collapse fork-infra churn
   (CI, workflow, or prose commits that adopt then revise then disable) to its final
   state rather than replaying each hop.

Stop and file a blocker when the overlay is non-linear, upstream refactored
touched files that need substantial re-authoring, or many core files conflict.
A fork hundreds of commits behind also needs a one-time manual catch-up outside
this workflow. That catch-up replays the real commits; never reconstruct the
overlay from a net diff. Diff each hand-ported `fix` against the fork's stable
branch. Put the inventory, conflict map, and distance behind upstream in the
blocker issue.

## Pin at the staged tip

This section applies only to non-TPU refreshes. The TPU group already stopped
after **Refresh the TPU vLLM release**.

Point Marin at `<branch>-next` so the e2e runs against the replayed code, then run
`uv run config/update-external.py` to regenerate
`lib/marin/src/marin/external_dependencies.py`; confirm only the intended pins change.
The stable `<branch>` remains at the old tip until an admin hard-swaps it after reviewing
the draft PR. Because `<branch>-next` and the eventual `<branch>` are the same commit,
the pin set here needs no change after that promotion.

- `pin = release:<path>` (`vllm-gpu`): the pin is a prebuilt wheel, so the refresh builds
  and promotes one through the fork's own release pipeline, then re-pins from the promoted
  manifest. Dispatching fork workflows needs `actions:write` on `marin-community/vllm`,
  which the fork-ferry profile grants.
  1. Push `main-next` to the fork.
  2. Build the candidate on that ref:
     `gh workflow run marin-gpu-candidate.yaml --repo marin-community/vllm --ref main-next`.
     The workflow otherwise builds on `push: main`; dispatching on `main-next` compiles the
     staged tip into both arches under a `marin-vllm-gpu-candidate-<sha>` prerelease.
  3. Once the candidate prerelease exists, promote it:
     `gh workflow run marin-gpu-release.yaml --repo marin-community/vllm --ref main-next -f candidate_tag=<tag>`.
     The release job validates the exact wheel bytes on real GPUs and publishes an immutable
     release carrying `marin-vllm-gpu-manifest.json`.
  4. Download that manifest and re-pin without hand-editing `gpu.toml`:
     `gh release download <release_tag> --repo marin-community/vllm --pattern marin-vllm-gpu-manifest.json`,
     then `uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json`.
     The helper writes `gpu.toml` (release tag, source commit, version, torch backend,
     per-arch url+sha256) and regenerates `external_dependencies.py`; it re-encodes the wheel
     URLs the way the pin loader validates, which a hand copy gets wrong.

  A base that crosses a CUDA/torch or vLLM stable-ABI boundary is a migration. Re-audit the
  wheel verifier and the fork's release gate for the extension name (`vllm._C_stable_libtorch`
  on CUDA 13) when the base moves across such a boundary.
- `pin = isolated_project` (`evalchemy`, `harbor`, `MarinSkyRL`): the uv source follows
  the fork's `main`, so `main` is the stable branch. Stage the rebase on `main-next`,
  review it from a compare link (`upstream_base..main-next`) on the Marin PR, and point
  the uv source at `main-next` to validate. After the e2e passes, run
  `uv run config/update-external.py <fork>` to lock `config/external/<fork>/uv.lock`
  against that exact tip. Keep the source on `main-next` in the draft PR while `main`
  still points at the old tip; the date tag keeps the staged SHA reachable. After an
  admin advances `main`, restore the source to `main`, rerun
  `uv run config/update-external.py <fork>`, and verify the lock still records the
  validated SHA. Push that follow-up to the same PR before marking it ready or merging
  it. Coordinate with the daily external-dependency bump, which also follows `main`.

Respect the section's `nuances`. Manual fixed-base overlay changes are a separate
workflow; see `docs/overlay-only-pr.md`.

## Check the fork's own suite

Run the fork's suite before the Marin end-to-end test, locally when supported or
through fork CI.

Derive the command from the fork's CI config verbatim; do not invent a marker
subset. A narrow marker such as `-m unit` can silently skip the thousands of unmarked
tests the CI's real expression (`-m "not runtime"`) collects, so the narrow run reads
green while most of the suite never ran. CI steps run in order under an implicit
`if: success()`: a later gated step (a `-m runtime` docker leg) does not run until an
earlier one is green, so its regressions stay hidden behind the first failure. Run
every step's marker in order.

On this VM, Docker bind mounts do not propagate into containers, so Harbor's
DOCKER-env golden tests must run in fork CI. Confirm the workflow supports
`workflow_dispatch` on the staged ref; otherwise it may silently skip a
non-`main` review branch.

For vLLM without its compiled CUDA/TPU stack, `py_compile` and a conflict-marker
sweep are structural checks only; do not call them behavioral validation.

For a version-sensitive golden, inspect the deciding code path and distinguish a
new dependency floor from a port defect. An upstream `litellm>=1.92` floor, for
example, can stale a golden without a fork-source change. Never downgrade below
the floor. Prefer a dependency-independent fixture that patches every seam the
trigger reads, such as both sync and async token counters. Otherwise regenerate
and verify it in one CI run with logs as artifacts.

## Validate

Run the descriptor's `e2e` before opening the PR:

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

- **`tests/cluster/vllm/test_snowball_backend_parity.py`** (`vllm-gpu`) — the
  Snowball-67B next-token logprob parity gate on H100s. It is `-m cluster` marked, so
  run it with `-o addopts= --import-mode=importlib`; the H100s live on CoreWeave and are
  reached through the marin federation hub (`target_cluster`), not a direct controller.
  Confirm the Levanter reference and `vllm-gpu-pp1` (single-node 8×H100) match the
  goldens within `max_probability_error` (`pp2` is a 16×H100 multi-node variant). Pair
  it with a bounded qwen3-0.6B GPU serve+eval to exercise the brokered serving path:

```sh
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```

- **`experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml`** (`evalchemy`) and
  **`experiments/evaluation/configs/harbor/aime-smoke.yaml`** (`harbor`) — one eval
  runner drives both; only the config source differs. Submit a bounded run and
  confirm it completes with metrics and no fork import/build/runtime traceback:

```sh
uv run python -m experiments.evaluation.cli launch --model qwen3-0.6b --limit 8 \
  --evalchemy-config experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml   # evalchemy
uv run python -m experiments.evaluation.cli launch --model qwen3-0.6b --limit 8 \
  --harbor-config experiments/evaluation/configs/harbor/aime-smoke.yaml          # harbor
```

  `--dry-run` resolves the model, backend, and task plan — a wiring pre-check, not
  fork validation. Depending on the eval it may not import the fork at all, or import
  it without exercising it, so a green dry-run says nothing about whether the new pin
  runs. Only the live run above validates the refreshed fork.

- **`experiments/post_training/iceball_micro.py`** (`MarinSkyRL`) — the micro
  post-training e2e. `--version` is required and `--run` builds the handles
  (without it the command only prints the plan); confirm the terminal stage
  completes:

```sh
uv run python -m experiments.post_training.iceball_micro --stage evaluation --version dev --run
```

For a non-TPU refresh, when an e2e fails, rerun the same workload against Marin's
current pin on the old stack, with the same target and priority. Fix only failures
that pass on the old stack and regress on the refreshed one. For TPU, preserve the
single physical receipt. Rerun only after a relevant producer, asset, requirement,
or launcher change, or when the first run failed before it exercised the release.

## Prepare the protected-branch promotion

This section applies only to non-TPU refreshes. Once the e2e passes on
`<branch>-next`, create the rollback tag for the current stable
tip and the date tag for the validated staged tip per `docs/promotion-protocol.md`.
Push and verify those tags, then leave the protected stable `<branch>` unchanged. The
draft Marin PR must identify each `<branch>-next` to `<branch>` hard swap that an admin
must complete before merge.

Keep the Marin PR draft until every required admin promotion is complete. After an
`isolated_project` promotion, restore its uv source from `main-next` to `main`, relock,
and confirm the resolved SHA did not change before marking the PR ready.

## Review and Open the PR

Do a PR-review pass over the fork commits and Marin diff using
`.agents/skills/review-pr/SKILL.md`, then run `./infra/pre-commit.py --review` and
fix or answer every finding. Check that each retained patch has a reason and a drop
condition, each dropped patch is truly obsolete, Marin edits are scoped to the
refresh, and no text overclaims validation evidence.

Open one draft `marin-community/marin` PR via `.agents/skills/commit/SKILL.md`,
request the descriptor's `blocker_assignee` as reviewer, then read back its title,
body, labels, base, head, and draft state. Take one non-blocking snapshot of CI,
comments, and reviews; do not start a monitoring loop. PR body: above the fold, the fork,
selected base, the staged tip SHA, its rollback and date tags, the pending admin
promotion (and the wheel release tag for `vllm-gpu`), e2e outcome, and unresolved
risks; in `<details>`, the base-selection evidence and the carry/drop/fix table with
dropped-patch reasons.
