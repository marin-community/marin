---
name: refresh-fork
description: Rebase one Marin fork onto newer upstream per its config/external/migration.toml descriptor, run the fork's e2e, and open the Marin PR or file a blocker issue.
---

# Skill: Refresh a Fork

Read first:

@AGENTS.md

## Mission

Every fork Marin pins under `config/external/` is a `marin-community` fork: our
commits on top of an upstream project. Refreshing a pin means rebasing our commits
onto a newer upstream base on a `<branch>-next` staging branch, validating with the
fork's e2e against that branch, re-pinning Marin at the exact staged tip, and
preparing the stable-branch promotion with rollback and date tags. Fork stable
branches are protected. An unattended run stops after opening the draft Marin PR
and identifies the admin promotion it needs; it never force-moves the stable branch.

Each pin refreshes independently. The one exception is a `group`, which refreshes as a
unit: its sections refresh together on one run and re-pin in one PR (each still rebases
onto its own base). Only the vllm/tpu-inference pair is grouped, because the TPU
launcher installs both pins at once and vllm's TPU base derives from the tpu-inference
release; splitting them could pin a mixed, unblessed stack. The vllm fork's GPU pin is
not in that group — it tracks upstream head on the fork's `main`, independent of the
`tpu` branch that carries the TPU pin. The weekly `ops-fork-ferry` workflow sends Weaver one request per
supported pin or group. That request names this skill and the forks to update; this
file owns the migration procedure.

Use the same algorithm in CI and local runs. In local/manual mode, ask before
external mutations: pushing fork branches, opening the Marin PR, or filing a GitHub
issue. Do not ask before the fork's required e2e.

## Read the Descriptor

Read the target fork's section in `config/external/migration.toml`. It gives:

- `upstream` — the repo we rebase onto. Every fork has one.
- `group` — if present, refresh every section in the group together in one PR
  (read them all now); if absent, this pin refreshes alone.
- `base_select` (+ `derived_from`) — how to choose the new upstream base.
- `pin` — where the resolved pin is recorded (`isolated_project` uv.lock,
  `descriptor:<path>#<section>` SHA, or `release:<path>` prebuilt wheel); drives the
  re-pin step.
- `branch` — the fork branch this pin tracks (`main` for a single-pin fork and for the
  vllm GPU pin; `tpu` for the vllm fork's TPU pin). The refresh stages on `<branch>-next`;
  an admin promotes that validated tip after reviewing the draft Marin PR.
- `e2e` — the Marin end-to-end that validates the refresh.
- `blocker_assignee` — who owns the "can't migrate" issue.
- `nuances` — constraints a human must respect (torch pins, known-good ceilings).

The descriptor records *how* to migrate; the pin source it names holds the actual
revision.

## Outcome

- If no newer base is selected and no pin metadata needs repair, exit successfully
  with a no-op summary.
- On success, create the rollback and date tags described in
  `docs/promotion-protocol.md`, then open exactly one draft PR in
  `marin-community/marin` for the fork or group after the e2e passes. A grouped
  refresh re-pins every group section at its staged tip in that single PR. State the
  exact `<branch>-next` to `<branch>` admin promotion still required, request the
  descriptor's `blocker_assignee` as reviewer, and monitor the PR per
  `.agents/skills/commit/SKILL.md`.
- On an unresolved blocker, do not open a PR. Create or update one
  `marin-community/marin` issue assigned to `blocker_assignee`, titled
  `Fork refresh blocked: <fork> — <short reason>`, with current pins, the selected
  base, branch names/SHAs if created, attempted fixes, the remaining failure, and
  artifacts.

## Scratch setup

- Scratch dir: `/tmp/marin-fork-refresh/<run-id>` (run id:
  `${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}` in Actions, else a UTC timestamp plus a
  short label).
- Clone the fork and add its `upstream` remote. The fork URL is `repository` in the
  pin source (`vllm/tpu-forks.toml` for descriptor pins, the `[tool.uv.sources]` git
  entry for isolated projects, the release-asset host in `vllm/gpu-release.toml` for the
  `vllm-gpu` release pin — the same `marin-community/vllm` repo); `<upstream>` is this
  section's `upstream`.

```sh
git clone <repository> <fork>
git -C <fork> remote add upstream <upstream>
git -C <fork> fetch --tags --multiple origin upstream   # --multiple fetches both remotes; without it "upstream" is read as a refspec
git -C <fork> remote set-head upstream -a                # so upstream/HEAD resolves
```

- Keep working notes as you go — decisions, selected bases, branch SHAs, validation
  outcomes, and sharp edges (surprising failures, compatibility traps). They feed the
  PR body: base-selection evidence and the carry/drop/fix table in `<details>`, and
  the unresolved risks above the fold.

## Select the base

- `base_select = upstream_main` (`evalchemy`, `harbor`, `MarinSkyRL`, `vllm-gpu`): the base is the
  tip of the `upstream` default branch. These pins rebase onto upstream `main`; there is no
  release to gate on. `vllm-gpu` tracks vLLM head this way, distinct from the TPU `vllm` pin's
  tpu-inference-blessed base.
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

Branch from the selected base as `<branch>-next` (the pin's `branch` with a `-next`
suffix). Use `tpu-next` for the vllm TPU pin and `main-next` otherwise; single-pin
forks and the vllm GPU pin both track `main`. This staging branch is disposable — a re-run
force-updates it — and is distinct from the protected stable `<branch>`, which the
unattended refresh leaves unchanged.

Find the base our commits currently sit on: `old_base` is the descriptor's
`upstream_base` (descriptor pins) or `git merge-base <fork>/<branch> upstream/HEAD`
(isolated and release pins, where it is not recorded). `old_tip` is the head of our
patches: the fork's `main` for isolated pins (Marin's recorded pin may lag `main`, so
rebase from `main` to cover the full patch set), or the pin's stable `<branch>` tip for
descriptor and release pins (the descriptor's `commit`, or `gpu-release.toml`'s
`source_commit`). Then, onto `new_base`:

1. Inventory our commits in order: `git log --reverse --no-merges old_base..old_tip`.
   Merge commits (especially merges of `upstream` into a feature branch) are not
   replayed — their content comes from the new base; drop them.
2. Classify each meaningful delta: `carry` (still needed, not upstreamed), `drop`
   (upstream absorbed it, obsolete, or temporary), `fix` (intent needed,
   implementation must change — re-author against the current layout when upstream
   moved or refactored the files it touches). Before carrying anything, check whether
   the new base already did it: grep the base for the symbols, APIs, or dependency
   pins the patch introduces. If they are present it is a `drop` — a backport that has
   landed, or a version ceiling the base now exceeds (a stale `<X` pin silently
   downgrades upstream). On a fast-moving fork this "did upstream absorb this?" pass is
   the highest-value step; it is what keeps the fork converging rather than re-adding
   duplicate or obsolete code.
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
   Statically cross-check every upstream symbol the overlay touches — constructor
   kwargs, function signatures, attributes, removed helpers — against `new_base`,
   including the overlay's tests. A vLLM refresh caught two breaks this way: a
   `FusedMoE` class replaced by a `FusedMoEFactory` function, and a removed
   `is_interleaved` config helper. A textual replay left both in place, and only a
   multi-hour GPU build would otherwise have surfaced them.
7. Keep history reviewable — no conflict artifacts, unrelated refactors, or
   preserved commits whose behavior is now `drop`. Collapse fork-infra churn
   (CI, workflow, or prose commits that adopt then revise then disable) to its final
   state rather than replaying each hop.

Stop and file a blocker instead of forcing a PR when the rebase is not a mechanical
replay: our overlay is non-linear (merge commits weaving upstream in), upstream
renamed or refactored files our `fix` commits touch (so they need re-authoring against
the new layout), or conflicts hit many core files at once. A fork hundreds of commits
behind upstream is usually in this state. The weekly cadence keeps a fork from ever
drifting this far; one that already has (see `nuances`) needs a one-time manual
catch-up outside this skill before it can be auto-migrated. That catch-up replays
the fork's real commits onto the new base; it never reconstructs the overlay from a
single net diff, which silently invents drift the fork never carried. Diff each
hand-ported `fix` file against the fork's `main` to confirm it matches intent. The
blocker issue carries the carry/drop/fix inventory, the conflict map, and how far
behind the fork is.

## Pin at the staged tip

Point Marin at `<branch>-next` so the e2e runs against the replayed code, then run
`uv run config/update-external.py` to regenerate
`lib/marin/src/marin/external_dependencies.py`; confirm only the intended pins change.
The stable `<branch>` remains at the old tip until an admin hard-swaps it after reviewing
the draft PR. Because `<branch>-next` and the eventual `<branch>` are the same commit,
the pin set here needs no change after that promotion.

- `pin = descriptor:<path>#<section>` (`vllm`, `tpu-inference`): push `<branch>-next` to
  the fork. Set the section's `commit` to its tip and `upstream_base` to the selected
  base in `vllm/tpu-forks.toml`. This stack resolves entirely inside the `uvx` env from
  the two forks, so there are no `uv.lock` changes and `jax`/`jaxlib`/`libtpu`/`torch`
  come from the forks' own dependencies — do not touch `marin-core`, `marin-levanter`,
  or `marin-fray`.
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
  4. Download that manifest and re-pin without hand-editing `gpu-release.toml`:
     `gh release download <release_tag> --repo marin-community/vllm --pattern marin-vllm-gpu-manifest.json`,
     then `uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json`.
     The helper writes `gpu-release.toml` (release tag, source commit, version, torch backend,
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

Run the fork's own test suite on the rebased branch before the marin e2e. The e2e
exercises one path; the fork suite is what catches re-porting drift across the whole
patch set. Run it locally where the runner supports it, otherwise dispatch it on the
fork's CI.

Derive the command from the fork's CI config verbatim; do not invent a marker
subset. A narrow marker such as `-m unit` can silently skip the thousands of unmarked
tests the CI's real expression (`-m "not runtime"`) collects, so the narrow run reads
green while most of the suite never ran. CI steps run in order under an implicit
`if: success()`: a later gated step (a `-m runtime` docker leg) does not run until an
earlier one is green, so its regressions stay hidden behind the first failure. Run
every step's marker in order.

Probe the runner before trusting a local run. This VM runs docker but its bind-mounts
do not propagate into containers, so a container-environment trial (harbor's
DOCKER-env golden tests) cannot complete here and has to run on the fork's CI. To
dispatch the suite there when the review PR sits on an `upstream-base/<sha>` branch:
gate workflows commonly filter `branches: ["main"]`, so such a PR gets no automatic
CI; only a workflow declaring `workflow_dispatch` and present on the default branch
can be dispatched against the branch ref; a workflow with neither `workflow_dispatch`
nor a matching push trigger never runs on that PR, so verify it locally.

A build-heavy fork raises a second ceiling. vLLM's suite needs its compiled CUDA/TPU
stack, and `import` fails outright when the runner has no torch, so the pre-e2e check
there is structural only: `py_compile` the replayed tree and sweep for leftover
conflict markers. A green structural pass is not behavioral validation — say so.

Deterministic golden tests are dependency-version fragile: an upstream-mandated
dependency floor (an upstream `litellm>=1.92` bump) can stale a golden even when the
fork source is byte-identical. Read the deciding code path rather than the golden
diff, and separate the dependency floor from a real port defect — never downgrade
below the floor to make a golden pass. Prefer making the fixture dependency-independent
by patching every version-sensitive seam the trigger reads (the sync and async token
counters both, not just one) over re-tuning the golden to a single version. Regenerate
in one CI run: a `workflow_dispatch` that runs the suite in update mode and then a
comparison-mode verify step, both non-failing, uploading the regenerated goldens and
logs as artifacts.

## Validate

Run the descriptor's `e2e` before opening the PR:

- **`experiments/evals/served_qwen3.py::QWEN3_TPU_INFERENCE`** (`vllm`,
  `tpu-inference`) — a bounded brokered TPU serve+eval smoke. Run TPU workloads
  through Iris on the `marin` cluster at interactive priority, `v6e-4` in GCP
  `europe-west4`. Confirm the proxy served completions, lm-eval wrote metrics and
  sample outputs, and no TPU/vLLM build, import, or runtime tracebacks occurred:

```sh
uv run iris --config lib/iris/config/marin.yaml job run \
  --job-name served-qwen3-<run-id> --cpu 1 --memory 2G --extra cpu \
  --priority interactive --no-wait -- python -c \
  "from dataclasses import replace; from fray.types import ResourceConfig; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.lm_eval_suite import lm_eval_suite; from experiments.evals.served_qwen3 import QWEN3_TPU_INFERENCE; inference = replace(QWEN3_TPU_INFERENCE, iris=replace(QWEN3_TPU_INFERENCE.iris, worker_resources=ResourceConfig.with_tpu('v6e-4', ram='96g', regions=['europe-west4']))); StepRunner().run([lower(lm_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>-dev', limit=8))])"
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

When an e2e fails, rerun the same workload against Marin's current pins on the old
fork stack, same target and priority. Fix only failures that pass on the old stack
and fail on the refreshed one. If the old stack is already broken, the fork's e2e
cannot gate this refresh: do not open a PR on an unvalidated pin — file or link a
blocker for the broken e2e and hold the refresh until it is fixed.

## Prepare the protected-branch promotion

Once the e2e passes on `<branch>-next`, create the rollback tag for the current stable
tip and the date tag for the validated staged tip per `docs/promotion-protocol.md`.
Push and verify those tags, then leave the protected stable `<branch>` unchanged. The
draft Marin PR must identify each `<branch>-next` to `<branch>` hard swap that an admin
must complete before merge. On the vllm fork the GPU and TPU pins promote onto their own
stable branches (`main` and `tpu`) independently.

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
request the descriptor's `blocker_assignee` as reviewer, and follow the commit
skill's monitoring loop to an exit condition. PR body: above the fold, the fork,
selected base, the staged tip SHA, its rollback and date tags, the pending admin
promotion (and the wheel release tag for `vllm-gpu`), e2e outcome, and unresolved
risks; in `<details>`, the base-selection evidence and the carry/drop/fix table with
dropped-patch reasons.

## Done Means

- The pin source named by `pin` carries the new revision (descriptor pins also carry
  `upstream_base`); `external_dependencies.py` is regenerated.
- `<branch>-next` points at the validated tip, the current stable tip has a rollback
  tag, and the validated tip has a date tag. The stable branch is unchanged, and the
  PR names the admin hard swap still required.
- A draft PR that temporarily follows `main-next` stays draft. After the admin
  promotion, its source follows `main` again and its lock still records the validated
  SHA before the PR is marked ready or merged.
- Retained patches explain why they exist; dropped patches are called out.
- The fork's e2e passed before the promotion tags and PR were created, or the blocker
  is in a Marin issue assigned to `blocker_assignee`.
- An opened Marin PR reaches a `commit` skill monitoring exit condition.
