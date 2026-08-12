---
name: refresh-fork
description: Rebase one Marin fork onto newer upstream per its config/external/migration.toml descriptor, run the fork's e2e, and open the Marin PR or file a blocker issue.
---

# Skill: Refresh a Fork

Read first:

@AGENTS.md

## Mission

Every fork Marin pins under `config/external/` is a `marin-community` fork: our
commits on top of an upstream project. Refreshing one means rebasing our commits
onto a newer upstream base, validating with the fork's e2e, and re-pinning Marin —
or, if a real blocker remains, filing one "can't migrate" issue.

Refresh one fork, or one atomic `group`, at a time. A `group` refreshes as a unit:
its sections refresh together on one date-stamped run and re-pin in one PR (each
still rebases onto its own base). The vllm/tpu-inference pair is grouped because the
TPU launcher installs both pins at once and vllm's base derives from the
tpu-inference release; splitting them could pin a mixed, unblessed stack. A weekly coordinator that walks the descriptor in
`depends_on` order is planned but not yet built; today a human runs it for a single
fork or group.

Use the same algorithm in CI and local runs. In local/manual mode, ask before
external mutations: pushing fork branches, opening the Marin PR, or filing a GitHub
issue. Do not ask before the fork's required e2e.

## Read the Descriptor

Read the target fork's section in `config/external/migration.toml`. It gives:

- `upstream` — the repo we rebase onto. Every fork has one.
- `group` — if present, refresh every section in the group together in one PR
  (read them all now); if absent, this fork refreshes alone.
- `base_select` (+ `derived_from`) — how to choose the new upstream base.
- `pin` — where the resolved pin is recorded (`isolated_project` uv.lock, or
  `descriptor:<path>#<section>`); drives the re-pin step.
- `e2e` — the Marin end-to-end that validates the refresh.
- `blocker_assignee` — who owns the "can't migrate" issue.
- `nuances` — constraints a human must respect (torch pins, known-good ceilings).

The descriptor records *how* to migrate; the pin source it names holds the actual
revision.

## Outcome

- If no newer base is selected and no pin metadata needs repair, exit successfully
  with a no-op summary.
- On success, open exactly one draft PR in `marin-community/marin` for the fork or
  group after the e2e passes — a grouped refresh re-pins every group section in that
  single PR. Request the descriptor's `blocker_assignee` as reviewer, and monitor it
  per `.agents/skills/commit/SKILL.md`.
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
  pin source (`vllm/tpu-forks.toml` for descriptor pins, the `[tool.uv.sources]`
  git entry for isolated projects); `<upstream>` is this section's `upstream`.

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

- `base_select = upstream_main` (`evalchemy`, `harbor`, `MarinSkyRL`): the base is the tip of the
  `upstream` default branch. These forks rebase onto upstream `main`; there is no
  release to gate on.
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

Branch from the selected base as `auto-refresh/<YYYYMMDD>/<base-id>-<shortsha>`
(`<base-id>` = the tpu-inference release tag, `lkg` for vLLM, or the upstream short
SHA; same date prefix across a group). Never rewrite an existing remote refresh
branch; on collision use the next `-rN` suffix.

Find the base our commits currently sit on: `old_base` is the descriptor's
`upstream_base` (descriptor pins) or `git merge-base <fork>/main upstream/HEAD`
(isolated pins). `old_tip` is the head of our patches: the fork's `main` for isolated
pins (Marin's recorded pin may lag `main`, so rebase from `main` to cover the full
patch set), or the pinned refresh branch for descriptor pins. Then, onto `new_base`:

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

## Re-pin

Re-pin per the section's `pin`, then run `uv run config/update-external.py` to
regenerate `lib/marin/src/marin/external_dependencies.py`; confirm only the intended
pins change.

- `pin = descriptor:<path>#<section>` (`vllm`, `tpu-inference`): push the refresh
  branch to the fork; do not move fork `main`. Set the section's `commit` to the
  pushed branch tip and `upstream_base` to the selected base in the descriptor. The
  pin references the reviewed branch; promoting it to fork `main` is the post-merge
  follow-up. This stack resolves entirely inside the `uvx` env from the two forks,
  so there are no `uv.lock` changes and `jax`/`jaxlib`/`libtpu`/`torch` come from the
  forks' own dependencies — do not touch `marin-core`, `marin-levanter`, or
  `marin-fray`.
- `pin = isolated_project` (`evalchemy`, `harbor`, `MarinSkyRL`): the uv source
  follows the fork's `main`, so the pin advances only when `main` does. Push the
  rebased history to the fork's `main` — a history rewrite, so coordinate: the daily
  external-dependency bump and this pin both follow `main`. Review the rebase from a
  compare link (`upstream_base..<new_tip>`) on the Marin PR before pushing. Then run
  `uv run config/update-external.py <fork>` to advance `config/external/<fork>/uv.lock`
  to the new `main`.

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
  "from dataclasses import replace; from fray.types import ResourceConfig; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.brokered_eval_suite import brokered_eval_suite; from experiments.evals.served_qwen3 import QWEN3_TPU_INFERENCE; inference = replace(QWEN3_TPU_INFERENCE, worker_resources=ResourceConfig.with_tpu('v6e-4', ram='96g', regions=['europe-west4'])); StepRunner().run([lower(brokered_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>-dev', limit=8))])"
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

## Review and Open the PR

Do a PR-review pass over the fork commits and Marin diff using
`.agents/skills/review-pr/SKILL.md`, then run `./infra/pre-commit.py --review` and
fix or answer every finding. Check that each retained patch has a reason and a drop
condition, each dropped patch is truly obsolete, Marin edits are scoped to the
refresh, and no text overclaims validation evidence.

Open one draft `marin-community/marin` PR via `.agents/skills/commit/SKILL.md`,
request the descriptor's `blocker_assignee` as reviewer, and follow the commit
skill's monitoring loop to an exit condition. PR body: above the fold, the fork,
selected base, fork branch/tip SHAs, e2e outcome, and unresolved risks; in
`<details>`, the base-selection evidence and the carry/drop/fix table with
dropped-patch reasons.

## Post-Merge Follow-Up

A `descriptor`-pinned fork keeps its `main` unchanged and pins a reviewed branch;
after the Marin PR merges, a separate operator promotes that branch to fork `main`
via `docs/post-merge-protocol.md`. An isolated fork already advanced its `main`
during the refresh, so it needs no promotion.

## Done Means

- The pin source named by `pin` carries the new revision (descriptor pins also carry
  `upstream_base`); `external_dependencies.py` is regenerated.
- Retained patches explain why they exist; dropped patches are called out.
- The fork's e2e passed before PR creation, or the blocker is in a Marin issue
  assigned to `blocker_assignee`.
- An opened Marin PR reaches a `commit` skill monitoring exit condition.
