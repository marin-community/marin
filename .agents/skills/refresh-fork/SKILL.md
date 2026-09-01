---
name: refresh-fork
description: Refresh a named Marin external fork pin onto a newer upstream base using its configured descriptor and required end-to-end test.
---

# Refresh a fork

Read `AGENTS.md`. Refresh a `group` as one unit and one PR. Per-fork guidance
lives beside this file: `docs/vllm.md` covers the `vllm`/`tpu-inference` group
and the GPU release pipeline, and `docs/xla.md` covers the XLA PJRT fork, which
is pinned outside `migration.toml` entirely.

In local mode, ask before pushing fork branches, opening the PR, or filing an
issue. The required end-to-end test needs no extra confirmation.

## Read the Descriptor

Read the target fork's section in `config/external/migration.toml`. It gives:

- `repository` — the Marin fork cloned as `origin`. When omitted, derive it from
  the pin source.
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
- On a successful branch-staged refresh, create the rollback and date tags described in
  `docs/promotion-protocol.md`, then open exactly one draft Marin PR after the e2e
  passes. State the exact `<branch>-next` to `<branch>` admin promotion still
  required and request the descriptor's `blocker_assignee` as reviewer.
- For a fork-specific release workflow, follow its guide, then return here for
  the common review and draft-PR closeout.
- On an unresolved blocker, do not open a PR. Create or update one
  `marin-community/marin` issue assigned to `blocker_assignee`, titled
  `Fork refresh blocked: <fork> — <short reason>`, with current pins, the selected
  base, branch names/SHAs if created, attempted fixes, the remaining failure, and
  artifacts.

## Scratch setup

- Scratch dir: `/tmp/marin-fork-refresh/<run-id>` (run id:
  `${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}` in Actions, else a UTC timestamp plus a
  short label).
- Clone the section's `repository` as `origin` and add the section's canonical
  `upstream` as a separate remote. If `repository` is omitted, derive it from the
  `[tool.uv.sources]` git entry for isolated projects or the release-asset host
  for release pins.

```sh
git clone <repository> <fork>
git -C <fork> remote add upstream <upstream>
git -C <fork> fetch --tags --multiple origin upstream   # --multiple fetches both remotes; without it "upstream" is read as a refspec
git -C <fork> remote set-head upstream -a                # so upstream/HEAD resolves
```

- Record selected bases, branch SHAs, carry/drop/fix decisions, validation, and
  unresolved risks for the PR.

## Select the base

- `base_select = upstream_main`: use the tip of the `upstream` default branch.
- `base_select = latest_release`: use GitHub Releases of the fork's `upstream`;
  do not use raw tags or branches. Select the newest stable release matching the
  descriptor's constraints and resolve it to a commit SHA.
- `base_select = derived`: read the SHA at `derived_from` from the release it
  names, verify it resolves in the fork's `upstream`, and use it as the base.
  Read the fork-specific guide for derivation details.

If the selected base matches the current one and no pin metadata needs repair, exit
no-op. Do not walk back to older releases when the latest eligible one fails; fix
the refresh or file a blocker issue.

For an isolated fork whose `upstream` base has not moved, there is nothing to rebase
and the refresh is a no-op — even if Marin's pin lags the fork's own `main`. Adopting
patches pushed to the fork since Marin last locked belongs to the daily
external-dependency bump (`ops-external-dependencies`); refresh-fork runs only when
there is a newer upstream base to rebase onto.

## Rebase the overlay

This section applies to branch-staged refreshes. A fork-specific release guide
may replace it with its own source replay.

Branch from the selected base as `<branch>-next` (the pin's `branch` with a `-next`
suffix). A pin on `main` uses `main-next`. This staging branch is disposable — a re-run
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

This section applies to branch-staged refreshes. A fork-specific release guide
may replace it with its own pinning step.

Point Marin at `<branch>-next` so the e2e runs against the replayed code, then run
`uv run config/update-external.py` to regenerate
`lib/marin/src/marin/external_dependencies.py`; confirm only the intended pins change.
The stable `<branch>` remains at the old tip until an admin hard-swaps it after reviewing
the draft PR. Because `<branch>-next` and the eventual `<branch>` are the same commit,
the pin set here needs no change after that promotion.

- `pin = release:<path>` (`vllm-gpu`): the pin is a prebuilt wheel, so the refresh builds
  and promotes one through the fork's own release pipeline, then re-pins from the promoted
  manifest. The candidate/promote/re-pin commands and the CUDA/torch ABI-boundary caveat
  are in `docs/vllm.md`.
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

For a version-sensitive golden, inspect the deciding code path and distinguish a
new dependency floor from a port defect. An upstream `litellm>=1.92` floor, for
example, can stale a golden without a fork-source change. Never downgrade below
the floor. Prefer a dependency-independent fixture that patches every seam the
trigger reads, such as both sync and async token counters. Otherwise regenerate
and verify it in one CI run with logs as artifacts.

## Validate

Run the descriptor's `e2e` before opening the PR:

- The vllm-family e2es — `experiments/evals/served_qwen3.py::QWEN3_TPU_INFERENCE`
  (`vllm`, `tpu-inference`) and `tests/cluster/vllm/test_snowball_backend_parity.py`
  (`vllm-gpu`) — are documented with their exact commands in `docs/vllm.md`.

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

When an e2e fails, rerun the same workload against Marin's current pin on the old
stack, with the same target and priority. Fix only failures that pass on the old
stack and regress on the refreshed one. A fork-specific guide may define stricter
rules for reusing or invalidating an existing physical receipt.

## Prepare the protected-branch promotion

This section applies to branch-staged refreshes. Once the e2e passes on
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

Open one draft `marin-community/marin` PR via `.agents/skills/commit/SKILL.md` and
request the descriptor's `blocker_assignee` as reviewer. For a newly opened draft,
verify its title, body, labels, base, head, and draft state. Take one non-blocking
snapshot of CI, issue comments, inline comments, and submitted reviews; address
anything already actionable; then return. Run the commit skill's monitoring loop
only when the caller explicitly requests monitoring, waiting, or babysitting.

Use the fork-specific guide for any additional body fields. Otherwise put the
fork, selected base, staged tip SHA, rollback and date tags, pending admin
promotion, e2e outcome, and unresolved risks above the fold; put detailed
base-selection evidence and the carry/drop/fix table in `<details>`.
