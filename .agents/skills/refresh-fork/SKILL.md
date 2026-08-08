---
name: refresh-fork
description: Refresh one Marin fork toward upstream per its config/external/migration.toml descriptor — advance the pin, run the fork's e2e, and open the Marin PR or file a blocker issue.
---

# Skill: Refresh a Fork

Read first:

@AGENTS.md

## Mission

Marin pins several forks under `config/external/`. Each has a section in
`config/external/migration.toml` describing how to migrate it toward upstream.
Use this skill to refresh **one** fork: advance its pin to a newer base, validate
with the fork's declared e2e, and open the Marin PR — or, if a real external
blocker remains, file one "can't migrate" issue.

Run one fork at a time. The weekly `ops-fork-ferry` coordinator invokes this
skill once per fork in `depends_on` order; a human runs it for a single fork.

Use the same algorithm in CI and local runs. In local/manual mode, ask before
external mutations: pushing fork branches, opening the Marin PR, or filing a
GitHub issue. Do not ask before the fork's required e2e.

## Read the Descriptor

Read the target fork's section in `config/external/migration.toml`. It gives:

- `kind` — `patch_free` or `overlay`; everything below branches on this.
- `pin` — where the pinned revision lives (`isolated_project`, or
  `descriptor:<path>#<section>`).
- `base_select` (+ `derived_from`) — how to choose the new base.
- `e2e` — the Marin end-to-end that validates the refresh.
- `blocker_assignee` — who owns the "can't migrate" issue.
- `nuances` — constraints a human must respect (torch pins, known-good ceilings).

The descriptor records *how* to migrate; the pin source it names holds the
actual revision. Never edit a fork's pin without updating the base it is compared
against.

## Outcome

- If no newer base is selected and no pin metadata needs repair, exit
  successfully with a no-op summary.
- On success, open exactly one draft PR in `marin-community/marin` after the e2e
  passes, request the descriptor's `blocker_assignee` as reviewer, and monitor it
  per `.agents/skills/commit/SKILL.md`.
- On an unresolved external blocker, do not open a PR. Create or update one
  `marin-community/marin` issue assigned to `blocker_assignee`, titled
  `Fork refresh blocked: <fork> — <short reason>`, with current pins, the
  selected base, branch names/SHAs if created, attempted fixes, the remaining
  failure, and artifacts.

## patch_free forks

`evalchemy`, `harbor`, `MarinSkyRL` track a branch with no Marin commits, so a
refresh only advances the pin and validates.

1. **Sync the base.** `base_select = upstream_main`: the fork's `main` should
   fast-forward to its `upstream` `main`. If it does not fast-forward, the fork
   carries local commits and is not patch-free — stop and reclassify it as
   `overlay` in the descriptor. `base_select = fork_main`: the fork is
   marin-native (no `upstream`); advance to its own `main`.
2. **Advance the pin.** `pin = isolated_project`: run
   `uv run config/update-external.py <fork>`, which bumps
   `config/external/<fork>/uv.lock` and regenerates
   `lib/marin/src/marin/external_dependencies.py`. Confirm no other files change.
3. **Validate** with the fork's `e2e` (below), then open the Marin PR.

## overlay forks

`vllm` and `tpu-inference` carry Marin patches on top of upstream. A refresh
selects a new upstream base, replays the overlays, audits the replay, and re-pins.
The `vllm`/`tpu-inference` pair is the worked example; keep the two forks on the
same date-stamped refresh.

### Scratch setup

- Scratch dir: `/tmp/marin-fork-refresh/<run-id>` (run id:
  `${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}` in Actions, else a UTC timestamp plus
  a short label).
- Clone each fork there and add its `upstream` remote:

```sh
git clone <repository> <fork>
git -C <fork> remote add upstream <upstream>
git -C <fork> fetch --tags origin upstream
```

- Keep two top-appended Markdown logs in the scratch dir: `notes-summary.md`
  (decisions, selected bases, branch SHAs, validation outcomes, final result) and
  `sharp-edges.md` (surprising failures, compatibility traps, memorable fixes).
  Curate for major learnings; skip routine transcripts.

### Select the base

- `base_select = latest_release` (`tpu-inference`): use GitHub Releases of the
  fork's `upstream`; do not use raw tags or branches. Select the newest release
  where `draft == false`, `prerelease == false`, and the tag is exactly
  `vMAJOR.MINOR.PATCH`. Resolve it to a commit SHA. If it matches the current
  `upstream_base`, exit no-op unless only repairing pin metadata.
- `base_select = derived` (`vllm`): read the SHA at `derived_from`
  (`tpu-inference:.buildkite/vllm_lkg.version`) from the selected `tpu-inference`
  release. That exact SHA is the base; verify it resolves in the fork's
  `upstream`. Inspect its TPU build metadata (`requirements/tpu.txt`,
  `pyproject.toml`, `setup.py`) for dependency implications.
- Do not walk back to older releases when the latest eligible one fails; fix the
  refresh or file a blocker issue.

### Rebuild overlays

Branch each fork from its selected base as
`auto-refresh/<YYYYMMDD>/<base-id>-<shortsha>` (`<base-id>` = the tpu-inference
release tag, or `lkg` for vLLM; same date prefix for the pair). Never rewrite an
existing remote refresh branch; on collision use the next `-rN` suffix.

For each fork, with `old_base` from the descriptor's `upstream_base`, `old_tip`
from its current pin, and `new_base` from selection:

1. Inventory the old overlay in order: `git log --reverse old_base..old_tip`.
2. Classify each meaningful delta: `carry` (still needed, not upstreamed),
   `drop` (upstream absorbed it, obsolete, or temporary), `fix` (intent needed,
   implementation must change).
3. Replay only `carry` and `fix` onto `new_base` in the old logical order:
   clean cherry-picks for carries; rewrite fixes as new commits referencing the
   original SHA(s).
4. In every retained commit body, state why it is still needed and its future
   drop condition. For non-obvious overlays, leave a short code-adjacent
   rationale.
5. Run `git range-diff old_base..old_tip new_base..<new_tip>` as the replay
   audit and explain every dropped or rewritten delta in the notes and PR.
6. Keep history reviewable — no conflict artifacts, unrelated refactors, or
   preserved commits whose behavior is now `drop`.

Push the finished branch to the corresponding `marin-community` fork. Do not open
fork PRs and do not move fork `main`; review happens via pushed branches and
compare links from the Marin PR.

### Re-pin

Update the descriptor named by `pin` — `config/external/vllm/tpu-forks.toml`, per
section — so `commit` is the exact fork branch tip and `upstream_base` is the
selected base. Then run `uv run config/update-external.py` to regenerate
`external_dependencies.py`. This stack is isolated (uvx), so no `uv.lock` changes.
PR-head SHAs are temporary and must be replaced by the landed fork `main` SHA
before undrafting.

Keep the stack isolated: it resolves entirely inside the `uvx` env from the two
forks, so `jax`/`jaxlib`/`libtpu`/`torch` come from the forks' own dependencies.
Do not touch `marin-core`, `marin-levanter`, or `marin-fray`, and do not
reintroduce a workspace `vllm`/`tpu-inference` dependency. Respect the section's
`nuances`.

Manual fixed-base overlay changes are a separate workflow; see
`docs/overlay-only-pr.md`.

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

- **`experiments/evaluation/cli.py`** (`evalchemy`) and
  **`experiments/evaluation/configs/harbor/aime-smoke.yaml`** (`harbor`) — run a
  bounded eval through the runner and confirm it completes with metrics.
- **`experiments/post_training/iceball_micro.py`** (`MarinSkyRL`) — the micro
  post-training e2e.

When an e2e fails, rerun the same workload against Marin's current pins on the old
fork stack, same target and priority. Fix only failures that pass on the old
stack and fail on the refreshed one. If the old stack is already broken, record it
as a baseline failure; do not rewrite that workload as part of this refresh.

## Review and Open the PR

Do a PR-review pass over the fork commits and Marin diff using
`.agents/skills/review-pr/SKILL.md`, then run `./infra/pre-commit.py --review` and
fix or answer every finding. Check that each retained overlay has a reason and a
drop condition, each dropped overlay is truly obsolete, Marin edits are scoped to
the refresh, and no text overclaims validation evidence.

Open one draft `marin-community/marin` PR via `.agents/skills/commit/SKILL.md`,
request the descriptor's `blocker_assignee` as reviewer, and follow the commit
skill's monitoring loop to an exit condition. PR body: above the fold, the fork,
selected base, fork branch/tip SHAs, e2e outcome, and unresolved risks; in
`<details>`, the base-selection evidence and (for overlays) the carry/drop/fix
table with dropped-overlay reasons.

## Post-Merge Follow-Up

This skill does not promote fork `main`. After the Marin PR merges and `main`
carries the new pins, a separate operator may run `docs/post-merge-protocol.md`
for overlay forks.

## Done Means

- The pin source named by `pin` carries the new revision, and (for overlays) its
  `upstream_base`; `external_dependencies.py` is regenerated.
- Fork `main` branches are unchanged and no fork PRs were opened.
- Retained overlays explain why they exist; dropped overlays are called out.
- The fork's e2e passed before PR creation, or the blocker is in a Marin issue
  assigned to `blocker_assignee`.
- An opened Marin PR reaches a `commit` skill monitoring exit condition.
