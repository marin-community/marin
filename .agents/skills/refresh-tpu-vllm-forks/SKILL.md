---
name: refresh-tpu-vllm-forks
description: Refresh Marin TPU-vLLM forks from a tpu-inference release/LKG pair, update exact SHA pins, run TPU smokes, and open the Marin PR.
---

# Skill: Refresh TPU-vLLM Fork Stack

Read first:

@AGENTS.md

## Mission

Marin maintains forks of `vllm` and `tpu-inference` with required patches. Use
this skill to update those forks to the latest tested upstream pair, reconcile
Marin overlay commits, then open the Marin PR that pins the refreshed fork tips.

Manual fixed-base overlay changes are a separate workflow; see
`docs/overlay-only-pr.md`.

Example run: [marin-community/marin#6453](https://github.com/marin-community/marin/pull/6453).

Use the same algorithm in CI and local runs. In local/manual mode, ask before
external mutations: pushing fork branches, publishing the logs Gist, opening the
Marin PR, or filing/updating a GitHub issue. Do not ask before required TPU
smoke tests.

| Repo | Role | Upstream |
| --- | --- | --- |
| [`marin-community/vllm`](https://github.com/marin-community/vllm) | Marin vLLM overlay branches. | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) |
| [`marin-community/tpu-inference`](https://github.com/marin-community/tpu-inference) | Marin TPU inference overlay branches. | [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) |
| [`marin-community/marin`](https://github.com/marin-community/marin) | Pins fork branch tips and receives the only PR. | n/a |

## Outcome

- If no newer upstream pair is selected and no pin metadata needs repair, exit
  successfully with a no-op summary.
- If the refresh succeeds, open exactly one draft PR in `marin-community/marin`
  after required smoke tests pass, and request `@yonromai` as reviewer.
- The PR updates Marin's fork tip SHAs, refreshes `uv.lock`, and reports bases,
  branches/tips, carried/dropped/fixed overlays, validation, and residual risk.
- Do not open fork PRs. Do not move either fork `main`; fork review happens via
  pushed branches and compare links from the Marin PR.

If a real external blocker remains after repair attempts, do not open a Marin
PR. Create or update one `marin-community/marin` issue assigned to `@yonromai`,
titled `TPU-vLLM fork refresh blocked: <short reason>`, with current pins,
selected release, branch names/SHAs if created, attempted fixes, remaining
failure, artifacts, and the logs Gist.

## Post-Merge Follow-Up

This skill does not run post-merge fork-main promotion. A successful refresh run
stops after opening the Marin PR; if blocked, it stops after filing or updating
the blocker issue.

After the Marin PR has merged and Marin `main` contains the new exact fork SHA
pins, a separate operator may run the post-merge protocol in
`docs/post-merge-protocol.md`.

## Workspace Setup

- Marin working copy:
  - GitHub Actions: use the checked-out `marin-community/marin` repo.
  - Local: use the human-provided Marin checkout/worktree. If it is a shared
    source checkout, create a dedicated worktree before editing.
- Run id:
  - GitHub Actions: `${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}`.
  - Local: UTC timestamp plus a short local label.
- Scratch dir: `/tmp/marin-fork-refresh/<run-id>`.
- Clone each fork there and add upstream:

```sh
git clone https://github.com/marin-community/vllm.git vllm
git -C vllm remote add upstream https://github.com/vllm-project/vllm.git
git -C vllm fetch --tags origin upstream

git clone https://github.com/marin-community/tpu-inference.git tpu-inference
git -C tpu-inference remote add upstream https://github.com/vllm-project/tpu-inference.git
git -C tpu-inference fetch --tags origin upstream
```

- Keep two top-appended Markdown logs in the scratch dir:
  - `notes-summary.md`: major decisions, selected bases, branch SHAs,
    validation outcomes, final no-op/PR/issue result.
  - `sharp-edges.md`: surprising failures, compatibility traps, memorable
    fixes, open operational risks.
- Curate the logs for major learnings only; skip routine command transcripts.
- Before opening a PR or issue, publish both logs to one GitHub Gist and link it
  from the PR/issue.

## Algorithm

### 1. Read Current Pins

- Prefer managed fork pins: read current `vllm` and `tpu-inference` SHAs from
  root `pyproject.toml` `tool.uv.sources`; check `uv.lock` against them.
- Read adjacent GitHub compare-link comments to recover each current upstream
  base. If comments are missing, compute `git merge-base <fork-sha>
  upstream/main` and include repaired compare comments in the Marin change.
- If Marin is still on legacy package pins such as `vllm-tpu==...` /
  `tpu-inference==...` in `lib/marin/pyproject.toml`, treat this as a one-time
  bootstrap migration: record the package versions, do not require old fork SHAs
  or compare comments, and migrate to exact fork SHA pins after validation.
- Resolve any old fork SHAs in the scratch fork clones before replaying overlays.

### 2. Select Bases

- Use GitHub Releases for `vllm-project/tpu-inference`; do not use raw tags,
  branches, or standalone latest vLLM releases as the selection signal.
- Select the newest release where `draft == false`, `prerelease == false`, and
  the tag is exactly `vMAJOR.MINOR.PATCH`.
- Resolve that release tag to a `tpu-inference` commit SHA. If it matches the
  current Marin `tpu-inference` upstream base, exit no-op unless only repairing
  pin metadata.
- Read `.buildkite/vllm_lkg.version` at the selected `tpu-inference` release.
  That exact SHA is the vLLM base; verify it resolves in `vllm-project/vllm`.
- Inspect the LKG vLLM TPU metadata for dependency/build implications:
  `requirements/tpu.txt`, `pyproject.toml`, and `setup.py`.
- Do not walk back to older `tpu-inference` releases when the latest eligible
  release fails; fix the refresh or file a blocking issue.
- Record the selected upstream base SHAs and the reason for action/no-op.

### 3. Rebuild Fork Overlays

Create one branch per fork from the selected upstream base:

```text
auto-refresh/<YYYYMMDD>/<base-id>-<shortsha>
```

Use the selected `tpu-inference` release tag as `<base-id>` for
`tpu-inference`, and `lkg` for vLLM. Keep the same date prefix for the pair.
Sanitize names. Never rewrite an existing remote refresh branch; on collision,
use the next `-rN` suffix.

For each fork:

1. Define `old_base` from the current compare-link comment, `old_tip` from the
   current Marin pin, and `new_base` from selected upstream metadata.
2. Inventory the old Marin overlay in order:
   `git log --reverse old_base..old_tip`.
3. For each meaningful old delta, decide whether its intent is already present
   in `new_base`, still required by Marin, or broken by new upstream APIs/deps.
   Use patch comparison and targeted diffs for the touched files.
4. Classify each delta:
   - `carry`: behavior is still needed and not upstreamed;
   - `drop`: upstream absorbed it, it is obsolete, or it was only temporary;
   - `fix`: intent is still needed, but implementation must change.
5. Replay only `carry` and `fix` deltas onto `new_base` in old logical order.
   Use clean cherry-picks for carries; rewrite fixes as new commits that
   reference the original commit SHA(s).
6. In every retained overlay commit body, explain why it is still needed:
   upstream gap, Marin dependency, validation signal, and future drop condition.
7. For major non-obvious overlays/fixes, also leave a short code-adjacent
   rationale tied to compatibility.
8. Run `git range-diff old_base..old_tip new_base..<new_tip>` as the replay
   audit and explain every dropped or rewritten delta in the notes/PR.
9. Keep history reviewable: no conflict artifacts, unrelated refactors, or
   preserved commits whose behavior is now `drop`.

For bootstrap migrations without pin-derived `old_base..old_tip`, create the
first managed branches from the selected upstream bases and replay only Marin
fork deltas whose source and intent are explicit.

Push the finished branch to the corresponding `marin-community` fork.

### 4. Wire Marin

Update Marin root `pyproject.toml` so `tool.uv.sources` pins exact fork branch
tip SHAs. Add adjacent compare comments that show the retained overlay commits
against the selected upstream base. Keep the short workflow pointer immediately
above the TPU-vLLM pins so future editors know which procedure owns the SHAs:

```toml
# Changes to TPU-vLLM fork SHAs must follow one of these protocols: if this is a
# release/LKG refresh, then follow .agents/skills/refresh-tpu-vllm-forks/SKILL.md;
# if this is a fixed-base overlay PR, then follow
# .agents/skills/refresh-tpu-vllm-forks/docs/overlay-only-pr.md. PR-head SHAs are
# temporary and must be replaced by the landed fork main SHA.
# https://github.com/marin-community/vllm/compare/<vllm-upstream-base-sha>...<vllm-branch-tip-sha>
vllm = { git = "https://github.com/marin-community/vllm.git", rev = "<vllm-branch-tip-sha>" }
# https://github.com/marin-community/tpu-inference/compare/<tpu-inference-upstream-base-sha>...<tpu-inference-branch-tip-sha>
tpu-inference = { git = "https://github.com/marin-community/tpu-inference.git", rev = "<tpu-inference-branch-tip-sha>" }
```

Also make only fork-stack update changes needed in Marin:

- remove any old `vllm-tpu==0.19.0` path;
- make `marin-core[vllm]` own the TPU-vLLM runtime stack;
- keep shared Marin workspace dependencies aligned by default. When the refresh
  requires a new version of packages shared with training/test stacks
  (`jax`, `jaxlib`, `libtpu`, `transformers`, `torch`, etc.), first update all
  relevant Marin workspace consumers together, including `marin-core`,
  `marin-levanter`, and `marin-fray`, then validate the unified graph;
- do not add resolver conflicts or package splits that isolate serving from
  training/test profiles unless the unified monorepo update has been attempted,
  the failure is understood, and the PR or blocker issue explains why the split
  is necessary and how it should be removed later;
- preserve worker/eval paths that intentionally combine `tpu` and `vllm`
  extras, unless refreshed-stack validation proves they must change;
- set `VLLM_TARGET_DEVICE=tpu` for TPU source-build workers.

Do not bundle unrelated usability, cleanup, or refactor work. Log those
separately if found.

### 5. Validate

Run before PR creation:

- resolver and lockfile checks;
- focused Marin dependency/eval/worker tests;
- TPU workloads through Iris on the `marin` cluster, always with interactive
  priority, targeting `v6e-4` in GCP region `europe-west4`;
- local troubleshooting loops on a persistent dev TPU node with the same
  `v6e-4` / `europe-west4` hardware before resubmitting Iris workloads;
- TPU import/build smoke;
- direct `vllm.LLM.generate` TPU smoke;
- bounded brokered Marin runtime smoke, preferring an existing script such as
  `experiments/evals/served_qwen3.py` over writing a new smoke.

Run the brokered suite with a bounded sample by composing its existing pieces:

```sh
uv run iris --config lib/iris/config/marin.yaml job run \
  --job-name served-qwen3-<run-id> --cpu 1 --memory 2G --extra cpu \
  --priority interactive --no-wait -- python -c \
  "from dataclasses import replace; from fray.types import ResourceConfig; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.brokered_eval_suite import brokered_eval_suite; from experiments.evals.served_qwen3 import QWEN3_INFERENCE; inference = replace(QWEN3_INFERENCE, worker_resources=ResourceConfig.with_tpu('v6e-4', ram='96g', regions=['europe-west4'])); StepRunner().run([lower(brokered_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>-dev', limit=8))])"
```

Inspect the Iris parent, broker, and worker logs; confirm the proxy served
completions, lm-eval wrote HumanEval metrics and sample outputs, and no
TPU/vLLM build, import, or runtime tracebacks occurred.

When a workload smoke fails, rerun the same workload against Marin's current
pins on the old fork stack, using the same Iris target/priority. Fix only
failures that pass on the old stack and fail on the refreshed stack. If the old
stack is already broken, record it as a baseline failure; do not rewrite that
workload or smoke test as part of this refresh.

### 6. Review Before PR

Do a PR-review-style pass over the fork commits and Marin diff. Use
`.agents/skills/review-pr/` as a checklist, then run
`./infra/pre-commit.py --review` before opening the PR and fix or respond to
every finding.

Check that:

- each retained overlay has a reason to exist and a future drop condition;
- each dropped overlay is truly upstreamed, obsolete, or temporary;
- Marin edits are scoped to fork-stack update issues;
- comments, commit messages, and PR text do not overclaim validation evidence;
- required tests and baseline comparisons support the final claim.

### 7. Open the Marin PR

After required smoke tests pass, publish the logs Gist, push the Marin branch,
and open one draft `marin-community/marin` PR. Request `@yonromai` as reviewer.

PR body:

- Above the fold: short summary, selected `tpu-inference` release, selected vLLM
  LKG, fork branch/tip SHAs, smoke-test outcome, logs Gist link, unresolved
  risks, and a one-line dropped-overlay summary.
- In GFM `<details>` blocks: base-selection evidence, carry/drop/fix table,
  explicit dropped-overlay reasons, smoke artifacts, and baseline-failure notes.
- Keep it readable; do not paste raw workflow logs or exhaustive command
  transcripts.

## Done Means

- Fork `main` branches are unchanged and no fork PRs are opened.
- Refreshed fork branches use stable names and exact SHA pins in Marin.
- Marin `pyproject.toml` includes overlay compare links and `uv.lock` is
  refreshed.
- Retained overlays explain why they still exist; dropped overlays are called
  out with reasons.
- Required smoke tests pass before PR creation, or the unresolved blocker is in
  a Marin issue assigned to `@yonromai`.
- The two curated logs are published as one Gist and linked from the PR/issue.
