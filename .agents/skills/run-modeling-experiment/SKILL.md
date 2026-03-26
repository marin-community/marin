---
name: run-modeling-experiment
description: Staged workflow for Marin modeling experiment lines. Use when asked to design, run, tune, compare, or interpret architecture changes, optimizer changes, train-loop changes, modeling ablations, or isoFLOP studies. Use it when the task needs an exact baseline, a Grug-first minimal-diff experiment path, a dedicated W&B view with `exp1234`-style tagging, and explicit small-budget promotion gates before spending larger compute. Layer it on top of `agent-research` for the branch/issue/logbook lifecycle.
---

# Run Modeling Experiment

## Overview

This skill specializes `.agents/skills/agent-research/SKILL.md` for modeling work. Use `agent-research` for branch/issue/logbook/snapshot cadence, then use this skill for baseline choice, Grug-first implementation, metric selection, staged tuning, and promotion gates.

## Default Starting Point

- Start from a Grug variant unless the user names another baseline or the work clearly belongs to an existing non-Grug line.
- For new architecture or train-loop changes, copy `experiments/grug/base/` to `experiments/grug/<variant>/` and keep the edits local. Follow `.agents/skills/change-grug/SKILL.md` and `experiments/grug/README.md`.
- For optimizer-only experiments, still prefer a Grug copy when the goal is a clean modeling comparison. If there is already a living line for the experiment, use that exact line as the baseline instead of force-migrating it.
- Name the exact baseline artifact. Do not compare against "main", "current default", or another fuzzy reference.

## Pin The Inputs Before Editing

Write these down in the issue and logbook before spending real compute:

1. Exact question and hypothesis.
2. Exact baseline config, run, branch, and commit.
3. Initial FLOP budget. Default to `3e18` only when there is no better local norm.
4. Primary small-budget success metric.
5. Promotion target: match baseline, beat baseline, or earn an isoFLOP sweep.
6. Change bucket: architecture, optimizer, train-loop, or another single modeling change.

If one of these is missing, infer the smallest reasonable default and write the assumption explicitly.

## Right-Size The Workflow

- Treat the full staged workflow as the default for open-ended modeling lines, not as mandatory ceremony for every request.
- If the user asks for a narrow slice of work, do the narrow slice.
- Common narrower cases:
  - sweep one hyperparameter around a fixed config,
  - run one method spike to `3e18`,
  - launch one explicit comparison matrix the user already specified,
  - verify whether an existing promising point reproduces.
- In those cases, do not force bracketing, Vizier, or isoFLOP gates unless the request actually calls for them.
- Still keep the basics:
  - exact baseline or fixed reference point,
  - explicit run tags,
  - a dedicated W&B view for the line,
  - a short note on what question this narrower run set is supposed to answer.

## Scout Prior Art First

- Search internal prior art before adding code:
  - `docs/reports/index.md`
  - related GitHub issues and PRs
  - `experiments/speedrun/`
  - `experiments/grug/variants.md`
  - `docs/reports/grug-archive.md`
- Search the repo for the exact knob or method before creating new config surface.
- Search external papers, repos, or blog posts when they materially sharpen the baseline or tuning plan, but do not let external prior art override a cleaner local comparison.
- Record a short scouting note: what already exists, what baseline seems strongest, and what failure modes or hyperparameter ranges are already known.

## Keep Experimental Diffs Small

- Change one bucket at a time. Do not mix cleanup into experimental diffs.
- Keep Grug template-first. Prefer copy-paste variants over reusable framework surface while the idea is still being tested.
- Keep non-swept knobs copied from the baseline and hard-coded where practical.
- If the method needs many new knobs to survive, it is probably not stable enough yet.
- Before broad review or larger sweeps, generate a diff report for the new variant:

```bash
uv run python scripts/grug_dir_diff.py \
  experiments/grug/base \
  experiments/grug/<variant> \
  --out /tmp/grug-diff \
  --no-open
```

## Open The Right Artifacts

- Reuse the `agent-research` artifact set: long-lived branch, experiment issue, and append-only logbook.
- Start from `.github/ISSUE_TEMPLATE/experiment.md`, but expand it beyond the minimal template.
- Create a dedicated W&B view or workspace up front, before the run set gets messy.
- Use the issue number as the canonical line tag name: `exp<issue number>`.
- Tag every run in the line with that exact tag. This includes:
  - the baseline verification run,
  - method sanity runs,
  - manual bracket runs,
  - Vizier trials,
  - any promoted isoFLOP runs for the same issue.
- Add a few more short tags that make slicing easy, for example `baseline` or `method`, the model family, and the method name.
- Configure the dedicated W&B view or workspace around that exact `exp<issue number>` tag so it exposes the full run line in one place.
- Record the canonical tag name and the W&B view link or query in:
  - the research logbook,
  - the issue body or summary section.
- Prefer this issue structure:
  - Motivation
  - Prior art
  - Exact baseline
  - Hypothesis
  - Initial FLOP target
  - Metrics
  - Planned configs
  - Bracketing plan
  - Vizier plan
  - IsoFLOP promotion gate
  - Notes / outcomes
- Write for a reader who understands ML systems but was not in the room.

## Use Marin's Default Metric Package

These exact keys already exist in current Marin W&B runs and should be the default dashboard panels:

- `train/loss` vs `throughput/total_tokens`
- `train/loss` vs `global_step`
- `train/loss` vs `throughput/total_gflops`
- `eval/paloma/c4_en/bpb` vs `throughput/total_tokens`
- `eval/paloma/c4_en/bpb` vs `global_step`
- `eval/paloma/c4_en/bpb` vs `throughput/total_gflops`
- `eval/bpb` vs `throughput/total_tokens`
- `eval/uncheatable_eval/bpb` vs `throughput/total_tokens`
- `throughput/mfu`
- `throughput/tokens_per_second`
- `throughput/gflops_per_second`

Add method-specific panels on top of these, not instead of them. Common extras:

- `optim/*` for optimizer or schedule work
- `grad/*`, `params/*`, `updates/*`, `opt_state/*` when watch metrics are enabled
- routing, load-balance, or auxiliary-loss metrics for MoE-like methods
- `mixture/*` if the method changes stage scheduling or data weighting

Relevant repo anchors:

- `experiments/grug/README.md` for the full Grug metric list
- `scripts/ferries/daily_analysis.py` for the canonical daily eval triplet
- `experiments/isoflop_sweep.py` for the default isoFLOP metric key (`eval/paloma/c4_en/bpb`)

For tagging, follow the repo's existing plain-tag pattern such as `exp600`, `exp1529`, and short method/family tags.

## Run The Staged Workflow

Use the stages below when the task is an open-ended experiment line. If the request is narrower, run only the subset that matches the request.

### 1) Verify The Baseline

- Reproduce or verify the baseline at the chosen small budget on the same code path and comparable hardware.
- Make sure the baseline actually logs the metrics you plan to compare.
- If eval metrics are missing, fix eval wiring before making quality claims.

### 2) Get The Idea To Beat The Baseline

- Do a quick YOLO phase first, but keep the goal pointed at beating the baseline at the small budget.
- Do not make this stage torturous. The point is to find out whether there is a credible path to winning, not to spend forever polishing a weak idea.
- Use informed best judgment to get the method to the point where the loss curve is interpretable and the run has a plausible shot against baseline.
- Fix obvious breakage: NaNs, prolonged spikes, throughput collapse, missing evals, or bad checkpoint wiring.
- If the method is still obviously behind after a reasonable amount of stabilization, either retune in a narrower loop or stop. Do not turn phase 2 into an endless rescue mission.

### 3) Bracket The Promising Region

- Manually sweep the 1-3 dominant hypers around the current point.
- Usual suspects:
  - learning rate
  - weight decay
  - initialization scale
  - method strength or auxiliary coefficient
  - routing temperature or similar method-specific knob
- Look for a basin, not a lucky point.
- If you cannot bracket a convincing neighborhood, do more manual stabilization before larger sweeps.

### 4) Narrow With Vizier

- Use Vizier only after the method is stable and roughly bracketed.
- Start from `experiments/references/reference_hyperparameter_sweep.py` when you need a local pattern.
- Keep the search space narrow and justified.
- Optimize one explicit objective. For baseline-modeling comparisons, default to a small-budget eval metric such as `eval/paloma/c4_en/bpb` unless the experiment clearly demands another target.
- Write down:
  - exposed parameters
  - search ranges
  - objective
  - number of trials
  - early-stop rules, if any

### 5) Pass The Small-Budget Gate

Only promote the idea if, after reasonable tuning:

- it trains reliably,
- the whole curve looks healthy,
- it is competitive with the baseline at the target budget,
- the result is robust enough that more compute seems justified.

If the endpoint is fine but the curve looks sick for a long time, treat that as a tuning failure, not a success.

This gate is for promotion decisions. It is not required when the user only asked for a bounded run set such as a one-knob sweep or a single `3e18` spike.

### 6) Promote To IsoFLOP Only When Earned

- Use `experiments/isoflop_sweep.py` and `lib/marin/src/marin/scaling_laws/` once the small-budget gate is passed.
- Keep `eval/paloma/c4_en/bpb` as the default comparison metric unless there is a stronger experiment-specific reason not to.
- Compare method vs baseline across the undertrained side, near-optimal region, and overtrained side.
- Prefer clean separation across the band. A tiny local win near the optimum is weak evidence.
- Do not build the whole claim on pathological edge cases at the very smallest budgets.

Skip this stage entirely when the spec does not call for isoFLOP evidence.

## Judge Runs By Curve Health, Not Endpoint Alone

Good signs:

- the loss curve is mostly well-behaved,
- spikes are brief and recoverable,
- neighboring settings behave predictably,
- throughput and MFU stay in a reasonable band relative to the baseline.

Bad signs:

- long spiky periods,
- repeated instability that only recovers late,
- gains that appear only after obvious curve pathology,
- throughput collapse or severe MFU regression,
- a result that depends on one fragile setting.

`trained eventually` is not enough.

## Close Out Honestly

By the end of the line, make sure the issue and logbook show:

- the exact baseline and experiment paths,
- what was run,
- what the stage gate decided,
- whether the recommendation is `proceed`, `revise and retune`, or `stop`.

If the idea wins and should become the new default Grug path, upstream it with `.agents/skills/change-grug/SKILL.md` and archive the experiment trail in `docs/reports/grug-archive.md`.

If the idea loses, keep the negative result in the record anyway.

## Avoid These Failure Modes

- fuzzy baselines
- sweeping a broken method
- changing method, tokenizer, data mix, and hardware all at once
- adding framework abstraction before the idea is proven
- declaring an isoFLOP win from one point
- letting W&B replace the issue and logbook as the decision record
