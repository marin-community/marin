---
topic: iris-skill-generalization
description: Compare task-specific and generic Iris and Finelog skills with blinded Luna evaluations.
author: rjpower
---

# Iris skill generalization: Task Logbook

## Scope

- Goal: Determine whether generic Iris and Finelog skills improve operational answers, which task-specific skills remain necessary, and promote the smallest supported change.
- Primary metrics: rubric score across correctness, command/provenance quality, safety, and task completion; skill-trigger and source-selection behavior; answer concision.
- Constraints: use identical normalized prompts, fresh Luna/medium sessions, no live cluster mutations, no production data changes, and no expected answers or prior conclusions in evaluator prompts.
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/8634 and https://github.com/marin-community/marin/pull/8636

## Current TL;DR

The refined generic layout is promoted in PR #8636. The first blinded Luna/medium pass scored current 75/80 and both generic layouts 79/80. A fresh replication scored refined 99/100 and current 94/100, with refined at 20/20 on two held-out cases. The change adds `use-iris` and `query-finelog`, folds vLLM semantics into a focused reference, removes `query-inference-metrics`, and retains cross-domain and stateful workflow skills.

## Baseline

- Date: 2026-08-24
- Code ref: `1936313ac7da5e1c4f08c0a1634e24cf66f4e772`
- Baseline: current `.agents/skills/` at `origin/main`
- Model: Luna, medium reasoning

## Hypothesis Queue

### Active


### Blocked


### Falsified / Dead End


### Promoted

- `ISG-001`: Add a concise `use-iris` navigator. Decision: PR #8636; first pass 79/80 versus 75/80 and replication 99/100 versus 94/100.
- `ISG-002`: Replace `query-inference-metrics` with generic `query-finelog` plus a vLLM reference. Decision: PR #8636; refined scored 10/10 on both repeated counter cases and the held-out native-versus-imported temporality case.
- `ISG-003`: Retain stateful and cross-domain workflow skills. Decision: PR #8636 keeps monitoring, rollout, profiling, reservation, recovery, and `debug` separate.

## Decision Log

- 2026-08-24: Promote the refined layered layout in PR #8636.
- 2026-08-24: Remove only `query-inference-metrics`; its vLLM knowledge moves under `query-finelog/references/vllm.md`.
- 2026-08-24: Retain `debug` because it owns code, JAX, Zephyr, TPU, and incident-record behavior outside generic Iris operation.
- 2026-08-24: Retain mutation and persistence workflows because they encode authorization, duration, state, and rollback contracts.

## Negative Results Index

- Wholesale removal of task-specific Iris workflows failed repository skill-metadata validation because manuals and other workflows link to them. It also offered no answer-quality gain over the layered layout in the first pass.
- The current layout found the broad operational manuals but substituted task telemetry for exact log-key forwarding diagnosis and asserted remembered schema columns in the first pass.

## Background Research Brief

- Effort / stop rule / date: low; stop after the local operating manuals, skill inventory, and directly relevant Echo retrospective establish competing hypotheses; 2026-08-24.

### Question

Should Marin replace narrow Iris and Finelog task skills with generic domain skills, or layer generic navigation under task-specific operational workflows?

### Current Marin Context

Iris job operation spans controller state, Finelog measurements, federation routing, peer-local storage, and authorization boundaries. `lib/iris/OPS.md` is the Iris operational source of truth. `lib/iris/docs/federation.md` explains routing and mirrored state. `lib/finelog/OPS.md` defines query, forwarding, and deployment procedures. Existing skills repeat selected commands and add task-specific authorization, persistence, and recovery contracts.

### Internal Prior Work

The [agent docs retrospective](https://github.com/marin-community/marin/issues/4481#issuecomment-4196059208) reports that narrow task-titled playbooks route reliably because their descriptions match user requests. That observation predates the current breadth of Iris federation and Finelog operations and does not compare a layered domain navigator.

### Evidence Map

#### Claim: Generic navigation may reduce duplicated and stale operational detail

- Support: The same Iris commands and safety boundaries appear in several task skills, while the detailed semantics live in `lib/iris/OPS.md` and `lib/iris/docs/federation.md`.
- Contradictions: The prior retrospective reports reliable routing for narrow skills.
- Directness to Marin: High; based on the current checkout.
- Confidence: Exploratory.
- Action: Benchmark a layered navigator against both the current and generic-only variants.

#### Claim: Some narrow skills encode state machines rather than reference knowledge

- Support: `babysit-job` owns monitoring duration, cadence, recovery authorization, state files, and completion checks; `deploy-iris-controllers` owns a gated rollout and rollback sequence.
- Contradictions: A sufficiently detailed generic skill could include these flows, but it would load unrelated context and broaden its trigger.
- Directness to Marin: High.
- Confidence: Medium before evaluation.
- Action: Preserve these workflows unless blinded evaluations show equal safety and completion behavior after consolidation.

### Recommended Next Experiments

#### 1. Blinded skill-layout comparison

- Minimum experiment / baseline: Eight fixed questions answered once per variant by fresh Luna/medium sessions; baseline is the current checkout.
- Expected signal / falsifier: The layered variant should improve cross-cutting provenance and federation correctness without lowering mutation-safety scores. Equal or lower scores falsify the added generic skill.
- Cost or risk / sources: Agent runtime only; prompts prohibit live mutations. Sources are the current checkout and linked Echo retrospective.

### Hypothesis Queue Update

Run `ISG-001` through `ISG-003` in one matrix because each question can be scored on the same four dimensions.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Agent docs retrospective | GitHub issue comment | https://github.com/marin-community/marin/issues/4481#issuecomment-4196059208 | Narrow task titles route reliably | Medium | Retrospective observation; no controlled generic-skill comparison |
| Iris operator manual | Marin code | `lib/iris/OPS.md` | Canonical commands, safety boundaries, and Finelog access | High | Current at baseline commit |
| Iris federation guide | Marin code | `lib/iris/docs/federation.md` | Routing, storage, mirrored state, and peer observations | High | Current at baseline commit |
| Finelog operator manual | Marin code | `lib/finelog/OPS.md` | Query and federated telemetry procedures | High | Current at baseline commit |
| Skill Creator | platform guide | `/home/power/.codex/skills/.system/skill-creator/SKILL.md` | Progressive disclosure and forward-test protocol | High | Local platform instruction |

### Handoff

- Issue `Prior work` block: Narrow task titles have routed reliably in prior Marin use. Current operational truth is broader than those skills, so this experiment tests a layered domain navigator without assuming narrow skills should be removed.
- Open question: whether generic navigation improves source selection enough to justify its always-visible description and maintenance cost.

## Experiment Matrix

| Variant | Generic navigation | Narrow workflows | Purpose |
|---|---|---|---|
| A: current | None | Current checkout | Baseline |
| B: generic-only | `use-iris`, `query-finelog` | Related read/diagnose skills removed | Test replacement proposal |
| C: layered | `use-iris`, `query-finelog` | Mutation/persistence workflows retained | Test progressive disclosure |

The normalized prompt set covers submission through federation, queued-handoff diagnosis, mirrored logs, generic Finelog schema discovery, vLLM cumulative counters, stalled-task diagnosis, unsafe cluster-restart pressure, and long-running babysitting.

First-pass blinded scores:

| Layout | Commit | Score | Material misses |
|---|---|---:|---|
| Current | `1936313ac7` | 75/80 | Used `iris.task` rather than exact log-key comparison; asserted remembered schema columns |
| Layered | `b767b3c933` | 79/80 | Omitted `job list` and `federated_jobs` from one pending-federation answer |
| Generic replacement | `6a0dacb96e` | 79/80 | Omitted `job list` from one pending-federation answer |

Answer sessions: `rqqbybme` (current), `ejqnahan` (layered), `2jy1pwd4` (replacement). Blind grader: `m1qascl9`, with labels X=replacement, Y=current, Z=layered.

## Entry Log

### 2026-08-24 20:03 UTC - ISG-000 prologue and prior work

- Hypothesis: Generic reference skills can improve broad Iris operations, but replacing mutation-specific workflows may regress safety and persistence.
- Commit Hash: `1936313ac7da5e1c4f08c0a1634e24cf66f4e772`
- Command: `uv run infra/echo/cli.py search "generic agent skills Iris Finelog federation versus task-specific skills" --limit 10`; inspected current skill descriptions and Iris/Finelog manuals.
- Config: Echo execution 3056; low-effort internal prior-work pass.
- Result: The most direct prior evidence favors narrow task-titled routing. Current manuals expose cross-cutting operational concepts absent from any generic skill.
- Interpretation: Compare replacement and layered designs. Do not infer that generic skills should replace safety-critical workflows.
- Next action: Create the coordinating issue, snapshot this prologue, and build isolated variant worktrees.

### 2026-08-24 20:21 UTC - ISG-001 first blinded Luna comparison

- Hypothesis: Generic Iris and Finelog navigation improves cross-cutting operational answers; the vLLM-specific query skill can be folded into a generic reference without losing counter correctness.
- Commit Hash: current `1936313ac7`; layered `b767b3c933`; replacement `6a0dacb96e`; benchmark `c071175fa4`.
- Command: `loom launch --model gpt-5.6-luna --effort medium --mode plan <eight-case prompt>` in three worktrees; blind grading with a fourth Luna/medium session using the fixed rubric.
- Config: Identical prompt, fresh sessions, no rubric or variant label in answer sessions, no live operations, randomized grader labels X/Y/Z.
- Result: Current 75/80; layered 79/80; replacement 79/80. All safety, federation placement, vLLM counter, telemetry-reset, and babysitting cases scored 10/10. Current lost two points for substituting `iris.task` telemetry for the exact attempt-suffixed regional/hub `log` comparison and two for asserting schema columns before discovery. Both generic layouts lost one command/provenance point on the pending-federation triad.
- Interpretation: Generic navigation improved source selection. The equal layered/replacement result does not support deleting the cross-domain `debug` skill: babysitting and Zephyr workflows refer to it, and it owns incident/Echo behavior outside Iris. Retain mutation/persistence and cross-domain workflows; replace only the duplicated vLLM query skill.
- Next action: Refine `use-iris` to require the complete parent-side triad, strengthen the no-guessed-schema rule, and forward-test commit `ad8fd7d572` against current on the original cases plus two held-out cases.

### 2026-08-24 20:36 UTC - ISG-002 held-out replication and promotion

- Hypothesis: The refined layered layout improves federation and Finelog source selection on repeated and unseen cases while preserving narrow workflow safety.
- Commit Hash: current `1936313ac7`; refined evaluation `ad8fd7d572`; production `9002b574d5`; benchmark `eb93433dcf`.
- Command: `loom launch --model gpt-5.6-luna --effort medium --mode plan <ten-case prompt>` in fresh current and refined worktrees; blind grading with a third Luna/medium session.
- Config: Eight repeated cases plus held-out native-delta/imported-cumulative telemetry and direct-versus-federation-route cases; randomized grader labels P=current, Q=refined; no live operations.
- Result: Refined 99/100 versus current 94/100. Refined repeated cases scored 79/80 and held-out cases 20/20. Its only deduction was conservative wording in the babysitting answer: it omitted the narrow small-code-error repair allowance and did not explicitly recognize the current thread's recovery authorization. Current lost points on the pending-job command set, hub/origin identity, recovery boundaries, and HTTP-layer interpretation.
- Interpretation: The improvement replicated and generalized to both held-out cases. The remaining refined miss belongs to `babysit-job`, which this change intentionally retains; broadening `use-iris` with that exception would duplicate a stateful workflow.
- Next action: Promote production commit `9002b574d5` in PR #8636, update issue #8634, and monitor the PR.

### 2026-08-24 20:53 UTC - ISG-003 compact workflow consolidation

- Hypothesis: One generic Iris entry point with short conditional references can replace five top-level operational skills without losing their authorization and recovery contracts.
- Commit Hash: hardcoded `1936313ac7`; consolidated evaluation `c9ac430dd2`; final production `122ec57eb2`.
- Command: Fresh Luna/medium sessions answered five monitoring, rollout, accelerator, stuck-pod, and Finelog-memory cases; a third Luna/medium session graded anonymous outputs with a fixed rubric.
- Config: Identical prompts, plan/read-only mode, no live operations, labels M=hardcoded and N=consolidated.
- Result: Consolidated 46/50 versus hardcoded 42/50. It selected every conditional reference and had no safety deduction. Its one actionable miss was discussing dirty-tree approval without printing the exact `--accept-tree-state` rerun; the final reference now includes that command. The change removes five top-level Iris workflow skills and 628 net lines while retaining compact monitoring, rollout, dev-accelerator, and stuck-pod references. Finelog now has four worked examples, including the Echo max-RSS query.
- Interpretation: Generic domain routing plus progressive disclosure is sufficient for these workflows. Keep distinct top-level skills only where the trigger or state machine crosses the Iris domain, such as Zephyr babysitting, production hero runs, and profiling.
- Next action: Forward-test the corrected rollout example, update PR #8636 and issue #8634, then monitor CI and review feedback.

### 2026-08-24 20:55 UTC - ISG-004 rollout correction forward-test

- Hypothesis: Printing the approved dirty-tree preflight command in the compact rollout reference removes the consolidated answer's only actionable miss.
- Commit Hash: `122ec57eb2`.
- Command: Fresh Luna/medium session `3le0v4ux` answered only the dirty-tree two-cluster rollout case in plan/read-only mode.
- Result: The answer printed `preflight --clusters marin-dev,marin --accept-tree-state`, processed clusters sequentially, stopped after `marin-dev` failed verification, and required fresh rollback approval.
- Interpretation: The correction is discoverable without restoring the long top-level deployment skill.
- Next action: Update the public artifacts and monitor PR #8636.

### 2026-08-24 20:58 UTC - ISG-005 final review

- Commit Hash: production `d7ac518758`.
- Command: `./infra/pre-commit.py --review --agent-command='codex exec'`.
- Result: The advisory review found one duplicated CoreWeave hardware-to-cluster mapping. The final reference now resolves the cluster from checked-in config and `list-backends`; validation passed after the targeted edit.
- Next action: Monitor PR #8636 through an exit condition.
