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
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/8634

## Current TL;DR

The baseline and variants have not run. Prior work favors narrow task-titled skills for routing, while current Iris operational knowledge is split between `lib/iris/OPS.md`, `lib/iris/docs/federation.md`, `lib/finelog/OPS.md`, and task-specific skills. The first experiment compares the current checkout with a generic-only variant and a layered variant that retains mutation-specific workflows.

## Baseline

- Date: 2026-08-24
- Code ref: `1936313ac7da5e1c4f08c0a1634e24cf66f4e772`
- Baseline: current `.agents/skills/` at `origin/main`
- Model: Luna, medium reasoning

## Hypothesis Queue

### Active

- `ISG-001`: A concise `use-iris` navigator improves cross-cutting job and federation answers because it routes agents to authoritative operational references. Next test: compare current skills with generic-only and layered variants on identical prompts.
- `ISG-002`: A generic `query-finelog` skill improves non-vLLM telemetry questions without reducing vLLM counter correctness. Next test: include generic schema-discovery and vLLM cumulative-snapshot cases.
- `ISG-003`: Mutation-specific skills remain necessary for monitoring, controller rollout, and destructive recovery because their authorization and persistence contracts are task-specific. Next test: score unsafe-restart and babysitting prompts separately from read-only diagnosis.

### Blocked


### Falsified / Dead End


### Promoted


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

## Entry Log

### 2026-08-24 20:03 UTC - ISG-000 prologue and prior work

- Hypothesis: Generic reference skills can improve broad Iris operations, but replacing mutation-specific workflows may regress safety and persistence.
- Commit Hash: `1936313ac7da5e1c4f08c0a1634e24cf66f4e772`
- Command: `uv run infra/echo/cli.py search "generic agent skills Iris Finelog federation versus task-specific skills" --limit 10`; inspected current skill descriptions and Iris/Finelog manuals.
- Config: Echo execution 3056; low-effort internal prior-work pass.
- Result: The most direct prior evidence favors narrow task-titled routing. Current manuals expose cross-cutting operational concepts absent from any generic skill.
- Interpretation: Compare replacement and layered designs. Do not infer that generic skills should replace safety-critical workflows.
- Next action: Create the coordinating issue, snapshot this prologue, and build isolated variant worktrees.
