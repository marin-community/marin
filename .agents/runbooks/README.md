# Ops Runbooks

A **runbook** is a markdown walkthrough of one recurring operational *situation*:
what sends you here, how to diagnose it, how to fix it, how to verify the fix,
and *why* it happens. Runbooks are the single source of truth for operational
**procedure**.

They sit beside the rest of the agent-knowledge tree:

| Tree | Owns | Shape |
|---|---|---|
| `.agents/runbooks/` | how to handle situation X (procedure) | living, undated |
| `.agents/skills/` | agent task playbooks (auto-surfaced) | living, invoked by name |
| `.agents/ops/` | what went wrong once (incident narratives) | dated postmortems |
| `lib/*/OPS.md` | command/SQL/connection **reference** | living catalogs |

## How runbooks relate to everything else

A runbook **owns the procedure**. Everything else **points at it** instead of
restating it:

- **`OPS.md` keeps the reference** — command catalogs, SQL schemas, connection
  selectors, state-code legends. Runbooks *link* to those sections and never
  copy them. The OPS.md troubleshooting matrix stays as an at-a-glance index;
  rows that have a runbook become a one-line pointer into it.
- **Skills point at runbooks.** A skill that is really a situation-walkthrough
  (e.g. `restart-iris`) is a thin pointer to its runbook. Skills that own genuine
  loop/lifecycle logic (`babysit-job`, `triage-canary`, `run-ferries`) stay as
  skills and only drop their duplicated *reference* listings to OPS.md / runbook
  links.
- **Postmortems feed runbooks.** When an incident in `.agents/ops/` reveals a
  recurring failure class, its diagnosis flow graduates into a runbook; the
  postmortem stays as the dated record and the runbook cites it under "Why".
- **`AGENTS.md` policies live once.** A cross-cutting guardrail (never restart a
  cluster without permission) keeps its bare prohibition in `AGENTS.md` and links
  to the runbook that owns the nuance.

The rule: **de-dup, don't just cross-link.** When a runbook takes ownership of a
procedure, the duplicated prose elsewhere is replaced by a tight pointer (plus
any one-line guardrail kept inline) — not left in place with a link bolted on.

## Conventions

- **Location:** `.agents/runbooks/<slug>.md`. One situation per file.
- **Naming:** undated kebab-case matching skill directory names
  (`deploy-controller-fix.md`). Runbooks are living docs, not dated incidents.
- **Cite, don't restate.** Point at the canonical command in OPS.md, the script
  docstring, or the code `file:line`. Explain the *why*; let the reference own
  the *how*. Prefer linking a runnable script over pasting its commands.
- **Lead with the guardrail.** Anything destructive (deletes a node, bounces a
  cluster, costs money) gets called out *before* the steps, gated on a human yes.
- **Voice:** calm, dry, direct — the `ops-steward` register. Point at the thing,
  say why it bites, give the smallest fix. No catastrophizing, no gold-plating.

## Template

```markdown
---
name: <slug>
description: <one line — the situation this runbook handles>
---

# Runbook: <Title>

**When you're here:** <the symptom/trigger that sends an operator or agent to
this runbook, 1–2 sentences>

**TL;DR:** <the fast path in one paragraph or three bullets>

## Before you touch anything

<guardrails and safety: what is destructive, what needs human approval, what
baseline to capture first. Omit only if nothing here is dangerous.>

## Diagnose

<the decision tree. State a hypothesis, give the command (link OPS.md for the
command reference), read the result, branch. Distinguish the look-alike causes.>

## Resolve

<the fix steps for each branch, least-destructive first.>

## Verify

<how to confirm it actually worked — not just that the process came back.>

## Why this happens

<the mechanism. Link the postmortem(s) and the code file:line. This is what a
reference catalog can't give you and why the runbook exists.>

## See also

<links: the OPS.md reference sections this leans on, related runbooks, the
skill(s) that point here, the postmortem(s) that feed it.>
```

## Index

| Runbook | Situation |
|---|---|
| [new-cluster](new-cluster.md) | Stand up a brand-new Iris cluster in a new region from scratch — the bootstrap order |
| [deploy-controller-fix](deploy-controller-fix.md) | Ship a merged controller/iris fix, or restart the controller, without shipping a stale `:latest` |
| [diagnose-stuck-pending-job](diagnose-stuck-pending-job.md) | A job sits PENDING — capacity wait vs frozen scheduler vs reservation-taint |
| [triage-tpu-worker-failure](triage-tpu-worker-failure.md) | Repeated TPU task failures — wedged container vs a genuinely bad node |
| [offline-checkpoint-analysis](offline-checkpoint-analysis.md) | Run expensive controller queries safely — offline checkpoint + parquet-over-GCS |
| [repair-task-attempts-invariant](repair-task-attempts-invariant.md) | tasks vs task_attempts invariant breaks — ghost co-tenants, orphan attempts, split slices |
| [finelog-rollout-rollback](finelog-rollout-rollback.md) | Roll out / roll back a finelog server; query archived parquet |
| [run-datakit-ferry-manually](run-datakit-ferry-manually.md) | Run, stop, and validate a datakit ferry smoke run by hand |
| [stand-up-coreweave-cluster](stand-up-coreweave-cluster.md) | Bring a CoreWeave GPU cluster to a running job; run the canary smoke; clean up nodepools |
| [tpu-ci-fleet](tpu-ci-fleet.md) | Operate the self-hosted TPU CI runner fleet — stand up, rotate PAT, recover runners |
