# Ops Runbooks

A **runbook** walks an operator through one recurring operational *task* — deploy
Iris on GCP, stand up a CoreWeave cluster, roll out finelog, run a ferry by hand.
It's what you open to *do the thing*: the guardrails, the order, the commands, and
how to confirm it worked. Runbooks are the single source of truth for operational
**procedure**.

A runbook answers an **ops** question — "how do I deploy iris?", "why is the
controller slow?" — an operator acting on the infrastructure. It is *not* the
place for a **user** question ("why is *my job* pending?" — troubleshooting a
workload) or a **reference** question ("how does the scheduler work?" — that's
`OPS.md` or the code). Those have their own homes:

| Tree | Owns |
|---|---|
| `.agents/runbooks/` | how to *do* an operational task (procedure) |
| `.agents/skills/` | agent task playbooks, auto-surfaced by description |
| `.agents/ops/` | dated incident postmortems |
| `lib/*/OPS.md` | command / SQL / connection **reference** |

## The rule: own the procedure, point at everything else

A runbook owns its procedure; everything else points at it instead of restating
it. `OPS.md` keeps the command/SQL reference and the runbook *links* to it. A
skill that is really a single situation (e.g. `restart-iris`) is a thin pointer to
its runbook; skills with genuine loop logic (`babysit-job`, `triage-canary`) stay
skills and just drop duplicated reference. A purely-mechanical spine becomes a
script the runbook drives (e.g. `scripts/iris/controller_deploy.py`).

**De-dup, don't cross-link.** When a runbook takes a procedure, delete the
duplicated prose elsewhere and replace it with a tight pointer — don't leave it in
place with a link bolted on. And **don't maintain an index or a TOC**: the file
listing in this directory is the index, and a hand-kept table goes stale the
moment a runbook is renamed.

## Conventions

- **One task per file**, `.agents/runbooks/<slug>.md`, undated kebab-case named
  for the action (`deploy-iris-gcp.md`, `run-datakit-ferry.md`).
- **Cite, don't restate.** Point at the canonical command in OPS.md or a script
  docstring; explain the *why* and let the reference own the *how*. Prefer linking
  a runnable script over pasting its commands. Anchor a citation on a section
  name, not a bare line number — line numbers drift.
- **Lead with the guardrail.** Anything destructive (deletes a node, bounces a
  cluster, costs money) is called out *before* the steps, gated on a human yes.
- **Keep `See also` to a couple of links, or drop it.** It is not a sitemap.
- **Voice:** calm, dry, direct. Point at the thing, say why it bites, give the
  smallest fix. No catastrophizing, no gold-plating.

## Template

Most runbooks are an *action* (the Steps shape below). A few are a *diagnosis* —
swap Steps for `Diagnose` (the decision tree) then `Resolve` (the fix per branch).

```markdown
---
name: <slug>
description: <one line — the task this runbook handles>
---

# Runbook: <Title>

**When you're here:** <the trigger that sends an operator here, 1–2 sentences>

**TL;DR:** <the fast path in one paragraph or a few bullets>

## Before you touch anything

<guardrails: what is destructive, what needs a human yes, what to capture first.
Omit only if nothing here is dangerous.>

## Steps

<the procedure, least-destructive first; link the command reference, don't paste
it. Number the steps when order is load-bearing.>

## Verify

<how to confirm it actually worked — not just that the process came back.>
```
