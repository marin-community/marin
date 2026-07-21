# Marin Ops Workflow

This directory implements the durable workflow behind `ops.oa.dev`: the
versioned database schema, authenticated Kubernetes Warning snapshot contract,
signal/case and agent-turn state machines, and the agent runner boundary.
Production integrations are controlled by explicit configuration gates.

The reviewed architecture and full contract live in
[`../../.agents/projects/ops_workflow/`](../../.agents/projects/ops_workflow/).

Run the focused tests with:

```bash
uv run --project infra/ops pytest infra/ops/tests
```
