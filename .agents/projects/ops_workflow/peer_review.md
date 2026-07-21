# Ops Workflow Peer Review

## Review history

The initial vertical slice was reviewed before PR #7460 and produced the durable queue, global serialization, prompt-injection boundary, read-only ops skill, and browser-flow improvements now present in the branch.

An unpublished follow-up proposed an internal webhook receiver, Cloud Tasks, and a loopback relay. Design review found that it solved reachability and retry concerns for an ingestion service that the workflow did not need. The design was replaced before publication.

## Current review target

The current patch makes Grafana PostgreSQL polling the only automatic alert source. Review should focus on:

1. Grafana 13 serialization and upgrade risk in `grafana_source.py`.
2. One-minute leader selection during rolling Cloud Run revisions.
3. Two-successful-absence resolution and failed-poll behavior.
4. The least-privilege `ops_grafana_reader` grants.
5. Warning mute timing versus error/critical Slack and email delivery.
6. IAP normalization for `*@openathena.ai` and `ops@openathena.ai`.
7. The local Grafana-shaped database and Playwright vertical slice.

## Code review disposition

The required `./infra/pre-commit.py --review --agent-command='claude -p'` review ran against commit `8538e135f7`. Four independent passes and a consolidation pass found no defect in the polling, IAP, SQL-role, or alert-routing design. The actionable findings were cleanup in the earlier spike implementation:

- a test-only scheduler, turn model, and fake runner duplicated the PostgreSQL and Loom path;
- reconciliation threaded an unnamed five-field tuple through several methods;
- archive outcomes conflated an active turn with an already archived or absent case;
- Grafana URL and turn lease values were duplicated literals;
- the Cloud Run component retained an unused public-access mode.

The follow-up deletes the parallel subsystem, introduces the named `TouchedSignal` and `ArchiveResult` types, splits case materialization into focused helpers, centralizes the constants, and makes the shared Cloud Run component IAP-only. The archive integration test covers active, first archive, repeat archive, missing case, and queued-turn cancellation. A second review found only an unused future mutation ledger and an unused outcome enum; both were removed before the initial migration was deployed.

A final review run was attempted after those removals, but every Claude lane exited before inspection because the account had reached its session limit. It emitted no findings and is not counted as a clean review. The two completed review rounds and their full dispositions remain the peer-review basis for rollout.
