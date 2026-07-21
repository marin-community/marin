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

The final code review command and disposition are recorded in the PR and Weaver peer-review artifact.
