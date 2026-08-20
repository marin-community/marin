---
name: task-snapshot
description: "Create stable commit or tag snapshots for task milestones, experiment results, and reproducible artifact links."
---

# Task Snapshot

Use a commit for an ordinary milestone and an annotated tag for a published
experiment checkpoint or long-lived handoff. Include only files relevant to the
snapshot; do not present a snapshot as evidence until it is reproducible.

When posting it to an issue/PR, include the commit/tag link, pinned GitHub links
to logbook/config/report/artifacts, exact commands, and durable environment
details (hardware/cluster/device count and non-secret variables). A pinned link
uses the snapshot in its path, for example:

```text
https://github.com/marin-community/marin/tree/<commit-or-tag>/.agents/logbooks/<topic>.md
```

Before handoff, verify commands, shapes/configs, and environment are explicit;
the logbook and issue point to the same result; and dense data is linked from an
artifact, W&B, or dashboard.
