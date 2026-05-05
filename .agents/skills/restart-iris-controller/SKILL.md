---
name: restart-iris-controller
description: Restart the Iris controller with state preservation. Use when asked to restart, redeploy, or upgrade the Iris controller or cluster.
---

Restart the Iris controller. Follow the procedures in `lib/iris/OPS.md` — specifically the "Controller Restart" section (dry-run validation, restart workflow, post-restart verification, error recovery).

Config shorthand: `marin` → `lib/iris/examples/marin.yaml`, `marin_dev` → `lib/iris/examples/marin-dev.yaml`, `coreweave` → `lib/iris/examples/coreweave.yaml`.

**NEVER do a full cluster restart** (`iris cluster restart`) without explicit user approval — this kills all running jobs.
