---
name: restart-iris
description: Restart the Iris controller with state preservation.
---

Restart the Iris controller. Follow the procedures in `lib/iris/OPS.md` — specifically the "Controller Restart" section (dry-run validation, restart workflow, post-restart verification, error recovery).

Config shorthand: `marin` → `lib/iris/config/marin.yaml`, `marin_dev` → `lib/iris/config/marin-dev.yaml`, `coreweave` → `lib/iris/config/coreweave.yaml`.

**NEVER do a full cluster restart** (`iris cluster restart`) without explicit user approval — this kills all running jobs.
