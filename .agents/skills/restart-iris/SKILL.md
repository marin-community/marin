---
name: restart-iris
description: Restart the Iris controller with state preservation.
---

Restart the Iris controller. Follow the **[deploy-controller-fix runbook](../../runbooks/deploy-controller-fix.md)** — it owns the full procedure: controller-only vs full restart, the rebuild-image-before-restart sequence (a restart alone re-pulls a stale `:latest` and ships nothing), the dry-run/baseline workflow, `--skip-checkpoint` recovery, and verifying the running git-hash. Command reference: `lib/iris/OPS.md` "Controller Restart".

Config shorthand: `marin` → `lib/iris/config/marin.yaml`, `marin_dev` → `lib/iris/config/marin-dev.yaml`, `coreweave` → `lib/iris/config/coreweave.yaml`.

**NEVER do a full cluster restart** (`iris cluster restart`) without explicit user approval — this kills all running jobs.
