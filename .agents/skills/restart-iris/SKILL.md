---
name: restart-iris
description: Restart the Iris controller with state preservation.
---

Restart the Iris controller. Follow the procedures in `lib/iris/OPS.md` —
specifically the "Controller Restart" section (restart workflow, post-restart
verification, rollback, error recovery).

Config shorthand: `marin` → `lib/iris/config/marin.yaml`, `marin_dev` →
`lib/iris/config/marin-dev.yaml`, `cw-us-east-02a` / `cw-rno2a` →
`lib/iris/config/cw-*.yaml` (CoreWeave).

Workflow per cluster:

1. Confirm the working tree is exactly the code to ship (`git status`,
   `git log -1`) — the restart builds and deploys the **current tree content**,
   including uncommitted changes.
2. Capture a baseline: `iris --cluster=<name> cluster status`.
3. `iris --cluster=<name> cluster controller restart` (add `--skip-checkpoint`
   only if the checkpoint step times out).
4. Verify with `iris --cluster=<name> cluster status`: the controller must be
   healthy AND running the `:<git-short-hash>` image you expect — not merely
   back up.

Notes:

- `iris cluster controller serve --dry-run` is NOT a pre-restart validation
  gate: it boots a full local controller that serves until killed (side
  effects suppressed), for interactive state inspection. Don't run it as a
  restart step. Rely on the unit suite / CI on the tree instead.
- GCE clusters (`marin`, `marin_dev`) restart over `gcloud compute ssh`
  (IAP). `Permission denied (publickey)` means your session's local username
  lacks SSH to the VM; the restart aborts safely (`cluster status` shows the
  old version still healthy — nothing deployed). Do not mint new SSH/OS Login
  keys from an unattended session; hand the restart to a session with working
  SSH. CoreWeave clusters restart via the Kubernetes API and need no SSH.
- **NEVER do a full cluster restart** (`iris cluster restart`) without
  explicit user approval — this kills all running jobs.
