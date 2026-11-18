# Controller checkpoint recovery

Use this procedure for a GCE controller whose local database causes repeated
OOMs or startup failure. Restarting reuses that database; restoring an older
checkpoint discards newer jobs and state. Obtain recovery authorization first.
For a reachable controller, prefer the recorded
[deployment rollback](../OPS.md#rolling-back-a-controller-deploy-migration-aware).

Read these values from the target cluster config:

- `STATE_DIR` — controller local state dir, default `/var/cache/iris/controller` (override: `storage.local_state_dir`). The DB lives in `$STATE_DIR/db`.
- `REMOTE` — `storage.remote_state_dir` (e.g. `gs://marin-us-central2/iris/state`). Checkpoints live at `$REMOTE/controller-state/<epoch_ms>/{controller.sqlite3.zst,auth.sqlite3.zst}`.

```bash
# 0. SSH to the controller VM (the GCE instance labelled iris-<prefix>-controller=true),
#    then set STATE_DIR/REMOTE from the cluster config so the commands below resolve.
gcloud compute ssh iris-controller-marin --zone <zone> --tunnel-through-iap
export STATE_DIR=/var/cache/iris/controller
export REMOTE=gs://<bucket>/iris/state

# 1. Pick a pre-spike checkpoint. The DB size is a good proxy for backlog/health:
#    a checkpoint much larger than its neighbours was already bloated — pick an
#    earlier, smaller one. Each subdir is named with its epoch_ms.
gcloud storage ls --long --readable-sizes "$REMOTE/controller-state/**/controller.sqlite3.zst"

# 2. Stop the controller (frees the RAM the bloated DB is consuming).
sudo docker stop iris-controller

# 3. Move the bloated DB ASIDE — never delete it. Startup reloads $STATE_DIR/db
#    if present, so this is what forces a fresh restore; keeping it makes the
#    rollback reversible.
sudo mv "$STATE_DIR/db" "$STATE_DIR/db.bloated.bak.$(date +%s)"

# 4. Restore the chosen checkpoint into $STATE_DIR/db using the controller image's
#    own download_checkpoint_to_local (handles the GCS pull, zstd decompress, and
#    the paired auth DB). Run it in a one-shot container so it reuses the VM's
#    ambient GCS credentials. Substitute <epoch_ms> from step 1.
IMAGE="$(sudo docker inspect --format='{{.Config.Image}}' iris-controller)"
sudo docker run --rm --network=host -v "$STATE_DIR:$STATE_DIR" "$IMAGE" \
    .venv/bin/python -c "from pathlib import Path; \
from iris.cluster.controller.checkpoint import download_checkpoint_to_local as restore; \
ok = restore('$REMOTE', Path('$STATE_DIR/db'), checkpoint_dir='$REMOTE/controller-state/<epoch_ms>'); \
raise SystemExit(0 if ok else 1)"

# 5. Confirm the restore actually produced a DB BEFORE starting (if it didn't, the
#    controller would reload the latest — often still-bloated — checkpoint on start).
test -f "$STATE_DIR/db/controller.sqlite3" || { echo 'RESTORE FAILED; restore the backup before starting'; exit 1; }

# 6. Start and verify it serves.
sudo docker start iris-controller
curl -sf http://localhost:10000/health && echo " controller healthy"
```

**Rollback cost.** Jobs and state created *after* the chosen checkpoint are dropped. Workers on separate VMs and other infrastructure are unaffected — they re-register with the recovered controller.

If recovery fails, stop the controller, move the failed restored database aside,
put the preserved database back at `$STATE_DIR/db`, and start the controller.
