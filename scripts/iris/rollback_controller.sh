#!/usr/bin/env bash
#
# Roll an Iris controller back to a previously-deployed image AND its
# pre-migration checkpoint.
#
# WHY BOTH: `iris cluster controller restart` applies forward-only DB migrations
# in place on the controller's on-VM state DB. Some are destructive (e.g. dropped
# tables). Once the new controller has migrated the DB, redeploying the OLD image
# alone is NOT safe — the old code loads a schema it does not understand and hits
# missing-table errors at runtime. A correct rollback therefore restores the
# pre-migration checkpoint (taken while the old code was still running) and then
# starts the old image on top of it.
#
# WHAT IT DOES (two phases, each individually safe):
#   Phase A  Redeploy OLD_IMAGE in place via `controller restart --image` while
#            the current controller is still reachable, so it takes the in-place
#            bootstrap path (it does NOT recreate/terminate the VM). The old
#            controller comes up on the still-migrated DB and typically fails its
#            health check — that is expected; Phase B fixes the DB.
#   Phase B  On the controller VM: stop the container, move the post-migration DB
#            aside (preserved, never deleted), restore CHECKPOINT_DIR into the DB
#            dir using the controller image's own download_checkpoint_to_local,
#            verify, then start the (old-image) container on the restored DB.
#
# PRECONDITION: the current controller must be REACHABLE (this script uses the
# in-place restart path). If the controller is wedged/unreachable, do NOT use
# this script — follow the fully-manual on-VM procedure in lib/iris/OPS.md
# ("Controller Checkpoint Rollback"), which never risks recreating the VM.
#
# SAFE BY DEFAULT: prints the plan and exits. Pass --execute to actually run.
#
# Usage:
#   OLD_IMAGE=ghcr.io/marin-community/iris-controller:<tree-hash> \
#   CHECKPOINT_DIR=gs://marin-us-central2/iris/marin/state/controller-state/<epoch_ms> \
#     scripts/iris/rollback_controller.sh --execute
#
# All parameters are env vars with marin-production defaults:
set -euo pipefail

CLUSTER="${CLUSTER:-marin}"
VM="${VM:-iris-controller-marin}"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-hai-gcp-models}"
STATE_DIR="${STATE_DIR:-/var/cache/iris/controller}"
REMOTE_STATE_DIR="${REMOTE_STATE_DIR:-gs://marin-us-central2/iris/marin/state}"

# Required: the known-good image to roll back to, and the checkpoint that was
# taken BEFORE the bad deploy (i.e. at the old schema).
OLD_IMAGE="${OLD_IMAGE:?set OLD_IMAGE=ghcr.io/marin-community/iris-controller:<tree-hash>}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR=gs://.../controller-state/<epoch_ms>}"

# The bare image tag (tree hash) — used to confirm Phase A actually deployed the
# old image before Phase B swaps the DB (guards against starting NEW code on the
# OLD DB, which would just re-migrate).
OLD_IMAGE_TAG="${OLD_IMAGE##*:}"

EXECUTE=0
[[ "${1:-}" == "--execute" || "${EXECUTE:-0}" == "1" ]] && EXECUTE=1

cat <<PLAN
Iris controller rollback plan
  cluster:          ${CLUSTER}
  controller VM:    ${VM} (zone=${ZONE}, project=${PROJECT})
  roll back to:     ${OLD_IMAGE}
  restore checkpt:  ${CHECKPOINT_DIR}
  local state dir:  ${STATE_DIR}
  remote state dir: ${REMOTE_STATE_DIR}

Phase A: iris --cluster=${CLUSTER} cluster controller restart --image ${OLD_IMAGE} --skip-checkpoint
Phase B: on ${VM}: stop container -> mv ${STATE_DIR}/db aside -> restore ${CHECKPOINT_DIR} -> start -> health check
PLAN

if [[ "$EXECUTE" != "1" ]]; then
  echo
  echo "DRY RUN — re-run with --execute to perform the rollback."
  exit 0
fi

echo
echo ">>> Pre-check: controller must be reachable (in-place restart path)."
if ! uv run iris --cluster="${CLUSTER}" cluster status >/dev/null 2>&1; then
  echo "ABORT: controller is not reachable. Do NOT use this script for a wedged" >&2
  echo "controller — follow lib/iris/OPS.md 'Controller Checkpoint Rollback'." >&2
  exit 1
fi

echo
echo ">>> Phase A: redeploy ${OLD_IMAGE} in place (health failure on the migrated DB is expected)."
uv run iris --cluster="${CLUSTER}" cluster controller restart --image "${OLD_IMAGE}" --skip-checkpoint \
  || echo "    Phase A restart returned non-zero (expected: old code on post-migration DB). Continuing."

echo
echo ">>> Phase B: restore the pre-migration checkpoint on ${VM} and start the old image."
gcloud compute ssh "${VM}" --zone="${ZONE}" --project="${PROJECT}" --tunnel-through-iap --command="
  set -euo pipefail
  IMAGE=\$(sudo docker inspect --format='{{.Config.Image}}' iris-controller)
  echo \"container image is now: \$IMAGE\"
  case \"\$IMAGE\" in
    *:${OLD_IMAGE_TAG}) : ;;
    *) echo \"ABORT: container image is not the expected rollback tag ${OLD_IMAGE_TAG}; Phase A did not deploy it. Not touching the DB.\" >&2; exit 1 ;;
  esac
  sudo docker stop iris-controller || true
  ts=\$(date +%s)
  echo \"preserving post-migration DB -> ${STATE_DIR}/db.postmigration.bak.\$ts\"
  sudo mv '${STATE_DIR}/db' \"${STATE_DIR}/db.postmigration.bak.\$ts\"
  echo \"restoring ${CHECKPOINT_DIR} -> ${STATE_DIR}/db (using \$IMAGE)\"
  sudo docker run --rm --network=host -v /var/cache/iris:/var/cache/iris \"\$IMAGE\" \
    .venv/bin/python -c \"from pathlib import Path; from iris.cluster.controller.checkpoint import download_checkpoint_to_local as r; raise SystemExit(0 if r('${REMOTE_STATE_DIR}', Path('${STATE_DIR}/db'), checkpoint_dir='${CHECKPOINT_DIR}') else 1)\"
  if [ ! -f '${STATE_DIR}/db/controller.sqlite3' ]; then
    echo \"RESTORE FAILED — not starting. Recover with: sudo mv ${STATE_DIR}/db.postmigration.bak.\$ts ${STATE_DIR}/db\" >&2
    exit 1
  fi
  sudo docker start iris-controller
  sleep 3
  curl -sf http://localhost:10000/health && echo ' controller healthy on rolled-back image + pre-migration DB'
"

echo
echo ">>> Rollback complete. Verify:"
echo "    uv run iris --cluster=${CLUSTER} cluster status   # expect Running: True, workers re-registering"
echo "    (the post-migration DB is preserved on the VM as ${STATE_DIR}/db.postmigration.bak.<ts>)"
