#!/usr/bin/env bash
# Submit one packed GLM task-generation batch to Iris/CoreWeave.
#
# This is intentionally a submit-side program: it resolves the mutable relay URL
# immediately before submission, reads credentials locally, and stages only the
# inputs the worker needs.  Credentials are passed to Iris with `-e`; they are
# never copied into the stage, manifests, or logs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$HERE/.." && pwd)"
# A continuation can pin its package and contracts to a prior approved source
# archive while retaining this submit-side transport/runtime implementation.
# Normal submissions stage the current checkout.
ROOT="${CAPABILITY_SOURCE_ROOT:-$DEFAULT_ROOT}"
ROOT="$(cd "$ROOT" && pwd)"
BUILD_ENVS="${BUILD_ENVS:-$HOME/openathena/build_envs}"
MARIN="${MARIN:-$HOME/openathena/marin}"
COMMON="$BUILD_ENVS/common"

usage() {
  cat <<'EOF'
Usage:
  scripts/submit.sh --pilot data/pilot.json --out runs/NAME [options] [-- CLI_ARGS...]

Options:
  --stage STAGE          generate, propose, synthesize, admit, checkpoint-revalidation, runtime-probe, reset-probe, daytona-health-probe, adaptive-test-pilot, composite-probe, composite-consensus-probe, composite-final-state-probe, context-budget-probe, c32-semantic-probe, image-migration-revalidation, judge-probe, evaluate, or regrade; default: propose
  --pilot PATH           capability manifest JSON (required for propose/generate)
  --source PATH          accepted-proposal input for synthesis/evaluation
  --out PATH             relative durable run prefix, e.g. runs/pilot-001
  --concurrency N        concurrent inference calls, default: 256
  --dispatch-limit N     explicit temporary active-call cap (never defaults below --concurrency)
  --tier TIER            interactive (default) or bulk
  --run-name NAME        Iris job suffix, default derives from --out
  --s3-root URI          durable object-store root
  --dry-run              complete local preflight and print the job command only
  --sandbox PROVIDER     daytona (default) or silo; silo stages vendor/silo_tools/dt.py and passes
                         SILO_API_TOKEN / SILO_BROKER_RESOLVE_URL from the submitter's environment
  --resume               require and restore a verified prior output snapshot
  --adopt-proposal-checkpoint PATH  fully restored proposal checkpoint (generate only)
  --adoption-source-archive PATH    original immutable controller archive
  --adoption-launch-receipt PATH    original proposal launch receipt
  -h, --help             show this help

Arguments after `--` are forwarded unchanged to capability_pipeline.cli.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
note() { printf '[cap-submit %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

PHASE=propose
PILOT=""
SOURCE=""
OUT=""
CONCURRENCY=256
DISPATCH_LIMIT=""
GLM_TIER=interactive
RUN_NAME=""
S3_ROOT="${S3_ROOT:-s3://marin-us-east-02a/users/muchanem/capability-pipeline}"
DRY_RUN=0
SANDBOX_PROVIDER="${CAPABILITY_SANDBOX_PROVIDER:-daytona}"
RESUME=0
ADOPT_CHECKPOINT=""; ADOPTION_ARCHIVE=""; ADOPTION_RECEIPT=""
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) PHASE="$2"; shift 2 ;;
    --pilot) PILOT="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --dispatch-limit) DISPATCH_LIMIT="$2"; shift 2 ;;
    --tier) GLM_TIER="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --s3-root) S3_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --sandbox) SANDBOX_PROVIDER="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --adopt-proposal-checkpoint) ADOPT_CHECKPOINT="$2"; shift 2 ;;
    --adoption-source-archive) ADOPTION_ARCHIVE="$2"; shift 2 ;;
    --adoption-launch-receipt) ADOPTION_RECEIPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "unknown option: $1" ;;
  esac
done

RESET_PROBE_ENV=none
if [ "$PHASE" = reset-probe ]; then
  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    [ "${#EXTRA_ARGS[@]}" -eq 2 ] && [ "${EXTRA_ARGS[0]}" = --environment ] || die "reset-probe accepts only --environment none|shellsim|docker"
    RESET_PROBE_ENV="${EXTRA_ARGS[1]}"
  fi
  case "$RESET_PROBE_ENV" in none|shellsim|docker) ;; *) die "reset-probe environment must be none, shellsim, or docker";; esac
fi

case "$PHASE" in generate|propose|synthesize|admit|checkpoint-revalidation|runtime-probe|reset-probe|daytona-health-probe|adaptive-test-pilot|composite-probe|composite-consensus-probe|composite-final-state-probe|context-budget-probe|c32-semantic-probe|image-migration-revalidation|judge-probe|evaluate|regrade) ;; *) die "--stage must be a supported phase";; esac
case "$GLM_TIER" in interactive|bulk) ;; *) die "--tier must be interactive or bulk";; esac
case "$SANDBOX_PROVIDER" in daytona|silo) ;; *) die "--sandbox must be daytona or silo";; esac
case "$CONCURRENCY" in ''|*[!0-9]*) die "--concurrency must be a positive integer";; esac
[ "$CONCURRENCY" -gt 0 ] || die "--concurrency must be positive"
if [ "$PHASE" = regrade ]; then
  [ "$CONCURRENCY" -le 8 ] || die "regrade requires explicit --concurrency matching its frozen plan (maximum 8)"
fi
if [ "$PHASE" = reset-probe ]; then
  [ "$CONCURRENCY" -eq 1 ] || die "reset protocol probe requires --concurrency 1"
fi
if [ "$PHASE" = checkpoint-revalidation ]; then
  [ "$CONCURRENCY" -eq 1 ] || die "checkpoint revalidation requires --concurrency 1"
  [ "$RESUME" = 0 ] || die "checkpoint revalidation requires a fresh output prefix"
  [ "${#EXTRA_ARGS[@]}" -eq 0 ] || die "checkpoint revalidation accepts no forwarded CLI arguments"
fi
if [ -n "$DISPATCH_LIMIT" ]; then
  case "$DISPATCH_LIMIT" in ''|*[!0-9]*) die "--dispatch-limit must be a positive integer";; esac
  [ "$DISPATCH_LIMIT" -gt 0 ] || die "--dispatch-limit must be positive"
  [ "$DISPATCH_LIMIT" -le "$CONCURRENCY" ] || die "--dispatch-limit cannot exceed --concurrency"
fi
[ -n "$OUT" ] || die "--out is required"
case "$OUT" in /*|*'..'*|*'//'*) die "--out must be a clean relative path";; esac
[ -n "$RUN_NAME" ] || RUN_NAME="cap-${OUT##*/}"
case "$RUN_NAME" in *[!A-Za-z0-9._-]*|'') die "--run-name contains unsupported characters";; esac
[ -d "$MARIN" ] || die "Marin workspace not found: $MARIN"
[ -f "$COMMON/submit_lib.sh" ] || die "shared submission library not found: $COMMON/submit_lib.sh"
[ -x "$HERE/worker.sh" ] || die "worker is not executable: $HERE/worker.sh"
[ -d "$ROOT/capability_pipeline" ] || die "capability_pipeline package has not been created yet"

if [ "$PHASE" = propose ] || [ "$PHASE" = generate ]; then
  [ -n "$PILOT" ] || die "--pilot is required for this stage"
  [ -f "$PILOT" ] || die "pilot file not found: $PILOT"
  PILOT="$(cd "$(dirname "$PILOT")" && pwd)/$(basename "$PILOT")"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - "$PILOT" <<'PY'
import sys
from capability_pipeline.catalog import load_pilot
load_pilot(sys.argv[1])
PY
fi
if [ "$PHASE" = context-budget-probe ]; then
  [ "$CONCURRENCY" -eq 1 ] || die "context-budget probe requires --concurrency 1"
  [ -n "$PILOT" ] && [ -f "$PILOT" ] || die "context-budget probe requires --pilot frozen request"
  PILOT="$(cd "$(dirname "$PILOT")" && pwd)/$(basename "$PILOT")"
  python3 "$HERE/run_context_budget_probe.py" --validate-only --input "$PILOT" || die "context-budget probe input is invalid"
fi
[ "$PHASE" != generate ] || [ -z "$SOURCE" ] || die "generate uses its own durable stage outputs; use --resume instead of --source"
ADOPTION_COUNT=0
for adoption_value in "$ADOPT_CHECKPOINT" "$ADOPTION_ARCHIVE" "$ADOPTION_RECEIPT"; do [ -n "$adoption_value" ] && ADOPTION_COUNT=$((ADOPTION_COUNT + 1)); done
[ "$ADOPTION_COUNT" = 0 ] || [ "$ADOPTION_COUNT" = 3 ] || die "proposal adoption requires checkpoint, source archive and launch receipt together"
if [ "$ADOPTION_COUNT" = 3 ]; then
  [ "$PHASE" = generate ] || die "proposal adoption is supported only by generate"
  [ "$RESUME" = 0 ] || die "proposal adoption starts a fresh generate output and cannot use --resume"
  [ -d "$ADOPT_CHECKPOINT" ] && [ ! -L "$ADOPT_CHECKPOINT" ] || die "proposal adoption checkpoint is missing or linked"
  for adoption_file in "$ADOPTION_ARCHIVE" "$ADOPTION_RECEIPT"; do [ -f "$adoption_file" ] && [ ! -L "$adoption_file" ] || die "proposal adoption input is missing or linked"; done
  ADOPT_CHECKPOINT="$(cd "$ADOPT_CHECKPOINT" && pwd)"
  ADOPTION_ARCHIVE="$(cd "$(dirname "$ADOPTION_ARCHIVE")" && pwd)/$(basename "$ADOPTION_ARCHIVE")"
  ADOPTION_RECEIPT="$(cd "$(dirname "$ADOPTION_RECEIPT")" && pwd)/$(basename "$ADOPTION_RECEIPT")"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - "$ADOPT_CHECKPOINT" "$ADOPTION_ARCHIVE" "$ADOPTION_RECEIPT" "$PILOT" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.proposal_adoption import validate_checkpoint
validate_checkpoint(*(Path(value) for value in sys.argv[1:]))
PY
fi
if [ -n "$SOURCE" ]; then
  [ -e "$SOURCE" ] || die "source input not found: $SOURCE"
  SOURCE="$(cd "$(dirname "$SOURCE")" && pwd)/$(basename "$SOURCE")"
fi
[ "$PHASE" != synthesize ] && [ "$PHASE" != admit ] && [ "$PHASE" != checkpoint-revalidation ] && [ "$PHASE" != image-migration-revalidation ] && [ "$PHASE" != evaluate ] && [ "$PHASE" != regrade ] || [ -n "$SOURCE" ] || die "--source is required for this stage"
if [ "$PHASE" = checkpoint-revalidation ]; then
  [ -d "$SOURCE" ] && [ ! -L "$SOURCE" ] || die "checkpoint source must be a bundle directory"
  SOURCE_SHA="$(shasum -a 256 "$SOURCE/manifest.json" | awk '{print $1}')"
  PLAN_SHA="$(shasum -a 256 "$SOURCE/request.json" | awk '{print $1}')"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - "$SOURCE" "$SOURCE_SHA" "$PLAN_SHA" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.checkpoint_revalidation import validate_checkpoint_bundle
validate_checkpoint_bundle(Path(sys.argv[1]), expected_manifest_sha256=sys.argv[2], expected_request_sha256=sys.argv[3])
PY
fi

MIGRATION_CANDIDATE_IMAGE='envreg.208261-marin-gpu.coreweave.app/capability-env-gen/c32-geometry-topology-repair-candidate@sha256:dcc31498e37351639cd2eaa89910388fb2f8cd5ccc0e97d272cdbfa655b08d1b'
MIGRATION_VERIFIER_IMAGE='envreg.208261-marin-gpu.coreweave.app/capability-env-gen/c32-geometry-topology-repair-verifier@sha256:25cda6cf8b745cbc02614069ee165e9af4816f672ffae7b41ed3d640c707ce29'
MIGRATION_MANIFEST_SHA256='641ed6358c3d307246972f77557c15b2ae6dcb951987894046f85f4a1270837e'
MIGRATION_RECEIPT_SHA256='200e07dd7774e31ee11547651f10494235eddcce6b7cb3d6bae8a5e0f8973428'
if [ "$PHASE" = image-migration-revalidation ]; then
  [ -d "$SOURCE" ] || die "image migration source must be the reviewed bundle directory"
  [ "$(shasum -a 256 "$SOURCE/manifest.json" | awk '{print $1}')" = "$MIGRATION_MANIFEST_SHA256" ] || die "reviewed image migration bundle manifest mismatch"
  [ "$(shasum -a 256 "$SOURCE/migration-receipt.json" | awk '{print $1}')" = "$MIGRATION_RECEIPT_SHA256" ] || die "reviewed image migration receipt mismatch"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - \
    "$SOURCE" "$MIGRATION_CANDIDATE_IMAGE" "$MIGRATION_VERIFIER_IMAGE" "$MIGRATION_MANIFEST_SHA256" "$MIGRATION_RECEIPT_SHA256" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.image_pointer_migration import validate_pointer_migration_bundle

validate_pointer_migration_bundle(
    Path(sys.argv[1]),
    expected_images={"candidate": sys.argv[2], "verifier": sys.argv[3]},
    expected_manifest_sha256=sys.argv[4],
    expected_receipt_sha256=sys.argv[5],
)
PY
fi

# Reuse the proven relay resolution, literal route inspection, and tier-aware
# readiness gate.  It deliberately does not use /v1/models: that endpoint can be
# green while the relay has no glm-5.3 route.
export BUILD_ENVS MARIN GLM_TIER
# shellcheck source=/dev/null
. "$COMMON/submit_lib.sh"

IRIS_PRIORITY=interactive
[ "$GLM_TIER" = bulk ] && IRIS_PRIORITY=batch
export IRIS_PRIORITY
if [ "$PHASE" = regrade ] || [ "$PHASE" = reset-probe ]; then
  CW_ID="$(gcloud secrets versions access latest --secret=cw-object-storage-key-id --project=hai-gcp-models 2>/dev/null)"
  CW_SEC="$(gcloud secrets versions access latest --secret=cw-object-storage-key-secret --project=hai-gcp-models 2>/dev/null)"
  BASE_URL=""
  workers=0
else
  load_secrets
fi
[ -n "${CW_ID:-}" ] && [ -n "${CW_SEC:-}" ] || die "CoreWeave object-store credentials are required for durable output"

if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then
BASE_URL="$(resolve_endpoint)"
# CAPABILITY_GLM_BASE_URL pins a specific relay that is registered for glm-5.3
# but is not the first registry row -- e.g. a second ingress when the first one
# stops accepting connections (relay E, 2026-09-22).  It is still checked below:
# the URL must be a registered glm-5.3 row owned by CAPABILITY_RELAY_ROUTE_JOB,
# and that relay's route is verified exactly as for the default one.
if [ -n "${CAPABILITY_GLM_BASE_URL:-}" ]; then
  BASE_URL="$CAPABILITY_GLM_BASE_URL"
  note "relay pinned by CAPABILITY_GLM_BASE_URL: $BASE_URL"
fi
[ -n "$BASE_URL" ] || die "could not resolve the GLM relay endpoint"
EXPECTED_RELAY_JOB="${CAPABILITY_RELAY_ROUTE_JOB:-/muchanem/glm53-relay-e}"
# The registry check follows RELAY_JOB (submit_lib's resolution source), so a
# per-cluster relay is checked against its own registration.  A job can only
# reach a relay on its own cluster: cross-cluster traffic is blocked (measured
# 2026-09-22), so rno2a jobs need RELAY_JOB=/muchanem/glm53-relay-rno2a and
# CAPABILITY_RELAY_ROUTE_JOB naming that relay's job.
registry_row="$(cd "$MARIN" && RELAY_ENDPOINT="$RELAY_JOB/glm-5.3" WANT_URL="${CAPABILITY_GLM_BASE_URL:-}" uv run --frozen python3 - <<'PY'
import os

from iris.cli.connect import open_iris_client

with open_iris_client(cluster_name="marin", workspace=None) as client:
    endpoints = client.list_endpoint_instances(os.environ["RELAY_ENDPOINT"])
if not endpoints:
    raise SystemExit(f"{os.environ['RELAY_ENDPOINT']} has no active registry rows")
want = os.environ.get("WANT_URL", "")
if want:
    matches = [e for e in endpoints if e.address == want]
    if not matches:
        raise SystemExit(f"pinned relay {want} is not a registered row of {os.environ['RELAY_ENDPOINT']}")
    endpoint = matches[0]
else:
    endpoint = endpoints[0]
print(f"{endpoint.address}\t{endpoint.task_id}")
PY
)"
IFS=$'\t' read -r REGISTERED_URL REGISTERED_TASK <<< "$registry_row"
[ "$REGISTERED_URL" = "$BASE_URL" ] || die "resolved GLM relay differs from its authoritative registry row"
ROUTE_LOG_JOB="${REGISTERED_TASK%/*}"
[ "$ROUTE_LOG_JOB" = "$EXPECTED_RELAY_JOB" ] || die "resolved GLM relay is owned by $ROUTE_LOG_JOB, expected $EXPECTED_RELAY_JOB"
route_logs=""
# Finelog can time out while endpoint discovery and inference remain healthy.
# Retry only the read; never infer a route from transport failure or stale owners.
for route_read_attempt in 1 2 3; do
  if route_logs="$(cd "$MARIN" && uv run --frozen iris --cluster=marin job logs "$ROUTE_LOG_JOB" --max-lines 100 --substring 'routes:' 2>/dev/null)"; then
    break
  fi
  route_logs=""
  note "route log read failed for $ROUTE_LOG_JOB (attempt $route_read_attempt/3)"
done
# `|| true`: under `set -euo pipefail` an empty read makes grep exit 1, which
# killed this script right here with exit 1 -- before the route check below
# could print why.  An absent routes line must reach the checks, not abort.
route_table="$(printf '%s\n' "$route_logs" | { grep -o 'routes: [^)]*' || true; } | tail -1)"
route=unknown
if [ -z "$route_table" ]; then
  # The relay prints its `routes:` table once, at startup.  On a relay that has
  # been up for days, finding it means substring-scanning days of logs, which
  # overruns finelog's 10 s read deadline once the log server is busy (observed
  # 2026-09-22: every preflight failed while the relay served normally).  Its
  # 60 s status line carries a per-model routed counter, which exists only when
  # the model is actually being routed, so a recent window is both cheaper and
  # more current evidence.  Match the key by name; never infer from transport.
  if counters="$(cd "$MARIN" && uv run --frozen iris --cluster=marin job logs "$ROUTE_LOG_JOB" --max-lines 20 --since-seconds 300 --substring 'model_routed:glm-5.3' 2>/dev/null)" \
       && printf '%s\n' "$counters" | grep -Fq '"model_routed:glm-5.3":'; then
    route=ok
    note "route verified from live relay counters (startup routes: line unreadable)"
  fi
fi
if [ "$route" = unknown ] && [ -n "$route_table" ]; then
  # Entries are `model->upstream`, comma separated.  A relay with no
  # MODEL_ROUTES reports `routes: none [+implicit glm-5.3->glm-router-orion]`:
  # its own ENDPOINT_MODEL routes implicitly, and leaving MODEL_ROUTES unset is
  # deliberate (setting it replaces the fallback; that 404'd glm-5.3 on
  # 2026-09-17).  Normalize the implicit marker, then still match by NAME.
  if printf '%s\n' "$route_table" | sed 's/^routes: //; s/\[+implicit[[:space:]]*/,/g; s/\]//g' | tr ',' '\n' \
       | sed 's/^[[:space:]]*//; s/->.*$//' | grep -Fxq glm-5.3; then
    route=ok
  else
    route=no-route
  fi
fi
[ "$route" = ok ] || die "relay route preflight is $route for glm-5.3; do not submit"
if ! endpoint_serving; then
  note "inference pool is not ready for tier=$GLM_TIER"
  fleet_probe_diagnosis >&2 || true
  die "health preflight failed; wait for workers.$(fleet_pool_key) >= ${GLM_MIN_WORKERS:-2}"
fi
workers="$(fleet_workers)"
fi

# Stage under one fixed, narrow leaf in Marin rather than the caller's cwd. A
# mkdir lock holds it until Iris has captured its source bundle, preventing one
# submission from changing another submission's snapshot.
STAGING_ROOT="${STAGING_ROOT:-$MARIN/capability-pipeline-staging}"
# Each no-wait Iris submission owns an immutable stage leaf. A later submit
# must never overwrite a live worker's current directory.
STAGE_ID="${RUN_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
STAGE="$STAGING_ROOT/submissions/$STAGE_ID"
LOCK="${CAPABILITY_SUBMISSION_LOCK:-$STAGING_ROOT/.submit.lock}"
mkdir -p "$STAGING_ROOT/submissions"
if ! mkdir "$LOCK" 2>/dev/null; then
  die "submission staging is in use: $LOCK (wait for the current Iris bundle to finish)"
fi
release_lock() { rmdir "$LOCK" 2>/dev/null || true; }
trap release_lock EXIT
mkdir -p "$STAGE" "$STAGE/capability_pipeline" "$STAGE/inputs" "$STAGE/data" "$STAGE/docs/audits"

# Copy only named runtime inputs.  `rsync --delete` is confined to the package leaf
# so a removed module cannot survive from a previous submission.
rsync -a --delete "$ROOT/capability_pipeline/" "$STAGE/capability_pipeline/"
find "$STAGE/capability_pipeline" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGE/capability_pipeline" -type f -name '*.py[co]' -delete
cp "$HERE/worker.sh" "$STAGE/worker.sh"
cp "$HERE/restore_seed.py" "$STAGE/restore_seed.py"
cp "$HERE/continuation_seed_transport.py" "$STAGE/continuation_seed_transport.py"
cp "$HERE/proposal_adoption_transport.py" "$STAGE/proposal_adoption_transport.py"
cp "$HERE/restore_bundle.py" "$STAGE/restore_bundle.py"
cp "$HERE/sync_results.py" "$STAGE/sync_results.py"
cp "$HERE/sync_supervisor.py" "$STAGE/sync_supervisor.py"
cp "$COMMON/bootstrap_tools.sh" "$STAGE/bootstrap_tools.sh"
cp "$COMMON/omp_env.sh" "$STAGE/omp_env.sh"
[ -f "$ROOT/data/revocations.json" ] || die "missing mandatory revocation registry"
cp "$ROOT/data/revocations.json" "$STAGE/data/revocations.json"
[ -f "$ROOT/docs/revocations.md" ] && cp "$ROOT/docs/revocations.md" "$STAGE/docs/revocations.md"
[ -f "$ROOT/docs/audits/c30_redesign_001.md" ] && cp "$ROOT/docs/audits/c30_redesign_001.md" "$STAGE/docs/audits/c30_redesign_001.md"
[ -f "$ROOT/docs/audits/c17_construction_001.md" ] && cp "$ROOT/docs/audits/c17_construction_001.md" "$STAGE/docs/audits/c17_construction_001.md"
if [ -n "$PILOT" ]; then cp "$PILOT" "$STAGE/pilot.json"; fi
if [ -n "$SOURCE" ] && [ "$PHASE" != checkpoint-revalidation ]; then
  rm -rf "$STAGE/inputs/source"
  if [ -d "$SOURCE" ]; then
    rsync -a --delete "$SOURCE/" "$STAGE/inputs/source/"
  else
    mkdir -p "$STAGE/inputs/source"
    cp "$SOURCE" "$STAGE/inputs/source/accepted.json"
  fi
else
  # Fixed staging must not let a previous continuation's contract manifest
  # constrain a source-less probe submitted later.
  rm -rf "$STAGE/inputs/source"
fi
if [ "$PHASE" = synthesize ]; then
  PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - "$STAGE" "$STAGE/inputs/source/accepted.json" <<'PY'
import sys
from pathlib import Path
stage, accepted = map(Path, sys.argv[1:])
sys.path.insert(0, str(stage))
from capability_pipeline import synthesis
if not Path(synthesis.__file__).resolve().is_relative_to(stage.resolve()):
    raise SystemExit("staged synthesis controller was not imported")
synthesis.load_accepted(accepted)
PY
fi
if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = runtime-probe ] || [ "$PHASE" = reset-probe ] || [ "$PHASE" = daytona-health-probe ] || [ "$PHASE" = adaptive-test-pilot ] || [ "$PHASE" = composite-probe ] || [ "$PHASE" = composite-consensus-probe ] || [ "$PHASE" = composite-final-state-probe ] || [ "$PHASE" = context-budget-probe ] || [ "$PHASE" = c32-semantic-probe ] || [ "$PHASE" = image-migration-revalidation ] || [ "$PHASE" = judge-probe ]; then
  # These are inputs to the agent's immutable per-proposal contract.  Copy them
  # into the fixed stage; never let a prior submission's vendor/docs survive.
  [ -f "$ROOT/vendor/task_spec/source.lock.json" ] || die "missing vendor/task_spec/source.lock.json"
  [ -f "$ROOT/docs/task_contract.md" ] || die "missing docs/task_contract.md"
  rm -rf "$STAGE/vendor" "$STAGE/docs" "$STAGE/daytona-tools"
  mkdir -p "$STAGE/vendor/task_spec" "$STAGE/docs" "$STAGE/daytona-tools"
  # Copy the complete pinned vendor contract, including any fail-closed overlay
  # patch named by its lock.  Copying only the source lock made a valid base
  # archive indistinguishable from an unavailable toolchain at runtime.
  rsync -a --delete "$ROOT/vendor/task_spec/" "$STAGE/vendor/task_spec/"
  TASKCOMPENDIUM_REV="$(sed -n 's/^[[:space:]]*"revision"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' "$ROOT/vendor/task_spec/source.lock.json" | head -1)"
  [ "${#TASKCOMPENDIUM_REV}" = 40 ] || die "invalid TaskCompendium revision in source lock"
  git -C "$MARIN" cat-file -e "$TASKCOMPENDIUM_REV^{commit}" 2>/dev/null || die "pinned TaskCompendium revision is unavailable in Marin"
  # The worker verifies every lock-listed file and the complete source set.  It
  # therefore receives this exact local archive and never needs a network clone.
  rm -rf "$STAGE/taskcompendium-source" "$STAGE/taskcompendium-source.tar.gz"
  if [ -n "${TASKCOMPENDIUM_SOURCE_ARCHIVE:-}" ]; then
    [ -f "$TASKCOMPENDIUM_SOURCE_ARCHIVE" ] || die "TaskCompendium source archive is missing"
    cp "$TASKCOMPENDIUM_SOURCE_ARCHIVE" "$STAGE/taskcompendium-source.tar.gz"
  else
    # macOS bsdtar writes AppleDouble members.  Build the portable worker
    # payload with a clean git archive rather than repacking a Finder-touched
    # directory; the worker's exact-file-set verifier remains fail-closed.
    mkdir -p "$STAGE/taskcompendium-source"
    git -C "$MARIN" archive --format=tar "$TASKCOMPENDIUM_REV" lib/taskcompendium \
      | COPYFILE_DISABLE=1 tar -x -C "$STAGE/taskcompendium-source" --strip-components=2
    COPYFILE_DISABLE=1 tar -C "$STAGE/taskcompendium-source" --exclude='._*' --exclude='.DS_Store' -czf "$STAGE/taskcompendium-source.tar.gz" .
    rm -rf "$STAGE/taskcompendium-source"
  fi
  TASKCOMPENDIUM_ARCHIVE_SHA="$(shasum -a 256 "$STAGE/taskcompendium-source.tar.gz" | awk '{print $1}')"
  printf '{"sha256":"%s","revision":"%s"}\n' "$TASKCOMPENDIUM_ARCHIVE_SHA" "$TASKCOMPENDIUM_REV" > "$STAGE/taskcompendium-bootstrap.json"
  rm -rf "$STAGE/taskcompendium-source"
  cp "$ROOT/docs/task_contract.md" "$STAGE/docs/task_contract.md"
  # This branch recreates docs, so copy every controller input after cleanup.
  [ -f "$ROOT/docs/revocations.md" ] && cp "$ROOT/docs/revocations.md" "$STAGE/docs/revocations.md"
  if [ -d "$ROOT/docs/audits" ]; then
    mkdir -p "$STAGE/docs/audits"
    rsync -a --delete "$ROOT/docs/audits/" "$STAGE/docs/audits/"
  fi
  [ -f "$ROOT/docs/build_acceptance_001.md" ] && cp "$ROOT/docs/build_acceptance_001.md" "$STAGE/docs/build_acceptance_001.md"
  if [ -d "$ROOT/docs/build_acceptance" ]; then
    mkdir -p "$STAGE/docs/build_acceptance"
    rsync -a --delete "$ROOT/docs/build_acceptance/" "$STAGE/docs/build_acceptance/"
  fi
  # These named controller inputs are consulted by construction/repair even
  # though they are not per-item contracts.  They must survive the docs cleanup
  # above and match the launch manifest when a continuation declares them.
  for controller_doc in builder_measurements.md builder_images.md partial_controls.md quality_review.md construction_repair.md construction_continuation.md; do
    [ -f "$ROOT/docs/$controller_doc" ] && cp "$ROOT/docs/$controller_doc" "$STAGE/docs/$controller_doc"
  done
  mkdir -p "$STAGE/scripts"
  cp "$HERE/build_runtime_probe.py" "$STAGE/scripts/build_runtime_probe.py"
  [ -f "$HERE/run_shellsim_snapshot_fixture.py" ] && cp "$HERE/run_shellsim_snapshot_fixture.py" "$STAGE/scripts/run_shellsim_snapshot_fixture.py"
  [ -f "$HERE/run_reset_protocol_probe.py" ] && cp "$HERE/run_reset_protocol_probe.py" "$STAGE/scripts/run_reset_protocol_probe.py"
  [ -f "$HERE/run_reset_conformance.py" ] && cp "$HERE/run_reset_conformance.py" "$STAGE/scripts/run_reset_conformance.py"
  [ -f "$HERE/probe_daytona_health.py" ] && cp "$HERE/probe_daytona_health.py" "$STAGE/scripts/probe_daytona_health.py"
  [ -f "$HERE/run_adaptive_test_pilot.py" ] && cp "$HERE/run_adaptive_test_pilot.py" "$STAGE/scripts/run_adaptive_test_pilot.py"
  [ -f "$HERE/build_judge_probe.py" ] && cp "$HERE/build_judge_probe.py" "$STAGE/scripts/build_judge_probe.py"
  [ -f "$HERE/run_composite_probe.py" ] && cp "$HERE/run_composite_probe.py" "$STAGE/scripts/run_composite_probe.py"
  [ -f "$HERE/run_composite_consensus_probe.py" ] && cp "$HERE/run_composite_consensus_probe.py" "$STAGE/scripts/run_composite_consensus_probe.py"
  [ -f "$HERE/run_composite_final_state_probe.py" ] && cp "$HERE/run_composite_final_state_probe.py" "$STAGE/scripts/run_composite_final_state_probe.py"
  [ -f "$HERE/run_context_budget_probe.py" ] && cp "$HERE/run_context_budget_probe.py" "$STAGE/scripts/run_context_budget_probe.py"
  [ -f "$HERE/run_c32_semantic_probe.py" ] && cp "$HERE/run_c32_semantic_probe.py" "$STAGE/scripts/run_c32_semantic_probe.py"
  [ -f "$HERE/c32_balanced_mutation.py" ] && cp "$HERE/c32_balanced_mutation.py" "$STAGE/scripts/c32_balanced_mutation.py"
  [ -f "$HERE/run_image_migration_revalidation.py" ] && cp "$HERE/run_image_migration_revalidation.py" "$STAGE/scripts/run_image_migration_revalidation.py"
  [ -f "$HERE/preflight_image_migration_cache.py" ] && cp "$HERE/preflight_image_migration_cache.py" "$STAGE/scripts/preflight_image_migration_cache.py"
  if [ "$PHASE" = c32-semantic-probe ]; then
    if [ -n "${C32_PROBE_WORKSPACE:-}" ]; then
      [ -f "${C32_PROBE_CONTRACT:-}" ] || die "custom c32 probe workspace requires a hash-bound C32_PROBE_CONTRACT"
      C32_WORKSPACE="$C32_PROBE_WORKSPACE"
    else
      C32_WORKSPACE="$ROOT/runs/synthesis-pilot-001/c32-full-pull-current-9902c78f/items/c32.geometry_topology_repair-7-75adb802f475/workspace"
    fi
    for required in task/specification.json task/renderings.json eval/evaluate.py eval/run_eval.py sandbox_runs/reference_python_cleaned.gpkg private/ground_truth.json fixtures/stormwater_conduits.gpkg; do
      [ -f "$C32_WORKSPACE/$required" ] || die "c32 semantic probe input missing: $required"
    done
    rm -rf "$STAGE/c32-probe-inputs"
    mkdir -p "$STAGE/c32-probe-inputs"
    if [ -n "${C32_PROBE_WORKSPACE:-}" ]; then
      cp "$C32_PROBE_CONTRACT" "$STAGE/c32-probe-inputs/probe-contract.json"
    fi
    cp "$C32_WORKSPACE/task/specification.json" "$STAGE/c32-probe-inputs/specification.json"
    cp "$C32_WORKSPACE/task/renderings.json" "$STAGE/c32-probe-inputs/renderings.json"
    cp "$C32_WORKSPACE/eval/evaluate.py" "$STAGE/c32-probe-inputs/evaluate.py"
    cp "$C32_WORKSPACE/eval/run_eval.py" "$STAGE/c32-probe-inputs/run_eval.py"
    cp "$C32_WORKSPACE/sandbox_runs/reference_python_cleaned.gpkg" "$STAGE/c32-probe-inputs/reference_cleaned.gpkg"
    cp "$C32_WORKSPACE/private/ground_truth.json" "$STAGE/c32-probe-inputs/ground_truth.json"
    cp "$C32_WORKSPACE/fixtures/stormwater_conduits.gpkg" "$STAGE/c32-probe-inputs/stormwater_conduits.gpkg"
  fi
  [ -f "$HERE/run_composite_runner_smoke.py" ] && cp "$HERE/run_composite_runner_smoke.py" "$STAGE/scripts/run_composite_runner_smoke.py"
  if [ "$PHASE" = checkpoint-revalidation ]; then
    cp "$HERE/run_checkpoint_revalidation.py" "$STAGE/scripts/run_checkpoint_revalidation.py"
    cp "$HERE/checkpoint_revalidation_transport.py" "$STAGE/checkpoint_revalidation_transport.py"
  fi
  DT_SRC="$BUILD_ENVS/envgen/dt.py"
  # silo ships a drop-in dt.py (same CLI, JSON and dt_calls.jsonl; client() backed
  # by silo; no snapshot-quota wait).  Every pipeline sandbox client, including the
  # Python runtime and verifier paths, is obtained from this file's client().
  [ "$SANDBOX_PROVIDER" = silo ] && DT_SRC="$ROOT/vendor/silo_tools/dt.py"
  [ -f "$DT_SRC" ] || die "missing sandbox dt.py for provider $SANDBOX_PROVIDER: $DT_SRC"
  for helper in dt.sh validate_env.py verify.py adapter.py; do
    [ -f "$BUILD_ENVS/envgen/$helper" ] || die "missing Daytona helper: $BUILD_ENVS/envgen/$helper"
    cp "$BUILD_ENVS/envgen/$helper" "$STAGE/daytona-tools/$helper"
  done
  cp "$DT_SRC" "$STAGE/daytona-tools/dt.py"
  if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = image-migration-revalidation ]; then
    # Trusted image capture stays outside the per-builder tools directory.
    # dtx imports dt.py from its parent; preserve and hash that staged layout.
    mkdir -p "$STAGE/daytona-tools/capture-tools"
    for helper in capture_rootfs.py in_sandbox_capture.py dtx.py cw_presign.py; do
      [ -f "$BUILD_ENVS/envgen/probes/$helper" ] || die "missing image capture helper: $helper"
      cp "$BUILD_ENVS/envgen/probes/$helper" "$STAGE/daytona-tools/capture-tools/$helper"
    done
    cp "$ROOT/scripts/capture_task_images.py" "$STAGE/scripts/capture_task_images.py"
    for helper in review_generic_task_image.py capture_generic_task_image.py publish_generic_task_image.py probe_generic_task_image.py migrate_generic_task_images.py image_publication_handoff.py; do
      cp "$ROOT/scripts/$helper" "$STAGE/scripts/$helper"
    done
  fi
  chmod +x "$STAGE/daytona-tools/dt.sh"
  # Runtime probes import only dt.py.  Also place it at the bundle root: Iris
  # reliably carries command-adjacent staged files across federation.
  cp "$DT_SRC" "$STAGE/dt.py"
  cat > "$STAGE/omp-research-overlay.yml" <<'YML'
# Credentials remain in process environment.  This only selects the research
# provider; browser automation is intentionally unavailable in packed workers.
providers:
  webSearchOrder: ["parallel"]
  webSearchTimeoutSeconds: 120
  fetch: parallel
browser:
  enabled: false
YML
else
  rm -rf "$STAGE/vendor" "$STAGE/docs" "$STAGE/daytona-tools" "$STAGE/omp-research-overlay.yml" "$STAGE/dt.py" "$STAGE/scripts" "$STAGE/c32-probe-inputs"
fi

# Controllers import the staged transport validator as scripts.restore_bundle.
# Marin also has a regular scripts package, so make this a regular package in
# every stage rather than leaving it as a lower-priority namespace package.
mkdir -p "$STAGE/scripts"
cp "$HERE/restore_bundle.py" "$STAGE/scripts/restore_bundle.py"
: > "$STAGE/scripts/__init__.py"
if [ "$PHASE" = reset-probe ]; then
  PYTHONSAFEPATH=1 PYTHONPATH="$STAGE" uv run --project "$MARIN" --frozen python3 - <<'PY'
from capability_pipeline.non_docker_reset import run_frozen_non_docker_reset
from scripts import run_reset_conformance
assert callable(run_frozen_non_docker_reset) and callable(run_reset_conformance.validate_plan_payload)
PY
fi

# Evaluation deliberately stages only the frozen input transport and the pinned
# TaskCompendium source. It needs neither OMP, Cargo, browser tooling, nor a
# controller-generated accepted.json.
if [ "$PHASE" = evaluate ] || [ "$PHASE" = regrade ]; then
  [ -d "$SOURCE" ] || die "evaluation/regrade source must be a self-contained bundle directory"
  [ -f "$ROOT/vendor/task_spec/source.lock.json" ] || die "missing TaskCompendium source lock"
  rm -rf "$STAGE/vendor" "$STAGE/taskcompendium-source" "$STAGE/taskcompendium-source.tar.gz"
  mkdir -p "$STAGE/vendor/task_spec"
  rsync -a --delete "$ROOT/vendor/task_spec/" "$STAGE/vendor/task_spec/"
  TASKCOMPENDIUM_REV="$(sed -n 's/^[[:space:]]*"revision"[[:space:]]*:[[:space:]]*"\([0-9a-f]*\)".*/\1/p' "$ROOT/vendor/task_spec/source.lock.json" | head -1)"
  [ "${#TASKCOMPENDIUM_REV}" = 40 ] || die "invalid TaskCompendium revision in source lock"
  git -C "$MARIN" cat-file -e "$TASKCOMPENDIUM_REV^{commit}" 2>/dev/null || die "pinned TaskCompendium revision is unavailable in Marin"
  mkdir -p "$STAGE/taskcompendium-source"
  git -C "$MARIN" archive --format=tar "$TASKCOMPENDIUM_REV" lib/taskcompendium | COPYFILE_DISABLE=1 tar -x -C "$STAGE/taskcompendium-source" --strip-components=2
  COPYFILE_DISABLE=1 tar -C "$STAGE/taskcompendium-source" --exclude='._*' --exclude='.DS_Store' -czf "$STAGE/taskcompendium-source.tar.gz" .
  rm -rf "$STAGE/taskcompendium-source"
  TASKCOMPENDIUM_ARCHIVE_SHA="$(shasum -a 256 "$STAGE/taskcompendium-source.tar.gz" | awk '{print $1}')"
  printf '{"sha256":"%s","revision":"%s"}\n' "$TASKCOMPENDIUM_ARCHIVE_SHA" "$TASKCOMPENDIUM_REV" > "$STAGE/taskcompendium-bootstrap.json"
  PLAN_SHA="$(shasum -a 256 "$SOURCE/plan.json" | awk '{print $1}')"
  BUNDLE_MANIFEST_SHA="$(shasum -a 256 "$SOURCE/manifest.json" | awk '{print $1}')"
  # Fail before Iris submission if staged bytes or controller identity no longer
  # match the frozen plan. This is metadata validation only.
  if [ "$PHASE" = evaluate ]; then
    PYTHONSAFEPATH=1 PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - \
      "$STAGE" "$STAGE/inputs/source/plan.json" "$PLAN_SHA" <<'PY'
import sys
from pathlib import Path
stage, plan, fingerprint = map(Path, sys.argv[1:4])
sys.path.insert(0, str(stage))
from capability_pipeline import evaluation
if not Path(evaluation.__file__).resolve().is_relative_to(stage.resolve()):
    raise SystemExit("staged evaluation controller was not imported")
evaluation.validate_plan(plan, fingerprint.name)
PY
    TRANSPORT_KIND=evaluation
  else
    PYTHONSAFEPATH=1 PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 - \
      "$STAGE" "$STAGE/inputs/source" "$STAGE/taskcompendium-source.tar.gz" "$PLAN_SHA" <<'PY'
import sys
import tarfile
import tempfile
from pathlib import Path
stage, source, archive = map(Path, sys.argv[1:4])
sys.path.insert(0, str(stage))
from capability_pipeline import regrade
if not Path(regrade.__file__).resolve().is_relative_to(stage.resolve()):
    raise SystemExit("staged regrade controller was not imported")
with tempfile.TemporaryDirectory() as temporary:
    with tarfile.open(archive, "r:gz") as member_archive:
        member_archive.extractall(temporary, filter="data")
    regrade.validate_plan_bundle(source, Path(temporary), sys.argv[4])
PY
    TRANSPORT_KIND=regrade
  fi
  uv run --project "$MARIN" --frozen python3 "$STAGE/restore_bundle.py" "pack-$TRANSPORT_KIND" \
    --source "$STAGE/inputs/source" --archive "$STAGE/inputs/$TRANSPORT_KIND-bundle.tar.gz" \
    --transport "$STAGE/inputs/$TRANSPORT_KIND.transport.json" --expected-manifest-sha256 "$BUNDLE_MANIFEST_SHA" --plan-sha256 "$PLAN_SHA"
  uv run --project "$MARIN" --frozen python3 "$STAGE/restore_bundle.py" "restore-$TRANSPORT_KIND" \
    --archive "$STAGE/inputs/$TRANSPORT_KIND-bundle.tar.gz" --transport "$STAGE/inputs/$TRANSPORT_KIND.transport.json" \
    --destination "$STAGE/inputs/$TRANSPORT_KIND-transport-check" --expected-manifest-sha256 "$BUNDLE_MANIFEST_SHA" --plan-sha256 "$PLAN_SHA"
  rm -rf "$STAGE/inputs/source" "$STAGE/inputs/$TRANSPORT_KIND-transport-check"
fi

# A reproducibility archive alone is insufficient if the worker stage omitted a
# declared continuation contract.  Check the exact staged bytes before source
# snapshotting or Iris bundling.
if [ -f "$STAGE/inputs/source/manifest.json" ]; then
  uv run --project "$MARIN" --frozen python3 - "$STAGE/inputs/source/manifest.json" "$STAGE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, stage = map(Path, sys.argv[1:])
document = json.loads(manifest_path.read_text())
required = document.get("required_contract_inputs", {})
if not isinstance(required, dict):
    raise SystemExit("source manifest required_contract_inputs must be a mapping")
for relative, expected in sorted(required.items()):
    path = stage / relative
    if not path.is_file():
        raise SystemExit(f"staged required contract input is missing: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"staged required contract input fingerprint mismatch: {relative}")
PY
fi
if [ -f "$STAGE/inputs/source/manifest.json" ] \
  && [ -d "$STAGE/inputs/source/restore-seed" ] \
  && uv run --project "$MARIN" --frozen python3 - "$STAGE/inputs/source/manifest.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1]))
raise SystemExit(0 if isinstance(document.get("seed_members"), dict) else 1)
PY
then
  uv run --project "$MARIN" --frozen python3 "$STAGE/restore_seed.py" --check-only \
    "$STAGE/inputs/source/manifest.json" "$STAGE/inputs/source/restore-seed"
  # Iris has demonstrated that it can drop deep continuation members in transit.
  # The single opaque archive is hash-bound by the same manifest and becomes the
  # only worker transport; retain full seed bytes in the durable source snapshot.
  if [ -f "$STAGE/inputs/source/restore-seed.tar.gz" ]; then
    rm -rf "$STAGE/inputs/source/restore-seed"
    uv run --project "$MARIN" --frozen python3 "$STAGE/restore_seed.py" --check-only \
      "$STAGE/inputs/source/manifest.json" "$STAGE/inputs/source/restore-seed"
    # Iris limits the complete source bundle to 25 MB.  Continuation seed bytes
    # remain immutable under their content digest outside this run's result
    # prefix, so new-run restore cannot mistake them for prior result state.
    # The staged receipt is small; the worker downloads the exact archive and
    # reuses restore_seed.py's manifest/member validation before copying files.
    (
      cd "$MARIN"
      CW_KEY_ID="$CW_ID" CW_KEY_SECRET="$CW_SEC" uv run --project "$MARIN" --frozen python3 \
        "$STAGE/continuation_seed_transport.py" upload \
        --archive "$STAGE/inputs/source/restore-seed.tar.gz" --root "$S3_ROOT" \
        --source-manifest-sha256 "$(shasum -a 256 "$STAGE/inputs/source/manifest.json" | awk '{print $1}')" \
        --receipt "$STAGE/inputs/source/restore-seed.transport.json"
    )
    rm -f "$STAGE/inputs/source/restore-seed.tar.gz"
  fi
fi
if [ "$PHASE" = image-migration-revalidation ]; then
  uv run --project "$MARIN" --frozen python3 "$STAGE/restore_bundle.py" pack \
    --source "$STAGE/inputs/source" \
    --archive "$STAGE/inputs/source-bundle.tar.gz" \
    --transport "$STAGE/inputs/source.transport.json" \
    --expected-manifest-sha256 "$MIGRATION_MANIFEST_SHA256" \
    --expected-receipt-sha256 "$MIGRATION_RECEIPT_SHA256"
  # Prove the exact archive can restore before removing the federation-sensitive
  # tree. The worker repeats this proof from the transported opaque bytes.
  uv run --project "$MARIN" --frozen python3 "$STAGE/restore_bundle.py" restore \
    --archive "$STAGE/inputs/source-bundle.tar.gz" \
    --transport "$STAGE/inputs/source.transport.json" \
    --destination "$STAGE/inputs/source-transport-check" \
    --expected-manifest-sha256 "$MIGRATION_MANIFEST_SHA256" \
    --expected-receipt-sha256 "$MIGRATION_RECEIPT_SHA256"
  rm -rf "$STAGE/inputs/source" "$STAGE/inputs/source-transport-check"
fi
if [ "$ADOPTION_COUNT" = 3 ]; then
  ADOPTION_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/proposal-adoption.XXXXXX")"
  cleanup_adoption_temp() { rm -rf "$ADOPTION_TEMP"; }
  trap cleanup_adoption_temp EXIT
  uv run --project "$MARIN" --frozen python3 "$HERE/proposal_adoption_transport.py" pack \
    --checkpoint "$ADOPT_CHECKPOINT" --source-archive "$ADOPTION_ARCHIVE" --launch-receipt "$ADOPTION_RECEIPT" \
    --archive "$ADOPTION_TEMP/adoption.tar.gz" --transport "$ADOPTION_TEMP/transport.json"
  (
    cd "$MARIN"
    CW_KEY_ID="$CW_ID" CW_KEY_SECRET="$CW_SEC" uv run --project "$MARIN" --frozen python3 "$HERE/proposal_adoption_transport.py" upload \
      --archive "$ADOPTION_TEMP/adoption.tar.gz" --transport "$ADOPTION_TEMP/transport.json" --root "$S3_ROOT" \
      --receipt "$STAGE/inputs/proposal-adoption.transport.json"
  )
  rm -rf "$ADOPTION_TEMP"
  trap - EXIT
  trap release_lock EXIT
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  printf '%s\0' "${EXTRA_ARGS[@]}" > "$STAGE/extra-args.nul"
else
  : > "$STAGE/extra-args.nul"
fi

# Source archives deliberately normalize member mtimes to zero for reproducible
# content.  Iris builds a ZIP, whose format rejects dates before 1980; normalize
# only the fixed staging copy after all content hashing/copy decisions.  This
# does not alter staged bytes or any source/object digest.
find "$STAGE" -type f -exec touch -t 198001010000 {} +

# The template is not used by plain proposal generation, but putting it beside the
# worker makes the same bounded stage ready for future OMP-based synthesis.
# OMP_MODELS_SOURCE swaps in a variant models.yml (e.g. a routing-key experiment arm).
LOCAL_MODELS="${OMP_MODELS_SOURCE:-$HOME/.omp/agent/models.yml}"
if [ -f "$LOCAL_MODELS" ]; then
  sed -E -e 's|^([[:space:]]*)baseUrl:.*|\1baseUrl: __BASE_URL__/v1|' \
         -e 's|^([[:space:]]*)apiKey:.*|\1apiKey: "!cat __TOKEN_PATH__"|' \
      "$LOCAL_MODELS" > "$STAGE/models.template.yml"
fi

PILOT_SHA=""
[ -n "$PILOT" ] && PILOT_SHA="$(shasum -a 256 "$PILOT" | awk '{print $1}')"
if [ "$PHASE" != checkpoint-revalidation ]; then SOURCE_SHA=""; fi
[ -n "$SOURCE" ] && [ -f "$SOURCE" ] && SOURCE_SHA="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
[ "$PHASE" != image-migration-revalidation ] || SOURCE_SHA="$MIGRATION_MANIFEST_SHA256"
[ "$PHASE" != evaluate ] && [ "$PHASE" != regrade ] || SOURCE_SHA="$BUNDLE_MANIFEST_SHA"
DAYTONA=""
PARALLEL_KEY=""
IRIS_EXTRA_ENV=()
if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = runtime-probe ] || [ "$PHASE" = daytona-health-probe ] || [ "$PHASE" = adaptive-test-pilot ] || [ "$PHASE" = composite-probe ] || [ "$PHASE" = composite-consensus-probe ] || [ "$PHASE" = composite-final-state-probe ] || [ "$PHASE" = c32-semantic-probe ] || [ "$PHASE" = image-migration-revalidation ] || [ "$PHASE" = evaluate ] || [ "$PHASE" = regrade ]; then
  if [ "$SANDBOX_PROVIDER" = silo ]; then
    [ -n "${SILO_API_TOKEN:-}" ] || die "--sandbox silo needs SILO_API_TOKEN (uv run python -m silo.launch worker-env)"
    [ -n "${SILO_BROKER_RESOLVE_URL:-}" ] || die "--sandbox silo needs SILO_BROKER_RESOLVE_URL (uv run python -m silo.launch worker-env)"
    SANDBOX_ENV=(-e CAPABILITY_SANDBOX_PROVIDER silo -e SILO_API_TOKEN "$SILO_API_TOKEN" -e SILO_BROKER_RESOLVE_URL "$SILO_BROKER_RESOLVE_URL")
  else
    DAYTONA="${DAYTONA_RL_API_KEY:-}"
    [ -n "$DAYTONA" ] || DAYTONA="$(gcloud secrets versions access latest --secret=DAYTONA_RL_API_KEY --project=hai-gcp-models 2>/dev/null)"
    [ -n "$DAYTONA" ] || die "no DAYTONA_RL_API_KEY available for synthesis"
    SANDBOX_ENV=(-e CAPABILITY_SANDBOX_PROVIDER daytona -e DAYTONA_API_KEY "$DAYTONA")
  fi
  if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = image-migration-revalidation ]; then
  PARALLEL_KEY="$(sed -n 's/^PARALLEL_KEY=//p' "$BUILD_ENVS/.parallel_key" 2>/dev/null | head -1 | tr -d '\r\n')"
  [ -n "$PARALLEL_KEY" ] || die "no PARALLEL_KEY in $BUILD_ENVS/.parallel_key for synthesis research"
  IRIS_EXTRA_ENV=("${SANDBOX_ENV[@]}" -e PARALLEL_API_KEY "$PARALLEL_KEY" -e GH_TOKEN "$GH" -e GITHUB_TOKEN "$GH")
  else
  IRIS_EXTRA_ENV=("${SANDBOX_ENV[@]}")
  fi
fi
if [ "$PHASE" = reset-probe ] && [ "$RESET_PROBE_ENV" = docker ]; then
  DAYTONA="${DAYTONA_RL_API_KEY:-}"
  [ -n "$DAYTONA" ] || DAYTONA="$(gcloud secrets versions access latest --secret=DAYTONA_RL_API_KEY --project=hai-gcp-models 2>/dev/null)"
  [ -n "$DAYTONA" ] || die "Docker reset probe requires Daytona credentials"
  IRIS_EXTRA_ENV=(-e DAYTONA_API_KEY "$DAYTONA")
fi
if [ -n "${CAPABILITY_MAX_REPAIR_ROUNDS:-}" ]; then
  case "$CAPABILITY_MAX_REPAIR_ROUNDS" in ''|*[!0-9]*) die "CAPABILITY_MAX_REPAIR_ROUNDS must be an integer" ;; esac
  [ "$CAPABILITY_MAX_REPAIR_ROUNDS" -le 5 ] || die "CAPABILITY_MAX_REPAIR_ROUNDS cannot exceed 5"
fi
STAGE_REL="${STAGE#"$MARIN"/}"
DEST="$S3_ROOT/$OUT"
if [ "${CAPABILITY_REQUIRE_EMPTY_DESTINATION:-0}" = 1 ]; then
  (
    cd "$MARIN"
    CW_KEY_ID="$CW_ID" CW_KEY_SECRET="$CW_SEC" uv run --frozen "$HERE/check_fleet_destination.py" --destination "$DEST"
  ) || die "fleet destination is not confirmed fresh; no job was submitted"
fi
if [ "$PHASE" = regrade ] || [ "$PHASE" = reset-probe ]; then
  note "preflight passed: protocol probe or fixed-response regrade, no GLM route or token required"
else
  note "preflight passed: tier=$GLM_TIER workers=$workers route=glm-5.3"
fi
note "job=$RUN_NAME destination=$DEST concurrency=$CONCURRENCY stage=$PHASE"

JOB_CMD="cd $(printf '%q' "$STAGE_REL") && bash worker.sh --stage $(printf '%q' "$PHASE") --out $(printf '%q' "$OUT") --concurrency $(printf '%q' "$CONCURRENCY") --tier $(printf '%q' "$GLM_TIER") --s3-dest $(printf '%q' "$DEST") --run-name $(printf '%q' "$RUN_NAME")"
[ -n "$DISPATCH_LIMIT" ] && JOB_CMD+=" --dispatch-limit $(printf '%q' "$DISPATCH_LIMIT")"
[ -n "$PILOT_SHA" ] && JOB_CMD+=" --pilot pilot.json --pilot-sha256 $(printf '%q' "$PILOT_SHA")"
[ "$RESUME" = 1 ] && JOB_CMD+=" --resume"
[ "$ADOPTION_COUNT" = 0 ] || JOB_CMD+=" --adoption-transport inputs/proposal-adoption.transport.json"
[ -n "$SOURCE" ] && JOB_CMD+=" --source inputs/source"
[ -n "$SOURCE_SHA" ] && JOB_CMD+=" --source-sha256 $(printf '%q' "$SOURCE_SHA")"
[ "$PHASE" != evaluate ] && [ "$PHASE" != regrade ] && [ "$PHASE" != checkpoint-revalidation ] || JOB_CMD+=" --plan-sha256 $(printf '%q' "$PLAN_SHA")"

if [ "$DRY_RUN" = 1 ]; then
  note "dry run: Iris job not submitted"
  printf '%s\n' "$JOB_CMD"
  exit 0
fi

if [ "$PHASE" = checkpoint-revalidation ]; then
  CHECKPOINT_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/checkpoint-transport.XXXXXX")"
  cleanup_checkpoint_temp() { rm -rf "$CHECKPOINT_TEMP"; }
  trap cleanup_checkpoint_temp EXIT
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$MARIN" --frozen python3 "$HERE/checkpoint_revalidation_transport.py" pack \
    --source "$SOURCE" --archive "$CHECKPOINT_TEMP/bundle.tar.gz" --transport "$STAGE/inputs/checkpoint.transport.json" \
    --manifest-sha256 "$SOURCE_SHA" --request-sha256 "$PLAN_SHA"
  (cd "$MARIN" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CW_KEY_ID="$CW_ID" CW_KEY_SECRET="$CW_SEC" uv run --project "$MARIN" --frozen python3 "$HERE/checkpoint_revalidation_transport.py" upload \
    --archive "$CHECKPOINT_TEMP/bundle.tar.gz" --transport "$STAGE/inputs/checkpoint.transport.json" \
    --root "$S3_ROOT" --receipt "$STAGE/inputs/checkpoint.blob.json")
  rm -rf "$CHECKPOINT_TEMP"
  trap - EXIT
  trap release_lock EXIT
  touch -t 198001010000 "$STAGE/inputs/checkpoint.transport.json" "$STAGE/inputs/checkpoint.blob.json"
fi

# Persist an immutable, credential-free source revision before launch.  This is
# deliberately separate from the fixed Iris stage: the archive can reproduce
# prompts and package logic without exposing template credentials or relying on
# the caller being a git checkout.
(
  cd "$MARIN"
  # Bind the reproducibility snapshot to the exact launch handoff rather than
  # merely the controller checkout.  Proposal seeds and continuation accepted
  # lists are deliberately named inputs; arbitrary neighbouring files remain
  # excluded by archive_source.py.
  SNAPSHOT_ARGS=(--root "$ROOT" --destination "$DEST")
  if [ -n "$SOURCE" ] && [ "$PHASE" != checkpoint-revalidation ]; then
    SNAPSHOT_ARGS+=(--input "$SOURCE")
    if uv run --project "$MARIN" --frozen python3 - "$SOURCE/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raise SystemExit(
    0
    if path.is_file()
    and json.loads(path.read_text()).get("schema_version")
    in {
        "capability-portable-runtime-bundle-v1",
        "capability-portable-runtime-bundle-v2",
        "capability-runtime-evaluation-bundle-v1",
        "capability-fixed-submission-regrade-bundle-v1",
    }
    else 1
)
PY
    then
      SNAPSHOT_ARGS+=(--complete-input)
    fi
  fi
  [ -n "$PILOT" ] && SNAPSHOT_ARGS+=(--pilot "$PILOT")
  CW_KEY_ID="$CW_ID" CW_KEY_SECRET="$CW_SEC" uv run --frozen "$HERE/archive_source.py" \
    "${SNAPSHOT_ARGS[@]}"
)

(
  cd "$MARIN"
  # Only the immutable leaf for this submission belongs in the Iris bundle.
  # Earlier leaves are retained locally for provenance but excluded from transport.
  BUNDLE_EXCLUDE="^capability-pipeline-staging/submissions/(?!${STAGE_ID}/)"
  # Git may ignore all submission leaves to keep unrelated jobs small. Include
  # only this exact immutable leaf through Iris's supported bundle API.
  BUNDLE_PREFLIGHT=(--workspace "$MARIN" --stage "$STAGE" --exclude "^capability-pipeline-staging/current/" --exclude "$BUNDLE_EXCLUDE")
  IRIS_CMD=(uv run --frozen iris --cluster=marin job run --include "$STAGE_REL/**/*" --exclude "^capability-pipeline-staging/current/" --exclude "$BUNDLE_EXCLUDE" \
    --priority "$IRIS_PRIORITY" --target-cluster "${TARGET_CLUSTER:-cw-us-east-02a}" \
    --job-name "$RUN_NAME" --no-wait --cpu "${CPU:-4}" --memory "${MEM:-16g}" \
    --max-retries "${MAX_RETRIES:-3}" \
    --disk "${DISK:-200GB}" \
    --enable-extra-resources \
    -e CW_KEY_ID "$CW_ID" -e CW_KEY_SECRET "$CW_SEC")
  if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then
    IRIS_CMD+=(-e GLM_API_TOKEN "$GLM" -e GLM_BASE_URL "$BASE_URL" -e GLM_TIER "$GLM_TIER")
  fi
  if [ "$PHASE" = image-migration-revalidation ]; then
    # The reviewed portable-runtime archive is intentionally complete and nearly
    # incompressible. Keep the Iris workspace below its transport ceiling by
    # retaining only Marin's locked workspace packages and this immutable leaf.
    # Tests, docs, and native source trees are not runtime inputs for this stage.
    COMPACT_EXCLUDE="^(?!(?:pyproject\\.toml|uv\\.lock|LICENSE|README\\.md|config/|lib/|infra/(?:pulumi|buckets|deploy|marina)/|capability-pipeline-staging/submissions/${STAGE_ID}/))|^lib/(?:[^/]+/)?(?:tests?|docs?|rust)(?:/|$)|^infra/(?:pulumi|buckets|deploy|marina)/(?:tests?|docs?)(?:/|$)"
    IRIS_CMD+=(--exclude "$COMPACT_EXCLUDE")
    BUNDLE_PREFLIGHT+=(--exclude "$COMPACT_EXCLUDE")
  fi
  uv run --frozen "$HERE/preflight_stage_bundle.py" "${BUNDLE_PREFLIGHT[@]}"
  if [ "${#IRIS_EXTRA_ENV[@]}" -gt 0 ]; then IRIS_CMD+=("${IRIS_EXTRA_ENV[@]}"); fi
  if { [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ]; } && [ -n "${CAPABILITY_MAX_REPAIR_ROUNDS:-}" ]; then
    IRIS_CMD+=(-e CAPABILITY_MAX_REPAIR_ROUNDS "$CAPABILITY_MAX_REPAIR_ROUNDS")
  fi
  # Per-item judge-calibration fan-out (synthesis default 64). Lower it for wide waves so
  # items calibrating in sync cannot burst hundreds of GLM requests in a minute.
  if { [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ]; } && [ -n "${CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY:-}" ]; then
    case "$CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY" in ''|*[!0-9]*) die "CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY must be an integer" ;; esac
    IRIS_CMD+=(-e CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY "$CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY")
  fi
  # BEGIN worker knob forwarding
  # Conveyor wait budgets and limits (capability_pipeline/conveyor.py validates them
  # before any work starts), and the image/publication/verifier/runtime-infrastructure
  # knobs the synthesis worker reads. Forward whichever the operator set, checked
  # against what the consumer accepts (int: whole number, num: non-negative decimal
  # such as 0.5, text: any non-empty value).
  if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ]; then
    # Keep in step with conveyor.DEFAULT_WAIT_BUDGETS (tests/test_submit_env_forwarding.py).
    WAIT_KINDS=" IMAGE_CAPTURE IMAGE_PUBLICATION IMAGE_COLD_PULL IMAGE_MIGRATION IMAGE_REVIEW IMAGE_REVIEW_TRANSPORT IMAGE_INFRASTRUCTURE BUILDER_PROCESS BUILDER_CONTINUATION CONTROLLER_EXCEPTION RUNTIME_INFRASTRUCTURE "
    WORKER_KNOBS=(
      CAPABILITY_CONVEYOR_MAX_CALLS_PER_ITEM:int
      CAPABILITY_CONVEYOR_MAX_ACTIVE_REENTRIES:int
      CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS:num
      CAPABILITY_PUBLICATION_QUEUE:text
      CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS:int
      CAPABILITY_IMAGE_COMMAND_TIMEOUT_SECONDS:num
      CAPABILITY_VERIFIER_PROVISIONING_SECONDS:int
      CAPABILITY_INFRA_PROBE_TTL_SECONDS:num
      CAPABILITY_RUNTIME_INFRA_MAX_RETRIES:int
    )
    forward_knob() {  # NAME KIND
      local knob_name="$1" knob_kind="$2" knob_value="${!1:-}"
      [ -n "$knob_value" ] || return 0
      case "$knob_kind" in
        int) [[ "$knob_value" =~ ^[0-9]+$ ]] || die "$knob_name must be a non-negative whole number: $knob_value" ;;
        num) [[ "$knob_value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || die "$knob_name must be a non-negative number: $knob_value" ;;
      esac
      IRIS_CMD+=(-e "$knob_name" "$knob_value")
    }
    for knob in "${WORKER_KNOBS[@]}"; do
      forward_knob "${knob%%:*}" "${knob##*:}"
    done
    while IFS= read -r name; do
      case "$name" in
        CAPABILITY_WAIT_*_BACKOFF_SECONDS) kind="${name#CAPABILITY_WAIT_}"; kind="${kind%_BACKOFF_SECONDS}"; value_kind=num ;;
        CAPABILITY_WAIT_*_ATTEMPTS) kind="${name#CAPABILITY_WAIT_}"; kind="${kind%_ATTEMPTS}"; value_kind=int ;;
        CAPABILITY_WAIT_*_SECONDS) kind="${name#CAPABILITY_WAIT_}"; kind="${kind%_SECONDS}"; value_kind=num ;;
        *) kind=""; value_kind="" ;;
      esac
      if [ -n "$kind" ] && [[ "$WAIT_KINDS" == *" $kind "* ]]; then
        forward_knob "$name" "$value_kind"
      elif [[ " ${WORKER_KNOBS[*]} " != *" $name:"* ]]; then
        printf 'warning: %s is not a knob the synthesis worker reads; not forwarded\n' "$name" >&2
      fi
    done < <(compgen -e | grep -E '^CAPABILITY_(WAIT|CONVEYOR)_' || true)
  fi
  # END worker knob forwarding
  IRIS_CMD+=(-- bash -c "$JOB_CMD")
  "${IRIS_CMD[@]}"
)
trap - EXIT
release_lock
