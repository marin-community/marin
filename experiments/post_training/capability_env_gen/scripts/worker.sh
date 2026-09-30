#!/usr/bin/env bash
# Packed Iris worker. Credentials remain environment-only; results are resumable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { printf '[cap-worker %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 2; }

PHASE=""; PILOT=""; PILOT_SHA=""; SOURCE=""; SOURCE_SHA=""; PLAN_SHA=""
RUN_OUT=""; CONCURRENCY=""; TIER=""; S3_DEST=""; RUN_NAME=""
DISPATCH_LIMIT=""
ADOPTION_TRANSPORT=""
RESUME=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) PHASE="$2"; shift 2 ;; --pilot) PILOT="$2"; shift 2 ;;
    --pilot-sha256) PILOT_SHA="$2"; shift 2 ;; --source) SOURCE="$2"; shift 2 ;;
    --source-sha256) SOURCE_SHA="$2"; shift 2 ;; --plan-sha256) PLAN_SHA="$2"; shift 2 ;; --out) RUN_OUT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;; --tier) TIER="$2"; shift 2 ;;
    --dispatch-limit) DISPATCH_LIMIT="$2"; shift 2 ;;
    --s3-dest) S3_DEST="$2"; shift 2 ;; --run-name) RUN_NAME="$2"; shift 2 ;;
    --adoption-transport) ADOPTION_TRANSPORT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$PHASE" in generate|propose|synthesize|admit|checkpoint-revalidation|runtime-probe|reset-probe|daytona-health-probe|adaptive-test-pilot|composite-probe|composite-consensus-probe|composite-final-state-probe|context-budget-probe|c32-semantic-probe|image-migration-revalidation|judge-probe|evaluate|regrade) ;; *) die "invalid stage";; esac
case "$TIER" in interactive|bulk) ;; *) die "invalid tier";; esac
case "$CONCURRENCY" in ''|*[!0-9]*) die "invalid concurrency";; esac
[ "$CONCURRENCY" -gt 0 ] || die "concurrency must be positive"
if [ -n "$DISPATCH_LIMIT" ]; then
  case "$DISPATCH_LIMIT" in ''|*[!0-9]*) die "invalid dispatch limit";; esac
  [ "$DISPATCH_LIMIT" -gt 0 ] && [ "$DISPATCH_LIMIT" -le "$CONCURRENCY" ] || die "dispatch limit must be within concurrency"
else
  DISPATCH_LIMIT="$CONCURRENCY"
fi
[ -n "$S3_DEST" ] && [ -n "$RUN_NAME" ] && [ -n "$RUN_OUT" ] || die "missing run metadata"
[ -z "$ADOPTION_TRANSPORT" ] || { [ "$PHASE" = generate ] && [ "$RESUME" = 0 ]; } || die "proposal adoption transport is generate-only and cannot resume"
if [ "$PHASE" != regrade ] && [ "$PHASE" != reset-probe ]; then
  [ -n "${GLM_BASE_URL:-}" ] && [ -n "${GLM_API_TOKEN:-}" ] || die "GLM credentials are absent"
fi
[ -n "${CW_KEY_ID:-}" ] && [ -n "${CW_KEY_SECRET:-}" ] || die "object-store credentials are absent"

WORK_ROOT="${WORK_ROOT:-/tmp/capability-pipeline/$RUN_NAME}"
RESULTS="$WORK_ROOT/results"; STATE="$WORK_ROOT/upload-state"
# Run the durable uploader from private copies outside the stage directory. The stage
# leaf (/app/capability-pipeline-staging/submissions/<run>) was observed disappearing
# mid-job (2026-09-29), after which every sync failed ("can't open file .../sync_results.py")
# and the job's results stopped reaching S3.
SYNC_BIN="$WORK_ROOT/sync-bin"
mkdir -p "$SYNC_BIN"
cp "$HERE/sync_supervisor.py" "$HERE/sync_results.py" "$SYNC_BIN/"
mkdir -p "$RESULTS" "$STATE"
if [ -z "${MARIN_PROJECT:-}" ]; then
  # A submission leaf can be nested arbitrarily under the bundled workspace.
  # Find the nearest ancestor that is actually a Marin project instead of
  # assuming a fixed number of parents from worker.sh.
  candidate="$HERE"
  while [ "$candidate" != / ]; do
    if [ -f "$candidate/pyproject.toml" ] && [ -d "$candidate/lib/rigging" ]; then
      MARIN_PROJECT="$candidate"
      break
    fi
    candidate="$(dirname "$candidate")"
  done
fi
[ -n "${MARIN_PROJECT:-}" ] && [ -f "$MARIN_PROJECT/pyproject.toml" ] || die "Marin project is unavailable: ${MARIN_PROJECT:-not found}"
export MARIN_PROJECT
UV=(uv run --project "$MARIN_PROJECT" --frozen)
exec > >(tee -a "$RESULTS/worker.log") 2>&1
# Self-contained durable-uploader runtime (2026-09-29). Agent shell commands (auto-approved,
# workdir /app) were observed deleting /app mid-job: the stage dir first, then /app/.venv
# and /app/lib. Syncs then failed with ModuleNotFoundError ('fsspec', 'rigging'), the final
# sync failed too, and all work was lost. Before any agent runs, copy exactly the installed
# packages sync_results.py needs (same versions as /app/.venv, plus the rigging source and
# cluster configs) into a venv under $WORK_ROOT. The uploader, the restore and the final
# sync use only that interpreter; nothing they load lives under /app afterwards.
# build-runtime runs from $MARIN_PROJECT so rigging resolves config/ while collecting deps,
# and it verifies the new venv imports nothing from outside itself or its base interpreter.
SYNC_ENV="$WORK_ROOT/sync-env"
(cd "$MARIN_PROJECT" && "${UV[@]}" python3 "$SYNC_BIN/sync_supervisor.py" build-runtime \
  --target "$SYNC_ENV" --config-dir "$MARIN_PROJECT/config" --outside "$MARIN_PROJECT") || die "could not build the self-contained sync runtime"
SYNC_PYTHON="$SYNC_ENV/bin/python"
[ -x "$SYNC_PYTHON" ] || die "sync runtime interpreter is missing: $SYNC_PYTHON"
# Import sync_results.py itself exactly as the supervisor will (-E -s, cwd under $WORK_ROOT).
(cd "$SYNC_BIN" && "$SYNC_PYTHON" -E -s "$SYNC_BIN/sync_results.py" --help >/dev/null) || die "sync runtime cannot load sync_results.py"
# Build-parallelism pins (2026-09-28). The pod has a CPU REQUEST but no CPU limit, so
# nproc / cpu_count report the whole node (100+ cores). cargo, make and friends default
# to that many jobs, and 30 concurrent agents building locally spike memory past the pod
# limit in seconds (OOMKilled at 64g and 192g). Pin every common parallelism knob.
BUILD_JOBS="${BUILD_JOBS:-4}"
export CARGO_BUILD_JOBS="$BUILD_JOBS" MAKEFLAGS="-j$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" RAYON_NUM_THREADS="$BUILD_JOBS" UV_CONCURRENT_BUILDS="$BUILD_JOBS" \
  UV_CONCURRENT_INSTALLS="$BUILD_JOBS" NPM_CONFIG_JOBS="$BUILD_JOBS" GOMAXPROCS="$BUILD_JOBS" \
  PYTEST_XDIST_AUTO_NUM_WORKERS="$BUILD_JOBS"
log "parallelism pins: cpus_online=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo ?) (nproc honours OMP_NUM_THREADS, so not used) BUILD_JOBS=$BUILD_JOBS"

if [ -n "$PILOT" ]; then
  [ -f "$PILOT" ] || die "staged pilot missing: $PILOT"
  actual="$(shasum -a 256 "$PILOT" | awk '{print $1}')"
  [ -z "$PILOT_SHA" ] || [ "$actual" = "$PILOT_SHA" ] || die "pilot fingerprint mismatch"
fi
if [ -n "$ADOPTION_TRANSPORT" ]; then
  [ -f "$ADOPTION_TRANSPORT" ] || die "proposal adoption transport receipt is missing"
  [ -f "$HERE/proposal_adoption_transport.py" ] || die "proposal adoption transport helper is missing"
  ADOPTION_ROOT="$WORK_ROOT/proposal-adoption-input"
  [ ! -e "$ADOPTION_ROOT" ] || die "proposal adoption destination must be new"
  "${UV[@]}" python3 "$HERE/proposal_adoption_transport.py" download \
    --receipt "$ADOPTION_TRANSPORT" --destination "$ADOPTION_ROOT"
  [ -d "$ADOPTION_ROOT/checkpoint" ] || die "proposal adoption checkpoint did not restore"
  PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${UV[@]}" python3 - \
    "$ADOPTION_ROOT/checkpoint" "$ADOPTION_ROOT/source-archive.tar.gz" "$ADOPTION_ROOT/launch-receipt.json" "$PILOT" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.proposal_adoption import validate_checkpoint
validate_checkpoint(*(Path(value) for value in sys.argv[1:]))
PY
fi
if [ "$PHASE" = evaluate ] || [ "$PHASE" = regrade ] || [ "$PHASE" = checkpoint-revalidation ]; then
  SOURCE="$WORK_ROOT/$PHASE-source"
fi
if [ -n "$SOURCE" ]; then
  if [ "$PHASE" = checkpoint-revalidation ]; then
    [ "$RESUME" = 0 ] && [ "$CONCURRENCY" = 1 ] && [ "$DISPATCH_LIMIT" = 1 ] || die "checkpoint revalidation requires a fresh single worker"
    [ ! -e "$SOURCE" ] || die "checkpoint source destination must be new"
    [ -f "$HERE/inputs/checkpoint.blob.json" ] && [ -f "$HERE/inputs/checkpoint.transport.json" ] || die "checkpoint transport receipts are missing"
    [ "${#SOURCE_SHA}" = 64 ] && [ "${#PLAN_SHA}" = 64 ] || die "checkpoint input fingerprints are missing"
  elif [ "$PHASE" = image-migration-revalidation ]; then
    [ ! -e "$SOURCE" ] || die "image migration source destination must be new"
    [ -f "$HERE/inputs/source-bundle.tar.gz" ] || die "image migration transport archive is missing"
    [ -f "$HERE/inputs/source.transport.json" ] || die "image migration transport manifest is missing"
  elif [ "$PHASE" = evaluate ]; then
    [ ! -e "$SOURCE" ] || die "evaluation source destination must be new"
    [ -f "$HERE/inputs/evaluation-bundle.tar.gz" ] || die "evaluation transport archive is missing"
    [ -f "$HERE/inputs/evaluation.transport.json" ] || die "evaluation transport manifest is missing"
    [ "${#SOURCE_SHA}" = 64 ] && [ "${#PLAN_SHA}" = 64 ] || die "evaluation input fingerprints are missing"
  elif [ "$PHASE" = regrade ]; then
    [ ! -e "$SOURCE" ] || die "regrade source destination must be new"
    [ -f "$HERE/inputs/regrade-bundle.tar.gz" ] || die "regrade transport archive is missing"
    [ -f "$HERE/inputs/regrade.transport.json" ] || die "regrade transport manifest is missing"
    [ "${#SOURCE_SHA}" = 64 ] && [ "${#PLAN_SHA}" = 64 ] || die "regrade input fingerprints are missing"
  elif [ ! -d "$SOURCE" ]; then
    die "staged source directory is missing: $SOURCE"
  elif [ "$PHASE" = propose ]; then
    for source_file in input_pilot.json plans.json proposals.json report.json; do
      [ -f "$SOURCE/$source_file" ] || die "staged proposal seed is missing: $SOURCE/$source_file"
    done
  elif [ "$PHASE" != adaptive-test-pilot ]; then
    source_file="$SOURCE/accepted.json"
    [ -f "$source_file" ] || die "staged accepted proposals missing: $source_file"
    if [ -n "$SOURCE_SHA" ]; then
      if [ "$PHASE" = image-migration-revalidation ]; then
        source_file="$SOURCE/manifest.json"
        [ -f "$source_file" ] || die "staged image migration manifest is missing"
      fi
      [ "$(shasum -a 256 "$source_file" | awk '{print $1}')" = "$SOURCE_SHA" ] || die "source fingerprint mismatch"
    fi
  fi
fi

sync_once() {
  local mode="${1:-}"
  local args=()
  [ -z "$mode" ] || args+=(--final)
  [ -x "$SYNC_PYTHON" ] || return 1
  # --python, not --project: the final sync must work after /app has been deleted.
  "$SYNC_PYTHON" -E -s "$SYNC_BIN/sync_supervisor.py" --source "$RESULTS" --destination "$S3_DEST" --state "$STATE" --python "$SYNC_PYTHON" "${args[@]}"
}
UPLOAD_PID=""
unpack_taskcompendium_source() {
  [ -f "$HERE/taskcompendium-source.tar.gz" ] || die "agentic stage requires TaskCompendium source archive"
  [ -f "$HERE/taskcompendium-bootstrap.json" ] || die "agentic stage requires TaskCompendium source archive manifest"
  local expected_archive_sha actual_archive_sha
  expected_archive_sha="$(sed -n 's/.*"sha256":"\([0-9a-f]*\)".*/\1/p' "$HERE/taskcompendium-bootstrap.json")"
  actual_archive_sha="$(shasum -a 256 "$HERE/taskcompendium-source.tar.gz" | awk '{print $1}')"
  [ "${#expected_archive_sha}" = 64 ] && [ "$actual_archive_sha" = "$expected_archive_sha" ] || die "TaskCompendium source archive fingerprint mismatch"
  rm -rf "$TASKCOMPENDIUM_SOURCE"
  mkdir -p "$TASKCOMPENDIUM_SOURCE"
  tar -xzf "$HERE/taskcompendium-source.tar.gz" -C "$TASKCOMPENDIUM_SOURCE" || die "could not unpack TaskCompendium source archive"
  [ -f "$TASKCOMPENDIUM_SOURCE/pyproject.toml" ] || die "TaskCompendium source archive is invalid"
}
finish() {
  local rc="$1"
  # The uploader PID is the Python supervisor, which reaps its sync child group
  # before exiting.  Waiting here excludes overlap with the final publisher.
  [ -n "$UPLOAD_PID" ] && kill -TERM "$UPLOAD_PID" 2>/dev/null || true
  [ -n "$UPLOAD_PID" ] && wait "$UPLOAD_PID" 2>/dev/null || true
  # Stdlib only, via the sync runtime: /app may be gone by now, and a failing command in
  # this trap (set -e) would skip the final sync below.
  "$SYNC_PYTHON" -E -s - "$RESULTS/terminal.json" "$rc" <<'PY' || log "could not write terminal.json"
import json, sys, time
with open(sys.argv[1], "w") as f:
    json.dump({"exit_code": int(sys.argv[2]), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    f.write("\n")
PY
  # A results tree that was never fully restored must not replace the durable snapshot:
  # 2026-09-29 shard-071/075 were preempted mid-restore and this trap published the
  # half-restored tree as latest.json, losing every status and quality verdict.
  if [ "$RESTORE_COMPLETE" != 1 ]; then
    log "restore did not complete; not publishing a partial results tree"
    exit "$rc"
  fi
  if ! sync_once final; then log "FATAL: final durable sync failed"; rc=75; fi
  exit "$rc"
}
RESTORE_COMPLETE=0
trap 'finish $?' EXIT

# Restore with the same self-contained runtime; this also proves it can reach S3 at startup.
(cd "$SYNC_BIN" && "$SYNC_PYTHON" -E -s "$SYNC_BIN/sync_results.py" restore --source "$S3_DEST" --destination "$RESULTS" \
  $([ "$RESUME" = 0 ] && printf '%s' --new-run)) \
  >> "$STATE/uploader.log" 2>&1 || die "could not restore prior durable results"
restore_seed() {
  # A revalidation has a new durable prefix, so S3 cannot restore the prior
  # item state.  Copy only a manifest-bound seed after the new-prefix check;
  # never merge it with an explicit resume of an existing output prefix.
  if [ -f "$SOURCE/restore-seed.transport.json" ]; then
    local source_root source_manifest_sha
    source_root="$(cd "$SOURCE" && pwd -P)" || die "continuation seed source directory is unavailable"
    source_manifest_sha="$(shasum -a 256 "$source_root/manifest.json" | awk '{print $1}')"
    [ "${#source_manifest_sha}" = 64 ] || die "continuation seed source manifest is unavailable"
    [ ! -e "$SOURCE/restore-seed.tar.gz" ] || die "continuation seed has both transport receipt and local archive"
    (
      cd "$MARIN_PROJECT"
      "${UV[@]}" python3 "$HERE/continuation_seed_transport.py" download \
        --receipt "$source_root/restore-seed.transport.json" \
        --expected-source-manifest-sha256 "$source_manifest_sha" \
        --destination "$source_root/restore-seed.tar.gz"
    )
  fi
  [ -d "$SOURCE/restore-seed" ] || [ -f "$SOURCE/restore-seed.tar.gz" ] || return 0
  # The image-migration executor validates and restores its seed into a fresh
  # nested output itself; pre-restoring here would bypass its archive checks.
  [ "$PHASE" != image-migration-revalidation ] || return 0
  [ "$PHASE" = synthesize ] || die "restore seed is only valid for synthesis"
  [ "$RESUME" = 0 ] || die "restore seed cannot be combined with --resume"
  "${UV[@]}" python3 "$HERE/restore_seed.py" "$SOURCE/manifest.json" "$SOURCE/restore-seed" "$RESULTS"
}
if [ -n "$SOURCE" ]; then restore_seed; fi
RESTORE_COMPLETE=1
"${UV[@]}" python3 - "$RESULTS/submission.json" "$PHASE" "$RUN_OUT" "$CONCURRENCY" "$DISPATCH_LIMIT" "$TIER" "$RUN_NAME" "$PILOT_SHA" <<'PY'
import json, sys, time
path, phase, out, concurrency, dispatch_limit, tier, run_name, pilot_sha = sys.argv[1:]
with open(path, "w") as f:
    json.dump({"phase": phase, "requested_out": out, "concurrency": int(concurrency), "dispatch_limit": int(dispatch_limit), "tier": tier,
               "run_name": run_name, "pilot_sha256": pilot_sha or None,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    f.write("\n")
PY
EXTRA_ARGS=(); [ -f "$HERE/extra-args.nul" ] && mapfile -d '' -t EXTRA_ARGS < "$HERE/extra-args.nul"

# Start durable log/artifact publication before any optional OMP, Cargo, or
# package bootstrap.  A setup failure is itself run evidence and must survive.
"$SYNC_PYTHON" -E -s "$SYNC_BIN/sync_supervisor.py" --source "$RESULTS" --destination "$S3_DEST" --state "$STATE" --python "$SYNC_PYTHON" \
  --loop --interval-seconds "${UPLOAD_INTERVAL_SECONDS:-15}" & UPLOAD_PID=$!

if [ "$PHASE" = image-migration-revalidation ]; then
  MIGRATION_MANIFEST_SHA256='641ed6358c3d307246972f77557c15b2ae6dcb951987894046f85f4a1270837e'
  MIGRATION_RECEIPT_SHA256='200e07dd7774e31ee11547651f10494235eddcce6b7cb3d6bae8a5e0f8973428'
  "${UV[@]}" python3 "$HERE/restore_bundle.py" restore \
    --archive "$HERE/inputs/source-bundle.tar.gz" \
    --transport "$HERE/inputs/source.transport.json" \
    --destination "$SOURCE" \
    --expected-manifest-sha256 "$MIGRATION_MANIFEST_SHA256" \
    --expected-receipt-sha256 "$MIGRATION_RECEIPT_SHA256"
  [ -f "$SOURCE/accepted.json" ] || die "restored image migration accepted input is missing"
  [ "$(shasum -a 256 "$SOURCE/manifest.json" | awk '{print $1}')" = "$SOURCE_SHA" ] || die "restored image migration source fingerprint mismatch"
fi

if [ "$PHASE" = evaluate ]; then
  "${UV[@]}" python3 "$HERE/restore_bundle.py" restore-evaluation \
    --archive "$HERE/inputs/evaluation-bundle.tar.gz" \
    --transport "$HERE/inputs/evaluation.transport.json" \
    --destination "$SOURCE" --expected-manifest-sha256 "$SOURCE_SHA" --plan-sha256 "$PLAN_SHA"
fi
if [ "$PHASE" = regrade ]; then
  "${UV[@]}" python3 "$HERE/restore_bundle.py" restore-regrade \
    --archive "$HERE/inputs/regrade-bundle.tar.gz" \
    --transport "$HERE/inputs/regrade.transport.json" \
    --destination "$SOURCE" --expected-manifest-sha256 "$SOURCE_SHA" --plan-sha256 "$PLAN_SHA"
fi
if [ "$PHASE" = checkpoint-revalidation ]; then
  "${UV[@]}" python3 "$HERE/checkpoint_revalidation_transport.py" download \
    --receipt "$HERE/inputs/checkpoint.blob.json" \
    --destination "$WORK_ROOT/checkpoint-bundle.tar.gz" \
    --manifest-sha256 "$SOURCE_SHA" --request-sha256 "$PLAN_SHA"
  "${UV[@]}" python3 "$HERE/checkpoint_revalidation_transport.py" restore \
    --archive "$WORK_ROOT/checkpoint-bundle.tar.gz" \
    --transport "$HERE/inputs/checkpoint.transport.json" \
    --destination "$SOURCE" \
    --manifest-sha256 "$SOURCE_SHA" --request-sha256 "$PLAN_SHA"
fi

if [ "$PHASE" = judge-probe ] || [ "$PHASE" = reset-probe ]; then
  [ -f "$HERE/vendor/task_spec/source.lock.json" ] || die "judge probe requires staged source lock"
  [ -f "$HERE/scripts/build_judge_probe.py" ] || die "judge probe builder is not staged"
  export TASKCOMPENDIUM_SOURCE="$HERE/taskcompendium-source"
  unpack_taskcompendium_source
fi

if [ "$PHASE" = evaluate ] || [ "$PHASE" = regrade ] || [ "$PHASE" = context-budget-probe ]; then
  [ -f "$HERE/vendor/task_spec/source.lock.json" ] || die "evaluation requires staged source lock"
  export TASKCOMPENDIUM_SOURCE="$HERE/taskcompendium-source"
  unpack_taskcompendium_source
fi

if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = runtime-probe ] || [ "$PHASE" = daytona-health-probe ] || [ "$PHASE" = adaptive-test-pilot ] || [ "$PHASE" = composite-probe ] || [ "$PHASE" = composite-consensus-probe ] || [ "$PHASE" = composite-final-state-probe ] || [ "$PHASE" = c32-semantic-probe ] || [ "$PHASE" = image-migration-revalidation ]; then
  if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = image-migration-revalidation ]; then
    [ -f "$HERE/models.template.yml" ] || die "synthesis requires a staged models template"
    [ -f "$HERE/omp-research-overlay.yml" ] || die "synthesis requires a staged research overlay"
  fi
  [ -f "$HERE/vendor/task_spec/source.lock.json" ] || die "agentic stage requires staged source lock"
  [ -f "$HERE/docs/task_contract.md" ] || die "agentic stage requires staged task contract"
  if [ "$PHASE" = c32-semantic-probe ]; then
    [ -f "$HERE/daytona-tools/dt.sh" ] || die "c32 semantic probe requires staged Daytona dt.sh"
    chmod +x "$HERE/daytona-tools/dt.sh" || die "could not make c32 Daytona helper executable"
    export CAPABILITY_DAYTONA_TOOLS="$HERE/daytona-tools"
  elif [ "$PHASE" = runtime-probe ] || [ "$PHASE" = daytona-health-probe ] || [ "$PHASE" = adaptive-test-pilot ] || [ "$PHASE" = composite-probe ] || [ "$PHASE" = composite-consensus-probe ] || [ "$PHASE" = composite-final-state-probe ]; then
    [ -f "$HERE/dt.py" ] || die "runtime probe requires staged Daytona dt.py"
    export CAPABILITY_DAYTONA_TOOLS="$HERE"
  else
    # Iris workspace federation can preserve helper bytes while normalizing mode
    # bits.  Verify the staged script exists, then establish the required mode
    # locally before synthesis invokes dt.sh.
    [ -f "$HERE/daytona-tools/dt.sh" ] || die "synthesis requires staged Daytona tools"
    chmod +x "$HERE/daytona-tools/dt.sh" || die "could not make staged Daytona helper executable"
    export CAPABILITY_DAYTONA_TOOLS="$HERE/daytona-tools"
  fi
  # Credentials follow the selected sandbox provider (capability_pipeline/
  # sandbox_provider.py); a silo run carries no Daytona key at all.
  case "${CAPABILITY_SANDBOX_PROVIDER:-daytona}" in
    silo)
      [ -n "${SILO_API_TOKEN:-}" ] || die "agentic stage on silo requires SILO_API_TOKEN"
      [ -n "${SILO_BROKER_RESOLVE_URL:-}${SILO_BROKER_URL:-}" ] || die "agentic stage on silo requires SILO_BROKER_RESOLVE_URL"
      ;;
    daytona) [ -n "${DAYTONA_API_KEY:-}" ] || die "agentic stage requires DAYTONA_API_KEY" ;;
    *) die "unknown CAPABILITY_SANDBOX_PROVIDER: ${CAPABILITY_SANDBOX_PROVIDER}" ;;
  esac
  export CAPABILITY_TASK_SPEC_LOCK="$HERE/vendor/task_spec/source.lock.json"
  export CAPABILITY_TASK_CONTRACT="$HERE/docs/task_contract.md"
  export CAPABILITY_OMP_CONFIG="$HERE/omp-research-overlay.yml"
  export TASKCOMPENDIUM_SOURCE="$HERE/taskcompendium-source"
  unpack_taskcompendium_source
  if [ "$PHASE" != daytona-health-probe ] && [ "$PHASE" != adaptive-test-pilot ] && [ "$PHASE" != composite-probe ] && [ "$PHASE" != composite-consensus-probe ] && [ "$PHASE" != composite-final-state-probe ] && [ "$PHASE" != c32-semantic-probe ]; then
    command -v cargo >/dev/null 2>&1 || die "agentic stage requires cargo for ShellSim bridge"
    # Keep cargo output outside the hashed source tree.  The toolchain verifier
    # rejects any extra source member, including a cargo target directory.
    SHELLSIM_TARGET="$WORK_ROOT/taskcompendium-shellsim-target"
    SHELLSIM_OVERLAY="$WORK_ROOT/taskcompendium-shellsim-overlay"
    "${UV[@]}" python3 - "$TASKCOMPENDIUM_SOURCE" "$SHELLSIM_OVERLAY" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.shellsim_snapshot_extension import apply_overlay

apply_overlay(Path(sys.argv[1]), Path(sys.argv[2]))
PY
    (cd "$SHELLSIM_OVERLAY/shellsim-bridge" && cargo build --locked --release --target-dir "$SHELLSIM_TARGET")
    export TASKCOMPENDIUM_SHELLSIM_BRIDGE="$SHELLSIM_TARGET/release/taskcompendium-shellsim"
    [ -x "$TASKCOMPENDIUM_SHELLSIM_BRIDGE" ] || die "ShellSim bridge build did not produce an executable"
  fi
  if [ "$PHASE" = synthesize ] || [ "$PHASE" = generate ] || [ "$PHASE" = checkpoint-revalidation ] || [ "$PHASE" = image-migration-revalidation ]; then
    export OMP_ENV_HERE="$HERE" MODELS_TEMPLATE="$HERE/models.template.yml"
    . "$HERE/omp_env.sh"
    omp_bootstrap_tools; omp_configure
    # Record only the executable/version and credential-free staged input digests.
    # The rendered models.yml can contain a token source and is intentionally never
    # copied into results or logged.
    mkdir -p "$RESULTS/controller"
    OMP_VERSION="$(omp --version 2>&1 | head -1 | tr -cd '[:alnum:] ._+-')"
    OMP_BINARY_SHA256="$(sha256sum "$(command -v omp)" | awk '{print $1}')"
    MODELS_TEMPLATE_SHA256="$(sha256sum "$HERE/models.template.yml" | awk '{print $1}')"
    OMP_OVERLAY_SHA256="$(sha256sum "$HERE/omp-research-overlay.yml" | awk '{print $1}')"
    printf '{"omp_version":"%s","omp_binary_sha256":"%s","models_template_sha256":"%s","research_overlay_sha256":"%s"}\n' \
      "$OMP_VERSION" "$OMP_BINARY_SHA256" "$MODELS_TEMPLATE_SHA256" "$OMP_OVERLAY_SHA256" > "$RESULTS/controller/omp-runtime.json"
    # Help is useful to bind option semantics to the worker version. Replace all
    # absolute paths before durable publication; no rendered provider config is read.
    omp --help 2>&1 | sed -E 's#(/[^[:space:]]+)+#<path>#g' > "$RESULTS/controller/omp-help.txt" || true
  fi
fi

# Native Harbor resolves project-local import paths from this staged package.
# Export it before every phase, including direct runtime scripts.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
# `uv run python -m` starts in the Marin checkout, which also owns a regular
# `scripts` package. Keep the staged package ahead of that checkout so every
# controller phase imports its own hash-bound transport validator.
export PYTHONSAFEPATH=1
log "starting phase=$PHASE tier=$TIER requested_concurrency=$CONCURRENCY dispatch_limit=$DISPATCH_LIMIT"
# Memory sampler (2026-09-28): construction jobs OOM at 64g and 192g. Log total RSS
# and the top processes by RSS every MEM_SAMPLE_SECONDS so the hog is visible in logs.
if [ "${MEM_SAMPLE_SECONDS:-10}" -gt 0 ]; then
  ( set +e +o pipefail  # the sampler must never die silently (the worker runs set -euo pipefail)
    while sleep "${MEM_SAMPLE_SECONDS:-10}"; do
      # /proc, not ps: the worker image has no procps.
      snap="$(awk 'FNR==1{n="?"} /^Name:/{n=$2} /^VmRSS:/{by[n]+=$2; c[n]++; t+=$2} END{printf "TOTAL %.1f\n", t/1048576; for (k in by) printf "%s:%d:%.1fg\n", k, c[k], by[k]/1048576}' /proc/[0-9]*/status 2>/dev/null)"
      top="$(printf '%s\n' "$snap" | grep -v '^TOTAL' | sort -t: -k3 -rn | head -8 | tr '\n' ' ')"
      cg="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)"
      tmpfs="$(df -k -t tmpfs 2>/dev/null | awk 'NR>1{u+=$3} END{printf "%.1f", u/1048576}')"
      node="$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "node_total_gb=%.0f node_avail_gb=%.0f", t/1048576, a/1048576}' /proc/meminfo 2>/dev/null)"
      printf '[cap-mem %s] %s host=%s cgroup_gb=%s total_rss_gb=%s tmpfs_used_gb=%s top: %s\n' "$(date -u +%H:%M:%S)" "$node" "$(hostname 2>/dev/null)" "$(awk -v b="$cg" 'BEGIN{printf "%.1f", b/1073741824}')" "$(printf '%s\n' "$snap" | awk '/^TOTAL/{print $2}')" "${tmpfs:-?}" "$top"
    done ) &
fi
# Memory reaper (2026-09-28). A single runaway process started by an agent's local command
# (the omp bash tool runs in this container) can take the pod from ~13 GB to the 192 GB
# limit in under 30 s; the kernel then OOM-kills the WHOLE container, all 30 sessions.
# Every 2 s: if the cgroup is above MEM_REAP_FRACTION of its limit, SIGKILL the largest
# process that is not omp, not the pipeline controller and not this worker, and log its
# command line. The agent sees one failed command instead of the job dying.
if [ "${MEM_REAP_FRACTION:-0.80}" != 0 ]; then
  ( set +e +o pipefail
    max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
    case "$max" in ''|max|*[!0-9]*) log "memory reaper: no cgroup memory limit readable ($max); disabled"; exit 0 ;; esac
    log "memory reaper: armed limit_gb=$(awk -v b="$max" 'BEGIN{printf "%.0f", b/1073741824}') fraction=${MEM_REAP_FRACTION:-0.80}"
    while sleep 2; do
      # Unreclaimable memory only (anon + shmem): page cache counts toward memory.current but
      # the kernel reclaims it before OOM, so it must not trigger a kill.
      cur="$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{print s+0}' /sys/fs/cgroup/memory.stat 2>/dev/null)"
      [ "${cur:-0}" -gt 0 ] 2>/dev/null || cur="$(awk '$1=="total_rss"||$1=="total_shmem"{s+=$2} END{print s+0}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)"
      awk -v c="$cur" -v m="$max" -v f="${MEM_REAP_FRACTION:-0.80}" 'BEGIN{exit !(c > m*f)}' || continue
      victim="$(for st in /proc/[0-9]*/status; do
          pid="${st#/proc/}"; pid="${pid%/status}"
          [ "$pid" = "$$" ] || [ "$pid" = "$BASHPID" ] && continue
          name="$(awk '/^Name:/{print $2; exit}' "$st" 2>/dev/null)"
          rss="$(awk '/^VmRSS:/{print $2; exit}' "$st" 2>/dev/null)"
          [ -n "$rss" ] || continue
          cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-200)"
          case "$name" in omp|tee|bash|sh|uv|sleep) continue ;; esac
          case "$cmd" in *capability_pipeline*|*sync_results*|*worker.sh*) continue ;; esac
          printf '%s %s %s\n' "$rss" "$pid" "$cmd"
        done | sort -rn | head -1)"
      [ -n "$victim" ] || { log "memory reaper: cgroup over threshold but no eligible victim"; continue; }
      vrss="${victim%% *}"; rest="${victim#* }"; vpid="${rest%% *}"; vcmd="${rest#* }"
      kill -9 "$vpid" 2>/dev/null && log "memory reaper: KILLED pid=$vpid rss_gb=$(awk -v k="$vrss" 'BEGIN{printf "%.1f", k/1048576}') cgroup_gb=$(awk -v b="$cur" 'BEGIN{printf "%.1f", b/1073741824}') cmd=$vcmd"
    done ) &
fi
if [ "$PHASE" = judge-probe ]; then
  PROBE_ROOT="$RESULTS/judge-probe"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$TASKCOMPENDIUM_SOURCE" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$TASKCOMPENDIUM_SOURCE" --frozen --extra harbor python "$HERE/scripts/build_judge_probe.py" \
    --source-lock "$HERE/vendor/task_spec/source.lock.json" --output "$PROBE_ROOT/evidence.json" --request-timeout 600
elif [ "$PHASE" = reset-probe ]; then
  [ "$DISPATCH_LIMIT" -eq 1 ] || die "reset protocol probe requires concurrency 1"
  [ -f "$HERE/scripts/run_reset_protocol_probe.py" ] || die "reset protocol probe runner is not staged"
  [ -f "$HERE/scripts/build_runtime_probe.py" ] || die "reset protocol fixture builder is not staged"
  export CAPABILITY_DAYTONA_TOOLS="$HERE/daytona-tools"
  export CAPABILITY_REMOTE_RESET_PROBE=1
  "${UV[@]}" python3 "$HERE/scripts/run_reset_protocol_probe.py" \
    --out "$RESULTS/reset-probe" --builder "$HERE/scripts/build_runtime_probe.py" "${EXTRA_ARGS[@]}"
elif [ "$PHASE" = runtime-probe ]; then
  [ -f "$HERE/scripts/build_runtime_probe.py" ] || die "runtime probe builder is not staged"
  PROBE_ROOT="$RESULTS/daytona-probe"
  # The probe package is built from the exact lock-verified TaskCompendium tree;
  # Harbor/daytona stays an ephemeral uv environment and no credentials enter it.
  # Iris bootstraps its own /app/.venv.  Do not let that unrelated environment
  # shadow the pinned TaskCompendium dependency set used by the probe.
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$TASKCOMPENDIUM_SOURCE" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$TASKCOMPENDIUM_SOURCE" --frozen --extra harbor --with daytona==0.200.2 python -c 'import msgspec, taskcompendium; import capability_pipeline.runtime_agents, capability_pipeline.daytona_environment, capability_pipeline.daytona_verifier; print({"msgspec": msgspec.__version__, "taskcompendium": taskcompendium.__file__, "runtime_agents": capability_pipeline.runtime_agents.__file__, "daytona_environment": capability_pipeline.daytona_environment.__file__, "daytona_verifier": capability_pipeline.daytona_verifier.__file__})' 2>&1 | tee "$RESULTS/runtime-import-preflight.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$TASKCOMPENDIUM_SOURCE" --frozen python "$HERE/scripts/build_runtime_probe.py" --out "$PROBE_ROOT" "${EXTRA_ARGS[@]}"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$TASKCOMPENDIUM_SOURCE" --frozen --extra harbor --with daytona==0.200.2 python "$HERE/capability_pipeline/runtime.py" \
    --package "$PROBE_ROOT/harbor" --bundle "$PROBE_ROOT/task" --controls "$PROBE_ROOT/task/controls.json" \
    --output "$PROBE_ROOT/runtime-evidence.json" --shellsim-bridge "$TASKCOMPENDIUM_SHELLSIM_BRIDGE"
elif [ "$PHASE" = daytona-health-probe ]; then
  [ -f "$HERE/scripts/probe_daytona_health.py" ] || die "Daytona health probe is not staged"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$TASKCOMPENDIUM_SOURCE" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$TASKCOMPENDIUM_SOURCE" --frozen --extra harbor --with daytona==0.200.2 python "$HERE/scripts/probe_daytona_health.py" \
    --output "$RESULTS/daytona-health.json" "${EXTRA_ARGS[@]}"
elif [ "$PHASE" = adaptive-test-pilot ]; then
  [ -f "$HERE/scripts/run_adaptive_test_pilot.py" ] || die "adaptive test pilot is not staged"
  ADAPTIVE_REPLAY_ARGS=(); [ -n "$SOURCE" ] && ADAPTIVE_REPLAY_ARGS=(--replay-root "$SOURCE")
  env -u VIRTUAL_ENV uv run --project "$MARIN_PROJECT" --frozen --prerelease=allow --with daytona==0.200.2 python "$HERE/scripts/run_adaptive_test_pilot.py" \
    --output "$RESULTS/adaptive-test-pilot" --concurrency "$DISPATCH_LIMIT" "${ADAPTIVE_REPLAY_ARGS[@]}" "${EXTRA_ARGS[@]}"
elif [ "$PHASE" = context-budget-probe ]; then
  [ "$DISPATCH_LIMIT" -eq 1 ] || die "context-budget probe requires one worker"
  [ -f "$HERE/scripts/run_context_budget_probe.py" ] || die "context-budget probe runner is not staged"
  [ -f "$PILOT" ] || die "context-budget probe request is not staged"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  export CAPABILITY_TASK_SPEC_LOCK="$HERE/vendor/task_spec/source.lock.json"
  export CAPABILITY_TASK_CONTRACT="$HERE/docs/task_contract.md"
  OVERLAY="$(PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${UV[@]}" python3 -c 'from pathlib import Path; from capability_pipeline.synthesis import OfficialToolchain; print(OfficialToolchain.resolve(Path("/tmp")).package_root)')"
  [ -f "$OVERLAY/pyproject.toml" ] || die "pinned TaskCompendium overlay is unavailable"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$OVERLAY" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --project "$OVERLAY" --frozen --extra harbor python "$HERE/scripts/run_context_budget_probe.py" \
      --input "$PILOT" --out "$RESULTS/context-budget-probe"
elif [ "$PHASE" = composite-probe ] || [ "$PHASE" = composite-consensus-probe ] || [ "$PHASE" = composite-final-state-probe ]; then
  if [ "$PHASE" = composite-probe ]; then
    PROBE_SCRIPT="$HERE/scripts/run_composite_probe.py"
    PROBE_ROOT="$RESULTS/composite-probe"
  elif [ "$PHASE" = composite-final-state-probe ]; then
    PROBE_SCRIPT="$HERE/scripts/run_composite_final_state_probe.py"
    PROBE_ROOT="$RESULTS/composite-final-state-probe"
  else
    PROBE_SCRIPT="$HERE/scripts/run_composite_consensus_probe.py"
    PROBE_ROOT="$RESULTS/composite-consensus-probe"
  fi
  [ -f "$PROBE_SCRIPT" ] || die "composite probe runner is not staged"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  # Resolve and copy the exact source before uv builds it.  `uv sync` writes
  # taskcompendium.egg-info under src/, which correctly makes the base archive
  # fail its exact-file-set verifier if resolution happens afterwards.
  OVERLAY="$(PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${UV[@]}" python3 -c 'from pathlib import Path; from capability_pipeline.synthesis import OfficialToolchain; print(OfficialToolchain.resolve(Path("/tmp")).package_root)')"
  [ -f "$OVERLAY/pyproject.toml" ] || die "composite TaskCompendium overlay was not created"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$OVERLAY" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$OVERLAY" --frozen --extra harbor --with daytona==0.200.2 "$PROBE_SCRIPT" --out "$PROBE_ROOT" "${EXTRA_ARGS[@]}"
elif [ "$PHASE" = c32-semantic-probe ]; then
  [ -f "$HERE/scripts/run_c32_semantic_probe.py" ] || die "c32 semantic probe runner is not staged"
  [ -f "$HERE/scripts/c32_balanced_mutation.py" ] || die "c32 semantic probe mutator is not staged"
  [ -f "$HERE/c32-probe-inputs/specification.json" ] || die "c32 semantic probe inputs are not staged"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  OVERLAY="$(PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${UV[@]}" python3 -c 'from pathlib import Path; from capability_pipeline.synthesis import OfficialToolchain; print(OfficialToolchain.resolve(Path("/tmp")).package_root)')"
  [ -f "$OVERLAY/pyproject.toml" ] || die "c32 semantic probe TaskCompendium overlay was not created"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$OVERLAY" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  # dt.sh deliberately delegates to DT_PYTHON. The interpreter resolved by a
  # standalone `uv run python -c` is an ephemeral build path, so bind it inside
  # the same uv command that runs the helper and probe.
  mkdir -p "$RESULTS/c32-semantic-probe"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$OVERLAY" --frozen --extra harbor --with daytona==0.200.2 bash -c '
    set -e
    export DT_PYTHON="$(command -v python)"
    "$1" --help > "$2"
    exec python "$3" --inputs "$4" --mutation-program "$5" --output "$6"
  ' _ "$HERE/daytona-tools/dt.sh" "$RESULTS/c32-semantic-probe/dt-help.txt" "$HERE/scripts/run_c32_semantic_probe.py" \
    "$HERE/c32-probe-inputs" "$HERE/scripts/c32_balanced_mutation.py" "$RESULTS/c32-semantic-probe"
elif [ "$PHASE" = checkpoint-revalidation ]; then
  [ -f "$HERE/scripts/run_checkpoint_revalidation.py" ] || die "checkpoint revalidation runner is not staged"
  mkdir -p "$RESULTS/controller"
  "${UV[@]}" python3 - "$RESULTS/controller/new-controller-provenance.json" <<'PY'
import json
import sys
from pathlib import Path
from capability_pipeline.checkpoint_revalidation import PROVENANCE_SCHEMA, running_controller_provenance

Path(sys.argv[1]).write_text(json.dumps({"schema_version": PROVENANCE_SCHEMA, **running_controller_provenance()}, sort_keys=True, indent=2) + "\n")
PY
  "${UV[@]}" python3 "$HERE/scripts/run_checkpoint_revalidation.py" \
    --bundle "$SOURCE" --out "$RESULTS/checkpoint-revalidation" \
    --expected-bundle-manifest-sha256 "$SOURCE_SHA" \
    --expected-request-sha256 "$PLAN_SHA" \
    --new-controller-provenance "$RESULTS/controller/new-controller-provenance.json" \
    --taskcompendium-source "$TASKCOMPENDIUM_SOURCE" \
    --daytona-tools "$CAPABILITY_DAYTONA_TOOLS" \
    --research-overlay "$CAPABILITY_OMP_CONFIG"
elif [ "$PHASE" = image-migration-revalidation ]; then
  MIGRATION_CANDIDATE_IMAGE='envreg.208261-marin-gpu.coreweave.app/capability-env-gen/c32-geometry-topology-repair-candidate@sha256:dcc31498e37351639cd2eaa89910388fb2f8cd5ccc0e97d272cdbfa655b08d1b'
  MIGRATION_VERIFIER_IMAGE='envreg.208261-marin-gpu.coreweave.app/capability-env-gen/c32-geometry-topology-repair-verifier@sha256:25cda6cf8b745cbc02614069ee165e9af4816f672ffae7b41ed3d640c707ce29'
  MIGRATION_MANIFEST_SHA256='641ed6358c3d307246972f77557c15b2ae6dcb951987894046f85f4a1270837e'
  MIGRATION_RECEIPT_SHA256='200e07dd7774e31ee11547651f10494235eddcce6b7cb3d6bae8a5e0f8973428'
  [ -f "$HERE/scripts/run_image_migration_revalidation.py" ] || die "image migration runner is not staged"
  [ -d "$SOURCE" ] || die "reviewed image migration bundle is not staged"
  TCS_VENV="$WORK_ROOT/taskcompendium-venv"
  OVERLAY="$(PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${UV[@]}" python3 -c 'from pathlib import Path; from capability_pipeline.synthesis import OfficialToolchain; print(OfficialToolchain.resolve(Path("/tmp")).package_root)')"
  [ -f "$OVERLAY/pyproject.toml" ] || die "image migration TaskCompendium overlay was not created"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv sync --project "$OVERLAY" --extra harbor --frozen 2>&1 | tee "$RESULTS/taskcompendium-uv-sync.log"
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$TCS_VENV" uv run --project "$OVERLAY" --frozen --extra harbor --prerelease=allow --with daytona==0.200.2 \
    "$HERE/scripts/run_image_migration_revalidation.py" \
    --bundle "$SOURCE" \
    --out "$RESULTS/revalidation" \
    --expected-candidate-image "$MIGRATION_CANDIDATE_IMAGE" \
    --expected-verifier-image "$MIGRATION_VERIFIER_IMAGE" \
    --expected-bundle-manifest-sha256 "$MIGRATION_MANIFEST_SHA256" \
    --expected-migration-receipt-sha256 "$MIGRATION_RECEIPT_SHA256" \
    --taskcompendium-source "$TASKCOMPENDIUM_SOURCE" \
    --daytona-tools "$CAPABILITY_DAYTONA_TOOLS" \
    --research-overlay "$CAPABILITY_OMP_CONFIG" \
    --model glm-orion/glm-5.3 \
    --session-time 28800 \
    --max-continuations 8 \
    --validation-timeout 14400
elif [ "$PHASE" = evaluate ]; then
  [ "$DISPATCH_LIMIT" -le 3 ] || die "evaluation concurrency cannot exceed 3"
  # The controller only validates and launches. OfficialToolchain creates the
  # separate pinned Harbor/Daytona environment for each untrusted runtime run.
  "${UV[@]}" python3 -m capability_pipeline.cli evaluate \
    --plan "$SOURCE/plan.json" --plan-sha256 "$PLAN_SHA" \
    --out "$RESULTS/evaluation" --taskcompendium-source "$TASKCOMPENDIUM_SOURCE" \
    --concurrency "$DISPATCH_LIMIT"
elif [ "$PHASE" = regrade ]; then
  [ "$DISPATCH_LIMIT" -le 8 ] || die "regrade parallelism cannot exceed 8"
  REGRADE_BINDING_KIND="$("${UV[@]}" python3 - "$SOURCE/input/bundle/binding.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["environment"]["kind"])
PY
)"
  if [ "$REGRADE_BINDING_KIND" = shellsim ]; then
    command -v cargo >/dev/null 2>&1 || die "ShellSim regrade requires cargo"
    SHELLSIM_OVERLAY="$WORK_ROOT/taskcompendium-shellsim-overlay"
    SHELLSIM_TARGET="$WORK_ROOT/taskcompendium-shellsim-target"
    "${UV[@]}" python3 - "$TASKCOMPENDIUM_SOURCE" "$SHELLSIM_OVERLAY" <<'PY'
import sys
from pathlib import Path
from capability_pipeline.shellsim_snapshot_extension import apply_overlay
apply_overlay(Path(sys.argv[1]), Path(sys.argv[2]))
PY
    (cd "$SHELLSIM_OVERLAY/shellsim-bridge" && cargo build --locked --release --target-dir "$SHELLSIM_TARGET")
    export TASKCOMPENDIUM_SHELLSIM_BRIDGE="$SHELLSIM_TARGET/release/taskcompendium-shellsim"
    [ -x "$TASKCOMPENDIUM_SHELLSIM_BRIDGE" ] || die "ShellSim regrade bridge is not executable"
  fi
  PLANNED_PARALLELISM="$("${UV[@]}" python3 - "$SOURCE/plan.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["parallelism"])
PY
)"
  [ "$PLANNED_PARALLELISM" = "$DISPATCH_LIMIT" ] || die "regrade concurrency differs from its frozen plan"
  export CAPABILITY_REMOTE_REGRADE=1
  "${UV[@]}" python3 -m capability_pipeline.cli regrade \
    --source "$SOURCE" --plan-sha256 "$PLAN_SHA" \
    --out "$RESULTS/regrade" --taskcompendium-source "$TASKCOMPENDIUM_SOURCE"
else
  # The unattended synthesis controller invokes the same remote-only fixed
  # grading diagnostic as the standalone regrade stage. Candidate and private
  # verifier code still execute inside separate Daytona sandboxes.
  export CAPABILITY_REMOTE_REGRADE=1
  CLI=("${UV[@]}" python3 -m capability_pipeline.cli "$PHASE" --out "$RESULTS" --concurrency "$DISPATCH_LIMIT" --tier "$TIER")
  [ -n "$PILOT" ] && CLI+=(--pilot "$PILOT")
  if [ -n "$SOURCE" ]; then
    case "$PHASE" in
      propose) CLI+=(--seed-run "$SOURCE") ;;
      admit) CLI+=(--candidates "$SOURCE/accepted.json") ;;
      *) CLI+=(--accepted "$SOURCE/accepted.json") ;;
    esac
  fi
  if [ -n "$ADOPTION_TRANSPORT" ]; then
    CLI+=(--adopt-proposal-checkpoint "$ADOPTION_ROOT/checkpoint" \
      --adoption-source-archive "$ADOPTION_ROOT/source-archive.tar.gz" \
      --adoption-launch-receipt "$ADOPTION_ROOT/launch-receipt.json")
  fi
  CLI+=("${EXTRA_ARGS[@]}")
  PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "${CLI[@]}"
fi
