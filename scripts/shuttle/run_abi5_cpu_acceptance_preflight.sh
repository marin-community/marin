#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly JOB_ROOT="${PWD}"
readonly MANIFEST="${JOB_ROOT}/acceptance-manifest.json"
readonly RESOLVED_LAUNCH="${JOB_ROOT}/resolved-launch.json"

if [[ ! -f "${MANIFEST}" ]]; then
  printf 'ABI 5 acceptance manifest is missing\n' >&2
  exit 2
fi
if [[ ! -f "${RESOLVED_LAUNCH}" ]]; then
  printf 'ABI 5 capsule is not launch-ready: resolved-launch.json is absent\n' >&2
  exit 2
fi

python3 - "${MANIFEST}" "${RESOLVED_LAUNCH}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result

manifest = json.loads(Path(sys.argv[1]).read_text(), object_pairs_hook=strict_object)
resolved = json.loads(Path(sys.argv[2]).read_text(), object_pairs_hook=strict_object)
if manifest["launch_ready"]:
    raise SystemExit("checked-in source manifest must remain non-launch-ready")
if manifest["retry_limits"] != {"failure": 0, "preemption": 0, "task_failure": 0}:
    raise SystemExit("all three retry ceilings must be zero")
required = set(manifest["unresolved_external_identities"])
if set(resolved) != required:
    raise SystemExit("resolved launch identities do not exactly close the reviewed placeholder set")
digest = re.compile(r"^sha256:[0-9a-f]{64}$")
for field in ("task_image_oci_digest", "init_image_oci_digest"):
    if not digest.fullmatch(resolved[field]):
        raise SystemExit(f"{field} must be an immutable OCI digest")
for field in ("bundle_content_sha256", "iris_config_sha256", "linux_dependency_lock_sha256"):
    if not re.fullmatch(r"[0-9a-f]{64}", resolved[field]):
        raise SystemExit(f"{field} must be a SHA-256")
for field in (
    "bundle_init_pinning_implementation_review",
    "exact_bundle_blob_submission_review",
    "iris_revision",
    "minimal_execution_environment_policy_review",
    "runner_implementation_review",
):
    if not re.fullmatch(r"[0-9a-f]{40}", resolved[field]):
        raise SystemExit(f"{field} must be a commit SHA")
if not re.fullmatch(r"CPython 3\.12\.[0-9]+ \([^)]+\) sha256:[0-9a-f]{64}", resolved["linux_python_identity"]):
    raise SystemExit("linux_python_identity must bind the exact CPython patch, build, and binary digest")
if resolved["bundle_content_sha256"] != os.environ.get("IRIS_BUNDLE_ID", ""):
    raise SystemExit("IRIS_BUNDLE_ID differs from the reviewed bundle digest")
PY

for variable in HF_TOKEN WANDB_API_KEY GCS_RESOLVE_REFRESH_SECS MARIN_PROVENANCE; do
  if [[ -n "${!variable-}" ]]; then
    printf 'forbidden inherited variable is set: %s\n' "${variable}" >&2
    exit 2
  fi
done
test ! -e "${JOB_ROOT}/.marin.yaml"
test ! -e "${JOB_ROOT}/coreweave.yaml"

printf 'ABI 5 external execution is intentionally not implemented by local preparation sources.\n' >&2
printf 'A separately reviewed runner commit must replace this stop after all identities resolve.\n' >&2
exit 2
