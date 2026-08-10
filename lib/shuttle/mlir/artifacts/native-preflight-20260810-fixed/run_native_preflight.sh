#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly EXPECTED_XLA_REVISION="9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
readonly EXPECTED_BAZEL_VERSION="7.7.0"
readonly BAZEL_URL="https://github.com/bazelbuild/bazel/releases/download/${EXPECTED_BAZEL_VERSION}/bazel-${EXPECTED_BAZEL_VERSION}-linux-x86_64"
readonly JOB_ROOT="${PWD}"
readonly LOG_ROOT="${JOB_ROOT}/logs"
readonly TOOL_ROOT="${JOB_ROOT}/tools"
readonly SOURCE_ROOT="${JOB_ROOT}/sources"
readonly XLA_ROOT="${SOURCE_ROOT}/xla"
readonly BAZEL_BIN="${TOOL_ROOT}/bazel-${EXPECTED_BAZEL_VERSION}"
readonly BAZEL_OUTPUT_ROOT="${JOB_ROOT}/bazel-output-user-root"
readonly BAZEL_REPOSITORY_CACHE="${JOB_ROOT}/bazel-repository-cache"

mkdir -p "${LOG_ROOT}" "${TOOL_ROOT}" "${SOURCE_ROOT}" \
  "${BAZEL_OUTPUT_ROOT}" "${BAZEL_REPOSITORY_CACHE}"

exec > >(tee "${LOG_ROOT}/run.log") 2>&1
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

finish() {
  local status=$?
  set +x
  printf 'preflight_exit_code=%s\n' "${status}"
  printf 'preflight_finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "${status}"
}
trap finish EXIT

test -f "${JOB_ROOT}/manifest.env"
# The manifest is generated only after both source revisions have been reviewed.
source "${JOB_ROOT}/manifest.env"
test "${XLA_REVISION}" = "${EXPECTED_XLA_REVISION}"
test "${BAZEL_VERSION}" = "${EXPECTED_BAZEL_VERSION}"
test "${DIALECT_REVIEW_STATUS}" = "GO"
test "${XLA_PATCH_REVIEW_STATUS}" = "GO"
test "${DIALECT_MARIN_SHA}" != "UNSET"
test "${XLA_PATCH_MARIN_SHA}" != "UNSET"
test -d "${JOB_ROOT}/lib/shuttle/mlir"
test -f "${JOB_ROOT}/lib/shuttle/mlir/xla_patch/0001-add-stablehlo-module-transform-hook.patch"

printf 'preflight_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'dialect_marin_sha=%s\n' "${DIALECT_MARIN_SHA}"
printf 'xla_patch_marin_sha=%s\n' "${XLA_PATCH_MARIN_SHA}"
printf 'xla_revision=%s\n' "${XLA_REVISION}"
printf 'bazel_version=%s\n' "${BAZEL_VERSION}"
printf 'cpu_limit=%s\n' "${PREFLIGHT_BAZEL_JOBS}"
printf 'bazel_ram_limit_mb=%s\n' "${PREFLIGHT_BAZEL_RAM_MB}"

uname -a
test -f /etc/os-release && sed -n '1,120p' /etc/os-release
getconf _NPROCESSORS_ONLN
cc --version
c++ --version
ld --version
git --version
python3 --version
curl --version

find "${JOB_ROOT}/lib/shuttle/mlir" -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | tee "${LOG_ROOT}/bundled-source-sha256.txt"

curl -fL "${BAZEL_URL}" -o "${BAZEL_BIN}"
chmod 0755 "${BAZEL_BIN}"
sha256sum "${BAZEL_BIN}" | tee "${LOG_ROOT}/bazel-binary-sha256.txt"
test "$("${BAZEL_BIN}" --version)" = "bazel ${EXPECTED_BAZEL_VERSION}"
"${BAZEL_BIN}" --version

git init "${XLA_ROOT}"
git -C "${XLA_ROOT}" remote add origin https://github.com/openxla/xla.git
git -C "${XLA_ROOT}" fetch --depth 1 origin "${EXPECTED_XLA_REVISION}"
git -C "${XLA_ROOT}" checkout --detach FETCH_HEAD
test "$(git -C "${XLA_ROOT}" rev-parse HEAD)" = "${EXPECTED_XLA_REVISION}"
git -C "${XLA_ROOT}" rev-parse HEAD
git -C "${XLA_ROOT}" status --short

readonly XLA_PATCH="${JOB_ROOT}/lib/shuttle/mlir/xla_patch/0001-add-stablehlo-module-transform-hook.patch"
git -C "${XLA_ROOT}" apply --check "${XLA_PATCH}"
git -C "${XLA_ROOT}" apply "${XLA_PATCH}"
git -C "${XLA_ROOT}" diff --check
git -C "${XLA_ROOT}" diff --stat
git -C "${XLA_ROOT}" diff --binary > "${LOG_ROOT}/applied-xla.patch"

if [[ -f "${XLA_ROOT}/WORKSPACE.bazel" ]]; then
  readonly XLA_WORKSPACE_FILE="${XLA_ROOT}/WORKSPACE.bazel"
elif [[ -f "${XLA_ROOT}/WORKSPACE" ]]; then
  readonly XLA_WORKSPACE_FILE="${XLA_ROOT}/WORKSPACE"
else
  printf 'Pinned XLA checkout has no WORKSPACE or WORKSPACE.bazel\n' >&2
  exit 1
fi

printf '\nlocal_repository(\n    name = "shuttle_mlir",\n    path = "%s",\n)\n' \
  "${JOB_ROOT}/lib/shuttle/mlir" >> "${XLA_WORKSPACE_FILE}"
tail -n 12 "${XLA_WORKSPACE_FILE}"

readonly -a BAZEL_STARTUP_FLAGS=(
  "--output_user_root=${BAZEL_OUTPUT_ROOT}"
)
readonly -a BAZEL_COMMAND_FLAGS=(
  "--repository_cache=${BAZEL_REPOSITORY_CACHE}"
  "--jobs=${PREFLIGHT_BAZEL_JOBS}"
  "--local_cpu_resources=${PREFLIGHT_BAZEL_JOBS}"
  "--local_ram_resources=${PREFLIGHT_BAZEL_RAM_MB}"
  "--noshow_progress"
  "--show_result=0"
)

cd "${XLA_ROOT}"
test -f .bazelversion && sed -n '1,20p' .bazelversion
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" release
readonly BAZEL_JAVA_HOME="$("${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" java-home)"
printf 'bazel_java_home=%s\n' "${BAZEL_JAVA_HOME}"
"${BAZEL_JAVA_HOME}/bin/java" -version
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" execution_root
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" output_base

"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" build "${BAZEL_COMMAND_FLAGS[@]}" \
  @shuttle_mlir//:shuttle_ops_inc_gen \
  2>&1 | tee "${LOG_ROOT}/bazel-build-shuttle-ops-inc-gen.log"

"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" build "${BAZEL_COMMAND_FLAGS[@]}" \
  @shuttle_mlir//:shuttle-opt \
  2>&1 | tee "${LOG_ROOT}/bazel-build-shuttle-opt.log"

"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test "${BAZEL_COMMAND_FLAGS[@]}" \
  --test_output=errors \
  @shuttle_mlir//:mlir_tests \
  2>&1 | tee "${LOG_ROOT}/bazel-test-shuttle-mlir.log"

"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test "${BAZEL_COMMAND_FLAGS[@]}" \
  --test_output=errors \
  //xla/pjrt:stablehlo_module_transform_test \
  //xla/pjrt:mlir_to_hlo_test \
  //xla/pjrt:mlir_to_hlo_unregistered_transform_test \
  //xla/pjrt:pjrt_executable_test \
  2>&1 | tee "${LOG_ROOT}/bazel-test-xla-hook.log"

find "${LOG_ROOT}" -type f \
  ! -name run.log \
  ! -name log-sha256.txt \
  -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | tee "${LOG_ROOT}/log-sha256.txt"
