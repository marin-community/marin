#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly EXPECTED_CANONICAL_MARIN_SHA="6340706df454d699124fc7b676499f5db9cccd4a"
readonly EXPECTED_XLA_REVISION="9b635916ecc6df6efee62d8e4b0c7ef87ef84d69"
readonly EXPECTED_JAX_REVISION="619764c15117fbefc4ba13ab941871cb514c23f6"
readonly EXPECTED_BAZEL_VERSION="7.7.0"
readonly BAZEL_URL="https://github.com/bazelbuild/bazel/releases/download/${EXPECTED_BAZEL_VERSION}/bazel-${EXPECTED_BAZEL_VERSION}-linux-x86_64"
readonly JOB_ROOT="${PWD}"
readonly LOG_ROOT="${JOB_ROOT}/logs"
readonly TOOL_ROOT="${JOB_ROOT}/tools"
readonly SOURCE_ROOT="${JOB_ROOT}/sources"
readonly XLA_ROOT="${SOURCE_ROOT}/xla"
readonly JAX_ROOT="${SOURCE_ROOT}/jax"
readonly BAZEL_BIN="${TOOL_ROOT}/bazel-${EXPECTED_BAZEL_VERSION}"
readonly XLA_BAZEL_OUTPUT_ROOT="${JOB_ROOT}/bazel-output-xla"
readonly JAX_BAZEL_OUTPUT_ROOT="${JOB_ROOT}/bazel-output-jax"
readonly BAZEL_REPOSITORY_CACHE="${JOB_ROOT}/bazel-repository-cache"
readonly ACCEPTANCE_VENV="${JOB_ROOT}/acceptance-venv"
readonly ACCEPTANCE_WORK="${JOB_ROOT}/acceptance-work"
readonly ACCEPTANCE_REPORT="${JOB_ROOT}/acceptance-report.json"
readonly WHEEL_ROOT="${JOB_ROOT}/wheel-dist"

mkdir -p "${LOG_ROOT}" "${TOOL_ROOT}" "${SOURCE_ROOT}" \
  "${XLA_BAZEL_OUTPUT_ROOT}" "${JAX_BAZEL_OUTPUT_ROOT}" \
  "${BAZEL_REPOSITORY_CACHE}" "${WHEEL_ROOT}"

exec > >(tee "${LOG_ROOT}/run.log") 2>&1
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

finish() {
  local status=$?
  set +Eeuxo pipefail
  printf 'preflight_exit_code=%s\n' "${status}"
  printf 'preflight_finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -d "${LOG_ROOT}" ]]; then
    find "${LOG_ROOT}" -type f \
      ! -name run.log \
      ! -name log-sha256.txt \
      -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum \
      > "${LOG_ROOT}/log-sha256.txt"
    cat "${LOG_ROOT}/log-sha256.txt"
  fi
  exit "${status}"
}
trap finish EXIT

test -f "${JOB_ROOT}/manifest.env"
source "${JOB_ROOT}/manifest.env"
test "${CANONICAL_MARIN_SHA}" = "${EXPECTED_CANONICAL_MARIN_SHA}"
test "${XLA_REVISION}" = "${EXPECTED_XLA_REVISION}"
test "${JAX_REVISION}" = "${EXPECTED_JAX_REVISION}"
test "${BAZEL_VERSION}" = "${EXPECTED_BAZEL_VERSION}"
test "${SOURCE_REVIEW_STATUS}" = "GO"
test "${XLA_PATCH_REVIEW_STATUS}" = "GO"
test "${JAX_PATCH_REVIEW_STATUS}" = "GO"
test "${ACCEPTANCE_REVIEW_STATUS}" = "GO"
test -d "${JOB_ROOT}/lib/shuttle/mlir"
test -f "${JOB_ROOT}/lib/shuttle/pyproject.toml"
test -d "${JOB_ROOT}/lib/shuttle/src/shuttle"
test ! -e "${JOB_ROOT}/lib/shuttle/mlir/artifacts"
test ! -e "${JOB_ROOT}/config"
test ! -e "${JOB_ROOT}/coreweave.yaml"

readonly BUNDLED_SOURCE_FILE_COUNT="$(
  find "${JOB_ROOT}/lib/shuttle" -type f | wc -l | tr -d '[:space:]'
)"
test "${BUNDLED_SOURCE_FILE_COUNT}" = "${BUNDLED_SHUTTLE_FILE_COUNT}"
readonly BUNDLED_MLIR_INPUT_COUNT="$(
  find "${JOB_ROOT}/lib/shuttle/mlir/test/Inputs" -maxdepth 1 -type f -name '*.mlir' \
    | wc -l \
    | tr -d '[:space:]'
)"
test "${BUNDLED_MLIR_INPUT_COUNT}" = "${EXPECTED_MLIR_INPUT_COUNT}"
test -f "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/shuttle_jaxlib_acceptance.py"
test -f "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/verify_acceptance_patch.py"
test -f "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/verify_acceptance_fixture_oracles.py"
test -f "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/fixture_audit_gate.py"
test -f "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/test_fixture_audit_gate.py"

readonly -a XLA_PATCHES=(
  "${JOB_ROOT}/lib/shuttle/mlir/xla_patch/0001-add-stablehlo-module-transform-hook.patch"
  "${JOB_ROOT}/lib/shuttle/mlir/xla_patch/0002-anchor-lit-labels-to-xla-repository.patch"
)
readonly -a JAX_PATCHES=(
  "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/0001-link-shuttle-xla-registry-adapter.patch"
  "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/0002-add-acceptance-observer-bridge.patch"
)
for patch in "${XLA_PATCHES[@]}" "${JAX_PATCHES[@]}"; do
  test -f "${patch}"
done

printf 'preflight_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'canonical_marin_sha=%s\n' "${CANONICAL_MARIN_SHA}"
printf 'xla_revision=%s\n' "${XLA_REVISION}"
printf 'jax_revision=%s\n' "${JAX_REVISION}"
printf 'bazel_version=%s\n' "${BAZEL_VERSION}"
printf 'cpu_limit=%s\n' "${PREFLIGHT_BAZEL_JOBS}"
printf 'bazel_ram_limit_mb=%s\n' "${PREFLIGHT_BAZEL_RAM_MB}"
printf 'bundled_shuttle_file_count=%s\n' "${BUNDLED_SOURCE_FILE_COUNT}"
printf 'expected_mlir_test_count=%s\n' "${EXPECTED_MLIR_TEST_COUNT}"
printf 'bundled_mlir_input_count=%s\n' "${BUNDLED_MLIR_INPUT_COUNT}"
printf 'expected_acceptance_test_count=%s\n' "${EXPECTED_ACCEPTANCE_TEST_COUNT}"

uname -a
test -f /etc/os-release && sed -n '1,120p' /etc/os-release
getconf _NPROCESSORS_ONLN
cc --version
c++ --version
ld --version
git --version
python3 --version
curl --version

find "${JOB_ROOT}/lib/shuttle" -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | tee "${LOG_ROOT}/bundled-source-sha256.txt"
sha256sum "${XLA_PATCHES[@]}" | tee "${LOG_ROOT}/xla-patch-sha256.txt"
sha256sum "${JAX_PATCHES[@]}" | tee "${LOG_ROOT}/jax-patch-sha256.txt"

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

git init "${JAX_ROOT}"
git -C "${JAX_ROOT}" remote add origin https://github.com/jax-ml/jax.git
git -C "${JAX_ROOT}" fetch --depth 1 origin "${EXPECTED_JAX_REVISION}"
git -C "${JAX_ROOT}" checkout --detach FETCH_HEAD
test "$(git -C "${JAX_ROOT}" rev-parse HEAD)" = "${EXPECTED_JAX_REVISION}"

for patch in "${XLA_PATCHES[@]}"; do
  git -C "${XLA_ROOT}" apply --check "${patch}"
  git -C "${XLA_ROOT}" apply "${patch}"
done
for ((patch_index=${#XLA_PATCHES[@]} - 1; patch_index >= 0; patch_index--)); do
  git -C "${XLA_ROOT}" apply --reverse --check "${XLA_PATCHES[patch_index]}"
done
git -C "${XLA_ROOT}" diff --check
git -C "${XLA_ROOT}" diff --stat | tee "${LOG_ROOT}/xla-patch-stat.txt"
git -C "${XLA_ROOT}" diff --binary > "${LOG_ROOT}/applied-xla.patch"

for patch in "${JAX_PATCHES[@]}"; do
  git -C "${JAX_ROOT}" apply --check "${patch}"
  git -C "${JAX_ROOT}" apply "${patch}"
done
for ((patch_index=${#JAX_PATCHES[@]} - 1; patch_index >= 0; patch_index--)); do
  git -C "${JAX_ROOT}" apply --reverse --check "${JAX_PATCHES[patch_index]}"
done
git -C "${JAX_ROOT}" diff --check
git -C "${JAX_ROOT}" diff --stat | tee "${LOG_ROOT}/jax-patch-stat.txt"
git -C "${JAX_ROOT}" diff --binary > "${LOG_ROOT}/applied-jax.patch"

readonly -a EXPECTED_LIT_RUNTIME_LABELS=(
  'Label("//xla/backends/gpu/target_config:all_gpu_specs")'
  'Label("//xla:lit.cfg.py")'
  'Label("//xla:sh_test_with_runfiles.py")'
  'Label("//xla/stream_executor/cuda:all_runtime")'
  'Label("//xla/tsl/cuda:nvshmem_stub")'
  'Label("//xla/tsl/cuda:nccl")'
  'Label("//xla:lit_google_cfg.py")'
)
{
  for label in "${EXPECTED_LIT_RUNTIME_LABELS[@]}"; do
    grep -Fn "${label}" "${XLA_ROOT}/xla/lit.bzl"
  done
  readonly LIT_RUNTIME_LABEL_COUNT="$(grep -Fc 'Label("//xla' "${XLA_ROOT}/xla/lit.bzl")"
  printf 'anchored_xla_runtime_label_count=%s\n' "${LIT_RUNTIME_LABEL_COUNT}"
  test "${LIT_RUNTIME_LABEL_COUNT}" = "7"
} | tee "${LOG_ROOT}/lit-label-proof.txt"

python3 -m venv "${ACCEPTANCE_VENV}"
readonly ACCEPTANCE_PYTHON="${ACCEPTANCE_VENV}/bin/python"
"${ACCEPTANCE_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${ACCEPTANCE_PYTHON}" -m pip install \
  'numpy>=2.0' \
  'scipy>=1.14' \
  'ml_dtypes>=0.5.0' \
  opt_einsum \
  'jax==0.10.1' \
  'jaxlib==0.10.1' \
  'pytest>=8.4'

PYTHONPATH="${JOB_ROOT}/lib/shuttle/mlir/jax_patch" \
  "${ACCEPTANCE_PYTHON}" -m pytest -q \
    "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/test_acceptance_contract.py" \
    "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/test_verify_acceptance_patch.py" \
    "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/test_fixture_audit_gate.py" \
    2>&1 | tee "${LOG_ROOT}/acceptance-contract-pytest.log"
grep -F "${EXPECTED_ACCEPTANCE_TEST_COUNT} passed" \
  "${LOG_ROOT}/acceptance-contract-pytest.log"
PYTHONPATH="${JOB_ROOT}/lib/shuttle/mlir/jax_patch" \
  "${ACCEPTANCE_PYTHON}" \
    "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/verify_acceptance_fixture_oracles.py" \
    2>&1 | tee "${LOG_ROOT}/acceptance-fixture-oracle-proof.log"

HERMETIC_PYTHON_VERSION=3.12 "${ACCEPTANCE_PYTHON}" \
  "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/verify_acceptance_patch.py" \
  --bazel "${BAZEL_BIN}" \
  --jax-source "${JAX_ROOT}" \
  --xla-source "${XLA_ROOT}" \
  --shuttle-mlir "${JOB_ROOT}/lib/shuttle/mlir" \
  --output-user-root "${JAX_BAZEL_OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/jax-acceptance-cquery-proof.log"
printf 'jax_acceptance_cquery_proof=PASS\n' \
  | tee -a "${LOG_ROOT}/jax-acceptance-cquery-proof.log"

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

readonly -a XLA_BAZEL_STARTUP_FLAGS=(
  "--output_user_root=${XLA_BAZEL_OUTPUT_ROOT}"
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
"${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" release
readonly BAZEL_JAVA_HOME="$(
  "${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" java-home
)"
printf 'bazel_java_home=%s\n' "${BAZEL_JAVA_HOME}"
"${BAZEL_JAVA_HOME}/bin/java" -version
"${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" execution_root
"${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" info "${BAZEL_COMMAND_FLAGS[@]}" output_base

readonly -a SHUTTLE_BUILD_GATES=(
  shuttle_ops_inc_gen
  ShuttleDialect
  ShuttlePasses
  ShuttleXlaRegistration
  ShuttleXlaRegistryAdapter
  ShuttleObserverTestBridge
  ShuttlePythonObserverTestBridge
  shuttle-opt
)
for gate in "${SHUTTLE_BUILD_GATES[@]}"; do
  "${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" build "${BAZEL_COMMAND_FLAGS[@]}" \
    "@shuttle_mlir//:${gate}" \
    2>&1 | tee "${LOG_ROOT}/bazel-build-${gate}.log"
done

"${ACCEPTANCE_PYTHON}" \
  "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/fixture_audit_gate.py" \
  --bazel "${BAZEL_BIN}" \
  --xla-source "${XLA_ROOT}" \
  --output-user-root "${XLA_BAZEL_OUTPUT_ROOT}" \
  --repository-cache "${BAZEL_REPOSITORY_CACHE}" \
  --jobs "${PREFLIGHT_BAZEL_JOBS}" \
  --ram-mb "${PREFLIGHT_BAZEL_RAM_MB}" \
  --python "${ACCEPTANCE_PYTHON}" \
  --generator "${JOB_ROOT}/lib/shuttle/mlir/test/Inputs/regenerate-jax-fixtures.py" \
  2>&1 | tee "${LOG_ROOT}/six-fixture-default-audit.log"
grep -F 'fixture_audit_normalizer=' "${LOG_ROOT}/six-fixture-default-audit.log"
grep -F '/shuttle-test-opt' "${LOG_ROOT}/six-fixture-default-audit.log"
grep -Fx 'six_fixture_default_audit=PASS' "${LOG_ROOT}/six-fixture-default-audit.log"

readonly -a SHUTTLE_NATIVE_TESTS=(
  xla_registration_test
  xla_registry_adapter_test
  pipeline_observer_test
  observer_test_bridge_test
)
for test_target in "${SHUTTLE_NATIVE_TESTS[@]}"; do
  "${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" test "${BAZEL_COMMAND_FLAGS[@]}" \
    --cache_test_results=no \
    --test_output=errors \
    "@shuttle_mlir//:${test_target}" \
    2>&1 | tee "${LOG_ROOT}/bazel-test-${test_target}.log"
done

"${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" build "${BAZEL_COMMAND_FLAGS[@]}" \
  @shuttle_mlir//:mlir_tests \
  2>&1 | tee "${LOG_ROOT}/bazel-build-shuttle-mlir-tests.log"
"${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" test "${BAZEL_COMMAND_FLAGS[@]}" \
  --cache_test_results=no \
  --test_output=errors \
  @shuttle_mlir//:mlir_tests \
  2>&1 | tee "${LOG_ROOT}/bazel-test-shuttle-mlir.log"
grep -F "Executed ${EXPECTED_MLIR_TEST_COUNT} out of ${EXPECTED_MLIR_TEST_COUNT} tests: ${EXPECTED_MLIR_TEST_COUNT} tests pass." \
  "${LOG_ROOT}/bazel-test-shuttle-mlir.log"

readonly -a XLA_TEST_TARGETS=(
  //xla/pjrt:stablehlo_module_transform_test
  //xla/pjrt:mlir_to_hlo_test
  //xla/pjrt:mlir_to_hlo_unregistered_transform_test
  //xla/pjrt:pjrt_executable_test
)
for test_target in "${XLA_TEST_TARGETS[@]}"; do
  test_name="${test_target##*:}"
  "${BAZEL_BIN}" "${XLA_BAZEL_STARTUP_FLAGS[@]}" test "${BAZEL_COMMAND_FLAGS[@]}" \
    --cache_test_results=no \
    --test_output=errors \
    "${test_target}" \
    2>&1 | tee "${LOG_ROOT}/bazel-test-xla-${test_name}.log"
done

cd "${JAX_ROOT}"
"${ACCEPTANCE_PYTHON}" build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --bazel_path="${BAZEL_BIN}" \
  --output_path="${WHEEL_ROOT}" \
  --bazel_startup_options="--output_user_root=${JAX_BAZEL_OUTPUT_ROOT}" \
  --bazel_options="--override_repository=xla=${XLA_ROOT}" \
  --bazel_options="--override_repository=shuttle_mlir=${JOB_ROOT}/lib/shuttle/mlir" \
  --bazel_options=--define=SHUTTLE_TEST_OBSERVER=1 \
  --bazel_options=--repo_env=ML_WHEEL_TYPE=release \
  --bazel_options="--repository_cache=${BAZEL_REPOSITORY_CACHE}" \
  --bazel_options="--jobs=${PREFLIGHT_BAZEL_JOBS}" \
  --bazel_options="--local_cpu_resources=${PREFLIGHT_BAZEL_JOBS}" \
  --bazel_options="--local_ram_resources=${PREFLIGHT_BAZEL_RAM_MB}" \
  --bazel_options=--noshow_progress \
  --bazel_options=--show_result=0 \
  2>&1 | tee "${LOG_ROOT}/jaxlib-wheel-build.log"

shopt -s nullglob
readonly -a JAXLIB_WHEELS=("${WHEEL_ROOT}"/jaxlib-0.10.1-cp312-*.whl)
test "${#JAXLIB_WHEELS[@]}" = "1"
readonly JAXLIB_WHEEL="${JAXLIB_WHEELS[0]}"
sha256sum "${JAXLIB_WHEEL}" | tee "${LOG_ROOT}/jaxlib-wheel-sha256.txt"
"${ACCEPTANCE_PYTHON}" -m zipfile -l "${JAXLIB_WHEEL}" \
  | tee "${LOG_ROOT}/jaxlib-wheel-contents.txt"

"${ACCEPTANCE_PYTHON}" -m pip install --no-deps --force-reinstall "${JAXLIB_WHEEL}"
JAX_RELEASE=1 "${ACCEPTANCE_PYTHON}" -m pip install --no-deps "${JAX_ROOT}"
"${ACCEPTANCE_PYTHON}" -m pip install --no-deps "${JOB_ROOT}/lib/shuttle"
"${ACCEPTANCE_PYTHON}" -m pip check | tee "${LOG_ROOT}/acceptance-pip-check.txt"
"${ACCEPTANCE_PYTHON}" -m pip freeze | LC_ALL=C sort \
  | tee "${LOG_ROOT}/acceptance-pip-freeze.txt"
"${ACCEPTANCE_PYTHON}" -c \
  'import jax, jaxlib, shuttle; print(f"jax={jax.__version__}"); print(f"jaxlib={jaxlib.__version__}"); print(f"shuttle={shuttle.__file__}")' \
  | tee "${LOG_ROOT}/acceptance-versions.txt"

test ! -e "${ACCEPTANCE_WORK}"
JAX_PLATFORMS=cpu \
PYTHONPATH="${JOB_ROOT}/lib/shuttle/mlir/jax_patch" \
  "${ACCEPTANCE_PYTHON}" \
    "${JOB_ROOT}/lib/shuttle/mlir/jax_patch/shuttle_jaxlib_acceptance.py" \
    --work-directory "${ACCEPTANCE_WORK}" \
    --report "${ACCEPTANCE_REPORT}" \
    2>&1 | tee "${LOG_ROOT}/jaxlib-acceptance.log"

cp "${ACCEPTANCE_REPORT}" "${LOG_ROOT}/acceptance-report.json"
find "${ACCEPTANCE_WORK}" -maxdepth 2 -type f -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  | tee "${LOG_ROOT}/acceptance-work-sha256.txt"
for report in \
  "${ACCEPTANCE_REPORT}" \
  "${ACCEPTANCE_WORK}/baseline.json" \
  "${ACCEPTANCE_WORK}/context_manager.json" \
  "${ACCEPTANCE_WORK}/concurrency.json" \
  "${ACCEPTANCE_WORK}/populate.json" \
  "${ACCEPTANCE_WORK}/reuse.json" \
  "${ACCEPTANCE_WORK}/cache-keys.json"; do
  printf '===== %s =====\n' "${report}"
  cat "${report}"
done
printf 'jaxlib_cpu_acceptance=PASS\n'
