#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXPECTED_XLA_REVISION="385ce9a909ddbd223ec8a9a92ca9b1ab6dc42aed"
readonly DIAGNOSTIC_PATCH="/app/experiments/grug/moe/standalone/ngc_xla_kernel_cache_diagnostic.patch"
readonly FIX_PATCH="/app/experiments/grug/moe/standalone/ngc_xla_kernel_cache_fix.patch"
readonly JAX_SOURCE="/opt/jax"
readonly XLA_SOURCE="/opt/xla"
readonly BAZEL_TARGET="@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
readonly BAZEL_ARTIFACT="/opt/jax/bazel-bin/external/xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
readonly ARTIFACT_PREFIX="${1:?usage: $0 S3_ARTIFACT_PREFIX}"
readonly BUILD_VARIANT="${2:-diagnostic}"
readonly BUILD_JOBS="${MARIN_NGC_XLA_BUILD_JOBS:-32}"

artifact_dir="$(mktemp -d)"
readonly artifact_dir
readonly artifact="${artifact_dir}/xla_cuda_plugin-${BUILD_VARIANT}.so"

if [[ "${BUILD_VARIANT}" != "diagnostic" && "${BUILD_VARIANT}" != "fix" ]]; then
  echo "variant must be diagnostic or fix, got ${BUILD_VARIANT}" >&2
  exit 1
fi

actual_revision="$(git -C "${XLA_SOURCE}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${EXPECTED_XLA_REVISION}" ]]; then
  echo "expected XLA ${EXPECTED_XLA_REVISION}, got ${actual_revision}" >&2
  exit 1
fi

git -C "${XLA_SOURCE}" apply --check "${DIAGNOSTIC_PATCH}"
git -C "${XLA_SOURCE}" apply "${DIAGNOSTIC_PATCH}"
if [[ "${BUILD_VARIANT}" == "fix" ]]; then
  git -C "${XLA_SOURCE}" apply --check "${FIX_PATCH}"
  git -C "${XLA_SOURCE}" apply "${FIX_PATCH}"
fi
cd "${JAX_SOURCE}"
echo "building ${BUILD_VARIANT} CUDA PJRT plugin"
bazel --batch --bazelrc="${JAX_SOURCE}/.bazelrc" \
  --output_user_root="${artifact_dir}/bazel-root" \
  build --jobs="${BUILD_JOBS}" "${BAZEL_TARGET}"
echo "copying ${BUILD_VARIANT} CUDA PJRT plugin"
test -f "${BAZEL_ARTIFACT}"
cp "${BAZEL_ARTIFACT}" "${artifact}"

echo "uploading ${BUILD_VARIANT} CUDA PJRT plugin and manifest"
/usr/bin/python -m pip install --disable-pip-version-check "s3fs==2026.1.0"
/usr/bin/python - "${artifact}" "${ARTIFACT_PREFIX}" "${actual_revision}" "${BUILD_VARIANT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import fsspec

artifact = Path(sys.argv[1])
prefix = sys.argv[2].rstrip("/")
xla_revision = sys.argv[3]
variant = sys.argv[4]


def upload(path: Path, destination: str) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source, fsspec.open(destination, "wb") as target:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
            target.write(chunk)
    return {"uri": destination, "sha256": digest.hexdigest(), "bytes": size}


manifest = {
    "ngc_image": "nvcr.io/nvidia/jax:26.06-py3",
    "xla_revision": xla_revision,
    "upstream_fix": "https://github.com/openxla/xla/commit/4c1b00509e646d13a9cf443cd10c866810ed923d",
    "variant": variant,
    "artifact": upload(artifact, f"{prefix}/{variant}/xla_cuda_plugin.so"),
}
manifest_uri = f"{prefix}/{variant}/manifest.json"
with fsspec.open(manifest_uri, "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print(json.dumps({"manifest_uri": manifest_uri, **manifest}, sort_keys=True))
PY
