#!/usr/bin/env bash
# Build a jaxlib==0.10.0 wheel with the WGMMAOp::verify() relaxation that allows the
# mixed E4M3xE5M2 FP8 wgmma operand pair (logbook GFP8-035/036). Runs inside the Iris
# iris-task image (python:3.12-slim, root, gcc-14 present; clang + bazel installed here).
#
# Only the base `jaxlib` wheel is built -- the mosaic-GPU MLIR dialect (incl. the verifier)
# lives in jaxlib, NOT in jax-cuda13-plugin, so this CPU-only build is paired at runtime
# with the STOCK jax-cuda13-plugin==0.10.0 for actual H100 execution.
#
# Produces: s3://marin-na/marin/tmp/mixed-fp8/jaxlib-0.10.0-*.whl (R2).
set -euo pipefail

JAX_TAG="jax-v0.10.0"
S3_DEST="s3://marin-na/marin/tmp/mixed-fp8/"
WORK=/tmp/jaxbuild
# bazel 7.7.0 (matches jax-v0.10.0 .bazelversion) is fetched below.

echo "===== 0. toolchain ====="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq clang
clang --version | head -1
curl -fsSL -o /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64
chmod +x /usr/local/bin/bazel
bazel --version

echo "===== 1. clone $JAX_TAG ====="
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
git clone --depth 1 --branch "$JAX_TAG" https://github.com/jax-ml/jax jax-src
cd jax-src
echo "HEAD: $(git rev-parse HEAD)"

echo "===== 2. patch WGMMAOp::verify() ====="
python3 - <<'PY'
p = "jaxlib/mosaic/dialect/gpu/mosaic_gpu.cc"
s = open(p).read()
old = (
    "  auto a_type = mlir::cast<mlir::ShapedType>(getA().getType());\n"
    "  auto b_type = getB().getType();\n"
    "  auto acc_type = getAccumulator().getType();\n"
    "\n"
    "  if (a_type.getElementType() != b_type.getElementType()) {\n"
    "    return error(\"The `a` and `b` inputs must have the same element type.\");\n"
    "  }"
)
new = (
    "  auto a_type = mlir::cast<mlir::ShapedType>(getA().getType());\n"
    "  auto b_type = getB().getType();\n"
    "  auto acc_type = getAccumulator().getType();\n"
    "\n"
    "  auto a_el = a_type.getElementType();\n"
    "  auto b_el = b_type.getElementType();\n"
    "  // FP8 is the explicit PTX-ISA exception to .atype == .btype: wgmma.mma_async\n"
    "  // accepts independent .e4m3/.e5m2 operand types on sm_90a (PTX ISA 9.7.16;\n"
    "  // CUTLASS emits ...f32.e4m3.e5m2). The ODS already lists both fp8 types as\n"
    "  // valid A/B operands; only this hand-written verifier blocked the pair.\n"
    "  bool both_fp8 =\n"
    "      llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(a_el) &&\n"
    "      llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(b_el);\n"
    "  if (a_el != b_el && !both_fp8) {\n"
    "    return error(\n"
    "        \"The `a` and `b` inputs must have the same element type \"\n"
    "        \"(except for the e4m3/e5m2 FP8 pair).\");\n"
    "  }"
)
n = s.count(old)
assert n == 1, f"expected exactly 1 anchor match, got {n}"
open(p, "w").write(s.replace(old, new))
print("patched WGMMAOp::verify() -- mixed e4m3/e5m2 fp8 now permitted")
PY
# show the patched region for the log
grep -n "both_fp8\|e4m3/e5m2 FP8 pair" jaxlib/mosaic/dialect/gpu/mosaic_gpu.cc

echo "===== 3. build jaxlib (CPU, hermetic py3.12) ====="
set +e
python3 build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --clang_path=/usr/bin/clang \
  --bazel_options=--jobs=96 \
  > /tmp/jaxlib_build.log 2>&1
rc=$?
set -e
echo "=== build rc=$rc; tail of build log ==="
tail -120 /tmp/jaxlib_build.log
if [ $rc -ne 0 ]; then
  echo "=== ERROR lines ==="; grep -iE "error:|fail|fatal|ERROR" /tmp/jaxlib_build.log | tail -60
  echo "BUILD_FAILED rc=$rc"; exit $rc
fi

echo "===== 4. locate wheel ====="
# build.py reports "Distribution path: .../dist/jaxlib-<ver>-cp312-...whl"; the version
# carries a .dev0+selfbuilt local suffix, so match jaxlib-*.whl (not jaxlib-0.10.0-*).
WHEEL="$(find "$WORK/jax-src/dist" "$WORK/jax-src" -name 'jaxlib-*.whl' 2>/dev/null | head -1)"
[ -z "$WHEEL" ] && WHEEL="$(find / -name 'jaxlib-*-cp312-*.whl' 2>/dev/null | grep -v site-packages | head -1)"
echo "WHEEL=$WHEEL"
[ -z "$WHEEL" ] && { echo "NO WHEEL FOUND"; exit 1; }
ls -la "$WHEEL"

echo "===== 5. upload to R2 ====="
# The active /app/.venv has no pip; use an ephemeral uv env with boto3 (uv is always
# present in the iris-task image). boto3 picks up AWS_ACCESS_KEY_ID/SECRET from the env.
WHEEL="$WHEEL" uv run --no-project --with boto3 python - <<'PY'
import boto3, os
wheel = os.environ["WHEEL"]
key = "marin/tmp/mixed-fp8/" + os.path.basename(wheel)
s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], region_name="auto")
s3.upload_file(wheel, "marin-na", key)
print(f"UPLOADED: s3://marin-na/{key}")
PY
echo "BUILD_OK"
