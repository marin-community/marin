#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Set up the GPU environment of a Baby RSI Iris task, then run Python with the arguments, for example
# `job.sh -m experiments.post_training.baby_rsi.run baseline ...`.
#
# vLLM runs from its own virtual environment, because its Torch pin differs from the task's environment. The task's
# environment gets the tokenizer library and pandas beside marin-taskforge. Submit with
# `--sync-package marin-taskforge --container-profile container_profile_privileged` (bubblewrap needs namespaces).
set -euo pipefail

# A privileged task sees every GPU of its node, but owns only the GPUs that Kubernetes assigned to it. The device
# plugin names them in NVIDIA_VISIBLE_DEVICES, or, with volume mounts, by the file names in the directory that it
# names. vLLM reads CUDA_VISIBLE_DEVICES as indices, so the UUIDs become the indices of nvidia-smi, in PCI bus order.
assigned="${NVIDIA_VISIBLE_DEVICES:-}"
if [ -d "$assigned" ]; then
  assigned=$(ls "$assigned" | paste -sd, -)
fi
if [ -z "$assigned" ] || [ "$assigned" = all ]; then
  echo "Cannot tell which GPUs this task owns: NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
indices=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', ' -v wanted="$assigned" 'BEGIN { n = split(wanted, list, ","); for (i = 1; i <= n; i++) keep[list[i]] = 1 }
      $2 in keep { print $1 }' | paste -sd, -)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$indices"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

VLLM_VENV=/tmp/vllm-venv
TASK_PYTHON="$PWD/.venv/bin/python"
# The Marin vLLM wheel of the SkyRL engines (vendor/skyrl/pyproject.toml), with their Torch and Transformers. It
# serves Qwen3 and the GrugMoE architecture of Snowball, and it streams s3:// weights with the Run:ai loader.
VLLM_RELEASE=https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20260929-39e62869693c
VLLM_WHEEL="$VLLM_RELEASE/vllm-0.0.0.dev20260929%2Bmarin.39e62869693c.cu132-cp38-abi3-manylinux_2_28_$(uname -m).whl"
CUDA_TOOLCHAIN=13.2.86
# The installs run outside the repository, so that its uv index settings do not select Torch builds.
(
  cd /tmp
  uv venv --quiet --python 3.12 "$VLLM_VENV"
  uv pip install --quiet --python "$VLLM_VENV/bin/python" --index-strategy unsafe-best-match \
    --index https://download.pytorch.org/whl/cu132 --index https://download.pytorch.org/whl/cpu \
    "vllm @ $VLLM_WHEEL" torch==2.13.0+cu132 transformers==5.16.1 "runai-model-streamer[s3]==0.16.1" \
    "nvidia-cuda-nvcc==$CUDA_TOOLCHAIN" "nvidia-cuda-crt==$CUDA_TOOLCHAIN" "nvidia-nvvm==$CUDA_TOOLCHAIN" ninja
  # The policy's tokenizer, and pandas for the script graders of the pool tasks.
  uv pip install --quiet --python "$TASK_PYTHON" transformers==4.57.1 pandas==2.3.3
)
# FlashInfer compiles kernels at run time, and the task image has no CUDA compiler. vLLM uses the compiler of the
# NVIDIA wheels, with the library names that the linker looks for (as marin.inference.vllm_server does).
cuda_home="$VLLM_VENV/lib/python3.12/site-packages/nvidia/cu13"
if [ ! -x "$cuda_home/bin/nvcc" ]; then
  echo "No nvcc in $cuda_home" >&2
  exit 1
fi
[ -e "$cuda_home/lib64" ] || ln -s lib "$cuda_home/lib64"
for library in libcudart libnvrtc; do
  if [ -f "$cuda_home/lib/$library.so.13" ] && [ ! -e "$cuda_home/lib/$library.so" ]; then
    ln -s "$library.so.13" "$cuda_home/lib/$library.so"
  fi
done
# The CoreWeave object store refuses path-style S3 requests, which the Run:ai loader sends by default.
printf '[default]\ns3 =\n    addressing_style = virtual\n' > "$VLLM_VENV/aws-config"
# Only the vLLM process gets these settings.
cat > "$VLLM_VENV/bin/baby-rsi-vllm" <<EOF
#!/usr/bin/env bash
# The JIT runs ninja and the compiler by name.
export CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$VLLM_VENV/bin:\$PATH" LIBRARY_PATH="$cuda_home/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}"
export AWS_CONFIG_FILE="$VLLM_VENV/aws-config"
exec "$VLLM_VENV/bin/vllm" "\$@"
EOF
chmod +x "$VLLM_VENV/bin/baby-rsi-vllm"
export BABY_RSI_VLLM="$VLLM_VENV/bin/baby-rsi-vllm"
exec .venv/bin/python "$@"
