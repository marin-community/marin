#!/bin/bash
# Build the JAX and PyTorch ROCm venvs and fetch MAMF-finder for the AMD matmul ceiling runs.
# Run once on the AMD HPC Fund login node from the synced checkout. Everything lands in
# $WORK/agents/<checkout folder name>/, because /home1 is too small for two ROCm venvs.
# --no-config keeps the repo's [tool.uv] settings, which route torch to the CPU index, out of these venvs.
set -euo pipefail

ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/
ROCM_VERSION=10.0.0
JAX_VERSION=0.11.0
# PyTorch build against the same ROCm release as the JAX plugin, so both use the same hipBLASLt.
TORCH_VERSION=2.13.0
MAMF_SHA=0359db89793c313e90e4f8a8bc8a2b1514ba00ae
MAMF_URL=https://raw.githubusercontent.com/stas00/ml-engineering/$MAMF_SHA/compute/accelerator/benchmarks/mamf-finder.py

checkout=$(cd "$(dirname "$0")/../.." && pwd)
agent_work="$WORK/agents/$(basename "$checkout")"
mkdir -p "$agent_work"/{logs,results,mamf}

uv venv --no-config --allow-existing -p 3.12 "$agent_work/venv-jax"
uv pip install --no-config --python "$agent_work/venv-jax/bin/python" --index-url "$ROCM_INDEX" \
  "rocm[libraries,device-gfx942,device-gfx950]==$ROCM_VERSION" \
  "jax_rocm10_plugin==$JAX_VERSION+rocm$ROCM_VERSION" \
  "jax_rocm10_pjrt==$JAX_VERSION+rocm$ROCM_VERSION"
uv pip install --no-config --python "$agent_work/venv-jax/bin/python" "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" numpy

uv venv --no-config --allow-existing -p 3.12 "$agent_work/venv-torch"
uv pip install --no-config --python "$agent_work/venv-torch/bin/python" --index-url "$ROCM_INDEX" \
  "torch==$TORCH_VERSION+rocm$ROCM_VERSION" \
  "amd-torch-device-gfx942==$TORCH_VERSION+rocm$ROCM_VERSION" \
  "amd-torch-device-gfx950==$TORCH_VERSION+rocm$ROCM_VERSION" \
  numpy
uv pip install --no-config --python "$agent_work/venv-torch/bin/python" packaging
# MAMF-finder reads power and clocks through amdsmi. Torch loads the ROCm SDK's own libamd_smi, and a second copy
# from /opt/rocm makes amdsmi report no GPUs, so use the SDK's binding in place: it finds the SDK library relative
# to its own file. A .pth file still applies under `python -I`, which MAMF runs with.
"$agent_work/venv-torch/bin/python" -I -c "import _rocm_sdk_core, pathlib, sysconfig
sdk_smi = pathlib.Path(_rocm_sdk_core.__file__).parent / 'share' / 'amd_smi'
pathlib.Path(sysconfig.get_path('purelib'), 'rocm_sdk_amdsmi.pth').write_text(f'{sdk_smi}\n')"

curl -fsSL -o "$agent_work/mamf/mamf-finder.py" "$MAMF_URL"
echo "$MAMF_SHA" >"$agent_work/mamf/SOURCE_SHA"
echo "Environments ready in $agent_work"
