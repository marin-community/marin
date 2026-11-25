#!/usr/bin/env bash
set -euo pipefail
spec_uri=$1
source_revision=$2
purpose=$3
# Use the published verifier fix in an environment isolated from running jobs.
export UV_CACHE_DIR
UV_CACHE_DIR=$(mktemp -d /tmp/mochi-dspark-uv.XXXXXX)
uv run --frozen --prerelease=allow \
  --with-editable vendor/skyrl \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://flashinfer.ai/whl/ \
  --index-strategy unsafe-best-match \
  --with 'torch==2.14.1+cu132' \
  --with 'torchaudio==2.11.0+cpu' \
  --with 'torchcodec==0.17.0+cu132' \
  --with 'torchvision==0.29.1+cu132' \
  --with 'cuda-toolkit[nvcc,cccl,nvrtc]==13.2.2' \
  --with 'flashinfer-cubin==0.6.18.post1' \
  --with 'runai-model-streamer[gcs,s3]==0.16.1' \
  --with 'vllm @ https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20261124-36cef9c0caf3/vllm-0.0.0.dev20261124+marin.36cef9c0caf3.cu132-cp38-abi3-manylinux_2_28_aarch64.whl' \
  --with 'speculators @ https://github.com/marin-community/speculators/releases/download/v0.8.0%2Bmarin.1/speculators-0.8.0%2Bmarin.1-py3-none-any.whl' \
  --with 'hs-connectors @ https://github.com/marin-community/speculators/releases/download/v0.8.0%2Bmarin.1/hs_connectors-0.8.0%2Bmarin.1-py3-none-any.whl' \
  python - "$spec_uri" "$source_revision" "$purpose" <<'PYTHON'
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

from iris.cluster.client.job_info import get_job_info
from marin.inference.vllm_server import _CUDA_NVCC_BOOTSTRAP, _write_virtual_hosted_s3_config
from rigging.filesystem.storage_path import StoragePath

spec_uri, revision, purpose = sys.argv[1:]
info = get_job_info()
if info is None or (info.worker_region is not None and info.worker_region.casefold() != "us-east-08a"):
    raise RuntimeError("Run only in the registered 08a Iris gang")
os.environ["AWS_CONFIG_FILE"] = _write_virtual_hosted_s3_config()
cache_root = Path(tempfile.mkdtemp(prefix="mochi-dspark-compile-"))
for variable, subdir in (("VLLM_CACHE_ROOT", "vllm"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                         ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
    os.environ[variable] = str(cache_root / subdir)
spec = json.loads(StoragePath(spec_uri).read_text())
runtime_digest = hashlib.sha256(Path("experiments/post_training/mochi_dspark_runtime.sh").read_bytes()).hexdigest()
if runtime_digest != spec["runtime_script_sha256"]:
    raise ValueError("Runtime script hash mismatch")
wheel_version = importlib.metadata.version("vllm")
if wheel_version != "0.0.0.dev20261124+marin.36cef9c0caf3.cu132":
    raise ValueError(f"Unexpected wheel: {wheel_version}")
(StoragePath(spec["output"]) / f"task-{info.task_index}-runtime.json").write_text(json.dumps({
    "source_revision": revision, "runtime_script_sha256": runtime_digest,
    "wheel_version": wheel_version, "purpose": purpose,
    "task": str(info.task_attempt), "worker_region": info.worker_region,
    "dependency_versions": {name: importlib.metadata.version(name) for name in (
        "torch", "torchaudio", "torchcodec", "torchvision", "transformers", "speculators",
        "flashinfer-python", "runai-model-streamer",
    )},
}, indent=2))
sys.argv = ["cuda-bootstrap", sys.executable, "-m", "experiments.post_training.mochi_dspark_gate",
            "--spec", spec_uri, "--source-revision", revision, "--purpose", purpose]
exec(_CUDA_NVCC_BOOTSTRAP)
PYTHON
