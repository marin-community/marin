# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview or submit the exact two-H100 E3.0 publication instrumentation gate."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

NODEID = "skyrl-train/tests/gpu/test_megatron_worker.py::test_megatron_policy_weight_sync[publication_stages_two_gpu]"
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "cloud/iris/bootstrap_runtime.sh",
    "skyrl-train/tests/gpu/test_megatron_worker.py",
    "skyrl-train/skyrl_train/weight_sync/publication_timing.py",
    "skyrl-train/skyrl_train/workers/megatron/megatron_worker.py",
    "skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py",
    "skyrl-train/skyrl_train/inference_engines/inference_engine_client.py",
    "skyrl-train/skyrl_train/fully_async_trainer.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--marin-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert len(args.marin_commit) == 40 and all(c in "0123456789abcdef" for c in args.marin_commit)
    assert args.job_name.startswith("async-rl-v2-publication-stage-")
    marin_root = Path(__file__).resolve().parents[2]
    assert (
        subprocess.check_output(["git", "-C", str(marin_root), "rev-parse", "HEAD"], text=True).strip()
        == args.marin_commit
    )
    assert not subprocess.check_output(["git", "-C", str(marin_root), "status", "--porcelain"], text=True)
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}
    assert "publication_stage_walls" in Path(SOURCES[-1]).read_text(), "driver trace hook must be present"
    body = f"""set -euo pipefail
publication_root="$PWD"
export PYTHONPATH="$publication_root/skyrl-train:$publication_root/skyrl-gym:$publication_root"
export HF_HOME=/tmp/oa-cache/huggingface
export UV_CACHE_DIR=/tmp/oa-cache/uv
export WANDB_MODE=disabled
export LOGURU_LEVEL=INFO
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path
expected=json.loads({json.dumps(json.dumps(hashes))})
for name,digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('PUBLICATION_SOURCE_VERIFIED',json.dumps({{'msr':{commit!r},'marin':{args.marin_commit!r},'sha256':expected}}),flush=True)
VERIFY
bash "$publication_root/cloud/iris/bootstrap_runtime.sh" "$publication_root" \\
  /tmp/oa-publication-env /tmp/oa-publication-runtime megatron development
source /tmp/oa-publication-runtime
/tmp/oa-publication-env/bin/python - <<'GPU'
import torch
assert torch.cuda.device_count()==2
assert all('H100' in torch.cuda.get_device_name(i) for i in range(2))
print('PUBLICATION_TWO_H100_PREFLIGHT_PASS',flush=True)
GPU
exec /tmp/oa-publication-env/bin/python -m pytest -s -q '{NODEID}'
"""
    command = [
        os.environ["IRIS"],
        "--cluster",
        "marin",
        "job",
        "run",
        "--job-name",
        args.job_name,
        "--target-cluster",
        "cw-us-east-02a",
        "--priority",
        "batch",
        "--gpu",
        "H100x2",
        "--replicas",
        "1",
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "100GB",
        "--enable-extra-resources",
        "--max-retries",
        "0",
        "--timeout",
        "1800",
        "--no-sync",
        "--no-wait",
        "--",
        "bash",
        "-c",
        body,
    ]
    preview = {
        "msr": commit,
        "marin": args.marin_commit,
        "sha256": hashes,
        "nodeid": NODEID,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "profile": "megatron/vllm/telemetry + frozen dev group",
        "max_task_gpu_hours": 1.0,
        "storage": "pinned public Qwen model to pod cache; no dataset or bucket I/O; durable Iris logs",
        "command": command,
    }
    print(json.dumps(preview, indent=2), flush=True)
    if args.execute:
        assert os.environ["IRIS_USER"] == "atqamar"
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
