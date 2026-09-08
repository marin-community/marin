# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview or run the isolated one-engine queue-persistence precursor."""

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

NODEID = (
    "skyrl-train/tests/gpu/test_publication_cap_precursor.py::test_cap8_preserves_queued_requests_across_original_pause"
)
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "cloud/iris/bootstrap_runtime.sh",
    "skyrl-train/tests/gpu/test_publication_cap_precursor.py",
    "skyrl-train/tests/gpu/publication_cap_protocol.py",
    "skyrl-train/tests/gpu/prepare_publication_cap_precursor.py",
    "skyrl-train/skyrl_train/config/ppo_base_config.yaml",
    "skyrl-train/skyrl_train/entrypoints/main_base.py",
    "skyrl-train/skyrl_train/inference_engines/ray_wrapped_inference_engine.py",
    "skyrl-train/skyrl_train/inference_engines/inference_engine_client.py",
    "skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py",
    "skyrl-train/skyrl_train/inference_engines/utils.py",
    "skyrl-train/skyrl_train/weight_sync/publication_accounting.py",
    "skyrl-train/skyrl_train/weight_sync/publication_version.py",
    "skyrl-train/skyrl_train/trajectory_runners/trajectory_processing.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--marin-commit", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    marin = Path(__file__).resolve().parents[2]
    assert (
        subprocess.check_output(["git", "-C", str(marin), "rev-parse", "HEAD"], text=True).strip() == args.marin_commit
    )
    assert not subprocess.check_output(["git", "-C", str(marin), "status", "--porcelain"], text=True)
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    msr = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert args.job_name.startswith("async-rl-v2-cap8-native-")
    spec = json.loads(args.input_spec.read_text())
    assert spec["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    if not args.cpu_only:
        assert len(spec["prompt_token_ids"]) == 64 and "train_parquet_sha256" in spec
    encoded = base64.b64encode(json.dumps(spec, sort_keys=True).encode()).decode()
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}
    stage = (
        "/tmp/oa-cap8-env/bin/python -m tests.gpu.prepare_publication_cap_precursor"
        if args.cpu_only
        else f"""
export PUBLICATION_CAP_RECEIPT=/tmp/oa-cap8-receipt.json
if /tmp/oa-cap8-env/bin/python -m pytest -s -q -m vllm {NODEID}; then cap8_status=0; else cap8_status=$?; fi
/tmp/oa-cap8-env/bin/python - <<'RECEIPT'
import hashlib,json
from pathlib import Path
path=Path('/tmp/oa-cap8-receipt.json')
if path.exists():
 data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 parts=[data[i:i+3072].decode() for i in range(0,len(data),3072)]
 for i,part in enumerate(parts):print('CAP8_RECEIPT_PART '+json.dumps(dict(sha256=digest,part=i,parts=len(parts),payload=part)),flush=True)
else:print('CAP8_RECEIPT_ABSENT before_native_test_receipt',flush=True)
RECEIPT
exit "$cap8_status"
"""
    )
    body = f"""set -euo pipefail
cap8_root="$PWD"
export PYTHONPATH="$cap8_root/skyrl-train:$cap8_root/skyrl-gym:$cap8_root"
export HF_HOME=/tmp/oa-cache/huggingface
export UV_CACHE_DIR=/tmp/oa-cache/uv
export WANDB_MODE=disabled
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path
expected={hashes!r}
for path,digest in expected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
print('CAP8_SOURCE_PASS '+json.dumps(dict(msr={msr!r},marin={args.marin_commit!r},sha256=expected)),flush=True)
VERIFY
bash "$cap8_root/cloud/iris/bootstrap_runtime.sh" "$cap8_root" /tmp/oa-cap8-env /tmp/oa-cap8-runtime megatron development
source /tmp/oa-cap8-runtime
export PUBLICATION_CAP_SPEC="$(python3 - <<'SPEC'
import base64
print(base64.b64decode({encoded!r}).decode())
SPEC
)"
{stage}
"""
    timeout = 900 if args.cpu_only else 600
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
        *([] if args.cpu_only else ["--gpu", "H100x1"]),
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
        str(timeout),
        "--no-sync",
        "--no-wait",
        "--",
        "bash",
        "-c",
        body,
    ]
    print(
        json.dumps(
            {
                "marin": args.marin_commit,
                "msr": msr,
                "source_hashes": hashes,
                "input_spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                "job_name": args.job_name,
                "nodeid": None if args.cpu_only else NODEID,
                "target_cluster": "cw-us-east-02a",
                "gpus": 0 if args.cpu_only else 1,
                "timeout_seconds": timeout,
                "task_gpu_hour_ceiling": 0 if args.cpu_only else timeout / 3600,
                "max_retries": 0,
                "runtime": "frozen megatron/vllm development",
                "storage": "pod cache and durable Iris log receipts; read family48 east data, no bucket writes",
                "command": command,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.execute:
        assert os.environ["IRIS_USER"] == "atqamar"
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
