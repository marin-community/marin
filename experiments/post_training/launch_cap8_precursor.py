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
from types import SimpleNamespace
from unittest.mock import patch

from iris.cli.connect import open_iris_client
from iris.cli.job import build_resources
from iris.client.client import IrisClient
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.timing import Duration

NODEID = (
    "skyrl-train/tests/gpu/test_publication_cap_precursor.py::test_cap4_preserves_queued_requests_across_original_pause"
)
SOURCES = (
    "uv.lock",
    "pyproject.toml",
    "cloud/iris/bootstrap_runtime.sh",
    "skyrl-train/tests/gpu/test_publication_cap_precursor.py",
    "skyrl-train/tests/gpu/publication_cap_protocol.py",
    "skyrl-train/tests/gpu/prepare_publication_cap_precursor.py",
    "skyrl-train/tests/gpu/stage_publication_cap_model.py",
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
    assert args.job_name.startswith("async-rl-v2-cap4-native-")
    spec = json.loads(args.input_spec.read_text())
    assert spec["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    if not args.cpu_only:
        assert len(spec["prompt_token_ids"]) == 64 and "train_parquet_sha256" in spec
    encoded = base64.b64encode(json.dumps(spec, sort_keys=True).encode()).decode()
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}
    stage = (
        "/tmp/oa-cap4-env/bin/python -m tests.gpu.prepare_publication_cap_precursor"
        if args.cpu_only
        else f"""
export PUBLICATION_CAP_RECEIPT=/tmp/oa-cap4-receipt.json
if /tmp/oa-cap4-env/bin/python -m pytest -s -q -m vllm {NODEID}; then cap4_status=0; else cap4_status=$?; fi
/tmp/oa-cap4-env/bin/python - <<'RECEIPT'
import hashlib,json
from pathlib import Path
path=Path('/tmp/oa-cap4-receipt.json')
if path.exists():
 data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
 parts=[data[i:i+3072].decode() for i in range(0,len(data),3072)]
 for i,part in enumerate(parts):
  row=dict(sha256=digest,part=i,parts=len(parts),payload=part)
  print('CAP4_RECEIPT_PART '+json.dumps(row),flush=True)
else:print('CAP4_RECEIPT_ABSENT before_native_test_receipt',flush=True)
RECEIPT
exit "$cap4_status"
"""
    )
    bootstrap = (
        'UV_PROJECT_ENVIRONMENT=/tmp/oa-cap4-env uv sync --quiet --frozen --project "$cap4_root" '
        "--python python3.12 --no-python-downloads --link-mode symlink "
        "--extra cpu --extra telemetry --group dev --group harbor-test"
        if args.cpu_only
        else 'bash "$cap4_root/cloud/iris/bootstrap_runtime.sh" "$cap4_root" '
        "/tmp/oa-cap4-env /tmp/oa-cap4-runtime megatron development\nsource /tmp/oa-cap4-runtime"
    )
    body = f"""set -euo pipefail
cap4_root="$PWD"
export PYTHONPATH="$cap4_root/skyrl-train:$cap4_root/skyrl-gym:$cap4_root"
export HF_HOME=/tmp/oa-cache/huggingface
export UV_CACHE_DIR=/tmp/oa-cache/uv
export WANDB_MODE=disabled
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
python3 - <<'VERIFY'
import hashlib,json
from pathlib import Path
expected={hashes!r}
for path,digest in expected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
print('CAP4_SOURCE_PASS '+json.dumps(dict(msr={msr!r},marin={args.marin_commit!r},sha256=expected)),flush=True)
VERIFY
{bootstrap}
export PUBLICATION_CAP_SPEC="$(python3 - <<'SPEC'
import base64
print(base64.b64decode({encoded!r}).decode())
SPEC
)"
export PUBLICATION_CAP_MODEL=/tmp/oa-cap4-model
/tmp/oa-cap4-env/bin/python -m tests.gpu.stage_publication_cap_model --output "$PUBLICATION_CAP_MODEL"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
{stage}
"""
    timeout = 900 if args.cpu_only else 600
    kwargs = dict(
        entrypoint=Entrypoint.from_command("bash", "-c", body),
        name=args.job_name,
        resources=build_resources(None, None if args.cpu_only else "H100x1", cpu=8, memory="64GB", disk="100GB"),
        environment=EnvironmentSpec(env_vars={"WANDB_MODE": "disabled"}, extras=[], setup_scripts=[]),
        constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-us-east-02a")],
        replicas=1,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        timeout=Duration.from_seconds(timeout),
        scheduling_timeout=Duration.from_seconds(300),
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
    )
    native = {}

    class Capture:
        def launch_job(self, request, **unused):
            assert request.max_retries_failure == request.max_retries_preemption == request.max_task_failures == 0
            assert request.timeout.milliseconds == timeout * 1000
            assert request.scheduling_timeout.milliseconds == 300000
            native.update(
                {
                    "max_retries_failure": request.max_retries_failure,
                    "max_retries_preemption": request.max_retries_preemption,
                    "max_task_failures": request.max_task_failures,
                    "timeout_milliseconds": request.timeout.milliseconds,
                    "scheduling_timeout_milliseconds": request.scheduling_timeout.milliseconds,
                    "gpu_count": request.resources.device.gpu.count,
                    "gpu_variant": request.resources.device.gpu.variant,
                    "cpu_millicores": request.resources.cpu_millicores,
                    "memory_bytes": request.resources.memory_bytes,
                    "disk_bytes": request.resources.disk_bytes,
                    "priority_band": request.priority_band,
                }
            )
            assert native["gpu_count"] == (0 if args.cpu_only else 1)
            return SimpleNamespace(job_id=request.name)

    # Preview the real serialization boundary without recording any environment
    # values; task pods supply their own east-object credentials.
    forbidden = {
        "WANDB_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
    with patch.dict(os.environ, {key: value for key, value in os.environ.items() if key not in forbidden}, clear=True):
        remote = RemoteClusterClient("http://127.0.0.1:1")
        remote._client = Capture()
        IrisClient(remote).submit(**kwargs)
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
                "runtime": (
                    "frozen cpu/telemetry dev/harbor-test" if args.cpu_only else "frozen megatron/vllm development"
                ),
                "storage": "pod cache and durable Iris log receipts; read family48 east data, no bucket writes",
                "entrypoint": ["bash", "-c", body],
                "native_request": native,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.execute:
        assert os.environ["IRIS_USER"] == "atqamar"
        with patch.dict(
            os.environ, {key: value for key, value in os.environ.items() if key not in forbidden}, clear=True
        ):
            with open_iris_client(
                config_file=marin / "lib/iris/config/marin.yaml", cluster_name="marin", workspace=Path.cwd()
            ) as client:
                job = client.submit(**kwargs)
                print(
                    "CAP4_SUBMITTED " + json.dumps({"job_id": str(job.job_id), "marin": args.marin_commit, "msr": msr}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
