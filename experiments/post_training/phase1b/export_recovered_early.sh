#!/usr/bin/env bash
set -euo pipefail

step=${1:?pass 10, 30, or 50}
mode=${2:?pass dry-run or submit}
case "$step" in 10|30|50) ;; *) echo "unexpected export step: $step" >&2; exit 2 ;; esac
case "$mode" in dry-run|submit) ;; *) echo "unexpected mode: $mode" >&2; exit 2 ;; esac
[[ "${MARIN_PREFIX:?}" == s3://marin-us-east-08a/marin && "${MARIN_TEMP_PREFIX:?}" == s3://marin-us-east-08a ]] || {
  echo "recovered exports require east-region output environment" >&2
  exit 2
}

cd /app/vendor/skyrl
bash cloud/iris/bootstrap_runtime.sh /app/vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron-export production
source /tmp/skyrl-runtime.sh
cd /app
export PYTHONPATH=/app
export EXPORT_STEP="$step" EXPORT_MODE="$mode"

python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

configure_coreweave_s3()
step = int(os.environ["EXPORT_STEP"])
mode = os.environ["EXPORT_MODE"]
old_root = (
    "s3://marin-rno2a/tmp/ttl=30d/skyrl/marin/users/benfeuer/checkpoints/phase1b/"
    "sft-theta_RJ_restored-lr2e-06-seed1/2026.11.11.1/checkpoints"
)
east_run = (
    "s3://marin-us-east-08a/marin/users/benfeuer/checkpoints/phase1b/"
    "sft-theta_RJ_restored-lr2e-06-seed1/2026.11.11.1"
)
request_uri = f"{old_root}/global_step_{step}/hf_export_request.json"
filesystem, path = filesystem_for(request_uri)
with filesystem.open(path) as source:
    request = json.load(source)
assert request["step"] == step and request["checkpoint_base_path"] == old_root
assert request["model_source_uri"].startswith("s3://marin-rno2a/")
assert request["num_nodes"] == 4 and request["gpus_per_node"] == 8

config_uri = f"{east_run}/resolved-launch.yaml"
filesystem, path = filesystem_for(config_uri)
with filesystem.open(path) as source:
    document = yaml.safe_load(source)
config = document["config"]
assert config["artifacts"]["checkpoint_root"].startswith("s3://marin-us-east-08a/")
assert config["artifacts"]["export_root"] == f"{east_run}/exports"
assert config["iris"]["cluster"] == "cw-rno2a"
launch_config = Path("/tmp/marinskyrl/phase1b-restored-s1-inner.yaml")
launch_config.parent.mkdir(parents=True, exist_ok=True)
launch_config.write_text(yaml.safe_dump(config, sort_keys=False))

command = [
    sys.executable, "-m", "cloud.iris.export_hf_checkpoint",
    "--ckpt_path", old_root,
    "--step", str(step),
    "--model_path", request["model_path"],
    "--model-source-uri", request["model_source_uri"],
    "--model-source-identity", request["model_source_identity"],
    "--num-nodes", "4", "--gpus-per-node", "8",
    "--export_path", f"{east_run}/exports",
    "--launch-config", str(launch_config),
    "--gpu-variant", "H100", "--priority", "production",
    "--job-name", f"opd-1b-rest1-early-export-s{step}-r2-20261111",
]
if mode == "dry-run":
    command.append("--dry-run")
subprocess.run(command, check=True)
PY
