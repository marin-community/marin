# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the SWE-rebench V2.1 fidelity gate on the maintained regional trial recipe.

Follows the ingest-A native-trial launcher exactly (their monitoring-state resubmit argv):
Daytona EVAL declarative trials on cw-us-east-02a, secrets isolated, marin-core env with the
vendored harbor[daytona] runtime editable, driver fetched from the regional bucket with a
pinned sha256 and exec'd. The driver runs per sampled row: nop + oracle on the upstream
digest-pinned image versus the condensed shared-base environment, identical F2P/P2P outcomes
required. Launch from a shell with the secrets env sourced.
"""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

OUTPUT_ROOT = "s3://marin-us-east-02a/marin/rl-data/atlas/swe-repair/2026.11.20.1/wave-v11"
SAMPLE_URI = f"{OUTPUT_ROOT}/fidelity-sample-manifest.json"
BINDINGS_URI = f"{OUTPUT_ROOT}/base-image-bindings.json"
DATASET_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/swe-rebench-v2/v2-1-data"
# The driver stages at a VERSIONED uri: a running wave job's retries pin the sha, so a
# re-stage at a shared uri breaks every retry's bootstrap assert (the wave1-full post-mortem).
DRIVER_URI = (
    "s3://marin-us-east-02a/marin/rl-data/inputs/swe-rebench-v2/gate/2026.11.20.2/swe_rebench_fidelity_driver.py"
)
DRIVER_SHA256 = "edc9563a982c6378add38ef46dd266e4095a6d982daa5eb3993283e1412495ba"
JOB_NAME = "swe-repair-swe-rebench-wave-v11"


def parse_args() -> dict:
    """Optional overrides: --job-name, --groups (comma list), --diagnose-instances (comma list)."""
    overrides = {}
    argv = sys.argv[1:]
    for index in range(0, len(argv) - 1, 2):
        if argv[index] == "--job-name":
            overrides["job_name"] = argv[index + 1]
        elif argv[index] == "--groups":
            overrides["groups"] = argv[index + 1]
        elif argv[index] == "--diagnose-instances":
            overrides["diagnose_instances"] = argv[index + 1]
        elif argv[index] == "--instances":
            overrides["instances"] = argv[index + 1]
        elif argv[index] == "--wave-groups":
            overrides["wave_groups"] = argv[index + 1]
        elif argv[index] == "--wave-limit":
            overrides["wave_limit"] = argv[index + 1]
        elif argv[index] == "--wave-exclusions":
            overrides["wave_exclusions"] = argv[index + 1]
        elif argv[index] == "--workers":
            overrides["workers"] = argv[index + 1]
        elif argv[index] == "--output-root":
            overrides["output_root"] = argv[index + 1]
        elif argv[index] == "--task-memory":
            overrides["task_memory"] = argv[index + 1]
    return overrides


def ghcr_auth() -> str:
    config = json.loads((Path.home() / ".docker" / "config.json").read_text())
    helper = config.get("credHelpers", {}).get("ghcr.io") or config["credsStore"]
    credential = json.loads(subprocess.check_output([f"docker-credential-{helper}", "get"], input=b"ghcr.io\n"))
    return base64.b64encode((credential["Username"] + ":" + credential["Secret"]).encode()).decode()


def main() -> None:
    overrides = parse_args()
    job_name = overrides.get("job_name", JOB_NAME)
    output_root = overrides.get("output_root", OUTPUT_ROOT)
    bootstrap = (
        "import hashlib;from rigging.filesystem.s3_compat import configure_coreweave_s3;"
        "from rigging.filesystem.storage_path import StoragePath;configure_coreweave_s3();"
        f"raw=StoragePath('{DRIVER_URI}').read_bytes();"
        f"assert hashlib.sha256(raw).hexdigest()=='{DRIVER_SHA256}';"
        "exec(compile(raw,'swe_rebench_fidelity_driver.py','exec'))"
    )
    environment = {**os.environ, "ATLAS_GHCR_DOCKER_CONFIG": json.dumps({"auths": {"ghcr.io": {"auth": ghcr_auth()}}})}
    args = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=cw-us-east-02a",
        "job",
        "run",
        "--secret",
        "DAYTONA_DATA_API_KEY",
        "env:DAYTONA_DATA_API_KEY",
        "--secret",
        "HF_TOKEN",
        "env:HF_TOKEN",
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
        "--no-wait",
        "--job-name",
        job_name,
        "--replicas",
        "1",
        "--cpu",
        "8",
        "--memory",
        overrides.get("task_memory", "32GB"),
        "--disk",
        "256GB",
        "--enable-extra-resources",
        "--sync-package",
        "marin-core",
        "--extra",
        "cpu",
        "--extra",
        "dedup",
        "--priority",
        "production",
        "--no-preemptible",
        "--timeout",
        "259200",
        "--max-retries",
        "3",
        "-e",
        "DAYTONA_TARGET",
        "us",
        "-e",
        "PYTHONPATH",
        "/app",
        "-e",
        "GATE_SAMPLE_URI",
        SAMPLE_URI,
        "-e",
        "GATE_BINDINGS_URI",
        BINDINGS_URI,
        "-e",
        "GATE_DATASET_URI",
        DATASET_URI,
        "-e",
        "GATE_OUTPUT_URI",
        output_root,
        *(["-e", "GATE_GROUPS", overrides["groups"]] if "groups" in overrides else []),
        *(["-e", "GATE_INSTANCE", overrides["instances"]] if "instances" in overrides else []),
        *(["-e", "WAVE_GROUPS", overrides["wave_groups"]] if "wave_groups" in overrides else []),
        *(["-e", "WAVE_LIMIT", overrides["wave_limit"]] if "wave_limit" in overrides else []),
        *(["-e", "WAVE_EXCLUSIONS_URI", overrides["wave_exclusions"]] if "wave_exclusions" in overrides else []),
        *(["-e", "GATE_WORKERS", overrides["workers"]] if "workers" in overrides else []),
        *(
            ["-e", "GATE_DIAGNOSTIC", "1", "-e", "GATE_INSTANCE", overrides["diagnose_instances"]]
            if "diagnose_instances" in overrides
            else []
        ),
        "--",
        "uv",
        "run",
        "--isolated",
        "--frozen",
        "--package",
        "marin-core",
        "--extra",
        "cpu",
        "--extra",
        "dedup",
        "--prerelease",
        "allow",
        "--with-editable",
        "./vendor/harbor[daytona]",
        "python",
        "-c",
        bootstrap,
    ]
    root = Path(__file__).parent
    with (root / "gate-driver-submit.log").open("w") as output:
        completed = subprocess.run(args, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"Fidelity gate driver job submitted: /benfeuer/{job_name}")


if __name__ == "__main__":
    main()
