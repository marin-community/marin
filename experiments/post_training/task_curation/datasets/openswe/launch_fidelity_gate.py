# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the OpenSWE fidelity gate job in-region.

The gate runs on the Daytona EVAL account: declarative image builds are a
validation capability (the RL account refuses them), and the factory builds
from the original Dockerfile context without creating named snapshots. The
launcher resolves the eval key at submission — from the operator environment,
or through GSM when the environment does not carry it — and the CLI redacts
secret-shaped values from its submit record.
"""

import argparse
import os
import subprocess

from rigging.secrets import resolve_secret_spec

RAW_SCRIPT = (
    "https://raw.githubusercontent.com/marin-community/marin/{revision}"
    "/experiments/post_training/task_curation/run_fidelity_gate.sh"
)
EVAL_KEY_SPEC = "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest"
CONDENSED_BINDINGS_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build"
UPSTREAM_BINDINGS_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build"
REPORT_ROOT = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/fidelity-gate"


def eval_key() -> str:
    """The Daytona eval account key, from the environment or GSM."""
    value = os.environ.get("DAYTONA_EVAL_API_KEY")
    if not value:
        value = resolve_secret_spec(EVAL_KEY_SPEC).value
    if not value:
        raise SystemExit("the Daytona eval key resolved empty; set DAYTONA_EVAL_API_KEY before launching")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument("--job-name", default="swe-zero-openswe-fidelity-gate-v1")
    parser.add_argument("--report-root", default=REPORT_ROOT)
    parser.add_argument("--condensed-bindings", default=CONDENSED_BINDINGS_URI)
    parser.add_argument("--upstream-bindings", default=UPSTREAM_BINDINGS_URI)
    parser.add_argument("--per-base", default=str(20))
    parser.add_argument("--concurrency", default="4")
    parser.add_argument("--bases", default="", help="A comma-separated base filter; empty runs every base.")
    parser.add_argument("--seed", default="")
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="256GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_fidelity_gate.sh && "
        f"bash /workspace/run_fidelity_gate.sh {args.revision} {args.condensed_bindings} "
        f"{args.upstream_bindings} {args.report_root} {args.per_base} {args.concurrency} "
        f"'{args.bases}' '{args.seed}'"
    )
    argv = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=cw-us-east-02a",
        "job",
        "run",
        "--secret",
        "HF_TOKEN",
        "env:HF_TOKEN",
        "--env-vars",
        "DAYTONA_API_KEY",
        eval_key(),
        "--env-vars",
        "DAYTONA_API_URL",
        "https://app.daytona.io/api",
        "--env-vars",
        "DAYTONA_TARGET",
        "us",
        "--no-wait",
        "--job-name",
        args.job_name,
        "--replicas",
        "1",
        "--cpu",
        args.cpu,
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--enable-extra-resources",
        "--priority",
        "interactive",
        "--no-preemptible",
        "--timeout",
        args.timeout,
        "--max-retries",
        "0",
        "--",
        "bash",
        "-c",
        command,
    ]
    completed = subprocess.run(argv)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"fidelity gate job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
