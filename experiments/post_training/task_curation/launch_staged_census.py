# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the staged-population census jobs for accepted conversion releases.

Each job measures its release's declared Daytona demands, verifies the cohort's
passing acceptance run live (fail closed), classifies through the declarative
path per the coordinator's ruling 2026-11-25 13:00 EST, and stages the census
artifact and the ActorCensusReceipt shards to the regional bucket for the
admission-time registry append.
"""

import argparse
import subprocess

RAW_SCRIPT = "https://raw.githubusercontent.com/marin-community/marin/{revision}/experiments/post_training/task_curation/run_staged_census.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument("--source-id", required=True, help="The cohort source id.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--acceptance-job", required=True, help="The cohort's passing acceptance Iris job name.")
    parser.add_argument("--census-root", required=True, help="The regional bucket root for the census artifacts.")
    parser.add_argument("--job-name", required=True, help="The Iris job name.")
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="128GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    census_root = args.census_root.rstrip("/")
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_staged_census.sh && "
        f"bash /workspace/run_staged_census.sh {args.revision} {args.source_id} "
        f"{args.release_repo} {args.acceptance_job} {census_root}"
    )
    argv = [
        "uv",
        "run",
        "--no-sync",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--target-cluster=cw-us-east-02a",
        "--no-sync",
        "--secret",
        "HF_TOKEN",
        "env:HF_TOKEN",
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
    print(f"staged census job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
