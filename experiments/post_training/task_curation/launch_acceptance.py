# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the acceptance-control jobs for published conversion releases in-region.

The acceptance runs the recorded negative controls against the converted graders
on the Daytona RL org (the backend the Harbor trials used), with the judge keys
resolved through the Iris secret sources: the local environment first, then GSM.
"""

import argparse
import subprocess

RAW_SCRIPT = "https://raw.githubusercontent.com/marin-community/marin/{revision}/experiments/post_training/task_curation/run_acceptance.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument("--source-id", required=True, help="The cohort source id.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--job-name", required=True, help="The Iris job name.")
    parser.add_argument("--sample", default="12", help="How many tasks to accept.")
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="128GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_acceptance.sh && "
        f"bash /workspace/run_acceptance.sh {args.revision} {args.source_id} {args.release_repo} {args.sample}"
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
        "--secret",
        "DAYTONA_API_KEY",
        "env:DAYTONA_RL_API_KEY",
        "--secret",
        "OPENAI_API_KEY",
        "gcp-secret://projects/hai-gcp-models/secrets/OPENAI_API_KEY/versions/latest",
        "--env",
        "DAYTONA_API_URL=https://app.daytona.io/api",
        "--env",
        "DAYTONA_TARGET=us",
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
        "batch",
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
    print(f"acceptance job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
