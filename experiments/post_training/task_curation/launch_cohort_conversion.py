# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the cohort conversion jobs in-region with an exact argv list.

The wave-1 TaskSpec-only conversion jobs run on the cluster that holds the
cohorts (cw-us-east-02a) at batch priority with the registry upload secret. The
command follows the intake-conversion launcher's bootstrap: fetch the landed
script over HTTP, let it download and verify the landed archive, then run the
conversion release module. The conversion scratch stays on the pod; the private
release is the durable artifact.
"""

import argparse
import subprocess

RAW_SCRIPT = "https://raw.githubusercontent.com/marin-community/marin/{revision}/experiments/post_training/task_curation/run_cohort_conversion.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to convert at.")
    parser.add_argument("--source-id", required=True, help="The Harbor cohort source id to convert.")
    parser.add_argument("--artifact-version", required=True, help="The release repo version suffix.")
    parser.add_argument("--job-name", required=True, help="The Iris job name.")
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="128GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_cohort_conversion.sh && "
        f"bash /workspace/run_cohort_conversion.sh {args.revision} {args.source_id} {args.artifact_version}"
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
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
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
    print(f"cohort conversion job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
