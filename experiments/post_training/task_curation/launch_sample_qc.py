# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the SAMPLE QC campaign for the taskspec re-pin's staged panels.

The driver runs as an Iris job at interactive priority; the pinned V4 reviewer
serves as the campaign's own child jobs at the same priority (the campaign
module owns that submission). The queue, the panels and the review artifacts
live in the region.
"""

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument("--queue-uri", required=True, help="The frozen queue URI (object storage).")
    parser.add_argument("--queue-sha256", required=True, help="The frozen queue's sha256.")
    parser.add_argument("--output-root", required=True, help="The review artifacts' output root.")
    parser.add_argument("--review-cache", required=True, help="The reviewer request cache root.")
    parser.add_argument("--job-name", default="atlas-taskspec-sample-qc")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--memory", default="64GB")
    parser.add_argument("--disk", default="300GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_uri = (
        "https://raw.githubusercontent.com/marin-community/marin/"
        f"{args.revision}/experiments/post_training/task_curation/run_sample_qc.sh"
    )
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_uri} -o /workspace/run_sample_qc.sh && "
        f"bash /workspace/run_sample_qc.sh {args.revision} {args.queue_uri} {args.queue_sha256} "
        f"{args.output_root} {args.review_cache}"
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
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
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
        "--job-name",
        args.job_name,
        "--",
        "bash",
        "-c",
        command,
    ]
    completed = subprocess.run(argv)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"sample qc job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
