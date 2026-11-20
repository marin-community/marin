# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the pending-intake conversion job with an exact argv list.

The iris CLI re-quotes ``--`` arguments, which mangled the earlier inline
command (exit 127: the whole multi-line string executed as one token). This
launcher passes the command as separate argv entries and keeps the revision,
output root and license-receipt uri in one place.
"""

import argparse
import subprocess

REPO = "https://github.com/marin-community/marin/archive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to convert at.")
    parser.add_argument("--output-root", required=True, help="The intake output root (object storage).")
    parser.add_argument("--license-uri", required=True, help="The license receipts object uri.")
    parser.add_argument("--job-name", default="swe-repair-swe-rebench-convert")
    args = parser.parse_args()

    script = "experiments/post_training/task_curation/datasets/swe_rebench_v2/run_intake_conversion.sh"
    command = (
        f"curl -fsSL {REPO}/{args.revision}.tar.gz | tar xz -C /workspace && "
        f"bash /workspace/marin-{args.revision}/{script} "
        f"{args.revision} {args.output_root} {args.license_uri}"
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
        "16",
        "--memory",
        "64GB",
        "--disk",
        "300GB",
        "--enable-extra-resources",
        "--priority",
        "batch",
        "--no-preemptible",
        "--timeout",
        "259200",
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
    print(f"intake conversion job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
