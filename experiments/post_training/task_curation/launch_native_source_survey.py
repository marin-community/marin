# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the native source-identity survey job at interactive priority.

The driver runs in the populations' region (cw-us-east-02a); it copies the
staged spec shards to the survey's durable prefix, fingerprints the three
re-pinned native conversions' populations, and runs the deterministic
content-NN neighbor survey focused on them. The summary it prints carries the
URIs and digests the FingerprintReference registry append consumes.
"""

import argparse
import subprocess

RAW_SCRIPT = (
    "https://raw.githubusercontent.com/marin-community/marin/"
    "{revision}/experiments/post_training/task_curation/run_native_source_survey.sh"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The revision carrying the runbook and the driver.")
    parser.add_argument("--survey-root", required=True, help="The regional source-identity prefix.")
    parser.add_argument("--served-survey-uri", required=True, help="The served identity survey artifact URI.")
    parser.add_argument("--served-survey-sha256", required=True, help="The served identity survey SHA-256.")
    parser.add_argument("--population-version", required=True)
    parser.add_argument("--source", action="append", required=True, help="source_id=expected_tasks")
    parser.add_argument("--job-name", default="native-source-survey-taskspec-repin-v1")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--memory", default="64GB")
    parser.add_argument("--disk", default="300GB")
    parser.add_argument("--timeout", default="21600", help="Job timeout in seconds (default: six hours).")
    parser.add_argument("--max-retries", default="1")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    assignments = ",".join(args.source)
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_native_source_survey.sh && "
        f"bash /workspace/run_native_source_survey.sh {args.revision} {args.survey_root} "
        f"{args.served_survey_uri} {args.served_survey_sha256} {args.population_version} "
        f"'{assignments}' {args.cpu} 4"
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
        args.max_retries,
        "--",
        "bash",
        "-c",
        command,
    ]
    completed = subprocess.run(argv)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"native source survey job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
