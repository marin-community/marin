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
    parser.add_argument("--cluster", default="cw-us-east-02a", help="The cluster the acceptance jobs ran on.")
    parser.add_argument(
        "--population-record",
        help="Measure the staged lexical population artifact at this path (ruling 2026-11-25 ~21:05 EST, "
        "option A); empty keeps the release tasks.parquet path.",
    )
    parser.add_argument(
        "--converter-revision",
        help="The converter revision to bind (the original census receipts' value); empty = this checkout.",
    )
    parser.add_argument(
        "--content-identity-proof",
        help="The staged per-task sha256 equality proof URI, referenced in the census basis and copied "
        "beside the receipts.",
    )
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
        f"{args.release_repo} {args.acceptance_job} {census_root} "
        f"'{args.population_record or ''}' '{args.converter_revision or ''}' '{args.content_identity_proof or ''}'"
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
    # Fail closed at submission: the census job's pod cannot resolve the
    # cluster config, so the passing-acceptance verification runs here, where
    # iris works. The census job records the verified evidence.
    describe = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "iris",
            f"--cluster={args.cluster}",
            "job",
            "describe",
            f"/benfeuer/{args.acceptance_job}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    state_line = next((line for line in describe.stdout.splitlines() if line.startswith("State:")), None)
    if state_line is None or "succeeded" not in state_line:
        raise SystemExit(
            f"REFUSED: the acceptance job {args.acceptance_job} is not succeeded "
            f"({(state_line or 'no state').strip()}); the cohort stays Pending for the repair cycle"
        )
    print(f"acceptance verified: {args.acceptance_job} succeeded")

    completed = subprocess.run(argv)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"staged census job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
