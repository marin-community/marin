# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the lexical scan for the staged conversion-release populations.

The scan consumes the staged populations manifest, rebuilds the DF masks for
the new cohort set (no frozen drop sets: the new populations are not in the
recorded artifact), measures against the pinned shared bloom, and lands the
lexical report artifact. Interactive priority per the operator directive.
"""

import argparse
import subprocess
from pathlib import Path

BLOOM = "s3://marin-us-east-02a/marin/datakit/bloom/_combined_fixed_eca06d42"
EVAL_CORPUS_VERSION = "aa-index-v4.1.1-v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to run at.")
    parser.add_argument(
        "--populations-manifest", required=True, help="The staged populations manifest (populations.json)."
    )
    parser.add_argument("--source-ids", required=True, help="The staged source-ids.json.")
    parser.add_argument("--population-version", required=True, help="The staged population version.")
    parser.add_argument("--artifact-version", required=True, help="The lexical artifact version.")
    parser.add_argument("--job-name", default="atlas-taskspec-lexical-scan")
    parser.add_argument("--max-workers", default="16")
    parser.add_argument("--cpu", default="16")
    parser.add_argument("--memory", default="128GB")
    parser.add_argument("--disk", default="300GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    root = "s3://marin-us-east-02a/marin/rl-data/atlas/lexical/text-cohort"
    plan_uri = f"{root.rstrip('/')}/lexical-scan/{args.population_version}/plan.tsv"
    populations_uri = f"{root.rstrip('/')}/lexical-scan/{args.population_version}/populations.json"
    source_ids_uri = f"{root.rstrip('/')}/lexical-scan/{args.population_version}/source-ids.json"
    report_uri = f"{root.rstrip('/')}/lexical-scan/{args.population_version}/lexical-report.json"
    plan_local = Path("build") / "lexical-scan-plan.tsv"
    plan_local.parent.mkdir(exist_ok=True)
    plan_local.write_text(
        f"populations\t{args.populations_manifest}\n"
        f"source_ids\t{args.source_ids}\n"
        f"populations_uri\t{populations_uri}\n"
        f"source_ids_uri\t{source_ids_uri}\n"
        f"report_uri\t{report_uri}\n"
        f"artifact_version\t{args.artifact_version}\n"
        f"bloom\t{BLOOM}\n"
        f"eval_corpus_version\t{EVAL_CORPUS_VERSION}\n"
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", str(plan_local), plan_uri],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", args.populations_manifest, populations_uri],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", args.source_ids, source_ids_uri],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    script_uri = (
        "https://raw.githubusercontent.com/marin-community/marin/"
        f"{args.revision}/experiments/post_training/task_curation/run_lexical_scan.sh"
    )
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_uri} -o /workspace/run_lexical_scan.sh && "
        f"bash /workspace/run_lexical_scan.sh {args.revision} {populations_uri} {source_ids_uri} "
        f"{args.population_version} {args.artifact_version} {args.max_workers} {report_uri}"
    )
    args_list = [
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
    completed = subprocess.run(args_list)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"lexical scan job submitted: /benfeuer/{args.job_name}")
    print(f"report will land at: {report_uri}")


if __name__ == "__main__":
    main()
