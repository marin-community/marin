# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the acceptance-control jobs for published conversion releases in-region.

The acceptance runs the recorded negative controls against the converted graders
on the Daytona RL org (the backend the Harbor trials used). The job environment
carries the org's API key and the judge key directly: the Daytona SDK and the
harnesses read plain environment variables, so the launcher resolves each value
at submission (the RL key from the operator environment, the judge key through
GSM first) and the CLI redacts secret-shaped values from its submit record.
"""

import argparse
import os
import subprocess

from rigging.secrets import resolve_secret_spec

RAW_SCRIPT = "https://raw.githubusercontent.com/marin-community/marin/{revision}/experiments/post_training/task_curation/run_acceptance.sh"
JUDGE_KEY_SPEC = "gcp-secret://projects/hai-gcp-models/secrets/OPENAI_API_KEY/versions/latest"


def _required_env(name: str, what: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing {what}: set {name} in the environment (source secrets.env) before launching")
    return value


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

    daytona_key = _required_env("DAYTONA_RL_API_KEY", "the Daytona RL org key")
    # The judge key comes from GSM; a missing key is a tooling bug, not a reason to skip.
    judge_key = resolve_secret_spec(JUDGE_KEY_SPEC).value

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
        "--env-vars",
        "DAYTONA_API_KEY",
        daytona_key,
        "--env-vars",
        "DAYTONA_API_URL",
        "https://app.daytona.io/api",
        "--env-vars",
        "DAYTONA_TARGET",
        "us",
        "--env-vars",
        "OPENAI_API_KEY",
        judge_key,
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
    print(f"acceptance job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
