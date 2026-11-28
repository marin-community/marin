# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the OpenSWE keep-only conversion job in-region.

The job converts the pinned keep-only population with the dedicated profile at
interactive priority. Docker Hub credentials — needed only by the minority of
rows whose FROM bases are third-party registry references — are read from the
operator's Docker Desktop credential helper at submission and passed to the job
as ``-e`` variables; the CLI redacts secret-shaped values from its submit
record and the credentials are never printed.
"""

import argparse
import json
import subprocess

from experiments.post_training.task_curation.datasets.openswe.launch_fidelity_gate import CONDENSED_BINDINGS_URI

RAW_SCRIPT = (
    "https://raw.githubusercontent.com/marin-community/marin/{revision}"
    "/experiments/post_training/task_curation/run_openswe_conversion.sh"
)


def docker_hub_credentials() -> dict[str, str]:
    """The Docker Hub username/token pair, read at submission and never logged."""
    credential = json.loads(
        subprocess.run(
            ["docker-credential-desktop", "get"],
            input=b"https://index.docker.io/v1/",
            capture_output=True,
            check=True,
        ).stdout
    )
    return {"DOCKERHUB_USERNAME": credential["Username"], "DOCKERHUB_TOKEN": credential["Secret"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="The landed marin revision to convert at.")
    parser.add_argument("--job-name", required=True, help="The Iris job name.")
    parser.add_argument("--gate-root", required=True, help="The staged fidelity gate report prefix.")
    parser.add_argument("--artifact-version", required=True, help="The release version suffix.")
    parser.add_argument("--condensed-bindings", default=CONDENSED_BINDINGS_URI)
    parser.add_argument("--cpu", default="8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--disk", default="256GB")
    parser.add_argument("--timeout", default="259200", help="Job timeout in seconds (default: three days).")
    args = parser.parse_args()

    script_url = RAW_SCRIPT.format(revision=args.revision)
    command = (
        f"mkdir -p /workspace && curl -fsSL {script_url} -o /workspace/run_openswe_conversion.sh && "
        f"bash /workspace/run_openswe_conversion.sh {args.revision} {args.condensed_bindings} "
        f"{args.gate_root} {args.artifact_version}"
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
    for name, value in docker_hub_credentials().items():
        argv.extend(("-e", name, value))
    completed = subprocess.run(argv)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"openswe conversion job submitted: /benfeuer/{args.job_name}")


if __name__ == "__main__":
    main()
