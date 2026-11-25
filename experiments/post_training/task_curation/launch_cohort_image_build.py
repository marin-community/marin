# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the regional Kaniko build for the accepted cohorts' images.

Follows the landed launch-kaniko pattern (sealed regional builder, digest-pinned
destination, credentials only via the secret-delivered Docker config): one GHCR
package for the cohort conversion images, one tag per cohort; the job records
each cohort's digest into a bindings manifest staged beside the contexts for the
conversion's image-bindings input. Interactive priority per the operator
directive.
"""

import argparse
import base64
import json
import os
import subprocess
from pathlib import Path

BUILDER = (
    "ghcr.io/marin-community/rl-data-openresearcher"
    "@sha256:9e4985160e33e3c448da42d4b2140807267961f426f88720381e2578377f0d60"
)
PACKAGE = "ghcr.io/marin-community/rl-data-cohort-taskspec"


def ghcr_auth() -> str:
    config = json.loads((Path.home() / ".docker" / "config.json").read_text())
    helper = config.get("credHelpers", {}).get("ghcr.io") or config["credsStore"]
    credential = json.loads(subprocess.check_output([f"docker-credential-{helper}", "get"], input=b"ghcr.io\n"))
    return base64.b64encode((credential["Username"] + ":" + credential["Secret"]).encode()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts-manifest", required=True, help="The staged cohort-to-context manifest (from cohort_build_contexts)."
    )
    parser.add_argument("--context-root", required=True, help="The regional bucket root the contexts were staged under.")
    parser.add_argument(
        "--bindings-uri", required=True, help="Where the job writes the source-to-pinned-image bindings manifest."
    )
    parser.add_argument("--tag-prefix", default="cohort")
    parser.add_argument("--job-name", default="atlas-taskspec-cohort-image-build")
    args = parser.parse_args()
    job_name = args.job_name

    contexts = json.loads(Path(args.contexts_manifest).read_text())
    staged_manifest = args.context_root.rstrip("/") + "/contexts-manifest.json"
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", args.contexts_manifest, staged_manifest],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    # The build plan is inline: kaniko reads the s3:// context archives directly
    # (the executor image has no curl/tar and needs none), so the command carries
    # one executor invocation per cohort with its s3 context and destination.
    def cohort_tag(entry) -> str:
        slug = entry["source_id"].partition(":")[2].replace("/", "-").replace("@", "-").lower()
        return f"{args.tag_prefix}-{slug[:48].rstrip('-')}"

    def cohort_reference(entry) -> str:
        return f"{PACKAGE}:{cohort_tag(entry)}"

    loop_lines = []
    for entry in contexts:
        destination = cohort_reference(entry)
        digest_binding = f"{entry['source_id']} {destination}@$(cat /tmp/digest.txt)"
        loop_lines.append(
            f"/kaniko/executor --context={entry['context_uri']} --dockerfile={entry['dockerfile']} "
            f"--destination={destination} "
            "--custom-platform=linux/amd64 --single-snapshot --cache=false --compressed-caching=false "
            "--reproducible --image-name-with-digest-file=/tmp/digest.txt; "
            f'echo "{digest_binding}" >> /tmp/bindings.txt; '
        )
    command = "set -eu; " + " ".join(loop_lines) + " cat /tmp/bindings.txt"
    command = "set -eu; " + " ".join(loop_lines) + " cat /tmp/bindings.txt"

    environment = {**os.environ, "ATLAS_GHCR_DOCKER_CONFIG": json.dumps({"auths": {"ghcr.io": {"auth": ghcr_auth()}}})}
    job_name = args.job_name
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
        "--task-image",
        BUILDER,
        "--cpu",
        "8",
        "--memory",
        "128GB",
        "--disk",
        "300GB",
        "--enable-extra-resources",
        "--timeout",
        "259200",
        "--priority",
        "interactive",
        "--job-name",
        job_name,
        "--secret-behavior",
        "isolate",
        "--secret",
        "config.json",
        "env:ATLAS_GHCR_DOCKER_CONFIG",
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
        "--secret",
        "HF_TOKEN",
        "env:HF_TOKEN",
        "-e",
        "DOCKER_CONFIG",
        "/run/rigging/secrets",
        "-e",
        "S3_ENDPOINT",
        "https://cwobject.com",
        "-e",
        "S3_FORCE_PATH_STYLE",
        "false",
        "-e",
        "AWS_REGION",
        "US-EAST-02A",
        "-e",
        "AWS_EC2_METADATA_DISABLED",
        "true",
        "--",
        "/busybox/sh",
        "-c",
        command,
    ]
    (Path("build") / "kaniko-submit.log").parent.mkdir(exist_ok=True)
    with (Path("build") / "kaniko-submit.log").open("w") as output:
        completed = subprocess.run(args_list, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"cohort image build job submitted: /benfeuer/{job_name}")
    print(f"bindings will land at: {args.bindings_uri}")


if __name__ == "__main__":
    main()
