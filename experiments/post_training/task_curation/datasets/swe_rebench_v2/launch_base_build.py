# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the regional Kaniko build for the ten shared SWE-rebench V2.1 base images.

Follows the landed launch-kaniko pattern from the ingest-B lane (sealed regional builder,
digest-pinned destination, INTERNAL visibility, credentials only via the secret-delivered
Docker config). One GHCR package per source rule: ``ghcr.io/marin-community/rl-data-swe-rebench-v2``
with one tag per group; the job records each group's digest into a bindings manifest staged
next to the context for the converter's ``base_images`` input.

Local validation builds must pass for every group before this job is submitted.
"""

import base64
import json
import os
import subprocess
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

BUILDER = (
    "ghcr.io/marin-community/rl-data-openresearcher"
    "@sha256:9e4985160e33e3c448da42d4b2140807267961f426f88720381e2578377f0d60"
)
PACKAGE = "ghcr.io/marin-community/rl-data-swe-rebench-v2"
CONTEXT_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/swe-rebench-v2/base-build/context.tar.gz"
# v1 pushed python/go/node; v2 pushed rust; v3 pushed jvm (jammy pin); v4 pushed php/julia/
# elixir; v5 pushed swift/minor. Content verification found the go and node bases lost their
# apt-installed binaries in the v1 kaniko build (dpkg records them; files absent); v6 rebuilds
# those two bases with binary guards.
GROUPS = (
    "jvm",
)
TAG_PREFIX = "base"


def ghcr_auth() -> str:
    config = json.loads((Path.home() / ".docker" / "config.json").read_text())
    helper = config.get("credHelpers", {}).get("ghcr.io") or config["credsStore"]
    credential = json.loads(subprocess.check_output([f"docker-credential-{helper}", "get"], input=b"ghcr.io\n"))
    return base64.b64encode((credential["Username"] + ":" + credential["Secret"]).encode()).decode()


def main() -> None:
    here = Path(__file__).parent
    root = here / "build"
    root.mkdir(parents=True, exist_ok=True)
    # Build context: the ten authored Dockerfiles, staged once to the regional bucket; each
    # build selects one file from the same archive.
    context = root / "context"
    context.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        (context / f"{group}.Dockerfile").write_bytes((here / "bases" / f"{group}.Dockerfile").read_bytes())
    archive = root / "context.tar.gz"
    subprocess.run(["tar", "-czf", str(archive), "-C", str(context), "."], check=True)
    staged = StoragePath(CONTEXT_URI)
    # fsutil resolves the regional store credentials from the cluster config; the secrets env
    # must be sourced by the caller.
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", str(archive), CONTEXT_URI],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    del staged
    environment = {**os.environ, "ATLAS_GHCR_DOCKER_CONFIG": json.dumps({"auths": {"ghcr.io": {"auth": ghcr_auth()}}})}
    command = "set -eu; echo 'GROUPS=" + ",".join(GROUPS) + "'; " + " ; ".join(
        "/kaniko/executor "
        f"--context={CONTEXT_URI} "
        f"--dockerfile={group}.Dockerfile "
        f"--destination={PACKAGE}:{TAG_PREFIX}-{group} "
        "--custom-platform=linux/amd64 --single-snapshot --cache=false --compressed-caching=false --reproducible "
        f"--image-name-with-digest-file=/tmp/{group}-digest.txt; "
        f'echo "{group} $(cat /tmp/{group}-digest.txt)"'
        for group in GROUPS
    )
    job_name = "swe-repair-swe-rebench-v2-bases-v22"
    args = [
        "uv", "run", "--no-sync", "iris", "--cluster=marin", "job", "run", "--no-wait",
        "--target-cluster=cw-us-east-02a", "--no-sync", "--task-image", BUILDER,
        "--cpu", "8", "--memory", "128GB", "--disk", "300GB", "--enable-extra-resources",
        "--timeout", "259200", "--priority", "batch", "--job-name", job_name,
        "--secret-behavior", "isolate", "--secret", "config.json", "env:ATLAS_GHCR_DOCKER_CONFIG",
        "--secret", "CW_KEY_ID", "env:CW_KEY_ID", "--secret", "CW_KEY_SECRET", "env:CW_KEY_SECRET",
        "-e", "DOCKER_CONFIG", "/run/rigging/secrets", "-e", "S3_ENDPOINT", "https://cwobject.com",
        "-e", "S3_FORCE_PATH_STYLE", "false", "-e", "AWS_REGION", "US-EAST-02A",
        "-e", "AWS_EC2_METADATA_DISABLED", "true", "--", "/busybox/sh", "-c", command,
    ]
    with (root / "kaniko-submit.log").open("w") as output:
        completed = subprocess.run(args, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"Regional Kaniko submission complete: {job_name}")


if __name__ == "__main__":
    main()
