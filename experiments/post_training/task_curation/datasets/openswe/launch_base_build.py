# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the regional Kaniko build for the condensed OpenSWE python bases.

Follows the landed launch-kaniko pattern (the swe-rebench-v2 base build): a
sealed regional builder, one INTERNAL GHCR package, one tag per Python version,
digest-pinned destination, credentials only via the secret-delivered Docker
config. The job records each version's digest into the bindings manifest the
converter's ``condensed_base_bindings`` input requires.

Local validation builds must pass for every version before this job is
submitted; a version the conda channels cannot serve (2.7 on a modern miniconda,
for instance) fails its own build and stays out of the bindings, and the rows
naming it convert to typed exclusions.
"""

import base64
import json
import os
import subprocess
from pathlib import Path

from experiments.post_training.task_curation.datasets.openswe import bases
from experiments.post_training.task_curation.datasets.openswe.sources import UPSTREAM_REVISION

BUILDER = (
    "ghcr.io/marin-community/rl-data-openresearcher"
    "@sha256:9e4985160e33e3c448da42d4b2140807267961f426f88720381e2578377f0d60"
)
PACKAGE = "ghcr.io/marin-community/rl-data-openswe"
CONTEXT_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build/context.tar.gz"
BINDINGS_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build/condensed-base-bindings.json"
TAG_PREFIX = "python"


def ghcr_auth() -> str:
    config = json.loads((Path.home() / ".docker" / "config.json").read_text())
    helper = config.get("credHelpers", {}).get("ghcr.io") or config["credsStore"]
    credential = json.loads(subprocess.check_output([f"docker-credential-{helper}", "get"], input=b"ghcr.io\n"))
    return base64.b64encode((credential["Username"] + ":" + credential["Secret"]).encode()).decode()


def builds_command(versions: tuple[str, ...], digest_dir: str = "/tmp") -> str:
    """The per-version Kaniko build commands; a failed version reports and does not stop."""
    return " ; ".join(
        f"/kaniko/executor "
        f"--context={CONTEXT_URI} "
        f"--dockerfile=python{version.replace('.', '_')}.Dockerfile "
        f"--destination={PACKAGE}:{TAG_PREFIX}-{version} "
        "--custom-platform=linux/amd64 --single-snapshot --cache=false --compressed-caching=false --reproducible "
        f"--image-name-with-digest-file={digest_dir}/python{version.replace('.', '_')}-digest.txt || "
        f"echo 'BASE_BUILD_FAILED python-{version}'; "
        f"cat {digest_dir}/python{version.replace('.', '_')}-digest.txt 2>/dev/null || true"
        for version in versions
    )


def bindings_collector(versions: tuple[str, ...], digest_dir: str = "/tmp") -> str:
    """The Python collector that records the built versions into the bindings manifest.

    Each build's digest line is kaniko's full destination with its digest
    (``ghcr.io/...:python-<version>@sha256:...``) — already a digest-pinned
    reference. A version without a digest file stayed unbuilt and stays out of
    the bindings, so the converter's rows naming it become typed exclusions.
    """
    version_list = json.dumps(list(versions))
    return f"""import json, pathlib
images = {{}}
for version in {version_list}:
    suffix = version.replace(".", "_")
    path = pathlib.Path({digest_dir!r}) / f"python{{suffix}}-digest.txt"
    if not path.exists():
        print(f"binding missing: python-{{version}}")
        continue
    images[version] = path.read_text().strip().splitlines()[-1]
bindings = {{"upstream_revision": {UPSTREAM_REVISION!r}, "miniconda_pin": {bases.MINICONDA3_PIN!r}, "images": images}}
pathlib.Path("condensed-base-bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\\n")
print(json.dumps(bindings, indent=2, sort_keys=True))
"""


def main() -> None:
    here = Path(__file__).parent
    root = here / "build"
    root.mkdir(parents=True, exist_ok=True)
    context = root / "context"
    context.mkdir(parents=True, exist_ok=True)
    for version in bases.PYTHON_VERSIONS:
        suffix = version.replace(".", "_")
        (context / f"python{suffix}.Dockerfile").write_text(bases.base_dockerfile(version))
    archive = root / "context.tar.gz"
    subprocess.run(["tar", "-czf", str(archive), "-C", str(context), "."], check=True)
    subprocess.run(
        ["uv", "run", "--no-sync", "fsutil", "cp", str(archive), CONTEXT_URI],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    environment = {**os.environ, "ATLAS_GHCR_DOCKER_CONFIG": json.dumps({"auths": {"ghcr.io": {"auth": ghcr_auth()}}})}
    collect = (
        "set -eu; "
        + builds_command(bases.PYTHON_VERSIONS)
        + " ; python3 - <<'PYEOF'\n"
        + bindings_collector(bases.PYTHON_VERSIONS)
        + "PYEOF"
    )
    job_name = "swe-zero-openswe-bases-v1"
    args = [
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
        "batch",
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
        collect,
    ]
    with (root / "kaniko-submit.log").open("w") as output:
        completed = subprocess.run(args, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"Regional Kaniko submission complete: {job_name}; bindings manifest: {BINDINGS_URI}")


if __name__ == "__main__":
    main()
