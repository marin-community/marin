# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the regional Kaniko build of the upstream-fidelity OpenSWE python bases.

The upstream ``openswe-python-*`` bases are never published, so the fidelity
gate builds its own instance of the upstream ``prepare_baseimg.py`` recipe —
the same bytes as the condensed bases — and pins the digests it records into
the ``upstream-base-bindings.json`` manifest. The gate's UPSTREAM arm binds
``FROM openswe-python-<v>`` to these references; building them as a separate
Kaniko instance keeps the gate's two arms on two independent builds of the
same recipe.

Follows the landed launch-kaniko pattern: a sealed regional builder, one
INTERNAL GHCR package, one tag per Python version, digest-pinned destination,
credentials only via the secret-delivered Docker config. The job's collector
prints the bindings manifest to the job log; the operator stages it to
``BINDINGS_URI`` after the job seals.
"""

import json
import os
import subprocess
from pathlib import Path

from experiments.post_training.task_curation.datasets.openswe import bases
from experiments.post_training.task_curation.datasets.openswe.launch_base_build import (
    BUILDER,
    ghcr_auth,
)
from experiments.post_training.task_curation.datasets.openswe.sources import UPSTREAM_REVISION

PACKAGE = "ghcr.io/marin-community/rl-data-openswe"
CONTEXT_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build/upstream-fidelity-context.tar.gz"
BINDINGS_URI = "s3://marin-us-east-02a/marin/rl-data/inputs/openswe/base-build/upstream-fidelity-bindings.json"
TAG_PREFIX = "upstream-python"
JOB_NAME = "swe-zero-openswe-upstream-fidelity-bases-v1"


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
    """The Python collector that records the built versions into the bindings manifest."""
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
pathlib.Path("upstream-base-bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\\n")
print(json.dumps(bindings, indent=2, sort_keys=True))
"""


def main() -> None:
    here = Path(__file__).parent
    root = here / "build"
    root.mkdir(parents=True, exist_ok=True)
    context = root / "upstream-fidelity-context"
    context.mkdir(parents=True, exist_ok=True)
    for version in bases.PYTHON_VERSIONS:
        suffix = version.replace(".", "_")
        (context / f"python{suffix}.Dockerfile").write_text(bases.base_dockerfile(version))
    archive = root / "upstream-fidelity-context.tar.gz"
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
        "interactive",
        "--no-preemptible",
        "--job-name",
        JOB_NAME,
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
    with (root / "upstream-kaniko-submit.log").open("w") as output:
        completed = subprocess.run(args, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"Regional Kaniko submission complete: {JOB_NAME}; bindings manifest: {BINDINGS_URI}")


if __name__ == "__main__":
    main()
