# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse or build the pinned Tau3 Pi/ACP image in GHCR."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

RECIPE = Path(__file__).parent
DEFAULT_REPOSITORY = "ghcr.io/marin-community/rollout-images"
BUILD_INPUTS = ("Dockerfile", "package.json", "package-lock.json", "requirements.lock")


@dataclass(frozen=True)
class Tau3Dependencies:
    platform: str
    base_image: str
    tau2_bench_commit: str
    node_version: str
    node_sha256: str
    runtime_images: dict[str, str]


DEPENDENCIES = Tau3Dependencies(
    platform="linux/amd64",
    base_image=(
        "ghcr.io/marin-community/rollout-images@sha256:17a0f60def260b484134ca7f137970735cd9d0a30900aeb1a037f59acac9484a"
    ),
    tau2_bench_commit="fc0055dc4e0a316c3f83133267fbd6faaa770992",
    node_version="22.23.4",
    node_sha256="b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406",
    runtime_images={
        "retail": (
            "ghcr.io/marin-community/rollout-images@sha256:25899b6e12e657f7850108b70dbae6ceb82dc9c988bd912355b22d7cdbdec24a"
        ),
        "airline": (
            "ghcr.io/marin-community/rollout-images@sha256:97798a165494fb42564dadae0eaa3b645c728d98004709e4e68c0cde2c8e6c98"
        ),
    },
)


def dependency_identity(base_image: str | None = None) -> tuple[Tau3Dependencies, str]:
    """Hash the image recipe, dependency locks, and declared build configuration."""
    dependencies = replace(DEPENDENCIES, base_image=base_image) if base_image is not None else DEPENDENCIES
    if "@sha256:" not in dependencies.base_image:
        raise ValueError("The parent image must be pinned by digest")
    identity = {
        "dependencies": asdict(dependencies),
        "inputs": {name: (RECIPE / name).read_text() for name in BUILD_INPUTS},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dependencies, digest


def registry_digest(image: str) -> str | None:
    """Resolve an image manifest, returning None only when the registry reports it absent."""
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image, "--format", "{{json .Manifest}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)["digest"]
    if "not found" in result.stderr.lower() or "manifest unknown" in result.stderr.lower():
        return None
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def tau3_image(repository: str = DEFAULT_REPOSITORY, base_image: str | None = None) -> str:
    """Return the prepared main image, building and pushing missing dependencies once."""
    dependencies, identity = dependency_identity(base_image)
    tag = f"{repository}:tau3-pi-{identity[:16]}"
    digest = registry_digest(tag)
    if digest is None:
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                f"--platform={dependencies.platform}",
                "--provenance=false",
                "--output=type=registry,compression=gzip,force-compression=true",
                f"--tag={tag}",
                f"--build-arg=BASE_IMAGE={dependencies.base_image}",
                f"--build-arg=NODE_VERSION={dependencies.node_version}",
                f"--build-arg=NODE_SHA256={dependencies.node_sha256}",
                str(RECIPE),
            ],
            check=True,
        )
        digest = registry_digest(tag)
        if digest is None:
            raise RuntimeError(f"The pushed image is missing: {tag}")
    return f"{repository}@{digest}"


def prepare_tau3_task(task: Path, repository: str = DEFAULT_REPOSITORY) -> None:
    """Pin a reviewed Tau3 task's images, preserving its server and configuration."""
    runtime = task / "environment/runtime-server"
    config = json.loads((runtime / "task_config.json").read_text())
    runtime_base = DEPENDENCIES.runtime_images[config["domain"]]
    main_image = tau3_image(repository)
    with TemporaryDirectory(prefix="tau3-image-") as directory:
        scratch = Path(directory)
        dockerfile = scratch / "Dockerfile"
        dockerfile.write_text(f"FROM {runtime_base}\nCOPY server.py task_config.json /app/\n")
        metadata = scratch / "metadata.json"
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--platform=linux/amd64",
                "--provenance=false",
                "--output=type=registry,compression=gzip,force-compression=true",
                f"--tag={repository}:tau3-runtime-{task.name}",
                f"--file={dockerfile}",
                f"--metadata-file={metadata}",
                str(runtime),
            ],
            check=True,
        )
        runtime_digest = json.loads(metadata.read_text())["containerimage.digest"]
    manifest = {
        "main": {"image": main_image, "workdir": "/app"},
        "tau3-runtime": {
            "image": f"{repository}@{runtime_digest}",
            "workdir": "/app",
            "agent_logs": "publish",
        },
    }
    (task / "environment/shellbox-images.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--base-image", help="Digest-pinned canonical Tau3 main image; included in the identity.")
    parser.add_argument("--task", type=Path, help="Prepare images and write shellbox-images.json for this Tau3 task.")
    parser.add_argument("--plan", action="store_true", help="Print the dependency tag without contacting GHCR.")
    args = parser.parse_args()
    if args.plan:
        dependencies, identity = dependency_identity(args.base_image)
        print(
            json.dumps(
                {"identity": identity, "tag": f"{args.repository}:tau3-pi-{identity[:16]}", **asdict(dependencies)},
                indent=2,
            )
        )
        return
    if args.task is not None:
        if args.base_image is not None:
            parser.error("--base-image applies only to the standalone main image build")
        prepare_tau3_task(args.task, args.repository)
    else:
        print(tau3_image(args.repository, args.base_image))


if __name__ == "__main__":
    main()
