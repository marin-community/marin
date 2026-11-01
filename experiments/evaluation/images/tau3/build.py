# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse or build the pinned Tau3 Pi/ACP image in GHCR."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

RECIPE = Path(__file__).parent
DEFAULT_REPOSITORY = "ghcr.io/marin-community/rollout-images"
LOCKS = ("package-lock.json", "mcp.lock", "acp.lock")


def dependency_identity(base_image: str | None = None) -> tuple[dict[str, Any], str]:
    """Hash dependency locks and declared build configuration, without reading source code."""
    dependencies = json.loads((RECIPE / "dependencies.json").read_text())
    if base_image is not None:
        dependencies["base_image"] = base_image
    if "@sha256:" not in dependencies["base_image"]:
        raise ValueError("The parent image must be pinned by digest")
    identity = {
        "dependencies": dependencies,
        "locks": {name: (RECIPE / name).read_text() for name in LOCKS},
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


def build_image(repository: str, base_image: str | None = None) -> dict[str, Any]:
    """Reuse the dependency tag, or build and push it, and return immutable image references."""
    dependencies, identity = dependency_identity(base_image)
    tag = f"{repository}:tau3-pi-{identity[:16]}"
    digest = registry_digest(tag)
    if digest is None:
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                f"--platform={dependencies['platform']}",
                "--provenance=false",
                "--output=type=registry,compression=gzip,force-compression=true",
                f"--tag={tag}",
                f"--build-arg=BASE_IMAGE={dependencies['base_image']}",
                f"--build-arg=NODE_VERSION={dependencies['node_version']}",
                f"--build-arg=NODE_SHA256={dependencies['node_sha256']}",
                str(RECIPE),
            ],
            check=True,
        )
        digest = registry_digest(tag)
        if digest is None:
            raise RuntimeError(f"The pushed image is missing: {tag}")
    return {
        "identity": identity,
        "tag": tag,
        "main": f"{repository}@{digest}",
        "runtime_images": dependencies["runtime_images"],
        "dependencies": dependencies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--base-image", help="Digest-pinned canonical Tau3 main image; included in the identity.")
    parser.add_argument("--output", type=Path, help="Write resolved image references as JSON.")
    parser.add_argument("--plan", action="store_true", help="Print the dependency tag without contacting GHCR.")
    args = parser.parse_args()
    if args.plan:
        dependencies, identity = dependency_identity(args.base_image)
        print(
            json.dumps(
                {"identity": identity, "tag": f"{args.repository}:tau3-pi-{identity[:16]}", **dependencies}, indent=2
            )
        )
        return
    if args.output is None:
        parser.error("--output is required when building or resolving an image")
    record = build_image(args.repository, args.base_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(record["main"])


if __name__ == "__main__":
    main()
