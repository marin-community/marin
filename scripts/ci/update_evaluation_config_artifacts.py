# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply verified Harbor and Evalchemy resolver manifests to Marin's pinned inputs.

The caller downloads release manifests itself. This script performs no network I/O: it validates
their structure, rewrites only explicitly delimited generated blocks, and reports whether source
inputs changed. The workflow then refreshes ``uv.lock`` and proposes the change in a Marin PR.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPin:
    package: str
    repository: str
    revision: str
    release_tag: str
    wheel_url: str
    wheel_sha256: str
    schema_fingerprint: str
    resolver_fingerprint: str | None

    @property
    def wheel_requirement(self) -> str:
        return f"{self.package} @ {self.wheel_url}#sha256={self.wheel_sha256}"


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {key!r} must be a non-empty string")
    return value


def load_harbor_manifest(path: Path) -> ArtifactPin:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or _string(data, "package") != "harbor-config":
        raise ValueError(f"{path}: not a harbor-config manifest")
    return ArtifactPin(
        package="harbor-config",
        repository=_string(data, "release_repository"),
        revision=_string(data, "parent_commit"),
        release_tag=_string(data, "release_tag"),
        wheel_url=_string(data, "wheel_url"),
        wheel_sha256=_string(data, "sha256"),
        schema_fingerprint=_string(data, "schema_fingerprint"),
        resolver_fingerprint=_string(data, "resolver_fingerprint"),
    )


def load_evalchemy_manifest(path: Path) -> ArtifactPin:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or _string(data, "package") != "evalchemy-config":
        raise ValueError(f"{path}: not an evalchemy-config manifest")
    wheel = data.get("wheel")
    if not isinstance(wheel, dict):
        raise ValueError(f"{path}: wheel must be an object")
    return ArtifactPin(
        package="evalchemy-config",
        repository="marin-community/evalchemy",
        revision=_string(data, "evalchemy_revision"),
        release_tag=_string(data, "release_tag"),
        wheel_url=_string(wheel, "url"),
        wheel_sha256=_string(wheel, "sha256"),
        schema_fingerprint=_string(data, "schema_fingerprint"),
        resolver_fingerprint=None,
    )


def _python_pin(name: str, artifact: ArtifactPin) -> str:
    resolver_fingerprint = (
        f'    resolver_fingerprint="{artifact.resolver_fingerprint}",\n' if artifact.resolver_fingerprint else ""
    )
    return (
        f"{name} = ConfigArtifact(\n"
        f'    package="{artifact.package}",\n'
        f'    repository="{artifact.repository}",\n'
        f'    revision="{artifact.revision}",\n'
        f'    release_tag="{artifact.release_tag}",\n'
        f'    wheel_url="{artifact.wheel_url}",\n'
        f'    wheel_sha256="{artifact.wheel_sha256}",\n'
        f'    schema_fingerprint="{artifact.schema_fingerprint}",\n'
        f"{resolver_fingerprint}"
        ")\n"
    )


def replace_generated_block(path: Path, begin: str, end: str, content: str) -> bool:
    text = path.read_text()
    start = text.find(begin)
    finish = text.find(end)
    if start < 0 or finish < 0 or finish < start:
        raise ValueError(f"{path}: missing generated block {begin!r}")
    replacement = f"{begin}\n{content}{end}"
    updated = text[:start] + replacement + text[finish + len(end) :]
    if updated == text:
        return False
    path.write_text(updated)
    return True


def apply_pins(repo_root: Path, harbor: ArtifactPin, evalchemy: ArtifactPin) -> bool:
    changed = False
    changed |= replace_generated_block(
        repo_root / "lib/marin/src/marin/evaluation/config_artifacts.py",
        "# BEGIN GENERATED CONFIG ARTIFACT PINS",
        "# END GENERATED CONFIG ARTIFACT PINS",
        _python_pin("HARBOR_CONFIG", harbor) + "\n" + _python_pin("EVALCHEMY_CONFIG", evalchemy),
    )
    changed |= replace_generated_block(
        repo_root / "lib/marin/pyproject.toml",
        "    # BEGIN GENERATED CONFIG ARTIFACT DEPENDENCIES",
        "    # END GENERATED CONFIG ARTIFACT DEPENDENCIES",
        f'    "{harbor.wheel_requirement}",\n    "{evalchemy.wheel_requirement}",\n',
    )
    changed |= replace_generated_block(
        repo_root / "pyproject.toml",
        "# BEGIN GENERATED HARBOR RUNTIME PIN",
        "# END GENERATED HARBOR RUNTIME PIN",
        f'harbor = {{ git = "https://github.com/{harbor.repository}.git", rev = "{harbor.revision}" }}\n',
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harbor-manifest", type=Path, required=True)
    parser.add_argument("--evalchemy-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    changed = apply_pins(
        args.repo_root, load_harbor_manifest(args.harbor_manifest), load_evalchemy_manifest(args.evalchemy_manifest)
    )
    print("changed" if changed else "unchanged")


if __name__ == "__main__":
    main()
