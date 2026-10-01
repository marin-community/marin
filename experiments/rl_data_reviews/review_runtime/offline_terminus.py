# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision Terminus-2 tools from checked local Debian packages for offline tasks."""

import json
import shlex
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2, Terminus2Options
from harbor.environments.base import BaseEnvironment
from review_io import digest, write_json

PACKAGE_DIR = "/tmp/atlas-agent-packages"


class OfflineTerminusOptions(Terminus2Options):
    debian_package_manifest: Path
    debian_package_manifest_sha256: str


class OfflineTerminus2(Terminus2):
    """Use native Terminus-2 after installing its tools without sandbox downloads."""

    options_model = OfflineTerminusOptions
    options: OfflineTerminusOptions

    async def setup(self, environment: BaseEnvironment) -> None:
        manifest = self.options.debian_package_manifest
        if digest(manifest) != self.options.debian_package_manifest_sha256:
            raise ValueError("Offline agent package manifest checksum mismatch")
        packages = json.loads(manifest.read_text())
        if not packages:
            raise ValueError("Provide the Debian packages required by Terminus-2")
        paths = []
        for package in packages:
            filename = package["filename"]
            if Path(filename).name != filename or not filename.endswith(".deb"):
                raise ValueError("Offline agent packages must be .deb filenames")
            path = manifest.parent / filename
            if digest(path) != package["sha256"]:
                raise ValueError(f"Offline agent package checksum mismatch: {filename}")
            paths.append(path)
        await environment.exec(f"mkdir -p {PACKAGE_DIR}", user="root", timeout_sec=30)
        targets = []
        for path in paths:
            target = f"{PACKAGE_DIR}/{path.name}"
            await environment.upload_file(path, target)
            targets.append(shlex.quote(target))
        result = await environment.exec("dpkg -i " + " ".join(targets) + " && tmux -V", user="root", timeout_sec=120)
        write_json(
            self.logs_dir / "offline-tooling.json",
            {
                "packages": packages,
                "return_code": result.return_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.return_code != 0:
            raise RuntimeError("Offline agent-tool installation failed; inspect agent/offline-tooling.json")
        await super().setup(environment)
