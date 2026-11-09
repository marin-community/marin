# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a production launch document in SkyRL's frozen CPU environment."""

import hashlib
import json
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf


def main(source: Path, destination: Path) -> None:
    project = Path(__file__).resolve().parents[3] / "vendor/skyrl"
    config = load_launch_config(source)
    compiled = OmegaConf.to_yaml(config, resolve=True)
    destination.write_text(compiled)
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                "validator": "cloud.iris.launch_config.load_launch_config",
                "runtime_entrypoint": config.runtime.entrypoint,
                "python": sys.version,
                "skyrl_lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
