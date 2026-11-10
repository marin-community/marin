# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a production launch document in SkyRL's frozen CPU environment."""

import hashlib
import json
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config
from cloud.iris.rl_config_translation import RL_ENTRYPOINTS, RLEntrypoint
from omegaconf import OmegaConf
from skyrl_train import objective  # noqa: F401 - populate the runtime policy-loss registry
from skyrl_train.utils.utils import validate_cfg, validate_generator_cfg


def main(source: Path, destination: Path) -> None:
    project = Path(__file__).resolve().parents[3] / "vendor/skyrl"
    config = load_launch_config(source)
    if config.runtime.entrypoint in {
        RL_ENTRYPOINTS[RLEntrypoint.GENERATE],
        RL_ENTRYPOINTS[RLEntrypoint.TERMINAL_BENCH_GENERATE],
    }:
        validate_generator_cfg(config.skyrl)
    else:
        validate_cfg(config.skyrl)
    compiled = OmegaConf.to_yaml(config, resolve=True)
    destination.write_text(compiled)
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
                "validator": "load_launch_config + skyrl_train.utils.utils.validate_cfg",
                "runtime_entrypoint": config.runtime.entrypoint,
                "python": sys.version,
                "skyrl_lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
