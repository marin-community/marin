"""Run a registered SkyRL entrypoint from a composed launch configuration."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, cast

from omegaconf import DictConfig, OmegaConf, open_dict

from cloud.iris.launch_config import load_launch_config
from cloud.iris.role_topology import derive_role_topology
from cloud.iris.rl_config_translation import registered_rl_entrypoint_module


def run_config(config_path: Path) -> None:
    """Load one launch config and call its registered SkyRL function."""
    config = load_launch_config(config_path)
    module = registered_rl_entrypoint_module(str(config.runtime.entrypoint))
    run = cast(Callable[[DictConfig], None], getattr(importlib.import_module(module), "run"))
    if config.skyrl.get("trainer", {}).get("mismatch_probe", {}).get("enabled"):
        with open_dict(config.skyrl):
            config.skyrl.runtime = {
                "checkpoint_path": str(config.skyrl.trainer.resume_path or config.inputs.model.uri),
            }
    topology = derive_role_topology(OmegaConf.to_container(config, resolve=True))
    if topology is not None:
        with open_dict(config.skyrl):
            if "runtime" not in config.skyrl:
                config.skyrl.runtime = {}
            config.skyrl.runtime.placement = {
                "policy": topology.selector("policy"),
                "engines": [topology.selector(role) for role in topology.roles if role != "policy"],
            }
    run(config.skyrl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
