# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose launch documents or read schema defaults in the SkyRL CPU environment."""

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

def verify_checkout(checkout: Path, *modules: ModuleType) -> None:
    """Require the native modules to come from the selected SkyRL checkout."""
    checkout = checkout.resolve()
    for module in modules:
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(checkout):
            raise RuntimeError(f"{module.__name__} loaded from outside the vendor SkyRL checkout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        required=True,
        help="checkout that supplies native modules",
    )
    parser.add_argument("--progress-file", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose", help="compose native launch documents")
    compose.add_argument("paths", nargs="*")
    compose.add_argument("--validate-driver", action="store_true")
    schema = commands.add_parser("schema-defaults", help="read native defaults for a set of configuration groups")
    schema.add_argument("groups_json", metavar="groups-json")
    args = parser.parse_args()

    def stage(message: str) -> None:
        temporary = args.progress_file.with_suffix(".tmp")
        temporary.write_text(message)
        temporary.replace(args.progress_file)

    stage("Import native modules")
    from omegaconf import OmegaConf

    import marinskyrl
    import skyrl_train
    from cloud.iris import launch_config, rl_config_translation

    verify_checkout(args.checkout, sys.modules[__name__], launch_config, rl_config_translation, skyrl_train, marinskyrl)

    if args.command == "compose":
        from rigging.filesystem.s3_compat import configure_coreweave_s3

        configure_coreweave_s3()
        configs = []
        for index, path in enumerate(args.paths, 1):
            stage(f"Compose document {index}/{len(args.paths)}")
            configs.append(launch_config.load_launch_config(path))
        stage("Convert native configurations")
        documents = [OmegaConf.to_container(config, resolve=True) for config in configs]
        if args.validate_driver:
            for index, config in enumerate(configs, 1):
                stage(f"Validate driver for document {index}/{len(configs)}")
                launch_config.validate_launch_driver_config(config)
        print(json.dumps(documents))
    else:
        stage("Read native schema defaults")
        groups = json.loads(args.groups_json)
        config = rl_config_translation.compose_config_groups(groups, config_name="ppo_base_config")
        defaults = OmegaConf.to_container(
            OmegaConf.structured(launch_config.SkyRLLaunchConfig), resolve=False, enum_to_str=True
        )
        defaults["skyrl"] = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
        print(json.dumps(defaults))


if __name__ == "__main__":
    main()
