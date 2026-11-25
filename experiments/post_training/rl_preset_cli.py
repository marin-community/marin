# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a model's default preset inside a Git archive."""

import argparse
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path


def describe(entry_module: str, model: str, preset_name: str) -> dict | None:
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415 - reserve stdout for JSON

    from experiments.post_training.rl_preset_plan import compose_configs, preset_launch_yaml  # noqa: PLC0415
    from experiments.post_training.rl_presets import ModelPresets  # noqa: PLC0415

    entry = import_module(entry_module)
    if not entry.__file__ or not Path(entry.__file__).resolve().is_relative_to(Path.cwd()):
        raise ValueError(f"{entry_module} did not load from the archive")
    table = getattr(entry, model.upper(), None)
    if not isinstance(table, ModelPresets) or table.model != model:
        # Historical entry modules can import the model's table under a local name.
        tables = {
            id(value): value
            for value in vars(entry).values()
            if isinstance(value, ModelPresets) and value.model == model
        }
        table = next(iter(tables.values())) if len(tables) == 1 else None
    if table is None or preset_name not in table.by_name:
        return None
    configure_coreweave_s3()
    preset = table.by_name[preset_name]
    with tempfile.TemporaryDirectory(prefix="rl-main-config-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(preset_launch_yaml(preset, name="main-base"))
        # The caller selected this checkout and already cleaned the Python environment.
        (config,) = compose_configs([str(path)], env=os.environ)
        return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe",))
    parser.add_argument("entry_module")
    parser.add_argument("model")
    parser.add_argument("preset")
    args = parser.parse_args()

    with redirect_stdout(sys.stderr):
        config = describe(args.entry_module, args.model, args.preset)
    print(json.dumps(config))


if __name__ == "__main__":
    main()
