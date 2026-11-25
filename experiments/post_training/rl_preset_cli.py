# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose a model's default preset inside a Git archive."""

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from importlib import import_module
from pathlib import Path


def describe(entry_module: str, model: str, preset_name: str) -> dict | None:
    from marin.execution.lazy import materialized_config  # noqa: PLC0415 - reserve stdout for JSON
    from marin.rl.skyrl import ArtifactHfModel, skyrl_cpu_command  # noqa: PLC0415
    from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

    from experiments.post_training import rl_compose as recipe  # noqa: PLC0415
    from experiments.post_training.rl_presets import ModelPresets  # noqa: PLC0415

    entry = import_module(entry_module)
    if not entry.__file__ or not Path(entry.__file__).resolve().is_relative_to(Path.cwd()):
        raise ValueError(f"{entry_module} did not load from the archive")
    tables = {
        id(table): table
        for table in vars(entry).values()
        if isinstance(table, ModelPresets) and table.model == model and preset_name in table.by_name
    }
    if len(tables) != 1:
        return None
    (table,) = tables.values()
    preset = table.by_name[preset_name]
    configure_coreweave_s3()
    spec = recipe.run_spec(preset, name="main-base", model_name=model)
    if isinstance(spec.model, ArtifactHfModel) and spec.model.relative_path is None:
        spec = replace(spec, model=replace(spec.model, relative_path="<terminal-hf-export>"))
    step = recipe.run_step(preset, spec)
    with tempfile.TemporaryDirectory(prefix="rl-main-config-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(materialized_config(step, recipe.run_prefix(preset)).launch_config_yaml)
        result = subprocess.run(
            skyrl_cpu_command(
                "python",
                "-I",
                "-m",
                "cloud.iris.config_cli",
                "--checkout",
                str(Path.cwd() / "vendor/skyrl"),
                "--progress-file",
                str(Path(directory) / "stage"),
                "compose",
                "--",
                str(path),
            ),
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            timeout=120,
        )
        (config,) = json.loads(result.stdout)
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
