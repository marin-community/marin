# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
from pathlib import Path

from cloud.iris import launch_config
from cloud.iris.rl_config_translation import RL_ENTRYPOINTS
from skyrlrecipe import RLEntrypoint


def check_vendored_launcher() -> None:
    source = Path(launch_config.__file__).resolve()
    expected = Path(__file__).resolve().parents[2] / "vendor/skyrl/cloud/iris/launch_config.py"
    if source != expected:
        raise ValueError(f"launcher module {source} differs from vendored source {expected}")


def load_documents(source: Path, output: Path) -> None:
    failures = {}
    with tempfile.TemporaryDirectory(prefix="skyrl-launch-documents-") as directory:
        for index, (case, yaml) in enumerate(json.loads(source.read_text()).items()):
            path = Path(directory) / f"{index}.yaml"
            path.write_text(yaml, encoding="utf-8")
            try:
                config = launch_config.load_launch_config(path)
                if (
                    case == "inventory/train"
                    and config.runtime.entrypoint != RL_ENTRYPOINTS[RLEntrypoint.TASKCOMPENDIUM]
                ):
                    raise ValueError("inventory training must resolve to the taskcompendium entrypoint")
            except Exception as error:
                failures[case] = f"{type(error).__name__}: {error}"
    output.write_text(json.dumps(failures))


def main() -> None:
    source, output = map(Path, sys.argv[1:])
    check_vendored_launcher()
    load_documents(source, output)


if __name__ == "__main__":
    main()
