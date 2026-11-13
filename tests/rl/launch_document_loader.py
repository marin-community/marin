# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
from pathlib import Path

from cloud.iris import launch_config
from yaml import safe_load


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
                loaded = launch_config.load_launch_config(path).run
                written = safe_load(yaml)["run"]
                if (loaded.id, loaded.attempt_id) != (written["id"], written["attempt_id"]):
                    raise ValueError(f"run.id and run.attempt_id load as {loaded.id!r} and {loaded.attempt_id!r}")
            except Exception as error:
                failures[case] = f"{type(error).__name__}: {error}"
    output.write_text(json.dumps(failures))


def main() -> None:
    source, output = map(Path, sys.argv[1:])
    check_vendored_launcher()
    load_documents(source, output)


if __name__ == "__main__":
    main()
