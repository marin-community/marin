# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
from pathlib import Path

from cloud.iris import launch_config
from omegaconf import OmegaConf


def installed_launcher() -> None:
    source = Path(launch_config.__file__).resolve()
    expected = Path(__file__).resolve().parents[2] / "vendor/skyrl/cloud/iris/launch_config.py"
    if source != expected:
        raise ValueError(f"launcher module {source} differs from vendored source {expected}")


def load_documents(source: Path, output: Path) -> None:
    loaded = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="skyrl-launch-documents-") as directory:
        for index, entry in enumerate(json.loads(source.read_text())):
            path = Path(directory) / f"{index}.yaml"
            path.write_text(json.dumps(entry["document"]))
            try:
                config = launch_config.load_launch_config(path)
                document = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
                loaded.append({"case": entry["case"], "document": document})
            except Exception as error:
                failures.append({"case": entry["case"], "error_type": type(error).__name__, "message": str(error)})
    output.write_text(
        json.dumps(
            {
                "documents": loaded,
                "failures": failures,
            }
        )
    )
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a launch-document batch with the installed launcher.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    installed_launcher()
    load_documents(args.input, args.output)


if __name__ == "__main__":
    main()
