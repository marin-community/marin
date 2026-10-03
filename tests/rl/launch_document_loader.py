# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path

from cloud.iris import launch_config
from omegaconf import OmegaConf


@dataclass(frozen=True)
class LauncherProvenance:
    commit: str
    source: Path


def installed_launcher(expected_commit: str) -> LauncherProvenance:
    metadata = distribution("marinskyrl")
    provenance = metadata.read_text("direct_url.json")
    if provenance is None:
        raise ValueError("installed launcher has no Git provenance")
    direct_url = json.loads(provenance)
    commit = direct_url["vcs_info"]["commit_id"]
    if commit != expected_commit:
        raise ValueError(f"installed launcher commit {commit} differs from {expected_commit}")
    source = Path(launch_config.__file__).resolve()
    if not source.is_relative_to(Path(metadata.locate_file("cloud")).resolve()):
        raise ValueError(f"launcher module is outside the installed distribution: {source}")
    return LauncherProvenance(commit, source)


def load_documents(source: Path, output: Path, launcher: LauncherProvenance) -> None:
    loaded = []
    failures = []
    start = time.monotonic()
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
                "installed_commit": launcher.commit,
                "launcher_source": str(launcher.source),
                "load_duration": time.monotonic() - start,
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
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    load_documents(args.input, args.output, installed_launcher(args.expected_commit))


if __name__ == "__main__":
    main()
