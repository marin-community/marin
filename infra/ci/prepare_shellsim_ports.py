# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Warm test ports before pytest and emit a catalog with local asset URLs."""

import argparse
import json
import os
from pathlib import Path

import fsspec
import shellsim

DEFAULT_CATALOG = "https://shellsim.rjp.io/ports/catalog.json"
TEST_PACKAGES = ("numpy", "pytest")


def prepare_test_catalog(catalog: str, output: Path) -> str:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "shellsim/ports"
    with fsspec.open(catalog, "rt") as source:
        data = json.load(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data))
    shellsim.Context(pypi_catalog=output, cache_dir=cache).Environment(pypi_packages=TEST_PACKAGES)

    # Shellsim stores portable assets at their catalog paths. Unprepared assets
    # also get local URLs, so a test cannot accidentally download a new port.
    def localize(value: object) -> None:
        if isinstance(value, dict):
            if "url" in value and "path" in value:
                value["url"] = (cache / "portable" / value["path"]).resolve().as_uri()
            for child in value.values():
                localize(child)
        elif isinstance(value, list):
            for child in value:
                localize(child)

    localize(data)
    output.write_text(json.dumps(data))
    return output.resolve().as_uri()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=Path("build/shellsim-tests/catalog.json"))
    args = parser.parse_args()
    print(prepare_test_catalog(args.catalog, args.output))


if __name__ == "__main__":
    main()
