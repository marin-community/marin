# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a recorded exchange using the worker's object-storage dependencies."""

import argparse
import tarfile
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    fs, path = filesystem_for(args.destination)
    with fs.open(path, "wb") as output, tarfile.open(fileobj=output, mode="w|gz") as archive:
        for file in sorted(args.source.iterdir()):
            archive.add(file, arcname=file.name)


if __name__ == "__main__":
    main()
