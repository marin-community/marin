# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage one intake input from object storage to a local path.

The pending-intake conversion job runs this before the driver so the driver's
``--license-receipts`` points at a local file (the receipt reader takes a plain
path).
"""

import argparse
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True, help="The object-storage uri to stage.")
    parser.add_argument("--output", required=True, type=Path, help="The local destination path.")
    args = parser.parse_args()

    configure_coreweave_s3()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(StoragePath(args.uri).read_bytes())
    print(f"staged {args.uri} -> {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
