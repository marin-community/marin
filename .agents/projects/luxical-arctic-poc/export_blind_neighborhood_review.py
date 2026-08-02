# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream one private blind-neighborhood review package from storage."""

import argparse
import base64
import gzip

from rigging.filesystem import StoragePath

REVIEW_CHUNK_MARKER = "BLIND_NEIGHBORHOOD_REVIEW_CHUNK="
REVIEW_CHUNK_SIZE = 8_000


def export_package(package_url: str) -> None:
    """Write a compressed package as bounded output chunks."""
    payload = StoragePath(package_url).read_text(compression="gzip").encode()
    encoded = base64.b64encode(gzip.compress(payload)).decode()
    chunks = [encoded[start : start + REVIEW_CHUNK_SIZE] for start in range(0, len(encoded), REVIEW_CHUNK_SIZE)]
    for index, chunk in enumerate(chunks):
        print(f"{REVIEW_CHUNK_MARKER}{index:04d}/{len(chunks):04d}:{chunk}")


def main() -> None:
    """Parse arguments and stream one review package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-url", required=True)
    args = parser.parse_args()
    export_package(args.package_url)


if __name__ == "__main__":
    main()
