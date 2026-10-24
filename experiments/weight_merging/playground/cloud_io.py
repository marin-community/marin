# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run fsutil with fail-closed CoreWeave routing for this experiment's S3 URLs."""

import sys
from urllib.parse import urlparse

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.fsutil.cli import main as fsutil_main


def require_coreweave(url: str) -> None:
    parsed = urlparse(url)
    spec = data_buckets().get(parsed.netloc)
    if spec is None or spec.store != StoreType.COREWEAVE or not spec.signing_region:
        raise ValueError(f"S3 destination lacks an explicit CoreWeave signing region: {url}")
    fs, _ = filesystem_for(url)
    host = urlparse(fs.endpoint_url or "").hostname or ""
    if host not in ("cwobject.com", "cwlota.com"):
        raise ValueError(f"Expected a CoreWeave endpoint for {url}; got {host!r}")


def main() -> None:
    for argument in sys.argv[1:]:
        if argument.startswith("s3://"):
            require_coreweave(argument)
    fsutil_main()


if __name__ == "__main__":
    main()
