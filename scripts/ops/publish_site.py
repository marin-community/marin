#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an analysis HTML site to durable public hosting (``gs://marin-public``).

Uploads a page and records it as an Artifact, then prints its public URL::

    uv run scripts/ops/publish_site.py report.html --user rav --slug dedup-examples \
        --version 2026.07.01 --title "Dedup examples"

``<source>`` is a single ``.html`` file or a directory whose ``index.html`` is the entrypoint.
"""

import argparse
import sys
from pathlib import Path

from marin.publish.sites import InvalidSiteError, publish_site


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish an analysis HTML site to gs://marin-public and record it.")
    parser.add_argument("source", help="local .html file or directory with index.html")
    parser.add_argument("--user", required=True, help="author handle (lowercase kebab)")
    parser.add_argument("--slug", required=True, help="site slug (lowercase kebab)")
    parser.add_argument("--version", required=True, help="CalVer version YYYY.MM.DD[.N]")
    parser.add_argument("--title", required=True, help="human label for the discovery index")
    parser.add_argument("--summary", default="", help="one-line description for the index")
    args = parser.parse_args()

    try:
        site = publish_site(
            Path(args.source),
            user=args.user,
            slug=args.slug,
            version=args.version,
            title=args.title,
            summary=args.summary,
        )
    except (InvalidSiteError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(site.url)


if __name__ == "__main__":
    main()
