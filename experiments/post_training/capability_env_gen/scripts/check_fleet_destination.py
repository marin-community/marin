#!/usr/bin/env python3
"""Reject a fleet destination that already contains any remote run state."""

from __future__ import annotations

import argparse


def require_empty(fs, prefix: str) -> None:
    if fs.exists(prefix.rstrip("/")):
        raise ValueError("fleet destination already exists; observe the prior run")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        import fsspec
        from rigging.filesystem.s3_compat import configure_coreweave_s3

        configure_coreweave_s3()
        fs, prefix = fsspec.core.url_to_fs(args.destination)
        require_empty(fs, prefix)
    except Exception as error:  # noqa: BLE001 - do not print credential-bearing transport errors.
        parser.exit(2, f"fleet destination preflight failed: {type(error).__name__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
