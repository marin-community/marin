# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI preflight check for Levanter's DeepEP JAX FFI integration."""

from __future__ import annotations

import argparse
import sys

from levanter.kernels.deepep.availability import (
    LAYOUT_REQUIRED_FILES,
    TRANSPORT_REQUIRED_FILES,
    deepep_install_help,
    deepep_preflight_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("layout", "transport"),
        default="transport",
        help="DeepEP FFI component to check. Transport includes layout dependencies.",
    )
    args = parser.parse_args(argv)

    required_files = LAYOUT_REQUIRED_FILES if args.component == "layout" else TRANSPORT_REQUIRED_FILES
    cache_component = "deepep_layout_ffi" if args.component == "layout" else "deepep_transport_ffi"
    status = deepep_preflight_status(required_files=required_files, component=cache_component)

    print("DeepEP preflight")
    print(f"  component: {args.component}")
    print(f"  source_root: {status.source_root or '<unset>'}")
    print(f"  cache_root: {status.cache_root}")
    print(f"  cuda_arch: {status.cuda_arch}")
    print(f"  nvcc: {status.nvcc_path or '<missing>'}")
    for warning in status.warnings:
        print(f"  warning: {warning}")
    for error in status.errors:
        print(f"  error: {error}")

    if not status.ok:
        print()
        print(deepep_install_help())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
