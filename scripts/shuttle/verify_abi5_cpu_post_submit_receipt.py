# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify an ABI 5 CPU post-submit receipt against reviewed preparation output."""

import argparse
from pathlib import Path

from prepare_abi5_cpu_acceptance import load_and_validate_post_submit_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-report", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-init-image", required=True)
    arguments = parser.parse_args()
    load_and_validate_post_submit_receipt(
        arguments.preparation_report,
        arguments.receipt,
        expected_init_image=arguments.expected_init_image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
