# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path


def validate_slack_summary(path: Path) -> None:
    """Require a non-empty Slack summary from canary triage."""
    if not path.is_file() or not path.read_text().strip():
        raise ValueError(f"Canary triage finished without writing a non-empty {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate_slack_summary(args.path)


if __name__ == "__main__":
    main()
