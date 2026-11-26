# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the static review queue package."""

from pathlib import Path
from shutil import copytree


def main() -> None:
    root = Path(__file__).parent
    copytree(root / "web", root / "dist", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
