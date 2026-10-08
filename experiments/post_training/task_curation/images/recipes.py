# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Container image recipes kept in this repository and built by ``images.build``."""

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


@dataclass(frozen=True)
class ImageRecipe:
    """An image built from ``context``, a directory holding its Dockerfile and ``requirements.lock``.

    ``packages`` are further directories the Dockerfile copies with ``COPY --from=<directory name>``.
    The bytes of every file in ``context`` and ``packages`` enter the image identity.
    """

    name: str
    context: Path
    packages: tuple[Path, ...] = ()


GRADER = ImageRecipe("grader", HERE / "grader", packages=(REPO_ROOT / "lib" / "verifyit" / "src" / "verifyit",))
"""The image every sandboxed grader in the catalog runs in."""

RECIPES = {recipe.name: recipe for recipe in (GRADER,)}
