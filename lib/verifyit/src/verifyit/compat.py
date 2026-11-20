# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backports for grader interpreters below Python 3.11.

The staged runtime imports verifyit on the grader machine's own interpreter, and the declared
grader environments include Python 3.10 images.
"""

import sys
from enum import Enum

# Ruff evaluates this module at the repo's 3.12 floor; verifyit runs on 3.10 grader machines.
if sys.version_info >= (3, 11):  # noqa: UP036
    import tomllib
    from enum import StrEnum
else:
    import tomli as tomllib

    class StrEnum(str, Enum):
        """The ``enum.StrEnum`` contract: members are strings and ``str()`` is the value."""

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum", "tomllib"]
