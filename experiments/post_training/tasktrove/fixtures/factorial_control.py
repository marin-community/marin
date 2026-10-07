# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


def calculate_factorial(number: int) -> int:
    """Return the factorial of a nonnegative integer number."""
    if number < 0:
        raise ValueError("negative number")
    if number == 0:
        return 1
    return number * calculate_factorial(number - 1)
