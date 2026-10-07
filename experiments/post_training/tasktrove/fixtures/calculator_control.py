# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


class Calculator:
    def add(self, a, b):
        """Return the sum of numbers a and b as a float."""
        return float(a + b)

    def subtract(self, a, b):
        """Return the difference of numbers a and b as a float."""
        return float(a - b)

    def multiply(self, a, b):
        """Return the product of numbers a and b as a float."""
        return float(a * b)

    def divide(self, a, b):
        """Return the quotient of numbers a and b as a float."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return float(a / b)
