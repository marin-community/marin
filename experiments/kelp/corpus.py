# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared program corpus for Kelp tree diffusion training and evaluation."""

# Toy corpus of 15 small Python functions used for training and evaluation.
# Each program is a standalone function covering basic arithmetic, comparisons,
# and control flow patterns.
TOY_CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n",
    "def div(a, b):\n    return a / b\n",
    "def neg(x):\n    return -x\n",
    "def square(x):\n    return x * x\n",
    "def double(x):\n    return x + x\n",
    "def is_positive(x):\n    return x > 0\n",
    "def is_zero(x):\n    return x == 0\n",
    "def identity(x):\n    return x\n",
    "def abs_val(x):\n    if x < 0:\n        return -x\n    return x\n",
    "def max_val(a, b):\n    if a > b:\n        return a\n    return b\n",
    "def min_val(a, b):\n    if a < b:\n        return a\n    return b\n",
    "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
    "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
]
