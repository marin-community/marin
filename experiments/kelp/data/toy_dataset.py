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

"""Toy dataset of simple Python functions for initial training.

Provides a hardcoded list of ~100 simple Python functions with docstrings
for testing the tree diffusion training pipeline.
"""

from dataclasses import dataclass
from collections.abc import Iterator

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray


@dataclass
class ToyProgram:
    """A toy Python program with docstring and code."""

    docstring: str
    signature: str
    body: str

    @property
    def full_code(self) -> str:
        """Get the complete function code."""
        return f'{self.signature}\n    """{self.docstring}"""\n{self.body}'

    @property
    def prompt(self) -> str:
        """Get the conditioning prompt (docstring + signature)."""
        return f'"""{self.docstring}"""\n{self.signature}'


# Simple arithmetic and math functions
ARITHMETIC_PROGRAMS = [
    ToyProgram(
        docstring="Return the sum of two numbers.",
        signature="def add(a: int, b: int) -> int:",
        body="    return a + b",
    ),
    ToyProgram(
        docstring="Return the difference between two numbers.",
        signature="def subtract(a: int, b: int) -> int:",
        body="    return a - b",
    ),
    ToyProgram(
        docstring="Return the product of two numbers.",
        signature="def multiply(a: int, b: int) -> int:",
        body="    return a * b",
    ),
    ToyProgram(
        docstring="Return the quotient of two numbers.",
        signature="def divide(a: float, b: float) -> float:",
        body="    return a / b",
    ),
    ToyProgram(
        docstring="Return the remainder of dividing a by b.",
        signature="def modulo(a: int, b: int) -> int:",
        body="    return a % b",
    ),
    ToyProgram(
        docstring="Return a raised to the power of b.",
        signature="def power(a: int, b: int) -> int:",
        body="    return a ** b",
    ),
    ToyProgram(
        docstring="Return the absolute value of n.",
        signature="def absolute(n: int) -> int:",
        body="    return abs(n)",
    ),
    ToyProgram(
        docstring="Return the maximum of two numbers.",
        signature="def maximum(a: int, b: int) -> int:",
        body="    return max(a, b)",
    ),
    ToyProgram(
        docstring="Return the minimum of two numbers.",
        signature="def minimum(a: int, b: int) -> int:",
        body="    return min(a, b)",
    ),
    ToyProgram(
        docstring="Return the average of two numbers.",
        signature="def average(a: float, b: float) -> float:",
        body="    return (a + b) / 2",
    ),
]

# Factorial, fibonacci, and recursion
RECURSION_PROGRAMS = [
    ToyProgram(
        docstring="Compute the factorial of n.",
        signature="def factorial(n: int) -> int:",
        body="    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    ),
    ToyProgram(
        docstring="Return the n-th Fibonacci number.",
        signature="def fibonacci(n: int) -> int:",
        body="    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    ),
    ToyProgram(
        docstring="Compute the sum of integers from 1 to n.",
        signature="def sum_to_n(n: int) -> int:",
        body="    if n <= 0:\n        return 0\n    return n + sum_to_n(n - 1)",
    ),
    ToyProgram(
        docstring="Compute the n-th triangular number.",
        signature="def triangular(n: int) -> int:",
        body="    return n * (n + 1) // 2",
    ),
    ToyProgram(
        docstring="Compute the greatest common divisor of a and b.",
        signature="def gcd(a: int, b: int) -> int:",
        body="    while b:\n        a, b = b, a % b\n    return a",
    ),
]

# String manipulation
STRING_PROGRAMS = [
    ToyProgram(
        docstring="Reverse a string.",
        signature="def reverse_string(s: str) -> str:",
        body="    return s[::-1]",
    ),
    ToyProgram(
        docstring="Check if a string is a palindrome.",
        signature="def is_palindrome(s: str) -> bool:",
        body="    return s == s[::-1]",
    ),
    ToyProgram(
        docstring="Count the number of vowels in a string.",
        signature="def count_vowels(s: str) -> int:",
        body='    return sum(1 for c in s.lower() if c in "aeiou")',
    ),
    ToyProgram(
        docstring="Convert a string to uppercase.",
        signature="def to_upper(s: str) -> str:",
        body="    return s.upper()",
    ),
    ToyProgram(
        docstring="Convert a string to lowercase.",
        signature="def to_lower(s: str) -> str:",
        body="    return s.lower()",
    ),
    ToyProgram(
        docstring="Capitalize the first letter of each word.",
        signature="def title_case(s: str) -> str:",
        body="    return s.title()",
    ),
    ToyProgram(
        docstring="Remove leading and trailing whitespace.",
        signature="def strip_whitespace(s: str) -> str:",
        body="    return s.strip()",
    ),
    ToyProgram(
        docstring="Count occurrences of a character in a string.",
        signature="def count_char(s: str, c: str) -> int:",
        body="    return s.count(c)",
    ),
    ToyProgram(
        docstring="Replace all occurrences of old with new in s.",
        signature="def replace_all(s: str, old: str, new: str) -> str:",
        body="    return s.replace(old, new)",
    ),
    ToyProgram(
        docstring="Split a string into words.",
        signature="def split_words(s: str) -> list[str]:",
        body="    return s.split()",
    ),
]

# List operations
LIST_PROGRAMS = [
    ToyProgram(
        docstring="Return the sum of elements in a list.",
        signature="def list_sum(lst: list[int]) -> int:",
        body="    return sum(lst)",
    ),
    ToyProgram(
        docstring="Return the length of a list.",
        signature="def list_length(lst: list) -> int:",
        body="    return len(lst)",
    ),
    ToyProgram(
        docstring="Find the maximum element in a list.",
        signature="def find_max(lst: list[int]) -> int:",
        body="    return max(lst)",
    ),
    ToyProgram(
        docstring="Find the minimum element in a list.",
        signature="def find_min(lst: list[int]) -> int:",
        body="    return min(lst)",
    ),
    ToyProgram(
        docstring="Reverse a list.",
        signature="def reverse_list(lst: list) -> list:",
        body="    return lst[::-1]",
    ),
    ToyProgram(
        docstring="Return the first element of a list.",
        signature="def first(lst: list):",
        body="    return lst[0]",
    ),
    ToyProgram(
        docstring="Return the last element of a list.",
        signature="def last(lst: list):",
        body="    return lst[-1]",
    ),
    ToyProgram(
        docstring="Return a list without duplicates.",
        signature="def unique(lst: list) -> list:",
        body="    return list(set(lst))",
    ),
    ToyProgram(
        docstring="Sort a list in ascending order.",
        signature="def sort_ascending(lst: list[int]) -> list[int]:",
        body="    return sorted(lst)",
    ),
    ToyProgram(
        docstring="Sort a list in descending order.",
        signature="def sort_descending(lst: list[int]) -> list[int]:",
        body="    return sorted(lst, reverse=True)",
    ),
    ToyProgram(
        docstring="Concatenate two lists.",
        signature="def concat_lists(a: list, b: list) -> list:",
        body="    return a + b",
    ),
    ToyProgram(
        docstring="Flatten a nested list.",
        signature="def flatten(lst: list[list]) -> list:",
        body="    return [x for sublist in lst for x in sublist]",
    ),
]

# Boolean and comparison
BOOLEAN_PROGRAMS = [
    ToyProgram(
        docstring="Check if n is even.",
        signature="def is_even(n: int) -> bool:",
        body="    return n % 2 == 0",
    ),
    ToyProgram(
        docstring="Check if n is odd.",
        signature="def is_odd(n: int) -> bool:",
        body="    return n % 2 != 0",
    ),
    ToyProgram(
        docstring="Check if n is positive.",
        signature="def is_positive(n: int) -> bool:",
        body="    return n > 0",
    ),
    ToyProgram(
        docstring="Check if n is negative.",
        signature="def is_negative(n: int) -> bool:",
        body="    return n < 0",
    ),
    ToyProgram(
        docstring="Check if n is zero.",
        signature="def is_zero(n: int) -> bool:",
        body="    return n == 0",
    ),
    ToyProgram(
        docstring="Check if n is a prime number.",
        signature="def is_prime(n: int) -> bool:",
        body="    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    ),
    ToyProgram(
        docstring="Check if a number is within a range.",
        signature="def in_range(n: int, low: int, high: int) -> bool:",
        body="    return low <= n <= high",
    ),
    ToyProgram(
        docstring="Check if a list is empty.",
        signature="def is_empty(lst: list) -> bool:",
        body="    return len(lst) == 0",
    ),
]

# Conversion functions
CONVERSION_PROGRAMS = [
    ToyProgram(
        docstring="Convert an integer to a string.",
        signature="def int_to_str(n: int) -> str:",
        body="    return str(n)",
    ),
    ToyProgram(
        docstring="Convert a string to an integer.",
        signature="def str_to_int(s: str) -> int:",
        body="    return int(s)",
    ),
    ToyProgram(
        docstring="Convert a list to a set.",
        signature="def list_to_set(lst: list) -> set:",
        body="    return set(lst)",
    ),
    ToyProgram(
        docstring="Convert Celsius to Fahrenheit.",
        signature="def celsius_to_fahrenheit(c: float) -> float:",
        body="    return c * 9 / 5 + 32",
    ),
    ToyProgram(
        docstring="Convert Fahrenheit to Celsius.",
        signature="def fahrenheit_to_celsius(f: float) -> float:",
        body="    return (f - 32) * 5 / 9",
    ),
    ToyProgram(
        docstring="Convert a binary string to an integer.",
        signature="def binary_to_int(s: str) -> int:",
        body="    return int(s, 2)",
    ),
    ToyProgram(
        docstring="Convert an integer to a binary string.",
        signature="def int_to_binary(n: int) -> str:",
        body="    return bin(n)[2:]",
    ),
]

# Simple algorithms
ALGORITHM_PROGRAMS = [
    ToyProgram(
        docstring="Return the n-th element of a list, or default if out of bounds.",
        signature="def safe_get(lst: list, n: int, default=None):",
        body="    return lst[n] if 0 <= n < len(lst) else default",
    ),
    ToyProgram(
        docstring="Compute the mean of a list of numbers.",
        signature="def mean(lst: list[float]) -> float:",
        body="    return sum(lst) / len(lst)",
    ),
    ToyProgram(
        docstring="Count elements that satisfy a predicate.",
        signature="def count_if(lst: list, pred) -> int:",
        body="    return sum(1 for x in lst if pred(x))",
    ),
    ToyProgram(
        docstring="Filter a list by a predicate.",
        signature="def filter_list(lst: list, pred) -> list:",
        body="    return [x for x in lst if pred(x)]",
    ),
    ToyProgram(
        docstring="Apply a function to each element of a list.",
        signature="def map_list(lst: list, fn) -> list:",
        body="    return [fn(x) for x in lst]",
    ),
    ToyProgram(
        docstring="Reduce a list with a binary function.",
        signature="def reduce_list(lst: list, fn, initial):",
        body="    result = initial\n    for x in lst:\n        result = fn(result, x)\n    return result",
    ),
    ToyProgram(
        docstring="Check if all elements satisfy a predicate.",
        signature="def all_satisfy(lst: list, pred) -> bool:",
        body="    return all(pred(x) for x in lst)",
    ),
    ToyProgram(
        docstring="Check if any element satisfies a predicate.",
        signature="def any_satisfy(lst: list, pred) -> bool:",
        body="    return any(pred(x) for x in lst)",
    ),
    ToyProgram(
        docstring="Find the index of an element in a list.",
        signature="def find_index(lst: list, x) -> int:",
        body="    return lst.index(x) if x in lst else -1",
    ),
    ToyProgram(
        docstring="Zip two lists together.",
        signature="def zip_lists(a: list, b: list) -> list:",
        body="    return list(zip(a, b))",
    ),
]

# All toy programs combined
TOY_PROGRAMS: list[ToyProgram] = (
    ARITHMETIC_PROGRAMS
    + RECURSION_PROGRAMS
    + STRING_PROGRAMS
    + LIST_PROGRAMS
    + BOOLEAN_PROGRAMS
    + CONVERSION_PROGRAMS
    + ALGORITHM_PROGRAMS
)


def create_toy_dataset() -> list[dict]:
    """Create the toy dataset as a list of dicts.

    Returns:
        List of dicts with 'prompt', 'code', 'docstring', 'signature', 'body' keys.
    """
    return [
        {
            "prompt": prog.prompt,
            "code": prog.full_code,
            "docstring": prog.docstring,
            "signature": prog.signature,
            "body": prog.body,
        }
        for prog in TOY_PROGRAMS
    ]


def toy_batch_iterator(
    batch_size: int,
    tokenizer,
    max_seq_len: int,
    key: PRNGKeyArray,
) -> Iterator[dict[str, Array]]:
    """Create an iterator over batches of toy data.

    Args:
        batch_size: Batch size.
        tokenizer: Tokenizer with encode method.
        max_seq_len: Maximum sequence length (will pad/truncate).
        key: PRNG key for shuffling.

    Yields:
        Dicts with 'tokens', 'prefix_len' keys.
    """
    dataset = create_toy_dataset()
    num_examples = len(dataset)

    while True:
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, jnp.arange(num_examples))

        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i : i + batch_size]
            if len(batch_indices) < batch_size:
                batch_indices = indices[:batch_size]

            batch_tokens = []
            batch_prefix_lens = []

            for idx in batch_indices:
                example = dataset[int(idx)]
                prompt_ids = tokenizer.encode(example["prompt"])
                code_ids = tokenizer.encode(example["code"])

                prefix_len = len(prompt_ids)
                tokens = code_ids

                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                    prefix_len = min(prefix_len, max_seq_len)
                else:
                    tokens = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))

                batch_tokens.append(tokens)
                batch_prefix_lens.append(prefix_len)

            yield {
                "tokens": jnp.array(batch_tokens),
                "prefix_len": jnp.array(batch_prefix_lens),
            }


# For compatibility with simple training scripts
def get_toy_examples(n: int | None = None) -> list[ToyProgram]:
    """Get toy program examples.

    Args:
        n: Number of examples to return (None for all).

    Returns:
        List of ToyProgram instances.
    """
    if n is None:
        return TOY_PROGRAMS
    return TOY_PROGRAMS[:n]
