# Copyright 2026 The Marin Authors
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

"""
Toy dataset for prototyping tree diffusion.

Contains simple Python functions for testing the tree diffusion pipeline.
"""

from experiments.kelp.ast_utils import TreeTensors, parse_python_to_tensors
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab

# Simple MBPP-style functions for prototyping
# Mix of functions with and without docstrings for conditioning experiments
TOY_PROGRAMS: list[str] = [
    # Arithmetic with docstrings
    '''def add(a, b):
    """Add two numbers together."""
    return a + b''',
    '''def subtract(a, b):
    """Subtract b from a."""
    return a - b''',
    '''def multiply(x, y):
    """Multiply two numbers."""
    return x * y''',
    '''def divide(x, y):
    """Divide x by y."""
    return x / y''',
    '''def square(n):
    """Return the square of n."""
    return n * n''',
    '''def cube(n):
    """Return the cube of n."""
    return n * n * n''',
    '''def double(x):
    """Double the input value."""
    return x * 2''',
    '''def triple(x):
    """Triple the input value."""
    return x * 3''',
    "def negate(x): return -x",
    '''def absolute(x):
    """Return the absolute value of x."""
    return x if x >= 0 else -x''',
    # Predicates with docstrings
    '''def is_even(n):
    """Check if n is even."""
    return n % 2 == 0''',
    '''def is_odd(n):
    """Check if n is odd."""
    return n % 2 != 0''',
    '''def is_positive(n):
    """Check if n is positive."""
    return n > 0''',
    '''def is_negative(n):
    """Check if n is negative."""
    return n < 0''',
    "def is_zero(n): return n == 0",
    "def is_equal(a, b): return a == b",
    "def is_greater(a, b): return a > b",
    "def is_less(a, b): return a < b",
    # Min/Max with docstrings
    '''def max_two(a, b):
    """Return the maximum of two values."""
    return a if a > b else b''',
    '''def min_two(a, b):
    """Return the minimum of two values."""
    return a if a < b else b''',
    '''def clamp(x, lo, hi):
    """Clamp x to be within [lo, hi]."""
    return max(lo, min(x, hi))''',
    # List operations with docstrings
    '''def first(lst):
    """Return the first element of a list."""
    return lst[0]''',
    '''def last(lst):
    """Return the last element of a list."""
    return lst[-1]''',
    '''def length(lst):
    """Return the length of a list."""
    return len(lst)''',
    '''def is_empty(lst):
    """Check if a list is empty."""
    return len(lst) == 0''',
    '''def sum_list(lst):
    """Return the sum of all elements in a list."""
    return sum(lst)''',
    '''def product_list(lst):
    """Return the product of all elements in a list."""
    result = 1
    for x in lst:
        result = result * x
    return result''',
    '''def reverse_list(lst):
    """Return a reversed copy of the list."""
    return lst[::-1]''',
    "def head(lst): return lst[0]",
    "def tail(lst): return lst[1:]",
    "def init(lst): return lst[:-1]",
    # Conditionals
    "def sign(n):\n    if n > 0:\n        return 1\n    elif n < 0:\n        return -1\n    else:\n        return 0",
    "def abs_diff(a, b): return a - b if a > b else b - a",
    "def max_three(a, b, c): return max(a, max(b, c))",
    "def min_three(a, b, c): return min(a, min(b, c))",
    # Loops
    "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result = result * i\n    return result",
    "def fibonacci(n):\n    a = 0\n    b = 1\n    for i in range(n):\n        a, b = b, a + b\n    return a",
    "def count_positive(lst):\n    count = 0\n    for x in lst:\n        if x > 0:\n            count = count + 1\n    return count",
    "def count_negative(lst):\n    count = 0\n    for x in lst:\n        if x < 0:\n            count = count + 1\n    return count",
    "def sum_positive(lst):\n    total = 0\n    for x in lst:\n        if x > 0:\n            total = total + x\n    return total",
    "def sum_even(lst):\n    total = 0\n    for x in lst:\n        if x % 2 == 0:\n            total = total + x\n    return total",
    "def sum_odd(lst):\n    total = 0\n    for x in lst:\n        if x % 2 != 0:\n            total = total + x\n    return total",
    # List comprehensions
    "def squares(n): return [i * i for i in range(n)]",
    "def evens(n): return [i for i in range(n) if i % 2 == 0]",
    "def odds(n): return [i for i in range(n) if i % 2 != 0]",
    "def double_all(lst): return [x * 2 for x in lst]",
    "def square_all(lst): return [x * x for x in lst]",
    "def filter_positive(lst): return [x for x in lst if x > 0]",
    "def filter_negative(lst): return [x for x in lst if x < 0]",
    "def filter_even(lst): return [x for x in lst if x % 2 == 0]",
    "def filter_odd(lst): return [x for x in lst if x % 2 != 0]",
    # String operations
    "def is_upper(s): return s.isupper()",
    "def is_lower(s): return s.islower()",
    "def to_upper(s): return s.upper()",
    "def to_lower(s): return s.lower()",
    "def str_length(s): return len(s)",
    "def concat(a, b): return a + b",
    "def repeat_str(s, n): return s * n",
    "def reverse_str(s): return s[::-1]",
    # Boolean logic
    "def and_op(a, b): return a and b",
    "def or_op(a, b): return a or b",
    "def not_op(a): return not a",
    "def xor_op(a, b): return (a or b) and not (a and b)",
    "def implies(a, b): return not a or b",
    # More arithmetic
    "def power(base, exp):\n    result = 1\n    for i in range(exp):\n        result = result * base\n    return result",
    "def gcd(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a",
    "def lcm(a, b): return a * b // gcd(a, b)",
    "def average(lst): return sum(lst) / len(lst)",
    "def median_three(a, b, c):\n    if a <= b <= c or c <= b <= a:\n        return b\n    elif b <= a <= c or c <= a <= b:\n        return a\n    else:\n        return c",
    # Index operations
    "def get_index(lst, i): return lst[i]",
    "def set_index(lst, i, v):\n    result = list(lst)\n    result[i] = v\n    return result",
    "def find_index(lst, x):\n    for i in range(len(lst)):\n        if lst[i] == x:\n            return i\n    return -1",
    "def contains(lst, x): return x in lst",
    "def count_occurrences(lst, x):\n    count = 0\n    for item in lst:\n        if item == x:\n            count = count + 1\n    return count",
    # More list operations
    "def append(lst, x): return lst + [x]",
    "def prepend(lst, x): return [x] + lst",
    "def remove_first(lst, x): return [item for item in lst if item != x]",
    "def unique(lst):\n    result = []\n    for x in lst:\n        if x not in result:\n            result.append(x)\n    return result",
    "def flatten(lst): return [x for sublist in lst for x in sublist]",
    "def zip_lists(a, b): return list(zip(a, b))",
    "def unzip_lists(pairs): return [p[0] for p in pairs], [p[1] for p in pairs]",
    # Range operations
    "def range_sum(n): return sum(range(n))",
    "def range_product(n):\n    result = 1\n    for i in range(1, n + 1):\n        result = result * i\n    return result",
    "def range_list(start, end): return list(range(start, end))",
    "def range_step(start, end, step): return list(range(start, end, step))",
    # Simple algorithms
    "def is_sorted(lst):\n    for i in range(len(lst) - 1):\n        if lst[i] > lst[i + 1]:\n            return False\n    return True",
    "def is_palindrome(s): return s == s[::-1]",
    "def all_equal(lst):\n    if len(lst) == 0:\n        return True\n    first = lst[0]\n    for x in lst:\n        if x != first:\n            return False\n    return True",
    "def all_positive(lst):\n    for x in lst:\n        if x <= 0:\n            return False\n    return True",
    "def any_positive(lst):\n    for x in lst:\n        if x > 0:\n            return True\n    return False",
    # Identity functions
    "def identity(x): return x",
    "def constant(x, y): return x",
    "def compose(f, g, x): return f(g(x))",
    "def apply(f, x): return f(x)",
    # Functions with type hints and docstrings
    '''def add_ints(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b''',
    '''def multiply_floats(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y''',
    '''def is_even_typed(n: int) -> bool:
    """Check if an integer is even."""
    return n % 2 == 0''',
    '''def list_sum(numbers: list) -> int:
    """Sum all numbers in a list."""
    return sum(numbers)''',
    '''def string_length(s: str) -> int:
    """Return the length of a string."""
    return len(s)''',
]


def load_toy_dataset(
    node_vocab: PythonNodeVocab | None = None,
    value_vocab: PythonValueVocab | None = None,
    max_nodes: int = 256,
    max_children: int = 16,
    max_value_len: int = 32,
) -> list[TreeTensors]:
    """Load the toy dataset as a list of TreeTensors.

    Args:
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values
        max_nodes: Maximum number of nodes per tree
        max_children: Maximum children per node
        max_value_len: Maximum value encoding length

    Returns:
        List of TreeTensors for each program in TOY_PROGRAMS
    """
    if node_vocab is None:
        node_vocab = PythonNodeVocab()
    if value_vocab is None:
        value_vocab = PythonValueVocab()

    tensors_list = []
    for code in TOY_PROGRAMS:
        try:
            tensors = parse_python_to_tensors(
                code,
                node_vocab=node_vocab,
                value_vocab=value_vocab,
                max_nodes=max_nodes,
                max_children=max_children,
                max_value_len=max_value_len,
            )
            tensors_list.append(tensors)
        except SyntaxError:
            # Skip programs that fail to parse (should not happen with our toy set)
            continue

    return tensors_list


def get_toy_programs() -> list[str]:
    """Get the list of toy program strings."""
    return TOY_PROGRAMS.copy()
