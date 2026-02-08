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

"""Programmatic augmentation for the subtree bank.

Expands the subtree bank beyond verbatim corpus fragments by generating
variations of existing subtrees. Analogous to data augmentation in computer
vision (color jitter, flipping, etc.), but for Python ASTs.

Three strategies are implemented:

1. **Variable renaming**: Replace identifiers with alternatives drawn from a
   pool of common names. This is the code equivalent of color jitter -- the
   structure is preserved, only the labels change.

2. **Operator perturbation**: Swap operators within the same category
   (arithmetic, comparison, boolean). Creates near-miss programs that the model
   must learn to distinguish and correct.

3. **Synthetic templates**: Generate subtrees from common code patterns
   (return expressions, conditionals, loops) with randomly filled slots. Fills
   gaps in the bank when the corpus is small.
"""

import ast
import logging
import random

from experiments.kelp.tree.subtree_bank import (
    EXPRESSION_TYPES,
    SubtreeBank,
    SubtreeEntry,
)

logger = logging.getLogger(__name__)

# --- Variable renaming ---

# Pools of replacement names, grouped by common usage patterns.
_SINGLE_LETTER_NAMES = list("abcdefghijklmnopqrstuvwxyz")
_DESCRIPTIVE_NAMES = [
    "result",
    "value",
    "item",
    "data",
    "total",
    "count",
    "index",
    "temp",
    "output",
    "current",
    "prev",
    "next_val",
    "acc",
    "elem",
    "key",
    "val",
    "num",
    "flag",
    "start",
    "end",
    "size",
    "length",
]
_NAME_POOL = _SINGLE_LETTER_NAMES + _DESCRIPTIVE_NAMES


class _NameRewriter(ast.NodeTransformer):
    """Renames non-builtin identifiers according to a mapping."""

    # Names we never rename -- builtins, keywords, common stdlib.
    PROTECTED = frozenset(
        {
            "True",
            "False",
            "None",
            "self",
            "cls",
            "super",
            "print",
            "len",
            "range",
            "int",
            "float",
            "str",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "type",
            "isinstance",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "min",
            "max",
            "sum",
            "abs",
            "any",
            "all",
            "open",
            "input",
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "StopIteration",
            "__init__",
            "__str__",
            "__repr__",
            "__len__",
            "__iter__",
            "__next__",
            "__getitem__",
            "__setitem__",
            "__contains__",
        }
    )

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            return ast.Name(id=self.mapping[node.id], ctx=node.ctx)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self.mapping:
            return ast.arg(arg=self.mapping[node.arg], annotation=node.annotation)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in self.mapping:
            node = ast.FunctionDef(
                name=self.mapping[node.name],
                args=node.args,
                body=node.body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
            )
            ast.copy_location(node, node)
        self.generic_visit(node)
        return node


def _collect_names(source: str) -> set[str]:
    """Collect all user-defined identifiers from source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _NameRewriter.PROTECTED:
            names.add(node.id)
        elif isinstance(node, ast.arg) and node.arg not in _NameRewriter.PROTECTED:
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef) and node.name not in _NameRewriter.PROTECTED:
            names.add(node.name)
    return names


def rename_variables(source: str, rng: random.Random) -> str | None:
    """Produce a variant of source with renamed identifiers.

    Each user-defined name is mapped to a random name from the pool.
    Returns None if renaming fails or produces identical source.
    """
    names = _collect_names(source)
    if not names:
        return None

    # Build a random mapping. Ensure no collisions.
    used = set(names) | _NameRewriter.PROTECTED
    mapping: dict[str, str] = {}
    pool = list(_NAME_POOL)
    rng.shuffle(pool)
    pool_iter = iter(pool)

    for name in sorted(names):
        for candidate in pool_iter:
            if candidate not in used:
                mapping[name] = candidate
                used.add(candidate)
                break
        else:
            # Pool exhausted; use name with suffix.
            mapping[name] = f"{name}_{rng.randint(0, 99)}"

    try:
        tree = ast.parse(source)
        rewriter = _NameRewriter(mapping)
        new_tree = rewriter.visit(tree)
        ast.fix_missing_locations(new_tree)
        result = ast.unparse(new_tree)
    except Exception:
        return None

    if result == source:
        return None

    # Verify the result parses.
    try:
        ast.parse(result)
    except SyntaxError:
        return None

    return result


# --- Operator perturbation ---

# Groups of interchangeable operators.
_ARITHMETIC_OPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
_COMPARE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_BOOL_OPS = (ast.And, ast.Or)
_UNARY_OPS = (ast.UAdd, ast.USub)


class _OpSwapper(ast.NodeTransformer):
    """Swaps operators within their category with a given probability."""

    def __init__(self, rng: random.Random, swap_prob: float = 0.3):
        self.rng = rng
        self.swap_prob = swap_prob
        self._changed = False

    @property
    def changed(self) -> bool:
        return self._changed

    def _maybe_swap(self, op: ast.AST, group: tuple) -> ast.AST:
        if type(op) in group and self.rng.random() < self.swap_prob:
            alternatives = [cls for cls in group if cls is not type(op)]
            if alternatives:
                self._changed = True
                return self.rng.choice(alternatives)()
        return op

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        node.op = self._maybe_swap(node.op, _ARITHMETIC_OPS)
        self.generic_visit(node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        node.ops = [self._maybe_swap(op, _COMPARE_OPS) for op in node.ops]
        self.generic_visit(node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        node.op = self._maybe_swap(node.op, _BOOL_OPS)
        self.generic_visit(node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
        node.op = self._maybe_swap(node.op, _UNARY_OPS)
        self.generic_visit(node)
        return node


def perturb_operators(source: str, rng: random.Random, swap_prob: float = 0.3) -> str | None:
    """Produce a variant with some operators swapped within their category.

    Returns None if no operators were changed or the result is invalid.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    swapper = _OpSwapper(rng, swap_prob=swap_prob)
    new_tree = swapper.visit(tree)

    if not swapper.changed:
        return None

    ast.fix_missing_locations(new_tree)
    try:
        result = ast.unparse(new_tree)
        ast.parse(result)
    except Exception:
        return None

    if result == source:
        return None

    return result


# --- Synthetic templates ---

# Templates for generating subtrees. Each template is a format string with
# slots filled from pools of expressions/names.

_EXPR_ATOMS = [
    "x",
    "y",
    "a",
    "b",
    "n",
    "i",
    "0",
    "1",
    "2",
    "True",
    "False",
    "None",
    "x + 1",
    "x - 1",
    "n + 1",
    "n - 1",
    "len(x)",
    "x[0]",
    "x[-1]",
]

_COND_ATOMS = [
    "x > 0",
    "x < 0",
    "x == 0",
    "n > 0",
    "n <= 1",
    "x is None",
    "x is not None",
    "len(x) > 0",
    "len(x) == 0",
    "a > b",
    "a < b",
    "a == b",
    "not x",
    "x and y",
    "x or y",
]

_RETURN_TEMPLATES = [
    "return {expr}",
    "return {expr1} + {expr2}",
    "return {expr1} - {expr2}",
    "return {expr1} * {expr2}",
    "return -{expr}",
    "return not {expr}",
]

_IF_TEMPLATES = [
    "if {cond}:\n    return {expr}",
    "if {cond}:\n    {var} = {expr}",
    "if {cond}:\n    return {expr1}\nreturn {expr2}",
    "if {cond1}:\n    return {expr1}\nif {cond2}:\n    return {expr2}",
]

_FOR_TEMPLATES = [
    "for {var} in range({expr}):\n    {var2} = {var2} + {var}",
    "for {var} in {iterable}:\n    if {cond}:\n        break",
    "for {var} in range(len({iterable})):\n    {iterable}[{var}] = {expr}",
]

_ASSIGN_TEMPLATES = [
    "{var} = {expr}",
    "{var} = {expr1} + {expr2}",
    "{var} = [{expr1}, {expr2}]",
    "{var} = ({expr1}, {expr2})",
    "{var} = {expr1} if {cond} else {expr2}",
]


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template string with random atoms."""
    # Collect all placeholders.
    result = template
    var_names = ["result", "val", "total", "temp", "acc", "item", "elem", "x", "y", "i"]
    iterable_names = ["items", "data", "values", "xs", "lst"]

    for placeholder in ["{var}", "{var2}"]:
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(var_names), 1)

    for placeholder in ["{iterable}"]:
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(iterable_names), 1)

    for placeholder in ["{expr}", "{expr1}", "{expr2}"]:
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(_EXPR_ATOMS), 1)

    for placeholder in ["{cond}", "{cond1}", "{cond2}"]:
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(_COND_ATOMS), 1)

    return result


def generate_synthetic_subtrees(
    rng: random.Random,
    count_per_category: int = 50,
) -> list[SubtreeEntry]:
    """Generate synthetic subtree entries from templates.

    Returns a list of SubtreeEntry objects ready to add to a SubtreeBank.
    """
    entries: list[SubtreeEntry] = []
    seen: set[str] = set()

    def _try_add(source: str, node_type: str) -> None:
        if source in seen:
            return
        try:
            tree = ast.parse(source)
            # Normalize via unparse for consistency with bank entries.
            normalized = ast.unparse(tree)
            if normalized in seen or len(normalized) < 5:
                return
            seen.add(normalized)
            stmt_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt))
            entries.append(
                SubtreeEntry(
                    source=normalized,
                    node_type=node_type,
                    stmt_count=stmt_count,
                )
            )
        except SyntaxError:
            return

    # Generate from each template category.
    for _ in range(count_per_category):
        template = rng.choice(_RETURN_TEMPLATES)
        _try_add(_fill_template(template, rng), "Return")

    for _ in range(count_per_category):
        template = rng.choice(_IF_TEMPLATES)
        _try_add(_fill_template(template, rng), "If")

    for _ in range(count_per_category):
        template = rng.choice(_FOR_TEMPLATES)
        _try_add(_fill_template(template, rng), "For")

    for _ in range(count_per_category):
        template = rng.choice(_ASSIGN_TEMPLATES)
        _try_add(_fill_template(template, rng), "Assign")

    # Generate expression subtrees.
    for _ in range(count_per_category):
        expr = rng.choice(_EXPR_ATOMS)
        expr2 = rng.choice(_EXPR_ATOMS)
        op = rng.choice(["+", "-", "*", "//"])
        combined = f"{expr} {op} {expr2}"
        try:
            tree = ast.parse(combined, mode="eval")
            normalized = ast.unparse(tree.body)
            node_type = type(tree.body).__name__
            if node_type in EXPRESSION_TYPES and normalized not in seen and len(normalized) >= 5:
                seen.add(normalized)
                entries.append(
                    SubtreeEntry(
                        source=normalized,
                        node_type=node_type,
                        stmt_count=0,
                    )
                )
        except SyntaxError:
            continue

    return entries


def augment_bank(
    bank: SubtreeBank,
    rng: random.Random,
    n_renamed: int = 2,
    n_perturbed: int = 2,
    synthetic_count: int = 50,
) -> SubtreeBank:
    """Augment a subtree bank with generated variations.

    For each existing entry, generates up to n_renamed variable-renamed variants
    and n_perturbed operator-perturbed variants. Also adds synthetic subtrees.

    Args:
        bank: Original subtree bank.
        rng: Random number generator.
        n_renamed: Number of renamed variants per entry.
        n_perturbed: Number of operator-perturbed variants per entry.
        synthetic_count: Number of synthetic subtrees per template category.

    Returns:
        New SubtreeBank with original entries plus augmented entries.
    """
    augmented = SubtreeBank()
    seen: set[tuple[str, str]] = set()

    # Copy originals.
    for node_type, entries in bank.entries.items():
        for entry in entries:
            augmented.add(entry)
            seen.add((node_type, entry.source))

    original_count = augmented.total_entries
    renamed_count = 0
    perturbed_count = 0

    # Generate renamed variants.
    for node_type, entries in bank.entries.items():
        for entry in entries:
            for _ in range(n_renamed):
                variant = rename_variables(entry.source, rng)
                if variant is not None and (node_type, variant) not in seen:
                    seen.add((node_type, variant))
                    augmented.add(
                        SubtreeEntry(
                            source=variant,
                            node_type=node_type,
                            stmt_count=entry.stmt_count,
                        )
                    )
                    renamed_count += 1

    # Generate operator-perturbed variants.
    for node_type, entries in bank.entries.items():
        for entry in entries:
            for _ in range(n_perturbed):
                variant = perturb_operators(entry.source, rng)
                if variant is not None and (node_type, variant) not in seen:
                    seen.add((node_type, variant))
                    augmented.add(
                        SubtreeEntry(
                            source=variant,
                            node_type=node_type,
                            stmt_count=entry.stmt_count,
                        )
                    )
                    perturbed_count += 1

    # Add synthetic templates.
    synthetics = generate_synthetic_subtrees(rng, count_per_category=synthetic_count)
    synthetic_added = 0
    for entry in synthetics:
        if (entry.node_type, entry.source) not in seen:
            seen.add((entry.node_type, entry.source))
            augmented.add(entry)
            synthetic_added += 1

    logger.info(
        f"Augmented subtree bank: {original_count} original + "
        f"{renamed_count} renamed + {perturbed_count} perturbed + "
        f"{synthetic_added} synthetic = {augmented.total_entries} total"
    )

    return augmented
