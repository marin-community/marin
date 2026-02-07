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

"""AST subtree bank for tree diffusion.

Extracts real Python code fragments from a corpus and indexes them by AST node
type. Used by the forward (noise) process to corrupt programs via realistic
subtree replacement, following the spirit of Tree Diffusion (Kapur et al., 2024)
but adapted for a real programming language.

Instead of randomly sampling from a CFG (which produces syntactically valid but
meaningless code for a language as large as Python), we draw replacement subtrees
from a bank of fragments extracted from real programs. This preserves the paper's
key guarantee -- every intermediate program is syntactically valid -- while
producing realistic mutations.
"""

import ast
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# AST node types worth extracting as subtrees. Grouped by category.
# We skip trivially small nodes (Name, Constant) that carry no structural info.
STATEMENT_TYPES = frozenset(
    {
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "Return",
        "Assign",
        "AugAssign",
        "AnnAssign",
        "For",
        "AsyncFor",
        "While",
        "If",
        "With",
        "AsyncWith",
        "Raise",
        "Try",
        "TryStar",
        "Assert",
        "Expr",
        "Global",
        "Nonlocal",
    }
)

EXPRESSION_TYPES = frozenset(
    {
        "BoolOp",
        "BinOp",
        "UnaryOp",
        "Lambda",
        "IfExp",
        "Dict",
        "Set",
        "ListComp",
        "SetComp",
        "DictComp",
        "GeneratorExp",
        "Await",
        "Compare",
        "Call",
        "JoinedStr",
        "Subscript",
        "Starred",
        "List",
        "Tuple",
    }
)

EXTRACTABLE_TYPES = STATEMENT_TYPES | EXPRESSION_TYPES


def count_statements(node: ast.AST) -> int:
    """Count the number of statement nodes in an AST subtree.

    This is the Python analogue of the paper's 'sigma' (primitive count).
    Used to control the size of extracted and replaced subtrees.
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.stmt):
            count += 1
    return count


def _node_type_name(node: ast.AST) -> str:
    return type(node).__name__


@dataclass
class SubtreeEntry:
    """A single extracted subtree."""

    source: str
    """Unparsed source code of the subtree."""

    node_type: str
    """AST node type name (e.g., 'If', 'Call', 'Assign')."""

    stmt_count: int
    """Number of statement nodes in this subtree."""


@dataclass
class SubtreeBank:
    """A bank of real Python code fragments indexed by AST node type.

    Used by the tree diffusion forward process to sample realistic replacement
    subtrees during program corruption.
    """

    entries: dict[str, list[SubtreeEntry]] = field(default_factory=dict)
    """Map from AST node type name to list of extracted subtrees."""

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self.entries.values())

    @property
    def node_types(self) -> list[str]:
        return sorted(self.entries.keys())

    def add(self, entry: SubtreeEntry) -> None:
        """Add a subtree entry to the bank."""
        if entry.node_type not in self.entries:
            self.entries[entry.node_type] = []
        self.entries[entry.node_type].append(entry)

    def sample(self, node_type: str, rng: random.Random) -> SubtreeEntry | None:
        """Sample a random subtree of the given AST node type.

        Returns None if no subtrees of that type are in the bank.
        """
        candidates = self.entries.get(node_type)
        if not candidates:
            return None
        return rng.choice(candidates)

    def sample_with_size(
        self,
        node_type: str,
        max_stmts: int,
        rng: random.Random,
        max_attempts: int = 20,
    ) -> SubtreeEntry | None:
        """Sample a subtree matching the node type with bounded statement count.

        Args:
            node_type: AST node type to match.
            max_stmts: Maximum number of statements allowed in the subtree.
            rng: Random number generator.
            max_attempts: Maximum sampling attempts before giving up.

        Returns:
            A matching SubtreeEntry, or None if no suitable candidate found.
        """
        candidates = self.entries.get(node_type)
        if not candidates:
            return None

        # Try random sampling first (fast path).
        for _ in range(max_attempts):
            entry = rng.choice(candidates)
            if entry.stmt_count <= max_stmts:
                return entry

        # Fall back to filtering (slower but guaranteed).
        filtered = [e for e in candidates if e.stmt_count <= max_stmts]
        if not filtered:
            return None
        return rng.choice(filtered)

    def has_type(self, node_type: str) -> bool:
        """Check whether the bank has entries for the given node type."""
        entries = self.entries.get(node_type)
        return entries is not None and len(entries) > 0

    @classmethod
    def from_corpus(
        cls,
        programs: Iterable[str],
        max_subtree_stmts: int = 5,
        max_entries_per_type: int = 10000,
    ) -> "SubtreeBank":
        """Build a subtree bank from a corpus of Python programs.

        Args:
            programs: Iterable of Python source code strings.
            max_subtree_stmts: Maximum statement count for extracted subtrees.
                Subtrees larger than this are skipped.
            max_entries_per_type: Cap on entries per node type to limit memory.

        Returns:
            Populated SubtreeBank.
        """
        bank = cls()
        seen: dict[str, set[str]] = {}  # Deduplicate by (type, source).
        programs_parsed = 0
        programs_failed = 0

        for source in programs:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                programs_failed += 1
                continue

            programs_parsed += 1
            _extract_from_tree(
                tree,
                source,
                bank,
                seen,
                max_subtree_stmts,
                max_entries_per_type,
            )

        logger.info(
            f"Built subtree bank: {bank.total_entries} entries across "
            f"{len(bank.entries)} node types from {programs_parsed} programs "
            f"({programs_failed} failed to parse)"
        )
        return bank

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
        max_subtree_stmts: int = 5,
        max_entries_per_type: int = 10000,
    ) -> "SubtreeBank":
        """Build a subtree bank from Python source files.

        Args:
            paths: Paths to .py files.
            max_subtree_stmts: Maximum statement count for extracted subtrees.
            max_entries_per_type: Cap on entries per node type.

        Returns:
            Populated SubtreeBank.
        """

        def _read_files():
            for path in paths:
                path = Path(path)
                if not path.exists() or not path.suffix == ".py":
                    continue
                try:
                    yield path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

        return cls.from_corpus(
            _read_files(),
            max_subtree_stmts=max_subtree_stmts,
            max_entries_per_type=max_entries_per_type,
        )

    def save(self, path: str | Path) -> None:
        """Save the bank to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for node_type, entries in self.entries.items():
            data[node_type] = [{"source": e.source, "stmt_count": e.stmt_count} for e in entries]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved subtree bank ({self.total_entries} entries) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SubtreeBank":
        """Load a bank from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        bank = cls()
        for node_type, entries in data.items():
            for entry_data in entries:
                bank.add(
                    SubtreeEntry(
                        source=entry_data["source"],
                        node_type=node_type,
                        stmt_count=entry_data["stmt_count"],
                    )
                )

        logger.info(f"Loaded subtree bank ({bank.total_entries} entries) from {path}")
        return bank

    def summary(self) -> str:
        """Return a human-readable summary of bank contents."""
        lines = [f"SubtreeBank: {self.total_entries} entries, {len(self.entries)} types"]
        for node_type in sorted(self.entries.keys()):
            entries = self.entries[node_type]
            lines.append(f"  {node_type}: {len(entries)} entries")
        return "\n".join(lines)


def _extract_from_tree(
    tree: ast.Module,
    source: str,
    bank: SubtreeBank,
    seen: dict[str, set[str]],
    max_subtree_stmts: int,
    max_entries_per_type: int,
) -> None:
    """Extract subtrees from a parsed AST into the bank."""
    for node in ast.walk(tree):
        type_name = _node_type_name(node)

        if type_name not in EXTRACTABLE_TYPES:
            continue

        # Check capacity for this type.
        existing = bank.entries.get(type_name, [])
        if len(existing) >= max_entries_per_type:
            continue

        stmt_count = count_statements(node)
        if stmt_count > max_subtree_stmts:
            continue

        # Unparse the subtree back to source. This normalizes formatting,
        # which is fine -- the model works with tokens, not formatting.
        try:
            unparsed = ast.unparse(node)
        except Exception:
            continue

        # Skip trivially short fragments (single names, bare constants).
        if len(unparsed) < 5:
            continue

        # Deduplicate.
        if type_name not in seen:
            seen[type_name] = set()
        if unparsed in seen[type_name]:
            continue
        seen[type_name].add(unparsed)

        # Verify the unparsed source round-trips through the parser.
        # This catches cases where ast.unparse produces invalid code.
        if not _is_parseable(unparsed, type_name):
            continue

        bank.add(
            SubtreeEntry(
                source=unparsed,
                node_type=type_name,
                stmt_count=stmt_count,
            )
        )


def _is_parseable(source: str, node_type: str) -> bool:
    """Check that a code fragment parses back successfully.

    Statements are parsed as-is. Expressions are wrapped in an assignment
    to form a valid statement for ast.parse().
    """
    try:
        if node_type in STATEMENT_TYPES:
            ast.parse(source)
        else:
            # Expressions need a statement wrapper to parse.
            ast.parse(f"__x = {source}")
        return True
    except SyntaxError:
        return False
