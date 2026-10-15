# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Check every number in the paper's tables against the dataset build. Reports; never edits the LaTeX.

`uv run python -m experiments.domain_phase_mix.release.verify_paper_numbers --build release_build
--sections <paper folder>/sections [--fail-on-untraced] [--allow-registered-skips]`

Tables. `parse_tables` reads, after stripping comments, every `table`, `table*`, `wraptable` and `minipage`
environment and every `\captionof{table}` up to the `longtable` or `tabular` that follows it, and returns each `tab:`
label's body: the rows after the header (everything up to the first `\midrule`, or up to a longtable's last head or
foot marker), split on unescaped `&`. A line without `&`, such as a section heading, is not a row. Cells keep their
LaTeX. A cell is numeric when it starts with a number once math mode, bold, italics, `\multicolumn` and daggers are
stripped; `printed_numbers` returns its numbers in print order with their printed decimals: a mean and its std, a
multiple and its saving, a fitted law's three constants. A number keeps its printed units: `358.3M` is 358.3,
`3.4\times10^{18}` is 3.4 and `30.1\%` is 30.1, so a query must return the same units.

Registry. `paper_cells.csv` has one row per printed number: `label, row, col, part, query, decimals, note`. `row` and
`col` index the parsed body from 0 (`col` counts `&`-separated cells, so a `\multicolumn` is one cell); `part` indexes
the numbers of the cell (default 0); `decimals` is the paper's printed decimals, at which the number is compared:
`round(value, decimals) == round(printed, decimals)`. A row with an empty query is a registered skip, whose note names
its kind before any colon: `constant` (a configuration constant typed into the paper), `deferred_two_phase` (two-phase
data this release omits) or `untraced` (the source is not in the build; the reason follows the colon). Every number
outside a table's first column needs a row; first-column numbers are row labels, checked only when registered. A row
that names no printed number, or other decimals than the paper prints, is an error, so the registry cannot drift from
the paper silently.

Queries. A query is `<table>[<filter>].<column>.<agg>`: a build table's name, a `DataFrame.query` filter (empty for
every row), a column, and `value` (exactly one row), `unique` (one value across the rows), `mean`, `std` (ddof 1), `se`
(std over the square root of the count), `median`, `min`, `max`, `sum` or `count`. A selected null raises rather than
being skipped. `A diff: B` is A minus B. Extensions, each needed by a paper table:

- `<column>` may be `{<expression>}`, evaluated per row by `DataFrame.eval` (`{1 - mariner_mse / panel_sd**2}`), or
  `mean_of(<selection>)`, the row-wise mean of the `bpb__<task>` columns for the task ids that another selection,
  `<table>[<filter>].<column>`, returns (`mean_of(tasks[family=='Math'].task_id)`).
- `concat(<selection>, <selection>, ...)` stacks selections, of one table or several, into one; a selection is
  `<table>[<filter>].<column>` or a `concat`. `concat(swarm[...].bpb, validation[...].bpb).std` is the std of a
  mixture's runs that two tables hold.
- `paired(<selection>, <selection>, <key>).<agg>` aggregates the differences of two selections paired on a key
  column, which must hold the same values once each: `paired(a.bpb, b.bpb, trainer_seed).se` is the standard error of
  a mean paired difference.
- Scalars combine with numbers through `+ - * /`, parentheses and `sqrt(...)`, for units and ratios:
  `100 * partition[bucket_id=='arxiv'].pool_tokens.value / partition[].pool_tokens.sum`.
"""

import argparse
import dataclasses
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import tables as release_tables

REGISTRY = Path(__file__).with_name("paper_cells.csv")
TABLE_ENV = re.compile(
    r"\\begin\{(?P<env>table\*?|wraptable|minipage)\}(?P<body>.*?)\\end\{(?P=env)\}"
    r"|\\captionof\{table\}(?P<captioned>.*?\\end\{(?:longtable|tabular)\})",
    re.S,
)
LABEL = re.compile(r"\\label\{(tab:[^}]+)\}")
TABULAR = re.compile(r"\\begin\{(?P<env>tabular|longtable)\}\s*(?:\[[^\]]*\])?\s*")
COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*$", re.M)
MIDRULE = re.compile(r"\\midrule(?:\[[^\]]*\])?")
LONGTABLE_HEADS = re.compile(r"\\end(?:firsthead|head|foot|lastfoot)(?![A-Za-z])")
ROW_END = re.compile(r"\\\\(?:\*|\[[^\]]*\])?")
RULES = re.compile(
    r"\\(?:toprule|midrule|bottomrule|hline|addlinespace|nopagebreak|pagebreak)(?:\[[^\]]*\])?"
    r"|\\cmidrule(?:\([^)]*\))?\{[^}]*\}"
)
CELL_SEPARATOR = re.compile(r"(?<!\\)&")
PRINTED = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
SKIP_KINDS = ("constant", "deferred_two_phase", "untraced")
UNREGISTERED = "unregistered"
AGGREGATES = ("value", "unique", "mean", "std", "se", "median", "min", "max", "sum", "count")
QUOTES = "'\"`"
NAME = re.compile(r"[A-Za-z_]\w*")
COLUMN = re.compile(r"[A-Za-z_][\w/]*")
LITERAL = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclasses.dataclass(frozen=True)
class Difference:
    label: str
    row: int
    col: int
    paper_value: float
    dataset_value: float
    query: str
    part: int = 0


@dataclasses.dataclass(frozen=True)
class Untraced:
    """A printed number the audit does not check: unregistered, or a registered skip of the given kind."""

    label: str
    row: int
    col: int
    text: str
    part: int = 0
    kind: str = UNREGISTERED
    note: str = ""


@dataclasses.dataclass(frozen=True)
class _Entry:
    query: str
    decimals: int
    kind: str
    note: str


@dataclasses.dataclass(frozen=True)
class _Selection:
    rows: pd.DataFrame
    values: pd.Series


def _balanced_end(text: str, start: int, opening: str, closing: str, quotes: str = "") -> int:
    """Index of the bracket that closes the one at `start`, skipping quoted strings."""
    if not text.startswith(opening, start):
        raise ValueError(f"expected {opening!r} at position {start} of {text!r}")
    depth, quote = 0, None
    for pos in range(start, len(text)):
        char = text[pos]
        if quote:
            quote = None if char == quote else quote
        elif char in quotes:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError(f"unbalanced {opening!r} in {text!r}")


def _replace_command(text: str, command: str, n_args: int, keep: int | None) -> str:
    """Replace each `command{a}{b}...` by its argument `keep` (or by nothing)."""
    while (start := text.find(command)) >= 0:
        pos, args = start + len(command), []
        for _ in range(n_args):
            while text[pos].isspace():
                pos += 1
            end = _balanced_end(text, pos, "{", "}")
            args.append(text[pos + 1 : end])
            pos = end + 1
        text = text[:start] + (args[keep] if keep is not None else "") + text[pos:]
    return text


def _plain(cell: str) -> str:
    """A cell's text without layout: math mode, spacing, emphasis, multicolumns, daggers and exponents of ten."""
    text = _replace_command(cell, "\\multicolumn", 3, keep=2)
    text = _replace_command(text, "\\rlap", 1, keep=None)
    text = re.sub(r"\^\{?\\dagger\}?|\\dagger", "", text)
    text = text.replace("\\pm", " \u00b1 ")
    text = re.sub(r"\\times\s*10\^\{?-?\d+\}?", "", text)  # a printed mantissa keeps its units
    text = re.sub(r"\\(?:times|boldmath|textbf|textit|emph|mathbf|mathrm|text|quad|qquad)(?![A-Za-z])", " ", text)
    text = re.sub(r"\\[!,;: ]", "", text)
    for token in ("\\%", "$", "{", "}", "~"):
        text = text.replace(token, " " if token == "~" else "")
    return " ".join(text.split())


def printed_numbers(cell: str) -> list[tuple[float, int]]:
    """The numbers a cell prints, with their printed decimals; none unless the cell starts with a number."""
    text = _plain(cell)
    if not PRINTED.match(text):
        return []
    numbers = []
    for match in PRINTED.finditer(text):
        printed = match.group(0)
        numbers.append((float(printed.replace(",", "")), len(printed.split(".")[1]) if "." in printed else 0))
    return numbers


def numeric(cell: str) -> float | None:
    """The first number a numeric cell prints, or None for a text cell."""
    numbers = printed_numbers(cell)
    return numbers[0][0] if numbers else None


def _tabular_body(block: str, label: str) -> tuple[str, str]:
    """The environment name and the rows text of the one tabular or longtable in a table block."""
    begins = list(TABULAR.finditer(block))
    if len(begins) != 1:
        raise ValueError(f"{label}: expected one tabular or longtable, found {len(begins)}")
    env, spec = begins[0].group("env"), begins[0].end()
    spec_end = _balanced_end(block, spec, "{", "}")
    end = block.find(f"\\end{{{env}}}", spec_end)
    if end < 0:
        raise ValueError(f"{label}: {env} is not closed")
    return env, block[spec_end + 1 : end]


def _body_rows(env: str, text: str) -> list[list[str]]:
    heads = list(LONGTABLE_HEADS.finditer(text)) if env == "longtable" else []
    if heads:
        body = text[heads[-1].end() :]
    else:
        split = MIDRULE.split(text, maxsplit=1)
        body = split[1] if len(split) == 2 else ROW_END.split(text, maxsplit=1)[-1]  # no midrule: one header row
    rows = []
    for line in ROW_END.split(body):
        cells = [cell.strip() for cell in CELL_SEPARATOR.split(RULES.sub("", line).strip())]
        if len(cells) > 1 and any(cells):
            rows.append(cells)
    return rows


def parse_tables(tex: str) -> dict[str, list[list[str]]]:
    """Each table label's body rows of cell strings, header rows excluded."""
    out: dict[str, list[list[str]]] = {}
    for env in TABLE_ENV.finditer(COMMENT.sub(r"\1", tex)):
        block = env.group("body") if env.group("env") else env.group("captioned")
        labels = LABEL.findall(block)
        if not labels:
            continue
        if len(labels) > 1 or labels[0] in out:
            raise ValueError(f"table labels must be unique, one per table: {labels}")
        out[labels[0]] = _body_rows(*_tabular_body(block, labels[0]))
    return out


def read_sections(sections: Path) -> str:
    return "\n".join(path.read_text() for path in sorted(sections.glob("*.tex")))


def _aggregate(values: pd.Series, agg: str) -> float:
    if agg == "count":
        return float(len(values))
    if values.isna().any():
        raise ValueError(f"the selection holds {int(values.isna().sum())} null value(s)")
    if values.empty:
        raise ValueError("the selection is empty")
    if agg == "value":
        if len(values) != 1:
            raise ValueError(f"query must select exactly one row, got {len(values)}")
        return float(values.iloc[0])
    if agg == "unique":
        distinct = values.unique()
        if len(distinct) != 1:
            raise ValueError(f"the selection holds {len(distinct)} distinct values, not one")
        return float(distinct[0])
    if agg in ("std", "se"):
        if len(values) < 2:
            raise ValueError(f"{agg} needs two or more rows, got {len(values)}")
        std = float(values.std(ddof=1))
        return std if agg == "std" else std / math.sqrt(len(values))
    return float(getattr(values, agg)())


def _paired(left: _Selection, right: _Selection, key: str) -> pd.Series:
    a = pd.Series(left.values.to_numpy(), index=left.rows[key].to_numpy())
    b = pd.Series(right.values.to_numpy(), index=right.rows[key].to_numpy())
    if not (a.index.is_unique and b.index.is_unique) or set(a.index) != set(b.index):
        raise ValueError(
            f"paired selections must hold the same {key} values once each: {list(a.index)}, {list(b.index)}"
        )
    return a - b.reindex(a.index)


class _Query:
    """Recursive-descent evaluation of one query (grammar in the module docstring)."""

    def __init__(self, text: str, tables: dict[str, pd.DataFrame]):
        self.text, self.pos, self.tables = text, 0, tables

    def evaluate(self) -> float:
        value = self._sum()
        if self._take("diff:"):
            value -= self._sum()
        self._space()
        if self.pos != len(self.text):
            raise self._error("unexpected text")
        return value

    def _error(self, problem: str) -> ValueError:
        return ValueError(f"malformed query, {problem} at {self.text[self.pos : self.pos + 30]!r}: {self.text}")

    def _space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _take(self, token: str) -> bool:
        self._space()
        if not self.text.startswith(token, self.pos):
            return False
        self.pos += len(token)
        return True

    def _expect(self, token: str) -> None:
        if not self._take(token):
            raise self._error(f"expected {token!r}")

    def _match(self, pattern: re.Pattern) -> str:
        self._space()
        match = pattern.match(self.text, self.pos)
        if not match:
            raise self._error(f"expected {pattern.pattern!r}")
        self.pos = match.end()
        return match.group(0)

    def _sum(self) -> float:
        value = self._product()
        while True:
            if self._take("+"):
                value += self._product()
            elif self._take("-"):
                value -= self._product()
            else:
                return value

    def _product(self) -> float:
        value = self._unary()
        while True:
            if self._take("*"):
                value *= self._unary()
            elif self._take("/"):
                value /= self._unary()
            else:
                return value

    def _unary(self) -> float:
        if self._take("-"):
            return -self._unary()
        if self._take("+"):
            return self._unary()
        return self._atom()

    def _atom(self) -> float:
        self._space()
        if LITERAL.match(self.text, self.pos):
            return float(self._match(LITERAL))
        if self._take("("):
            value = self._sum()
            self._expect(")")
            return value
        name = self._match(NAME)
        if name == "sqrt":
            self._expect("(")
            value = self._sum()
            self._expect(")")
            return math.sqrt(value)
        if name == "paired":
            self._expect("(")
            left = self._selection(self._match(NAME))
            self._expect(",")
            right = self._selection(self._match(NAME))
            self._expect(",")
            key = self._match(NAME)
            self._expect(")")
            return _aggregate(_paired(left, right, key), self._agg())
        return _aggregate(self._selection(name).values, self._agg())

    def _agg(self) -> str:
        self._expect(".")
        agg = self._match(NAME)
        if agg not in AGGREGATES:
            raise self._error(f"unknown aggregate {agg!r}")
        return agg

    def _selection(self, table: str) -> _Selection:
        if table == "concat" and self._take("("):
            parts = [self._selection(self._match(NAME))]
            while self._take(","):
                parts.append(self._selection(self._match(NAME)))
            self._expect(")")
            return _Selection(pd.concat([p.rows for p in parts]), pd.concat([p.values for p in parts]))
        if table not in self.tables:
            raise self._error(f"no build table {table!r}")
        self._space()
        if not self.text.startswith("[", self.pos):
            raise self._error("expected '['")
        end = _balanced_end(self.text, self.pos, "[", "]", QUOTES)
        condition = self.text[self.pos + 1 : end].strip()
        self.pos = end + 1
        frame = self.tables[table]
        rows = frame.query(condition, engine="python") if condition else frame
        self._expect(".")
        return _Selection(rows, self._column(rows))

    def _column(self, rows: pd.DataFrame) -> pd.Series:
        self._space()
        if self.text.startswith("{", self.pos):
            end = _balanced_end(self.text, self.pos, "{", "}", QUOTES)
            expression = self.text[self.pos + 1 : end]
            self.pos = end + 1
            values = rows.eval(expression, engine="python")
            return values if isinstance(values, pd.Series) else pd.Series(values, index=rows.index)
        if self._take("mean_of("):
            tasks = self._selection(self._match(NAME)).values
            self._expect(")")
            return rows[[f"bpb__{task}" for task in tasks]].mean(axis=1, skipna=False)
        return rows[self._match(COLUMN)]


def evaluate(cell_query: str, tables: dict[str, pd.DataFrame]) -> float:
    """The value of one registry query over the build tables."""
    return _Query(cell_query, tables).evaluate()


def _entries(registry: pd.DataFrame, parsed: dict[str, list[list[str]]]) -> dict[tuple[str, int, int, int], _Entry]:
    """Registry rows by (label, row, col, part); raises listing every row that names no printed number."""
    frame = registry.copy()
    if "part" not in frame:
        frame["part"] = 0
    frame[["query", "note"]] = frame[["query", "note"]].fillna("").astype(str)
    entries: dict[tuple[str, int, int, int], _Entry] = {}
    problems = []
    for r in frame.itertuples(index=False):
        key = (str(r.label), int(r.row), int(r.col), int(r.part))
        where = "{} r{} c{} p{}".format(*key)
        rows = parsed.get(key[0], [])
        cells = rows[key[1]] if 0 <= key[1] < len(rows) else []
        printed = printed_numbers(cells[key[2]]) if 0 <= key[2] < len(cells) else []
        query, kind = r.query.strip(), r.note.split(":", 1)[0].strip()
        if key in entries:
            problems.append(f"{where}: registered twice")
        elif not 0 <= key[3] < len(printed):
            problems.append(f"{where}: not a printed number of the paper's table")
        elif int(r.decimals) != printed[key[3]][1]:
            problems.append(f"{where}: the paper prints {printed[key[3]][1]} decimals, the registry says {r.decimals}")
        elif query and kind in SKIP_KINDS:
            problems.append(f"{where}: a skipped cell with a query")
        elif not query and kind not in SKIP_KINDS:
            problems.append(f"{where}: no query, and the note names no skip kind {SKIP_KINDS}")
        entries[key] = _Entry(query, int(r.decimals), "" if query else kind, r.note)
    if problems:
        raise ValueError("the registry does not match the paper's tables:\n" + "\n".join(problems))
    return entries


def compare(
    tex: str, registry: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> tuple[list[Difference], list[Untraced]]:
    """Every registered number that differs from its query at the printed decimals, and every number not checked."""
    parsed = parse_tables(tex)
    entries = _entries(registry, parsed)
    differences, untraced = [], []
    for label, rows in parsed.items():
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                for k, (paper, _) in enumerate(printed_numbers(cell)):
                    entry = entries.get((label, i, j, k))
                    if entry is None:
                        if j > 0:
                            untraced.append(Untraced(label, i, j, cell, k))
                        continue
                    if entry.kind:
                        untraced.append(Untraced(label, i, j, cell, k, entry.kind, entry.note))
                        continue
                    try:
                        value = evaluate(entry.query, tables)
                    except Exception as error:
                        raise ValueError(f"{label} r{i} c{j} p{k}: {error}") from error
                    if round(value, entry.decimals) != round(paper, entry.decimals):
                        differences.append(Difference(label, i, j, paper, value, entry.query, k))
    return differences, untraced


def load_build(build: Path) -> dict[str, pd.DataFrame]:
    """Every table of a build by name: CSVs with exact float parsing, Parquet as written."""
    out = {}
    for path in sorted(build.glob("data/*/*.csv")):
        out[path.stem] = release_tables.read_csv(path)
    for path in sorted(build.glob("data/*/*.parquet")):
        out[path.stem] = pd.read_parquet(path)
    return out


def read_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path, dtype=str, keep_default_na=False)
    return registry.astype({c: int for c in ("row", "col", "part", "decimals") if c in registry})


def audit(build: Path, sections: Path, registry_path: Path = REGISTRY) -> tuple[list[Difference], list[Untraced]]:
    return compare(read_sections(sections), read_registry(registry_path), load_build(build))


def _summary(tex: str, registry: pd.DataFrame, differences: list[Difference], untraced: list[Untraced]) -> list[str]:
    """Per table: its checked numbers (registered or outside the label column), traced, differing and skipped."""
    parsed = parse_tables(tex)
    traced = Counter(key[0] for key, entry in _entries(registry, parsed).items() if not entry.kind)
    differ = Counter(d.label for d in differences)
    lines = []
    for label in parsed:
        kinds = Counter(u.kind for u in untraced if u.label == label)
        numbers = traced[label] + sum(kinds.values())
        skips = "".join(f", {kinds[kind]} {kind}" for kind in (*SKIP_KINDS, UNREGISTERED) if kinds[kind])
        lines.append(f"{label}: {numbers} numbers, {traced[label]} traced, {differ[label]} differ{skips}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", type=Path, required=True, help="the dataset build (its data/<config>/ tables)")
    parser.add_argument("--sections", type=Path, required=True, help="the paper's sections/ folder")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--fail-on-untraced", action="store_true", help="fail when a printed number is not checked")
    parser.add_argument(
        "--allow-registered-skips",
        action="store_true",
        help="with --fail-on-untraced, accept the registry's constant and deferred_two_phase cells",
    )
    args = parser.parse_args(argv)
    tex, registry = read_sections(args.sections), read_registry(args.registry)
    differences, untraced = compare(tex, registry, load_build(args.build))
    for d in differences:
        cell = f"{d.label} r{d.row} c{d.col} p{d.part}"
        print(f"DIFF {cell}: paper {d.paper_value} dataset {d.dataset_value:.6f} <- {d.query}")
    unresolved = [u for u in untraced if u.kind in (UNREGISTERED, "untraced")]
    skipped = [u for u in untraced if u.kind not in (UNREGISTERED, "untraced")]
    for u in unresolved:
        print(f"UNTRACED {u.label} r{u.row} c{u.col} p{u.part}: {u.text}" + (f" ({u.note})" if u.note else ""))
    for line in _summary(tex, registry, differences, untraced):
        print(line)
    print(f"{len(differences)} differences, {len(unresolved)} untraced cells, {len(skipped)} registered skips")
    failing = unresolved + ([] if args.allow_registered_skips else skipped)
    return 1 if differences or (args.fail_on_untraced and failing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
