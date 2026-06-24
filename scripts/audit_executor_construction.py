# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scope the executor build-phase migration.

Imports every ``experiments/`` module inside an ``audit_construction()`` block and
reports each ``ExecutorStep`` / ``StepSpec`` built outside an ``executor_context()``
— i.e. the module-import-scope construction that the guard will reject. Use it to
size the rewrite and to drive the per-file migration.

    uv run --package marin-core python scripts/audit_executor_construction.py
    uv run --package marin-core python scripts/audit_executor_construction.py --traceback  # full stacks

Import-time construction is captured directly. Construction that only runs under a
module's ``if __name__ == "__main__"`` block (already a ``build_steps()`` flow) is
not exercised by import and is not reported here — those sites just need their
entrypoint wrapped in ``executor_context()``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections import defaultdict
from pathlib import Path

from marin.execution.context import ContextlessConstruction, audit_construction

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def discover_modules(root: Path) -> list[str]:
    """Return dotted module names for every importable ``.py`` under ``root``."""
    modules = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def audit_all(modules: list[str]) -> tuple[list[ContextlessConstruction], dict[str, str]]:
    """Import every module under one audit window; return constructions and import errors."""
    import_errors: dict[str, str] = {}
    with audit_construction() as sink:
        for modname in modules:
            try:
                importlib.import_module(modname)
            except Exception as e:
                import_errors[modname] = f"{type(e).__name__}: {e}"
    return sink, import_errors


def migration_site(record: ContextlessConstruction) -> tuple[str, int]:
    """Where the rewrite happens: the innermost ``experiments/`` frame on the stack.

    Library factories (``download_*_step``, ``perplexity_gap_step``, …) are correct —
    the defect is the experiment calling them at module scope, so attribute the
    construction to that experiment line, not the factory. Constructions with no
    experiment frame (a module-level step inside ``lib/``) fall back to their site.
    """
    for frame in reversed(record.stack):
        if f"{EXPERIMENTS_ROOT.name}/" in frame.filename.replace(str(REPO_ROOT) + "/", ""):
            return (frame.filename, frame.lineno or 0)
    site = record.site
    return (site.filename, site.lineno or 0) if site else ("<unknown>", 0)


def report(records: list[ContextlessConstruction], import_errors: dict[str, str], *, show_traceback: bool) -> None:
    by_file: dict[str, list[ContextlessConstruction]] = defaultdict(list)
    for record in records:
        filename = migration_site(record)[0]
        by_file[filename].append(record)

    distinct_sites = {migration_site(r) for r in records}

    print("\n=== Contextless executor-step construction (by experiment to migrate) ===\n")
    for filename in sorted(by_file, key=_relativize):
        rel = _relativize(filename)
        rows = sorted(by_file[filename], key=lambda r: migration_site(r)[1])
        lines = sorted({migration_site(r)[1] for r in rows})
        print(f"{rel}  ({len(rows)} construction(s) across {len(lines)} line(s))")
        if show_traceback:
            for record in rows:
                print(f"    L{migration_site(record)[1]:<5} {record.kind:<12} {record.name}")
                for line in record.stack.format():
                    print(f"        {line.rstrip()}")
        print()

    print("=== Summary ===")
    print(f"  constructions   : {len(records)}")
    print(f"  distinct lines  : {len(distinct_sites)}")
    print(f"  files to migrate: {len(by_file)}")
    if import_errors:
        print(f"\n  {len(import_errors)} module(s) could not be imported (not audited):")
        for modname, err in sorted(import_errors.items()):
            print(f"    {modname}: {err}")


def _relativize(filename: str) -> str:
    try:
        return str(Path(filename).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traceback", action="store_true", help="print the full stack for each construction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)  # silence noisy experiment imports
    modules = discover_modules(EXPERIMENTS_ROOT)
    print(f"Auditing {len(modules)} experiment modules under {EXPERIMENTS_ROOT.relative_to(REPO_ROOT)}/ ...")
    records, import_errors = audit_all(modules)
    report(records, import_errors, show_traceback=args.traceback)


if __name__ == "__main__":
    main()
