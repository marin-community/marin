# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point: run replay scenarios and dump DB + SQL trace per scenario.

::

    uv run python -m iris.cluster.controller.replay.run \\
        [--seed=PATH/TO/CHECKPOINT_DIR] \\
        [--scenario=NAME[,NAME...]] \\
        --out=PATH/TO/OUTDIR

For every scenario, writes ``OUTDIR/<scenario>/db.json`` and
``OUTDIR/<scenario>/sql.txt``. With ``--seed``, each scenario starts
from the supplied checkpoint via ``ControllerDB.replace_from``.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.replay.db_dump import deterministic_dump
from iris.cluster.controller.replay.scenarios import SCENARIO_NAMES, run_scenario
from iris.cluster.controller.replay.sql_trace import sql_tracing
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions

logger = logging.getLogger(__name__)


def _resolve_scenarios(arg: str | None) -> list[str]:
    if arg is None:
        return list(SCENARIO_NAMES)
    requested = [name.strip() for name in arg.split(",") if name.strip()]
    unknown = [name for name in requested if name not in SCENARIO_NAMES]
    if unknown:
        raise SystemExit(f"unknown scenario(s): {unknown}; available: {SCENARIO_NAMES}")
    return requested


def _execute_scenario(name: str, *, seed: Path | None, out: Path) -> None:
    sql_log: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"iris-replay-{name}-") as db_dir_str:
        db_dir = Path(db_dir_str)
        with sql_tracing(sql_log):
            db = ControllerDB(db_dir=db_dir)
            try:
                if seed is not None:
                    db.replace_from(seed)
                store = ControllerStore(db)
                transitions = ControllerTransitions(store)
                run_scenario(name, transitions)
                dump = deterministic_dump(db)
            finally:
                db.close()

    scenario_out = out / name
    scenario_out.mkdir(parents=True, exist_ok=True)
    (scenario_out / "db.json").write_text(json.dumps(dump, indent=2, sort_keys=True) + "\n")
    (scenario_out / "sql.txt").write_text("\n".join(sql_log) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iris.cluster.controller.replay.run", description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="Optional path or URI loaded via ControllerDB.replace_from before each scenario.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Comma-separated scenario names. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory; one subdirectory per scenario containing db.json and sql.txt.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Python logging level (default WARNING; verbose logs would mix with replay output).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    scenarios = _resolve_scenarios(args.scenario)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in scenarios:
        logger.info("Running scenario: %s", name)
        _execute_scenario(name, seed=args.seed, out=args.out)
        print(f"wrote {args.out / name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
