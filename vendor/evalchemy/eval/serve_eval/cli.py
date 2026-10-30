"""Dispatch Evalchemy's public console commands."""

from __future__ import annotations

import sys
from typing import Sequence

from eval.serve_eval.validate_config import main as validate_config


def main(argv: Sequence[str] | None = None) -> None:
    """Run a validation command or the legacy evaluation CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "validate-config":
        validate_config(args[1:])
        return

    from eval.eval import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
