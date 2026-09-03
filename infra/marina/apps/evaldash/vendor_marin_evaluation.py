# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy the record contract and statistics engine out of lib/marin into this app.

EvalDash shares five modules with the evaluation runners. Installing ``marin-core`` would
pull JAX and the training stack into the Marina image and its test environment, so the
modules are vendored under ``marin_evaluation/`` with their package imports rewritten.
``tests/test_vendored_modules.py`` fails when the copies drift from the library; run this
script to refresh them:

    uv run apps/evaldash/vendor_marin_evaluation.py
"""

from pathlib import Path

APP = Path(__file__).resolve().parent
REPO_ROOT = APP.parents[3]
LIBRARY = REPO_ROOT / "lib" / "marin" / "src" / "marin" / "evaluation"
VENDORED = APP / "marin_evaluation"
MODULES = ("archive", "records", "eval_stats", "eval_measurements", "lm_eval_samples")
LIBRARY_PACKAGE = "marin.evaluation."
VENDORED_PACKAGE = "evaldash.marin_evaluation."


def vendored_source(module: str) -> str:
    """The library module's source as it must appear in the vendored copy."""
    return (LIBRARY / f"{module}.py").read_text(encoding="utf-8").replace(LIBRARY_PACKAGE, VENDORED_PACKAGE)


def main() -> None:
    VENDORED.mkdir(exist_ok=True)
    for module in MODULES:
        (VENDORED / f"{module}.py").write_text(vendored_source(module), encoding="utf-8")
        print(f"vendored {module}")


if __name__ == "__main__":
    main()
