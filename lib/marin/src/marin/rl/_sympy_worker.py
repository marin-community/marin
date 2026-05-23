# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolated sympy worker for spawn-context multiprocessing.

Lives in marin.rl (which has no __init__.py — namespace package) instead of
marin.rl.environments.tinker_environments because the latter triggers
``marin.rl.environments.__init__`` -> ``.base`` -> ``import jax`` on import.
With jax 0.9.2 + libtpu 0.0.39 the JAX TPU plugin's import-time init blocks
when run under a spawn subprocess whose parent (the rollout worker) already
holds TPU device locks, causing ``parse_expr`` calls to time out at exactly
the wrapper's 10s deadline.

By keeping this module's imports to sympy only, the spawn subprocess's
re-import of the worker function is fast (~1-2s) and avoids touching TPU
state. See rl_blog logbook F12 for the full diagnosis.
"""

from sympy.parsing import sympy_parser


def parse_expr_worker(py_expr: str):
    """Top-level worker for subprocess multiprocessing — must be picklable."""
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(*sympy_parser.standard_transformations, sympy_parser.implicit_multiplication_application),
    )


def sympy_prewarm() -> None:
    """Trivial task used to force this module's import inside a spawn worker.

    Importing the module re-runs ``from sympy.parsing import sympy_parser`` at
    the top, which is what makes per-call sympy parses fast after the first
    call. Used by the persistent pool warm-up path; see math_grading.py F13.
    """
    return None
