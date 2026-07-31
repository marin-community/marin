# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "scipy",
# ]
# ///
"""Paired bootstrap of continuous shape optimization on genuinely new policies.

The censored arm in ``bootstrap_continuous_shape_20260726.py`` hides the best rows of
the fit panel, which tests extrapolation in the target but not in mixture space: the
hidden policies still come from the same two design series. The 300M heldout split
contains 111 policies at least 0.005 total variation from every fit-panel aggregate, so
they test the other axis, and nothing in either arm's selection ever reads them.

This is the direct test of whether a lower out-of-fold criterion is a better model or
just a better-tuned criterion. Draws are constructed by
``bootstrap_continuous_shape_20260726.make_draws`` with the same seed, so the resamples
here are the same resamples used there and the two files describe one experiment.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_continuous_shape_20260726 import (  # noqa: E402
    BUILDERS,
    DRAWS,
    L2_GRIDS,
    SEED,
    SELECTORS,
    Draw,
    make_draws,
    paired_summary,
)
from dual_objective_harness_20260726 import build_benchmark, fit_on  # noqa: E402
from swarm39_harness_20260725 import TABLE9, UNCHEATABLE, Model  # noqa: E402
from swarm39_models_20260725 import _state_shapes  # noqa: E402

_STATE: dict[str, Any] = {}


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def score_novel(bench, model: Model, target: str, criterion: str, selector, draw: Draw) -> dict:
    """Fit on a resample of the 300M fit panel, score the fixed novel heldout policies."""
    panel = bench.fit_300m.subset(draw.fit_300m)
    shape, l2, info = selector(panel, model, target, criterion, None)
    fitted = fit_on(panel, model, target, shape, l2)
    novel = bench.heldout_300m.subset(bench.novel_rows)
    predicted = fitted.predict(novel)
    observed = novel.targets[target]
    keep = np.isfinite(predicted) & np.isfinite(observed)
    residual = predicted[keep] - observed[keep]
    return {
        "selected_oof_rmse": float(info["selected_score"]),
        "novel_rmse": float(np.sqrt(np.mean(residual**2))),
        "novel_bias": float(np.mean(residual)),
        "novel_abs_bias": float(abs(np.mean(residual))),
        "novel_spearman": float(np.corrcoef(_ranks(predicted[keep]), _ranks(observed[keep]))[0, 1]),
    }


def _initializer(model_name: str, l2_name: str, target: str, criterion: str) -> None:
    build, include_threshold = BUILDERS[model_name]
    _STATE.update(
        bench=build_benchmark(),
        model=Model(model_name, build, (lambda i=include_threshold: _state_shapes(i)), l2_grid=L2_GRIDS[l2_name]),
        target=target,
        criterion=criterion,
    )


def _worker(payload: tuple[int, Draw]) -> tuple[int, dict, dict]:
    index, draw = payload
    args = (_STATE["bench"], _STATE["model"], _STATE["target"], _STATE["criterion"])
    return index, score_novel(*args, SELECTORS["grid"], draw), score_novel(*args, SELECTORS["continuous"], draw)


def run(model_name: str, l2_name: str, target: str, draws: int, processes: int) -> dict:
    bench = build_benchmark()
    draw_list, _ = make_draws(bench, target, draws, SEED)
    grid_rows: list[dict | None] = [None] * draws
    continuous_rows: list[dict | None] = [None] * draws
    with Pool(processes, initializer=_initializer, initargs=(model_name, l2_name, target, "rmse")) as pool:
        for index, grid, continuous in pool.imap_unordered(_worker, list(enumerate(draw_list)), chunksize=1):
            grid_rows[index] = grid
            continuous_rows[index] = continuous
    assert all(row is not None for row in grid_rows), "a bootstrap draw did not return"
    return {
        "model": model_name,
        "l2_grid": l2_name,
        "target": target,
        "draws": draws,
        "n_novel": int(bench.novel_rows.sum()),
        "paired": paired_summary(grid_rows, continuous_rows),  # type: ignore[arg-type]
    }


def main() -> None:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "bootstrap_novel_generalization.json"
    results = []
    for model_name in ("hierarchical_phase_replay", "compact_retained_state"):
        for target in (UNCHEATABLE, TABLE9):
            result = run(model_name, "no_ridge", target, DRAWS, 14)
            results.append(result)
            print(f"=== {model_name} no_ridge {target} ({result['draws']} draws, {result['n_novel']} novel)", flush=True)
            for key, row in result["paired"].items():
                print(
                    f"  {key:18s} grid={row['grid_mean']:+.5f} cont={row['continuous_mean']:+.5f} "
                    f"diff={row['diff_mean']:+.5f} [{row['diff_lo']:+.5f},{row['diff_hi']:+.5f}] "
                    f"frac_better={row['fraction_continuous_better']:.3f}",
                    flush=True,
                )
            destination.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
