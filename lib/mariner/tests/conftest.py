# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: a synthetic swarm drawn from a known MARINER model, a miniature snapshot in mariner-data's layout,
and the local mariner-data snapshot."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.model import Anchor, Head, Objective, ObjectiveFit, Shape, Swarm, TaskFit

SYNTHETIC_SEED = 20261028
EPOCHS_PER_UNIT_WEIGHT = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SAMPLED_RUNS = 160  # at 80, t1's floor search ends flat and fresh mixtures leave the swarm's coverage
TRUE_SHAPES = (Shape(0.5, 1.0, 2.0), Shape(1.0, 0.7, 3.0), Shape(0.25, 1.0, 1.0))
TRUE_INTERCEPT = math.log(0.3)
TRUE_PHI = 0.8
TASK_WEIGHTS = (0.5, 0.3, 0.2)
NOISE_SD = 0.002


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--mariner-data", default=None, help="local mariner-data snapshot read by data_integration tests")


@pytest.fixture(scope="session")
def snapshot(request: pytest.FixtureRequest) -> Path:
    root = request.config.getoption("--mariner-data")
    if root is None:
        pytest.fail("pass --mariner-data <snapshot dir>; see lib/mariner/README.md")
    return Path(root)


@pytest.fixture
def synthetic_problem() -> tuple[Swarm, Objective, dict[str, Anchor], ObjectiveFit]:
    """A swarm whose outcomes follow a known three-task MARINER model plus Gaussian noise.

    Six buckets; row 0 is the proportional mixture and the calibration row, rows 1..160 are Dirichlet draws, and every
    pool fraction is one. Returns the swarm, its objective, anchors at the model's value for the proportional mixture
    (repeat SD the noise SD) and the generating model.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    epochs = np.asarray(EPOCHS_PER_UNIT_WEIGHT)
    buckets = tuple(f"b{index}" for index in range(len(epochs)))
    proportional = (1 / epochs) / np.sum(1 / epochs)
    weights = np.vstack([proportional, rng.dirichlet(np.ones(len(epochs)), size=SAMPLED_RUNS)])
    task_names = tuple(f"t{index}" for index in range(len(TRUE_SHAPES)))
    tasks = []
    for task, shape in zip(task_names, TRUE_SHAPES, strict=True):
        benefit_amplitudes = rng.uniform(0.2, 1.0, len(epochs))
        harm_amplitudes = rng.uniform(0.0, 0.3, len(epochs))
        head = Head(TRUE_INTERCEPT, np.concatenate([benefit_amplitudes, harm_amplitudes]), TRUE_PHI)
        # gamma, nu and the inner-CV RMSE describe a fit; a generating model has none.
        tasks.append(TaskFit(task, shape, math.nan, math.nan, False, math.nan, head))
    truth = ObjectiveFit("synthetic", buckets, epochs, np.asarray(TASK_WEIGHTS), tuple(tasks), {})
    clean = truth.predict_tasks(weights)
    outcomes = pd.DataFrame(clean + rng.normal(0.0, NOISE_SD, size=clean.shape), columns=list(task_names))
    swarm = Swarm(
        runs=tuple(f"run_{row:03d}" for row in range(len(weights))),
        buckets=buckets,
        weights=weights,
        epochs_per_unit_weight=epochs,
        pool_fractions=np.ones_like(weights),
        outcomes=outcomes,
        calibration=np.arange(len(weights)) == 0,
    )
    objective = Objective("synthetic", task_names, np.asarray(TASK_WEIGHTS))
    at_proportional = truth.predict_tasks(proportional)[0]
    anchors = {task: Anchor(float(value), NOISE_SD) for task, value in zip(task_names, at_proportional, strict=True)}
    return swarm, objective, anchors, truth


RELEASE_SEED = 20261029
RELEASE_PROPORTIONAL = (4 / 7, 2 / 7, 1 / 7)  # pools of 400, 200 and 100 tokens


def mixture_columns(weights: Sequence[float], pool_fractions: Sequence[float] = (1.0, 1.0, 1.0)) -> dict[str, float]:
    return {
        **{f"w__b{bucket}": weight for bucket, weight in enumerate(weights)},
        **{f"pf__b{bucket}": fraction for bucket, fraction in enumerate(pool_fractions)},
    }


def write_release_table(root: Path, config: str, table: str, frame: pd.DataFrame, file_format: str) -> dict[str, str]:
    """Write ``frame`` where the dataset keeps ``table`` and return the table's MANIFEST.json entry."""
    path = Path("data") / config / f"{table}.{file_format}"
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        frame.to_parquet(root / path, index=False)
    else:
        frame.to_csv(root / path, index=False)
    return {"config": config, "table": table, "path": path.as_posix(), "format": file_format}


@pytest.fixture
def release_snapshot(tmp_path: Path) -> Path:
    """A miniature mariner-data snapshot, with the tables and MANIFEST.json entries the loader reads.

    - partition: buckets b0, b1 and b2 with pools of 400, 200 and 100 tokens and 2, 4 and 8 epochs per unit weight.
    - tasks: t0 and t1 in obe_easy (weight 1/2 each), u0 in uncheatable.
    - swarm_qwen3_360m_1p6b: the fit rows fit_00 to fit_12 in that order, with the proportional repeats repeat_0,
      repeat_1 and repeat_2 after fit_02, fit_06 and fit_12. fit_01 is the proportional baseline; fit_05 has weights
      (0.5, 0.25, 0.25) and draws half of b2's pool. Fit row k has bpb__t0 = 1 + k / 100; the repeats have t0 (and
      t1) 1.0, 1.1 and 1.2 and u0 2.0, 2.2 and 2.4.
    - ladder: the June proportional run at 3e18 (t0 and t1 5.0, u0 2.6), a copy of repeat_0, a MARINER run at 3e18
      and a proportional run at 2e19, all at the proportional mixture from whole pools.
    - heldout: h0 (t0 1.0, t1 0.8) and h2 (t0 1.2, t1 1.0) at exposure e0, h1 (t0 1.4, t1 1.2) at e1, which draws
      half of b2's pool, and h3 at e2. bpb__uncheatable records the Uncheatable aggregate (2.0, 2.4, 2.2, 2.6); h3
      has it but no u0, as five of the release's held-out runs do. The table has no bpb__obe_easy.
      fits_heldout_predictions, the one Parquet table, predicts h0 to h2 for both objectives and h3 for uncheatable
      only.
    """
    rng = np.random.default_rng(RELEASE_SEED)
    fit_rows = []
    for row in range(13):
        label, kind, weights, fractions = f"run_{row:05d}", "sampled", rng.dirichlet(np.ones(3)), (1.0, 1.0, 1.0)
        if row == 1:
            label, kind, weights = "baseline_proportional", "baseline", RELEASE_PROPORTIONAL
        elif row == 5:
            weights, fractions = (0.5, 0.25, 0.25), (1.0, 1.0, 0.5)
        fit_rows.append(
            {
                "observation_id": f"fit_{row:02d}",
                "run_label": label,
                "design_kind": kind,
                "fit_inclusion": True,
                **mixture_columns(weights, fractions),
                "bpb__t0": 1 + row / 100,
                "bpb__t1": 2 + row / 100,
                "bpb__u0": 3 + row / 100,
            }
        )
    repeats = [
        {
            "observation_id": f"repeat_{index}",
            "run_label": f"proportional_repeat_{index:02d}",
            "design_kind": "repeat",
            "fit_inclusion": False,
            **mixture_columns(RELEASE_PROPORTIONAL),
            "bpb__t0": value,
            "bpb__t1": value,
            "bpb__u0": uncheatable,
        }
        for index, (value, uncheatable) in enumerate(zip((1.0, 1.1, 1.2), (2.0, 2.2, 2.4), strict=True))
    ]
    swarm = pd.DataFrame([*fit_rows[:3], repeats[0], *fit_rows[3:7], repeats[1], *fit_rows[7:], repeats[2]])
    ladder = pd.DataFrame(
        [
            {"observation_id": "june_3e18", "method": "proportional", "compute_flops": 3e18, "bpb__u0": 2.6},
            {"observation_id": "repeat_0", "method": "proportional", "compute_flops": 3e18, "bpb__u0": 2.0},
            {"observation_id": "mariner_3e18", "method": "mariner", "compute_flops": 3e18, "bpb__u0": 1.5},
            {"observation_id": "proportional_2e19", "method": "proportional", "compute_flops": 2e19, "bpb__u0": 1.9},
        ]
    ).assign(bpb__t0=[5.0, 1.0, 1.5, 1.9], bpb__t1=[5.0, 1.0, 1.5, 1.9], **mixture_columns(RELEASE_PROPORTIONAL))
    heldout = pd.DataFrame(
        [
            {"observation_id": "h0", "exposure_id": "e0", **mixture_columns((0.2, 0.3, 0.5)), "bpb__t0": 1.0},
            {
                "observation_id": "h1",
                "exposure_id": "e1",
                **mixture_columns((0.6, 0.2, 0.2), (1.0, 1.0, 0.5)),
                "bpb__t0": 1.4,
            },
            {"observation_id": "h2", "exposure_id": "e0", **mixture_columns((0.2, 0.3, 0.5)), "bpb__t0": 1.2},
            {"observation_id": "h3", "exposure_id": "e2", **mixture_columns((0.1, 0.1, 0.8)), "bpb__t0": 1.6},
        ]
    ).assign(bpb__t1=[0.8, 1.2, 1.0, 1.4], bpb__u0=[2.0, 2.4, 2.2, math.nan], bpb__uncheatable=[2.0, 2.4, 2.2, 2.6])
    predictions = pd.DataFrame(
        [
            {"objective": objective, "observation_id": run, "task_id": task, "predicted_bpb": 1.0}
            for objective, runs, tasks in (
                ("obe_easy", ("h0", "h1", "h2"), ("t0", "t1")),
                ("uncheatable", ("h0", "h1", "h2", "h3"), ("u0",)),
            )
            for run in runs
            for task in tasks
        ]
    )
    partition = pd.DataFrame(
        {"bucket_id": ["b0", "b1", "b2"], "pool_tokens": [400, 200, 100], "epochs_per_unit_weight": [2.0, 4.0, 8.0]}
    )
    tasks = pd.DataFrame(
        {
            "task_id": ["t0", "t1", "u0"],
            "objective": ["obe_easy", "obe_easy", "uncheatable"],
            "aggregate_weight": [0.5, 0.5, 1.0],
        }
    )
    entries = [
        write_release_table(tmp_path, "partition", "partition", partition, "csv"),
        write_release_table(tmp_path, "tasks", "tasks", tasks, "csv"),
        write_release_table(tmp_path, "swarm_qwen3_360m_1p6b", "swarm_qwen3_360m_1p6b", swarm, "csv"),
        write_release_table(tmp_path, "ladder", "ladder", ladder, "csv"),
        write_release_table(tmp_path, "heldout", "heldout", heldout, "csv"),
        write_release_table(tmp_path, "fits", "fits_heldout_predictions", predictions, "parquet"),
    ]
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"tables": entries}, indent=1))
    return tmp_path
