# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild mixture-selection's bundled inputs from the dataset build, and refit MARINER on the two archived swarms.

`uv run python -m experiments.domain_phase_mix.release.regenerate_mixture_selection_data bundle --out DIR`
`uv run python -m experiments.domain_phase_mix.release.regenerate_mixture_selection_data archived-fits`

`bundle` writes the five bundle files that the build holds, in the bundle's layout (internal bucket and metric names
from the build's `partition` and `tasks` tables, the hash-less training run name in `run`, the `calibration` flag):
`swarm_weights.csv`, `swarm_outcomes.csv` and `objectives.csv` byte for byte; `buckets.csv` with the dataset's exact
D_TGT / pool_tokens, where the bundle's file divides a budget a relative 5.0e-7 larger; and `anchors.csv`, each task's
mean and sample SD over the proportional mixture's repeats: the ten repeats for OlmoBaseEval Easy, and for Uncheatable
those ten and the June ladder run at 3e18 (the `ladder` row that no swarm table holds). The bundle's manifest names the
swarm's own proportional run as the eleventh; the June run is the one that reproduces the bundle's anchors.

`archived-fits` writes the same layout for the fit rows of each archived swarm into `<out>/<swarm>/`, with anchors from
that swarm's ten repeats and no fold table, so mixture-selection recomputes the inner folds; runs `mixture_selection.py
fit` for both objectives into `<out>/fit_<swarm>_<objective>.json`; and writes the per-task parameters to the committed
ARCHIVED_FITS, which `configs/fits.py` appends to `fits_task_parameters`. mixture-selection runs in an isolated
environment built from its lockfile, so its checkout stays untouched.
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import paths, tables

MAIN_SWARM = "swarm_qwen3_360m_1p6b"
ARCHIVED_SWARMS = ("swarm_llama_160m_1p2b", "swarm_llama_200m_6b")
OBJECTIVES = {"uncheatable": "uncheatable", "obe_easy": "table9"}  # public name -> mixture-selection's
CALIBRATION_RUN = "baseline_proportional"  # the proportional fit row, which anchors the floors and is never validated
REPEAT = "repeat"  # design_kind of the proportional repeats
SWARM_SEED_BASE = 7_141_000  # the main swarm's launch trained run NNN with data seed 7,141,000 + NNN
JUNE_RUN_FLOPS = 3e18
# mixture-selection's inner folds without a fold table: three mixture blocks by k-means (50 restarts, seed 20,350,902)
# on the square-root weights, the proportional run pinned to training. The bundle's fold table, behind the reference
# fit, holds the same labels.
INNER_FOLDS = "kmeans3_sqrt_weights_seed20350902"
ARCHIVED_FITS = paths.REPO / "experiments/domain_phase_mix/release/archived_swarm_fits.csv"
TASK_PARAMETERS = [
    "swarm",
    "objective",
    "task_id",
    "rate",
    "power",
    "threshold",
    "ridge",
    "kappa",
    "floor",
    "inner_cv_rmse",
    "inner_folds",
]
# `python` in an environment built from mixture-selection's lockfile, its optional `folds` extra (scikit-learn, which
# recomputes inner folds) included; `--isolated` leaves the checkout's own environment untouched.
MIXSEL_PYTHON = ["uv", "run", "--project", str(paths.MIXSEL), "--extra", "folds", "--frozen", "--isolated", "python"]
HEADER = """\
# MARINER's per-task fits to the archived swarms' fit rows: mixture_selection.py fit, both objectives, with anchors from
# each swarm's ten proportional repeats and inner folds recomputed (inner_folds), written by
# experiments/domain_phase_mix/release/regenerate_mixture_selection_data.py archived-fits.
# The archived swarms' published Uncheatable aggregates use the legacy corpus weights; these fits weight the corpora by
# the paper's fixed byte shares (objectives.csv), which enter the fitted aggregate, not the per-task parameters below.
# generated_from_commit: {marin} (mixture-selection {mixsel}; dataset build of {build})
"""


def _table(build: Path, name: str) -> pd.DataFrame:
    return tables.read_csv(paths.require(build / "data" / name / f"{name}.csv"))


def _repeats(swarm: pd.DataFrame) -> pd.DataFrame:
    repeats = swarm[swarm.design_kind == REPEAT]
    assert len(repeats) == 10, f"expected ten proportional repeats, found {len(repeats)}"
    return repeats


def _write_layout(
    out: Path,
    fit: pd.DataFrame,
    run: pd.Series,
    build: Path,
    anchor_runs: dict[str, pd.DataFrame],
) -> None:
    """mixture-selection's five input files for the fit rows `fit`, named `run`, with each objective's anchors over the
    proportional runs in `anchor_runs` (keyed by public objective name)."""
    partition, tasks = _table(build, "partition"), _table(build, "tasks")
    fit = fit.reset_index(drop=True)
    head = pd.DataFrame({"run": run.to_numpy(), "calibration": (fit.run_label == CALIBRATION_RUN).to_numpy()})
    assert head.calibration.sum() == 1, "one calibration row: the proportional fit run"
    weights = fit[[f"w__{b}" for b in partition.bucket_id]].set_axis(list(partition.internal_name), axis=1)
    outcomes = fit[[f"bpb__{t}" for t in tasks.task_id]].set_axis(list(tasks.internal_metric), axis=1)
    objectives = pd.DataFrame(
        {
            "objective": [OBJECTIVES[o] for o in tasks.objective],
            "component": tasks.internal_metric,
            "weight": tasks.aggregate_weight,
        }
    )
    buckets = pd.DataFrame(
        {
            "bucket": partition.internal_name,
            "pool_tokens": partition.pool_tokens,
            "epochs_per_unit_weight": partition.epochs_per_unit_weight,
        }
    )
    anchors = []
    for task, metric, objective in zip(tasks.task_id, tasks.internal_metric, tasks.objective, strict=True):
        values = anchor_runs[objective][f"bpb__{task}"]
        assert values.notna().all(), f"{task}: a proportional run lacks its value"
        anchors.append(
            {
                "objective": OBJECTIVES[objective],
                "component": metric,
                "proportional_bpb": values.mean(),
                "repeat_sd": values.std(ddof=1),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    # pandas' default float repr is the shortest round-trip string, the bundle's own formatting.
    pd.concat([head, weights], axis=1).to_csv(out / "swarm_weights.csv", index=False)
    pd.concat([head[["run"]], outcomes], axis=1).to_csv(out / "swarm_outcomes.csv", index=False)
    objectives.to_csv(out / "objectives.csv", index=False)
    buckets.to_csv(out / "buckets.csv", index=False)
    pd.DataFrame(anchors).to_csv(out / "anchors.csv", index=False)


def write(build: Path, out: Path) -> None:
    """The bundle's swarm, objective, bucket and anchor files, from the build's main swarm, ladder and name tables."""
    swarm = _table(build, MAIN_SWARM)
    number = swarm.data_seed - SWARM_SEED_BASE
    fit = swarm[swarm.fit_inclusion].assign(number=number).sort_values("number")
    assert fit.number.between(0, 999).all() and fit.number.is_unique, "data seeds outside the swarm's seed block"
    run = pd.Series([f"singleavg_fit_{n:03d}_{label}" for n, label in zip(fit.number, fit.run_label, strict=True)])
    ladder = _table(build, "ladder")
    june = ladder[
        (ladder.method == "proportional")
        & (ladder.compute_flops == JUNE_RUN_FLOPS)
        & ~ladder.observation_id.isin(swarm.observation_id)
    ]
    assert len(june) == 1, f"expected one proportional ladder run at 3e18 outside the swarm, found {len(june)}"
    repeats = _repeats(swarm)
    _write_layout(out, fit, run, build, {"obe_easy": repeats, "uncheatable": pd.concat([repeats, june])})


def write_archived(build: Path, swarm_table: str, out: Path) -> None:
    """mixture-selection's inputs for an archived swarm's fit rows, with anchors from its ten proportional repeats; no
    fold table, so mixture-selection recomputes the inner folds."""
    swarm = _table(build, swarm_table)
    fit = swarm[swarm.fit_inclusion]
    repeats = _repeats(swarm)
    _write_layout(out, fit, fit.run_label, build, {"obe_easy": repeats, "uncheatable": repeats})


def run_mixture_selection(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """`python <args>` in mixture-selection's locked environment."""
    return subprocess.run([*MIXSEL_PYTHON, *args], cwd=cwd, capture_output=True, text=True)


def fit_command(inputs: Path, objective: str, out: Path) -> list[str]:
    """Arguments of mixture-selection's `fit` on the bundle-layout files in `inputs`; `inputs` holds no fold table, so
    the fit recomputes the inner folds."""
    splits = inputs / "splits.csv"
    assert not splits.exists(), f"{splits} would replace the recomputed inner folds"
    names = ("swarm-weights", "swarm-outcomes", "objectives", "buckets", "anchors", "splits")
    files = ("swarm_weights.csv", "swarm_outcomes.csv", "objectives.csv", "buckets.csv", "anchors.csv", "splits.csv")
    inputs_args = [arg for name, file in zip(names, files, strict=True) for arg in (f"--{name}", str(inputs / file))]
    script = str(paths.MIXSEL / "mixture_selection.py")
    return [script, "fit", "--objective", objective, *inputs_args, "--out", str(out)]


def task_parameters(fit: dict, swarm: str, build: Path) -> pd.DataFrame:
    """The TASK_PARAMETERS rows of a mixture-selection fit JSON, named by the build's task ids."""
    tasks = _table(build, "tasks")
    task_ids = dict(zip(tasks.internal_metric, tasks.task_id, strict=True))
    objective = {internal: public for public, internal in OBJECTIVES.items()}[fit["objective"]]
    rows = [
        {
            "swarm": swarm,
            "objective": objective,
            "task_id": task_ids[t["component"]],
            "rate": t["shape"]["rate"],
            "power": t["shape"]["power"],
            "threshold": t["shape"]["threshold"],
            "ridge": t["ridge"],
            "kappa": t["kappa"],
            "floor": t["floor"],
            "inner_cv_rmse": t["inner_cv_rmse"],
            "inner_folds": INNER_FOLDS,
        }
        for t in fit["tasks"]
    ]
    return pd.DataFrame(rows, columns=TASK_PARAMETERS)


def _fit(inputs: Path, objective: str, out: Path) -> None:
    result = run_mixture_selection(fit_command(inputs, objective, out), cwd=inputs)
    if result.returncode:
        raise RuntimeError(f"mixture-selection fit {out.name} failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    print(f"wrote {out}", flush=True)


def _commit(repo: Path, *files: str) -> str:
    """`repo`'s checked-out commit; raises when any of `files` has uncommitted changes, so the commit holds the code
    that ran."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *files], cwd=repo, capture_output=True, text=True, check=True
    )
    if status.stdout.strip():
        raise ValueError(f"commit before generating: {status.stdout.strip()}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    return head.stdout.strip()


def fit_archived(build: Path, out: Path) -> pd.DataFrame:
    """Refit both objectives on both archived swarms (four fits in parallel) and return their per-task parameters."""
    jobs = []
    for table in ARCHIVED_SWARMS:
        swarm = table.removeprefix("swarm_")
        write_archived(build, table, out / swarm)
        jobs += [(swarm, out / swarm, name, out / f"fit_{swarm}_{name}.json") for name in OBJECTIVES.values()]
    with ThreadPoolExecutor(len(jobs)) as pool:
        list(pool.map(lambda job: _fit(*job[1:]), jobs))
    return pd.concat(
        [task_parameters(json.loads(path.read_text()), swarm, build) for swarm, _, _, path in jobs], ignore_index=True
    )


def write_archived_fits(build: Path, out: Path) -> None:
    """Run `fit_archived` and write ARCHIVED_FITS with its provenance header."""
    marin = _commit(paths.REPO, str(Path(__file__).resolve().relative_to(paths.REPO)))
    mixsel = _commit(paths.MIXSEL, "mixture_selection.py", "pyproject.toml")
    build_commit = json.loads(paths.require(build / "MANIFEST.json").read_text())["commit"]
    parameters = fit_archived(build, out)
    header = HEADER.format(marin=marin, mixsel=mixsel, build=build_commit)
    ARCHIVED_FITS.write_text(header + parameters.to_csv(index=False))
    print(f"wrote {ARCHIVED_FITS} ({len(parameters)} rows)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle", help="the bundle's swarm, objective, bucket and anchor files")
    bundle.add_argument("--build", type=Path, default=paths.BUILD)
    bundle.add_argument("--out", type=Path, required=True)
    archived = commands.add_parser("archived-fits", help="refit MARINER on the archived swarms")
    archived.add_argument("--build", type=Path, default=paths.BUILD)
    archived.add_argument("--out", type=Path, default=paths.BUILD / "fits")
    args = parser.parse_args(argv)
    if args.command == "bundle":
        write(args.build, args.out)
    else:
        write_archived_fits(args.build, args.out)


if __name__ == "__main__":
    main()
