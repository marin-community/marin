# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``mariner`` command, and the self-test that re-derives the paper's fits and trained mixtures from mariner-data.

The commands that read a swarm take either a released swarm, ``--data DIR --swarm NAME``, or one's own tables,
``--runs CSV --partition CSV --tasks CSV [--target-budget TOKENS]``, in which case the runs whose ``design_kind`` is
``repeat`` anchor the floors (``mariner.data`` describes the tables). ``fit`` writes a fit as JSON
(``ObjectiveFit.to_json``), which ``predict``, ``optimize`` and ``evaluate`` read. A proposed mixture is written with
one row per bucket: the continuous optimum's weight, the blocks of the sampler's 2,048-block grid, the weight they
give, and the epochs that weight materializes in the target run. A command whose input it cannot use, such as a
missing file or a table without a ``w__``, ``pf__`` or ``bpb__`` column it needs, stops with a message on stderr and
exit status 2, as argparse does for a malformed argument, and so does a command that needs an extra not installed.
``figures`` redraws the paper's main-text figures from a snapshot as PDFs; it needs the figures extra.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mariner import olmix, optimize
from mariner.data import (
    DATASET_REPO,
    DESIGN_KIND,
    HELDOUT_PREDICTIONS_TABLE,
    HELDOUT_TABLE,
    MAIN_SWARM,
    OBJECTIVES,
    OBSERVATION_ID,
    OLMIX_CAP,
    POOL_FRACTION_PREFIX,
    PROPORTIONAL,
    REPEAT,
    SWARM_PREDICTIONS_TABLE,
    SWARMS,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    WEIGHT_PREFIX,
    anchors_from_runs,
    download,
    mixture_weights,
    objective_from_tasks,
    pool_fractions,
    read_csv,
    read_partition,
    read_table,
    release_anchors,
    release_heldout_bank,
    release_partition,
    release_runs,
    release_swarm,
    release_tasks,
    released_olmix_laws,
    released_task_predictions,
    swarm_from_runs,
    trained_counts,
)
from mariner.design import STRATIFIED, UNIMAX, proportional, removals, sobol_candidates, stratified, swarm_table, unimax
from mariner.evaluate import (
    MEASURED_COLUMN,
    PREDICTED_COLUMN,
    REPEAT_COLUMN,
    cross_validate,
    heldout_metrics,
    selection_metrics,
)
from mariner.fit import fit_objective
from mariner.model import Anchor, Objective, ObjectiveFit, Swarm, check_buckets, order_difference, shown_values

logger = logging.getLogger(__name__)

# BPB: the released fits used epoch multipliers a relative 5.0e-7 above D_tgt / P_i, which moves their
# predictions by up to 3.0e-7.
PREDICTION_TOLERANCE = 1e-6
EXACT = 0.0  # buckets in which a re-derived mixture may differ from the trained one
EPOCHS_TOLERANCE = 1e-12  # relative: one partition and target budget give the same epochs up to floating rounding
TASK_WEIGHT_TOLERANCE = 1e-12  # relative: one tasks table gives the same weights up to floating rounding
UNIMAX_MAX_EPOCHS = 1.0  # the recipe's UniMax baseline repeats no bucket
PREDICTION_PREFIX = "predicted_"
MIXTURES_TABLE = "mixtures"  # predict's --mixtures, in error messages
DIAGNOSTICS_SUFFIX = ".diagnostics.json"
LAWS_SUFFIX = ".laws.json"
CHECK_FAILED_STATUS = 1
INPUT_ERROR_STATUS = 2  # argparse's status for a malformed argument
LOG_FORMAT = "%(asctime)s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
OBJECTIVE_HELP = f"objective of the tasks table ({' or '.join(OBJECTIVES)} in mariner-data)"
MATPLOTLIB = "matplotlib"
FONTTOOLS = "fontTools"
MATPLOTLIB_MISSING = "redrawing the figures needs matplotlib, which marin-mariner's figures extra installs"


# ---------------------------------------------------------------------------------------------------------------
# Self-test: the paper's fits and trained mixtures from a snapshot
# ---------------------------------------------------------------------------------------------------------------


class OlmixLaws(StrEnum):
    """The self-test's Olmix laws: the snapshot's ``fits_olmix_task_laws``, or refitted (about 10 more minutes)."""

    RELEASED = "released"
    REFIT = "refit"


@dataclasses.dataclass(frozen=True)
class Check:
    """One self-test comparison: ``value``, a largest gap or a number of differing buckets, against ``limit``."""

    name: str
    value: float
    limit: float

    @property
    def passed(self) -> bool:
        return self.value <= self.limit


def largest_gap(predicted: np.ndarray, released: pd.DataFrame) -> float:
    return float(np.abs(predicted - released.to_numpy()).max())


def differing_buckets(counts: np.ndarray, trained: np.ndarray) -> float:
    return float(np.count_nonzero(counts != trained))


def self_test(
    root: Path, objectives: Sequence[str] = OBJECTIVES, olmix_laws: OlmixLaws = OlmixLaws.RELEASED
) -> list[Check]:
    """Re-derive the paper's MARINER fits and trained mixtures from the mariner-data snapshot at ``root``.

    For each objective, MARINER is refitted to the main swarm. Its predictions at the swarm's runs and at the held-out
    runs must match the released ones within 1e-6, and its unconstrained proposal must equal the trained MARINER
    mixture block for block. The Olmix proposal from the released or refitted laws, at the trained Olmix mixture's
    epoch cap and KL weight, must equal that mixture block for block. Raises ``ImportError`` before any fit when the
    olmix extra is not installed.
    """
    olmix_laws = OlmixLaws(olmix_laws)  # also takes the value, "released" or "refit"
    olmix.require_cvxpy()
    heldout_runs = read_table(root, HELDOUT_TABLE).set_index(OBSERVATION_ID)
    checks = []
    for name in objectives:
        objective = objective_from_tasks(release_tasks(root), name)
        tasks = list(objective.tasks)
        swarm = release_swarm(root, MAIN_SWARM, name)
        logger.info("%s: refitting MARINER on the %d runs of %s", name, len(swarm.runs), MAIN_SWARM)
        fit = fit_objective(swarm, objective, release_anchors(root, MAIN_SWARM, name), log=logger.info)
        in_sample = released_task_predictions(root, SWARM_PREDICTIONS_TABLE, name).loc[list(swarm.runs), tasks]
        heldout = released_task_predictions(root, HELDOUT_PREDICTIONS_TABLE, name)[tasks]
        runs = heldout_runs.loc[heldout.index]
        heldout_predictions = fit.predict_tasks(
            mixture_weights(runs, fit.buckets, HELDOUT_TABLE), pool_fractions(runs, fit.buckets, HELDOUT_TABLE)
        )
        checks.append(
            Check(
                f"{name} in-sample prediction gap",
                largest_gap(fit.predict_tasks(swarm.weights, swarm.pool_fractions), in_sample),
                PREDICTION_TOLERANCE,
            )
        )
        checks.append(
            Check(f"{name} held-out prediction gap", largest_gap(heldout_predictions, heldout), PREDICTION_TOLERANCE)
        )

        logger.info("%s: proposing the MARINER mixture", name)
        mariner_mixture = TRAINED_MARINER[name]
        proposal = optimize.propose(fit, swarm.weights)
        checks.append(
            Check(
                f"{name} MARINER buckets off {mariner_mixture}",
                differing_buckets(proposal.counts, trained_counts(root, mariner_mixture)),
                EXACT,
            )
        )

        logger.info("%s: proposing the Olmix mixture from the %s laws", name, olmix_laws)
        if olmix_laws == OlmixLaws.RELEASED:
            laws = released_olmix_laws(root, name)
        else:
            laws = olmix.fit_laws(swarm, objective)
        trained = TRAINED_OLMIX[name]
        olmix_proposal = olmix.propose(laws, objective, swarm.epochs_per_unit_weight, OLMIX_CAP, trained.kl)
        checks.append(
            Check(
                f"{name} Olmix ({olmix_laws} laws) buckets off {trained.mixture_id}",
                differing_buckets(olmix_proposal.counts, trained_counts(root, trained.mixture_id)),
                EXACT,
            )
        )
    return checks


# ---------------------------------------------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------------------------------------------


def check_swarm_arguments(args: argparse.Namespace) -> None:
    """Raise ``ValueError`` unless the arguments name a released swarm or one's own tables, without mixing them."""
    if args.data is not None:
        source = "--data"
        required = {"--swarm": args.swarm}
        excluded = {"--partition": args.partition, "--tasks": args.tasks, "--target-budget": args.target_budget}
    else:
        source = "--runs"
        required = {"--partition": args.partition, "--tasks": args.tasks}
        excluded = {"--swarm": args.swarm}
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"{source} needs {' and '.join(missing)}")
    unused = [flag for flag, value in excluded.items() if value is not None]
    if unused:
        raise ValueError(f"{source} does not take {', '.join(unused)}")


class SwarmTables(NamedTuple):
    """A swarm as the commands read it: its fit rows, the objective whose outcomes they hold, and its run table."""

    swarm: Swarm
    objective: Objective
    runs: pd.DataFrame


def read_swarm(args: argparse.Namespace, objective_name: str) -> SwarmTables:
    """The swarm the arguments name, with the outcomes of the objective ``objective_name``."""
    check_swarm_arguments(args)
    if args.data is not None:
        tasks, runs = release_tasks(args.data), release_runs(args.data, args.swarm)
        partition = release_partition(args.data)
    else:
        tasks, runs = read_csv(args.tasks), read_csv(args.runs)
        partition = read_partition(args.partition, args.target_budget)
    objective = objective_from_tasks(tasks, objective_name)
    return SwarmTables(swarm_from_runs(runs, partition, objective.tasks), objective, runs)


def read_anchors(args: argparse.Namespace, tables: SwarmTables) -> dict[str, Anchor]:
    """The released swarm's anchors, or the mean and SD of the ``repeat`` runs of one's own run table."""
    if args.data is not None:
        return release_anchors(args.data, args.swarm, tables.objective.name)
    repeats = tables.runs[tables.runs[DESIGN_KIND] == REPEAT]
    return anchors_from_runs(tables.swarm, repeats, tables.objective.tasks)


def read_fit(path: Path) -> ObjectiveFit:
    """The fit JSON at ``path``; raises ``ValueError`` naming the path unless the file is UTF-8 JSON of a fit.

    A fit is a JSON object that ``ObjectiveFit.from_json`` reads: it has every field, and each field holds a value of
    the type the field needs.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8 text: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a fit JSON of mariner fit: its top level is not an object")
    try:
        return ObjectiveFit.from_json(payload)
    except KeyError as error:
        raise ValueError(f"{path} is not a fit JSON of mariner fit: it has no field {error}") from error
    except (AttributeError, OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a fit JSON of mariner fit: a field's value cannot be read ({error})") from error


def check_epochs(fit: ObjectiveFit, epochs_per_unit_weight: np.ndarray, source: str) -> None:
    """Raise ``ValueError`` unless ``source``'s epochs per unit weight are the fit's within ``EPOCHS_TOLERANCE``."""
    if np.allclose(epochs_per_unit_weight, fit.epochs_per_unit_weight, rtol=EPOCHS_TOLERANCE, atol=0.0):
        return
    ratio = np.max(np.abs(epochs_per_unit_weight / fit.epochs_per_unit_weight - 1.0))
    raise ValueError(
        f"the epochs per unit weight of {source} differ from the fit's by up to a relative {ratio:.3g}: the fit read"
        " another partition or target budget"
    )


def check_tasks(fit: ObjectiveFit, objective: Objective) -> None:
    """Raise ``ValueError`` unless the fit's tasks and task weights are ``objective``'s, in order and within 1e-12."""
    tasks = tuple(task.task for task in fit.tasks)
    if tasks != objective.tasks:
        raise ValueError(
            f"the fit's tasks are not those of {objective.name} in its order: {order_difference(objective.tasks, tasks)}"
        )
    differing = ~np.isclose(fit.weights, objective.weights, rtol=TASK_WEIGHT_TOLERANCE, atol=0.0)
    if differing.any():
        named = shown_values([task for task, differs in zip(tasks, differing, strict=True) if differs])
        raise ValueError(f"the fit's weights of {named} differ from those of {objective.name}")


def check_swarm_of_fit(fit: ObjectiveFit, swarm: Swarm) -> None:
    """Raise ``ValueError`` unless ``swarm`` has the buckets, epochs per unit weight and runs, in order, of ``fit``.

    optimize scores mixtures at the fit's epochs per unit weight and draws its SLSQP starts from the swarm's rows:
    given tables at another target budget it would ignore that budget, and given another swarm's tables it would
    start from runs the fit was not fitted on.
    """
    check_buckets(fit, swarm.buckets, "the swarm")
    check_epochs(fit, swarm.epochs_per_unit_weight, "the swarm")
    if tuple(map(str, swarm.runs)) != tuple(fit.inner_fold_labels):
        raise ValueError("the swarm's runs are not the runs of the fit in the fit's order; give the fit's run table")


def sibling_path(path: Path, suffix: str) -> Path:
    """The file in the directory of ``path`` named by its stem and ``suffix``: mixture.csv -> mixture<suffix>."""
    return path.with_name(f"{path.stem}{suffix}")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)
    logger.info("wrote %s", path)


def write_json(payload: object, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", path)


def mixture_table(
    buckets: Sequence[str], continuous: np.ndarray, counts: np.ndarray, epochs_per_unit_weight: np.ndarray
) -> pd.DataFrame:
    runtime = counts / optimize.MIXTURE_BLOCK_SIZE
    return pd.DataFrame(
        {
            "bucket": list(buckets),
            "continuous_weight": continuous,
            "runtime_count": counts,
            "runtime_weight": runtime,
            "materialized_epochs": runtime * epochs_per_unit_weight,
        }
    )


# ---------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------


def run_download(args: argparse.Namespace) -> int:
    print(download(args.out, args.repo, args.revision))
    return 0


def run_fit(args: argparse.Namespace) -> int:
    tables = read_swarm(args, args.objective)
    swarm, objective = tables.swarm, tables.objective
    logger.info("fitting the %d tasks of %s on %d runs", len(objective.tasks), objective.name, len(swarm.runs))
    fit = fit_objective(swarm, objective, read_anchors(args, tables), log=logger.info)
    write_json(fit.to_json(), args.out)
    return 0


def run_predict(args: argparse.Namespace) -> int:
    fit = read_fit(args.fit)
    mixtures = read_csv(args.mixtures)
    weights = mixture_weights(mixtures, fit.buckets, MIXTURES_TABLE)
    read = [f"{WEIGHT_PREFIX}{bucket}" for bucket in fit.buckets]
    fraction_columns = [f"{POOL_FRACTION_PREFIX}{bucket}" for bucket in fit.buckets]
    fractions = None  # whole pools, unless the file has a pf__ column, which then needs every bucket's
    if mixtures.columns.str.startswith(POOL_FRACTION_PREFIX).any():
        fractions = pool_fractions(mixtures, fit.buckets, MIXTURES_TABLE)
        read += fraction_columns
    predicted = {f"{PREDICTION_PREFIX}{fit.objective}": fit.predict(weights, fractions)}
    write_csv(mixtures.drop(columns=read).assign(**predicted), args.out)
    return 0


def run_optimize(args: argparse.Namespace) -> int:
    fit = read_fit(args.fit)
    swarm = read_swarm(args, fit.objective).swarm
    check_swarm_of_fit(fit, swarm)
    proposal = optimize.propose(fit, swarm.weights, args.cap, args.kl)
    write_csv(mixture_table(fit.buckets, proposal.continuous, proposal.counts, fit.epochs_per_unit_weight), args.out)
    diagnostics = {"summary": proposal.summary.to_json(), "starts": [record.to_json() for record in proposal.starts]}
    write_json(diagnostics, sibling_path(args.out, DIAGNOSTICS_SUFFIX))
    return 0


def run_cross_validate(args: argparse.Namespace) -> int:
    tables = read_swarm(args, args.objective)
    anchors = read_anchors(args, tables)
    table = cross_validate(tables.swarm, tables.objective, anchors, args.repeats, log=logger.info)
    write_csv(table, args.out)
    for repeat, rows in table.groupby(REPEAT_COLUMN):
        metrics = selection_metrics(rows[PREDICTED_COLUMN].to_numpy(), rows[MEASURED_COLUMN].to_numpy())
        print(json.dumps({REPEAT_COLUMN: int(repeat), **dataclasses.asdict(metrics)}))
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    fit = read_fit(args.fit)
    # The held-out runs share the main swarm's model and data scale, so they test only a fit of that swarm's runs, in
    # any order, at its epochs per unit weight, for a released objective.
    swarm = release_swarm(args.data, MAIN_SWARM, fit.objective)
    if set(fit.inner_fold_labels) != set(map(str, swarm.runs)):
        raise ValueError(f"evaluate scores a fit of the main swarm ({MAIN_SWARM}), and {args.fit} fits other runs")
    source = f"the main swarm ({MAIN_SWARM})"
    check_buckets(fit, swarm.buckets, source)
    check_epochs(fit, swarm.epochs_per_unit_weight, source)
    check_tasks(fit, objective_from_tasks(release_tasks(args.data), fit.objective))
    metrics = heldout_metrics(fit, release_heldout_bank(args.data, fit.objective))
    print(json.dumps(dataclasses.asdict(metrics), indent=2))
    return 0


def run_olmix(args: argparse.Namespace) -> int:
    olmix.require_cvxpy()
    tables = read_swarm(args, args.objective)
    swarm, objective = tables.swarm, tables.objective
    # Before the laws, whose fit takes minutes on a released swarm.
    optimize.check_cap_and_kl(swarm.epochs_per_unit_weight, args.cap, args.kl, olmix.CAP_ROUNDING_SLACK)
    logger.info(
        "fitting the Olmix laws of the %d tasks of %s on %d runs", len(objective.tasks), objective.name, len(swarm.runs)
    )
    laws = olmix.fit_laws(swarm, objective)
    proposal = olmix.propose(laws, objective, swarm.epochs_per_unit_weight, args.cap, args.kl)
    logger.info("Olmix proposal: %s, predicted %.6f", proposal.solver, proposal.predicted)
    write_csv(mixture_table(swarm.buckets, proposal.continuous, proposal.counts, swarm.epochs_per_unit_weight), args.out)
    payload = {
        "objective": objective.name,
        "buckets": list(swarm.buckets),
        "tasks": [
            {"task": task, "log_c": law.log_c, "coefficients": law.coefficients.tolist(), "huber_loss": law.huber_loss}
            for task, law in zip(objective.tasks, laws, strict=True)
        ],
    }
    write_json(payload, sibling_path(args.out, LAWS_SUFFIX))
    return 0


def run_design(args: argparse.Namespace) -> int:
    partition = read_partition(args.partition, args.target_budget)
    base = proportional(partition.pool_tokens)
    budget = args.unimax_budget_fraction * args.target_budget
    baselines = {
        PROPORTIONAL: base,
        UNIMAX: unimax(partition.pool_tokens, budget, UNIMAX_MAX_EPOCHS),
        STRATIFIED: stratified(len(partition.bucket_ids)),
    }
    # A candidate within the least distance of a baseline or a removal would train a near-duplicate of that run.
    fixed = np.vstack([*baselines.values(), removals(base)])
    candidates = sobol_candidates(args.candidates, len(partition.bucket_ids), args.seed, existing=fixed)
    table = swarm_table(
        partition.bucket_ids,
        candidates,
        baselines,
        removal_base=base,
        repeats=args.repeats,
        data_seed_start=args.seed,
    )
    write_csv(table, args.out)
    return 0


def run_figures(args: argparse.Namespace) -> int:
    # mariner.figures imports matplotlib, an optional dependency; find it without importing it, as olmix finds cvxpy.
    if importlib.util.find_spec(MATPLOTLIB) is None:
        raise ImportError(MATPLOTLIB_MISSING)
    from mariner.figures import write_figures  # noqa: PLC0415  (optional dependency: the figures extra)

    logging.getLogger(FONTTOOLS).setLevel(logging.WARNING)  # it logs each font it subsets into a PDF at INFO
    for path in write_figures(args.data, args.out, args.only):
        print(path)
    return 0


def run_self_test(args: argparse.Namespace) -> int:
    checks = self_test(args.data, olmix_laws=OlmixLaws.REFIT if args.refit_olmix else OlmixLaws.RELEASED)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.value:.3g} (limit {check.limit:g})")
    return 0 if all(check.passed for check in checks) else CHECK_FAILED_STATUS


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def add_swarm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "swarm", "a released swarm (--data, --swarm) or one's own tables (--runs, --partition, --tasks)"
    )
    source = group.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, metavar="DIR", help="mariner-data snapshot directory")
    source.add_argument("--runs", type=Path, metavar="CSV", help="run table of one's own swarm")
    group.add_argument(
        "--swarm", choices=SWARMS, metavar="NAME", help=f"released swarm, with --data: {', '.join(SWARMS)}"
    )
    group.add_argument("--partition", type=Path, metavar="CSV", help="partition table, with --runs")
    group.add_argument("--tasks", type=Path, metavar="CSV", help="tasks table, with --runs")
    group.add_argument(
        "--target-budget",
        type=float,
        metavar="TOKENS",
        help="the target run's tokens, with --runs: epochs per unit weight are its ratio to each pool_tokens"
        " (default: the partition's epochs_per_unit_weight)",
    )


def argument_parser() -> argparse.ArgumentParser:
    """The ``mariner`` command's parser: one subcommand per step, each bound to its handler as ``run``."""
    parser = argparse.ArgumentParser(
        prog="mariner", description="Fit MARINER to a swarm of proxy runs and propose a pretraining data mixture."
    )
    parser.set_defaults(out=None)  # evaluate and self-test write no file
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    command = commands.add_parser("download", help="download the mariner-data tables this package reads")
    command.add_argument("--out", type=Path, required=True, metavar="DIR", help="directory to download into")
    command.add_argument(
        "--repo", default=DATASET_REPO, metavar="ID", help=f"dataset repository (default {DATASET_REPO})"
    )
    command.add_argument("--revision", metavar="REV", help="dataset revision (default: the latest)")
    command.set_defaults(run=run_download)

    command = commands.add_parser("fit", help="fit MARINER to a swarm and write the fit as JSON")
    add_swarm_arguments(command)
    command.add_argument("--objective", required=True, metavar="NAME", help=OBJECTIVE_HELP)
    command.add_argument("--out", type=Path, required=True, metavar="FIT.json", help="fit JSON to write")
    command.set_defaults(run=run_fit)

    command = commands.add_parser("predict", help="predict a fit's objective at mixtures")
    command.add_argument("--fit", type=Path, required=True, metavar="FIT.json", help="fit JSON")
    command.add_argument(
        "--mixtures",
        type=Path,
        required=True,
        metavar="CSV",
        help="mixtures (CSV): w__<bucket_id> for each bucket of the fit, and pool fractions pf__<bucket_id>"
        " (default: whole pools)",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="CSV",
        help="CSV to write: the mixtures' other columns and predicted_<objective>",
    )
    command.set_defaults(run=run_predict)

    command = commands.add_parser("optimize", help="propose the mixture that minimizes a fit's objective")
    command.add_argument("--fit", type=Path, required=True, metavar="FIT.json", help="fit JSON")
    add_swarm_arguments(command)
    command.add_argument(
        "--cap", type=float, metavar="EPOCHS", help="most epochs of any bucket (default: none, as in the paper)"
    )
    command.add_argument(
        "--kl",
        type=float,
        metavar="WEIGHT",
        default=0.0,
        help="weight of a KL penalty toward the proportional mixture (default 0, as in the paper)",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="CSV",
        help=f"mixture CSV to write; the optimizer's diagnostics go beside it, as <name>{DIAGNOSTICS_SUFFIX}",
    )
    command.set_defaults(run=run_optimize)

    command = commands.add_parser("cross-validate", help="predict a swarm's runs out of fold on the paper's folds")
    add_swarm_arguments(command)
    command.add_argument("--objective", required=True, metavar="NAME", help=OBJECTIVE_HELP)
    command.add_argument(
        "--repeats", type=int, default=1, metavar="N", help="repeats of the five outer folds (default 1)"
    )
    command.add_argument("--out", type=Path, required=True, metavar="CSV", help="out-of-fold predictions to write")
    command.set_defaults(run=run_cross_validate)

    command = commands.add_parser(
        "evaluate", help=f"score a fit of the main swarm ({MAIN_SWARM}) on the paper's held-out runs, as JSON"
    )
    command.add_argument(
        "--fit",
        type=Path,
        required=True,
        metavar="FIT.json",
        help=f"fit JSON: a fit of the main swarm ({MAIN_SWARM}) at its target budget, for a released objective",
    )
    command.add_argument("--data", type=Path, required=True, metavar="DIR", help="mariner-data snapshot directory")
    command.set_defaults(run=run_evaluate)

    command = commands.add_parser("olmix", help="fit the Olmix baseline to a swarm and propose its mixture")
    add_swarm_arguments(command)
    command.add_argument("--objective", required=True, metavar="NAME", help=OBJECTIVE_HELP)
    command.add_argument(
        "--kl",
        type=float,
        metavar="WEIGHT",
        required=True,
        help="weight of the KL penalty toward the proportional mixture (the paper: 0.05 for uncheatable, 0.005 for"
        " obe_easy)",
    )
    command.add_argument(
        "--cap",
        type=float,
        required=True,
        metavar="EPOCHS",
        help="most epochs of any bucket, inf for none (the paper: 4)",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="CSV",
        help=f"mixture CSV to write; the fitted laws go beside it, as <name>{LAWS_SUFFIX}",
    )
    command.set_defaults(run=run_olmix)

    command = commands.add_parser("design", help="write the run table of a new swarm")
    command.add_argument(
        "--partition", type=Path, required=True, metavar="CSV", help="partition table: bucket_id, pool_tokens"
    )
    command.add_argument("--target-budget", type=float, required=True, metavar="TOKENS", help="the target run's tokens")
    command.add_argument(
        "--candidates", type=int, required=True, metavar="N", help="number of Sobol' candidate mixtures"
    )
    command.add_argument(
        "--seed", type=int, required=True, help="seed of the candidates; the runs' data seeds count up from it"
    )
    command.add_argument(
        "--repeats",
        type=int,
        required=True,
        metavar="N",
        help="runs of the proportional mixture that anchor the floors (2 or more)",
    )
    command.add_argument(
        "--unimax-budget-fraction",
        type=float,
        metavar="FRACTION",
        required=True,
        help="share of the target budget that the UniMax baseline spreads at one epoch at most (the paper: 0.8)",
    )
    command.add_argument("--out", type=Path, required=True, metavar="CSV", help="run table to write")
    command.set_defaults(run=run_design)

    command = commands.add_parser("figures", help="redraw the paper's main-text figures from a snapshot as PDFs")
    command.add_argument("--data", type=Path, required=True, metavar="DIR", help="mariner-data snapshot directory")
    command.add_argument("--out", type=Path, required=True, metavar="DIR", help="directory to write <stem>.pdf into")
    command.add_argument(
        "--only",
        nargs="+",
        metavar="LABEL",
        help="the paper's labels of the figures to draw, such as fig:b5 (default: all)",
    )
    command.set_defaults(run=run_figures)

    command = commands.add_parser("self-test", help="re-derive the paper's fits and trained mixtures from a snapshot")
    command.add_argument("--data", type=Path, required=True, metavar="DIR", help="mariner-data snapshot directory")
    command.add_argument(
        "--refit-olmix",
        action="store_true",
        help="refit the Olmix laws instead of reading the released ones (about 10 more minutes)",
    )
    command.set_defaults(run=run_self_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``mariner`` command with ``argv`` (default: the process's arguments) and return its exit status.

    A malformed argument exits through argparse with status 2. The directory of ``--out`` is created before the
    command runs, and the command runs with one thread in each BLAS and OpenMP pool. An input the command cannot use,
    raised as a ``ValueError`` (such as a table without a bucket's ``w__`` column) or an ``OSError`` (such as a missing
    file), or an extra it needs and lacks, raised as an ``ImportError`` that names the extra, stops it with a message
    on stderr and status 2. Other exceptions, a solver's ``RuntimeError`` among them, propagate. ``self-test`` returns
    1 when a check fails.
    """
    args = argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    try:
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)  # before the compute, which a bad path would waste
        # NumPy, SciPy and scikit-learn each start a pool of one thread per core, and a fit is many small NNLS and
        # least-squares solves, where those threads only contend: on 16 Linux cores a fit took over ten times longer.
        with threadpool_limits(limits=1):
            return args.run(args)
    except (ImportError, OSError, ValueError) as error:
        print(f"mariner {args.command}: error: {error}", file=sys.stderr)
        return INPUT_ERROR_STATUS
