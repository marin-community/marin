# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate perplexity ↔ downstream-accuracy datapoints across protein-docs runs.

The protein-docs distance-masked size sweep (see
``experiments/protein/train_protein_*_distance_masked.py``) produces:

* In-training perplexity (``eval/protein-docs-cd-val/loss``) at every
  ``steps_per_eval`` step, logged to W&B.
* Offline downstream metrics from
  ``experiments/protein/eval_protein_distogram.py`` (per-(target, N) MAE) and
  ``experiments/protein/eval_protein_contacts.py`` (per-contact-type F1),
  written as ``summary.json`` to gs://.

This script joins those two sources by ``(model_name, checkpoint_step)`` and
writes two CSVs to ``OUTPUT_DIR``:

* ``perplexity.csv`` — one row per ``(model, checkpoint_step)``.
* ``accuracy.csv`` — one row per ``(model, checkpoint_step, target, eval_kind)``.

The plotting helper ``plot_perplexity_vs_accuracy.py`` reads both CSVs and
joins them in pandas to produce the perplexity-vs-MAE / perplexity-vs-F1
scatter plots.

Add new training runs to the ``RUNS`` registry below as they complete and
re-run this script to refresh the CSVs.

Usage::

    WANDB_API_KEY=... uv run python -m experiments.protein.aggregate_perplexity_vs_accuracy
"""

import argparse
import csv
import io
import json
import logging
import sys
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)


OUTPUT_DIR = "gs://marin-us-east5/eval/protein-perplexity-vs-accuracy/v1"
WANDB_ENTITY = "timodonnell"
WANDB_PROJECT = "marin"
PERPLEXITY_METRIC = "eval/protein-docs-cd-val/loss"


@dataclass(frozen=True)
class RunEntry:
    """One protein-docs training run + its evaluation artifacts.

    Attributes:
        model_name: Short label (used in CSVs and plots), e.g. ``"30m"``.
        model_size_params: Approximate parameter count (informational, used for
            coloring/sizing in plots).
        wandb_run_name: W&B run name. Defaults to the run name marin generates
            from ``default_train(name=...)``; override if you renamed it.
        distogram_eval_dirs: List of gs:// dirs that each contain a
            ``summary.json`` from ``eval_protein_distogram.py``. Each dir
            corresponds to a (PDB target, sequence-variant) eval.
        contacts_eval_dirs: List of gs:// dirs that each contain a
            ``summary.json`` from ``eval_protein_contacts.py``.
    """

    model_name: str
    model_size_params: int
    wandb_run_name: str
    distogram_eval_dirs: tuple[str, ...] = ()
    contacts_eval_dirs: tuple[str, ...] = ()


# Registry of protein-docs training runs to aggregate. Add an entry per run as
# training completes and offline evals are produced. The ``1b`` entry is the
# only currently completed run (the continuation file's 7d355e bucket).
RUNS: list[RunEntry] = [
    RunEntry(
        model_name="1b",
        model_size_params=985_000_000,
        # Run name as W&B records it. The marin step name is
        # ``protein-contacts-1b-3.5e-4-distance-masked``; W&B appends the
        # version hash → ``...-7d355e``. Update if your run uses a different
        # name (e.g. when default_train auto-suffixes).
        wandb_run_name="protein-contacts-1b-3.5e-4-distance-masked-7d355e",
        distogram_eval_dirs=(),
        contacts_eval_dirs=(),
    ),
    # TODO: append entries for 30m, 100m, 400m, 420m_deep, 1_5b, 3b runs as
    # they finish training and have offline evals.
]


# ---- Perplexity (from W&B) ----


@dataclass(frozen=True)
class PerplexityRow:
    model_name: str
    model_size_params: int
    checkpoint_step: int
    val_loss: float
    val_perplexity: float


def fetch_perplexity(entry: RunEntry) -> list[PerplexityRow]:
    """Pull ``eval/protein-docs-cd-val/loss`` from W&B for every logged step."""
    import math

    import wandb

    api = wandb.Api()
    run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{entry.wandb_run_name}"
    logger.info("Fetching W&B run %s", run_path)
    run = api.run(run_path)

    rows: list[PerplexityRow] = []
    # ``run.scan_history`` streams every logged step; ``samples=1e6`` is a no-op
    # ceiling that prevents API truncation.
    for record in run.scan_history(keys=["_step", PERPLEXITY_METRIC], page_size=10000):
        step = record.get("_step")
        loss = record.get(PERPLEXITY_METRIC)
        if step is None or loss is None:
            continue
        rows.append(
            PerplexityRow(
                model_name=entry.model_name,
                model_size_params=entry.model_size_params,
                checkpoint_step=int(step),
                val_loss=float(loss),
                val_perplexity=float(math.exp(loss)),
            )
        )
    logger.info("  %d perplexity rows", len(rows))
    return rows


# ---- Accuracy (from GCS summary.json) ----


@dataclass(frozen=True)
class AccuracyRow:
    model_name: str
    model_size_params: int
    checkpoint_step: int
    eval_kind: str  # "distogram" | "contacts"
    target: str
    n_prompt_contacts: int | None  # None for contacts eval
    distogram_expected_mae_a: float | None = None
    distogram_argmax_mae_a: float | None = None
    distogram_contact_corr: float | None = None
    contact_long_f1: float | None = None
    contact_med_f1: float | None = None
    contact_short_f1: float | None = None
    source_summary_path: str = ""


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_distogram_summary(entry: RunEntry, summary_path: str, summary: dict) -> list[AccuracyRow]:
    """One row per (target, N) from a single ``eval_protein_distogram`` run."""
    target = (summary.get("pdb_id") or "?").lower()
    # ``checkpoint_step`` isn't always written into summary.json; let the user
    # encode it in the eval directory name and parse here if needed. For now,
    # treat as -1 ("unknown step at eval time").
    checkpoint_step = int(summary.get("inference", {}).get("checkpoint_step", -1))
    rows: list[AccuracyRow] = []
    for per_n in summary.get("per_n", []):
        n = int(per_n["n_prompt_contacts"])
        m = per_n["metrics"]
        rows.append(
            AccuracyRow(
                model_name=entry.model_name,
                model_size_params=entry.model_size_params,
                checkpoint_step=checkpoint_step,
                eval_kind="distogram",
                target=target,
                n_prompt_contacts=n,
                distogram_expected_mae_a=float(m.get("expected_mean_abs_err_A")),
                distogram_argmax_mae_a=float(m.get("argmax_mean_abs_err_A")),
                distogram_contact_corr=float(m.get("contact_prob_auc_proxy_corr")),
                source_summary_path=summary_path,
            )
        )
    return rows


def parse_contacts_summary(entry: RunEntry, summary_path: str, summary: dict) -> AccuracyRow:
    """One row per contacts-eval (with all three contact types as F1 columns)."""
    target = (summary.get("pdb_id") or "?").lower()
    checkpoint_step = int(summary.get("inference", {}).get("checkpoint_step", -1))
    per_type = summary.get("per_type", {})

    def _consensus_f1(type_tok: str) -> float | None:
        c = per_type.get(type_tok, {}).get("consensus") or {}
        if "precision" not in c or "recall" not in c:
            return None
        return _f1(float(c["precision"]), float(c["recall"]))

    return AccuracyRow(
        model_name=entry.model_name,
        model_size_params=entry.model_size_params,
        checkpoint_step=checkpoint_step,
        eval_kind="contacts",
        target=target,
        n_prompt_contacts=None,
        contact_long_f1=_consensus_f1("<long-range-contact>"),
        contact_med_f1=_consensus_f1("<medium-range-contact>"),
        contact_short_f1=_consensus_f1("<short-range-contact>"),
        source_summary_path=summary_path,
    )


def fetch_accuracy(entry: RunEntry) -> list[AccuracyRow]:
    rows: list[AccuracyRow] = []
    for d in entry.distogram_eval_dirs:
        path = f"{d.rstrip('/')}/summary.json"
        try:
            summary = _read_json(path)
        except FileNotFoundError:
            logger.warning("Missing distogram summary at %s; skipping.", path)
            continue
        rows.extend(parse_distogram_summary(entry, path, summary))
    for d in entry.contacts_eval_dirs:
        path = f"{d.rstrip('/')}/summary.json"
        try:
            summary = _read_json(path)
        except FileNotFoundError:
            logger.warning("Missing contacts summary at %s; skipping.", path)
            continue
        rows.append(parse_contacts_summary(entry, path, summary))
    logger.info("  %d accuracy rows for %s", len(rows), entry.model_name)
    return rows


# ---- CSV output ----


def write_csv(path: str, rows: list, fieldnames: list[str]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        # Dataclass → dict; restrict to declared fieldnames.
        d = {k: getattr(row, k) for k in fieldnames}
        writer.writerow(d)
    with fsspec.open(path, "w") as f:
        f.write(buf.getvalue())
    logger.info("Wrote %d rows to %s", len(rows), path)


PERPLEXITY_FIELDS = [
    "model_name",
    "model_size_params",
    "checkpoint_step",
    "val_loss",
    "val_perplexity",
]

ACCURACY_FIELDS = [
    "model_name",
    "model_size_params",
    "checkpoint_step",
    "eval_kind",
    "target",
    "n_prompt_contacts",
    "distogram_expected_mae_a",
    "distogram_argmax_mae_a",
    "distogram_contact_corr",
    "contact_long_f1",
    "contact_med_f1",
    "contact_short_f1",
    "source_summary_path",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Skip W&B fetch (useful for offline iteration on accuracy CSV).",
    )
    args = parser.parse_args(argv)

    perplexity_rows: list[PerplexityRow] = []
    accuracy_rows: list[AccuracyRow] = []

    for entry in RUNS:
        logger.info("Processing run %s (%s params)", entry.model_name, f"{entry.model_size_params:,}")
        if not args.skip_wandb:
            try:
                perplexity_rows.extend(fetch_perplexity(entry))
            except Exception as exc:
                logger.warning("W&B fetch failed for %s: %s", entry.model_name, exc)
        accuracy_rows.extend(fetch_accuracy(entry))

    out = args.output_dir.rstrip("/")
    write_csv(f"{out}/perplexity.csv", perplexity_rows, PERPLEXITY_FIELDS)
    write_csv(f"{out}/accuracy.csv", accuracy_rows, ACCURACY_FIELDS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
