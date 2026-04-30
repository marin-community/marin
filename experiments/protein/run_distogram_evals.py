# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a grid of distogram evals across all protein-docs runs.

The grid is ``models x targets x variants``:

* **models** — one entry per protein-docs training run (registered in
  ``EVAL_RUNS`` below). Each entry points at an HF checkpoint (gs:// path
  produced by ``export_protein_<size>_distance_masked.py``).
* **targets** — the three benchmark PDBs at the top of
  ``protein_distogram_eval.TARGETS``: 1QYS (top7), 7BNY, 1UBQ (ubiquitin).
* **variants** — native sequence + two SolubleMPNN redesigns (idx 0 and 1).
  Redesigns come from the ``soluble-v2`` JSONL in GCS.

Each cell is submitted as one iris vLLM job. Output layout:

    gs://marin-us-east5/eval/protein-distogram/v1/
        <model_label>/
            <target_label>/
                <variant_label>/
                    summary.json
                    distogram_n{0..5}.npz

``plot_combined_distogram_report.py`` reads this layout to produce the
cross-run report.

Usage::

    # Print commands without submitting
    uv run python -m experiments.protein.run_distogram_evals --dry-run

    # Submit only 30m + 1QYS native (handy for sanity-checking the pipeline
    # before launching all 63 cells)
    uv run python -m experiments.protein.run_distogram_evals \\
        --filter-model 30m --filter-target top7 --filter-variant native

    # Full grid
    uv run python -m experiments.protein.run_distogram_evals
"""

import argparse
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


OUTPUT_PREFIX = "gs://marin-us-east5/eval/protein-distogram/v1"
SOLUBLE_REDESIGNS_SOURCE = "gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/soluble-v2/redesigns.jsonl"
IRIS_CONFIG = "lib/iris/examples/marin.yaml"
EVAL_ZONE = "us-east5-a"
PROMPT_CONTACT_COUNTS = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class EvalRunEntry:
    """One protein-docs training run + its HF checkpoint."""

    model_label: str
    hf_checkpoint_path: str


@dataclass(frozen=True)
class EvalTarget:
    pdb_id: str
    label: str
    chain_id: str | None = None


@dataclass(frozen=True)
class EvalVariant:
    """Either native sequence (no override) or a specific redesign."""

    label: str
    redesigns_source: str | None = None
    method: str | None = None
    redesign_idx: int | None = None


# ---- Configuration ----

# Add an entry per training run as its HF export completes. ``hf_checkpoint_path``
# is the directory written by ``export_protein_<size>_distance_masked.py``
# (i.e. the marin auto-computed `hf/...-step-50000-<hash>` path), or any other
# HF checkpoint (e.g. an intermediate-step export).
#
# The existing 1B run's HF checkpoint already exists at the path below; the
# rest are placeholders to be filled in once exports finish.
EVAL_RUNS: list[EvalRunEntry] = [
    EvalRunEntry(
        model_label="1b_continue_train",
        hf_checkpoint_path=(
            "gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-31337"
        ),
    ),
    # TODO: fill in once exports run. Path layout for fresh runs is
    # ``gs://marin-us-east5/hf/protein-contacts-<size>-distance-masked-step-<N>-<hash>``.
    # EvalRunEntry(model_label="30m", hf_checkpoint_path="gs://..."),
    # EvalRunEntry(model_label="100m", hf_checkpoint_path="gs://..."),
    # EvalRunEntry(model_label="400m", hf_checkpoint_path="gs://..."),
    # EvalRunEntry(model_label="420m_deep", hf_checkpoint_path="gs://..."),
    # EvalRunEntry(model_label="1_5b", hf_checkpoint_path="gs://..."),
    # EvalRunEntry(model_label="3b", hf_checkpoint_path="gs://..."),
]

EVAL_TARGETS: list[EvalTarget] = [
    EvalTarget(pdb_id="1QYS", label="top7"),
    EvalTarget(pdb_id="7BNY", label="7bny"),
    EvalTarget(pdb_id="1UBQ", label="ubiquitin"),
]

EVAL_VARIANTS: list[EvalVariant] = [
    EvalVariant(label="native"),
    EvalVariant(
        label="soluble-0",
        redesigns_source=SOLUBLE_REDESIGNS_SOURCE,
        method="soluble",
        redesign_idx=0,
    ),
    EvalVariant(
        label="soluble-1",
        redesigns_source=SOLUBLE_REDESIGNS_SOURCE,
        method="soluble",
        redesign_idx=1,
    ),
]


# ---- Submission ----


def cell_output_dir(run: EvalRunEntry, target: EvalTarget, variant: EvalVariant) -> str:
    return f"{OUTPUT_PREFIX}/{run.model_label}/{target.label}/{variant.label}"


def build_iris_command(run: EvalRunEntry, target: EvalTarget, variant: EvalVariant) -> list[str]:
    """The full ``iris job run ... -- python -m eval_protein_distogram ...`` argv."""
    output_dir = cell_output_dir(run, target, variant)
    inner = [
        "python",
        "-m",
        "experiments.protein.eval_protein_distogram",
        "--model",
        run.hf_checkpoint_path,
        "--pdb-id",
        target.pdb_id,
        "--prompt-contact-counts",
        *(str(n) for n in PROMPT_CONTACT_COUNTS),
        "--output-dir",
        output_dir,
    ]
    if target.chain_id is not None:
        inner.extend(["--chain-id", target.chain_id])
    if variant.redesigns_source is not None:
        assert variant.method is not None and variant.redesign_idx is not None
        inner.extend(
            [
                "--sequence-override-source",
                variant.redesigns_source,
                "--sequence-override-target-label",
                target.label,
                "--sequence-override-method",
                variant.method,
                "--sequence-override-idx",
                str(variant.redesign_idx),
            ]
        )

    return [
        "uv",
        "run",
        "iris",
        f"--config={IRIS_CONFIG}",
        "job",
        "run",
        "--tpu=v5p-8",
        "--memory=64GB",
        "--disk=64GB",
        "--cpu=16",
        "--extra=vllm",
        "--extra=tpu",
        f"--zone={EVAL_ZONE}",
        "--no-wait",
        "--",
        *inner,
    ]


def cells_iter(
    runs: list[EvalRunEntry],
    targets: list[EvalTarget],
    variants: list[EvalVariant],
    *,
    filter_model: list[str] | None,
    filter_target: list[str] | None,
    filter_variant: list[str] | None,
):
    for run in runs:
        if filter_model and run.model_label not in filter_model:
            continue
        for target in targets:
            if filter_target and target.label not in filter_target:
                continue
            for variant in variants:
                if filter_variant and variant.label not in filter_variant:
                    continue
                yield (run, target, variant)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print iris commands without submitting.",
    )
    parser.add_argument(
        "--filter-model",
        action="append",
        help="Only submit for model labels in this list (repeatable). Default: all in EVAL_RUNS.",
    )
    parser.add_argument(
        "--filter-target",
        action="append",
        help="Only submit for target labels in this list (repeatable). Default: all in EVAL_TARGETS.",
    )
    parser.add_argument(
        "--filter-variant",
        action="append",
        help="Only submit for variant labels in this list (repeatable). Default: all in EVAL_VARIANTS.",
    )
    args = parser.parse_args(argv)

    if not EVAL_RUNS:
        parser.error("EVAL_RUNS registry is empty; add an entry before submitting.")

    if "WANDB_API_KEY" not in os.environ and not args.dry_run:
        logger.warning(
            "WANDB_API_KEY not set in env; iris submission may fail at marin's "
            "pre-flight check. Run: "
            "export WANDB_API_KEY=$(awk '/^machine api.wandb.ai/{flag=1; next} "
            "flag && /password/{print $2; exit}' ~/.netrc)"
        )

    cells = list(
        cells_iter(
            EVAL_RUNS,
            EVAL_TARGETS,
            EVAL_VARIANTS,
            filter_model=args.filter_model,
            filter_target=args.filter_target,
            filter_variant=args.filter_variant,
        )
    )
    if not cells:
        logger.warning("No cells matched the filters; nothing to do.")
        return 0
    logger.info("Submitting %d eval cells", len(cells))

    submitted: list[str] = []
    for run, target, variant in cells:
        cmd = build_iris_command(run, target, variant)
        cell_label = f"{run.model_label}/{target.label}/{variant.label}"
        if args.dry_run:
            logger.info("[dry-run] %s", cell_label)
            print(shlex.join(cmd))
            continue
        logger.info("Submitting %s", cell_label)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("Submit failed for %s: %s", cell_label, proc.stderr.strip())
            return proc.returncode
        # iris prints the job id on the last line of stdout.
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        job_id = stdout_lines[-1] if stdout_lines else "<unknown>"
        submitted.append(f"{cell_label} → {job_id}")
        logger.info("  job_id=%s", job_id)

    if submitted:
        logger.info("Submitted %d jobs:", len(submitted))
        for line in submitted:
            logger.info("  %s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
