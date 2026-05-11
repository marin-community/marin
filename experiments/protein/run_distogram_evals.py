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
    """One protein-docs training run + a checkpoint to evaluate.

    The checkpoint can be either an HF dir (existing post-export workflow) or
    a Levanter checkpoint dir (the eval container converts it to HF in-place
    on CPU before vLLM loads). Set exactly one of ``hf_checkpoint_path`` or
    (``levanter_checkpoint_path`` + ``levanter_model_spec``).
    """

    model_label: str
    hf_checkpoint_path: str | None = None
    levanter_checkpoint_path: str | None = None
    # Importable ``module.attribute`` for an LmConfig — required when using
    # the Levanter path since Levanter checkpoints don't carry an architecture
    # spec. Example: "experiments.protein.train_protein_30m_distance_masked.protein_llama_30m".
    levanter_model_spec: str | None = None

    def __post_init__(self):
        has_hf = self.hf_checkpoint_path is not None
        has_levanter = self.levanter_checkpoint_path is not None
        if has_hf == has_levanter:
            raise ValueError(
                "EvalRunEntry must specify exactly one of hf_checkpoint_path or " "levanter_checkpoint_path."
            )
        if has_levanter and self.levanter_model_spec is None:
            raise ValueError("levanter_checkpoint_path requires levanter_model_spec.")

    @property
    def model_path(self) -> str:
        """The path passed to ``--model``."""
        return self.hf_checkpoint_path or self.levanter_checkpoint_path  # pyrefly: ignore


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

# Add an entry per training run. Each entry can specify either an HF checkpoint
# directory (legacy snapshot-1 path) or a Levanter checkpoint directory (the
# eval container converts it to HF in-place on CPU before vLLM loads).
# ``model_label`` includes the rough training step at the time of submission;
# the actual checkpoint loaded is the latest one found by
# ``discover_latest_checkpoint`` when the eval container starts.
#
# Snapshot 8 (2026-05-11 ~01:10 UTC): 1b_unmasked and 420m_deep finished.
# Updated still-training models to latest checkpoints. 1b_all_docs not yet
# included — only has eval@1000, too early.
EVAL_RUNS: list[EvalRunEntry] = [
    EvalRunEntry(
        # Finished at step-50000. Unchanged.
        model_label="30m-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-30m-distance-masked-a7457a/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_30m_distance_masked.protein_llama_30m",
    ),
    EvalRunEntry(
        # Finished at step-50000. Unchanged.
        model_label="100m-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-100m-distance-masked-917586/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_100m_distance_masked.protein_llama_100m",
    ),
    EvalRunEntry(
        # Finished at step-49999. Final eval loss=1.0455, bpb=0.1115.
        model_label="400m-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-400m-distance-masked-0de2c1/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_400m_distance_masked.protein_llama_400m",
    ),
    EvalRunEntry(
        # Finished at step-49999. Final eval loss=1.224, bpb=0.131.
        # Diverged briefly at step 32500 then recovered.
        model_label="420m_deep-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-420m-deep-distance-masked-81e865/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_420m_deep_distance_masked.protein_llama_420m_deep",
    ),
    EvalRunEntry(
        # Latest eval@28500 loss=1.065.
        model_label="1_5b-step-29000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_1_5b_distance_masked.protein_llama_1_5b",
    ),
    EvalRunEntry(
        # Latest eval@13000 loss=1.226 — slow but progressing.
        model_label="3b-step-13000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-3b-distance-masked-ef3aa5/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_3b_distance_masked.protein_llama_3b",
    ),
    EvalRunEntry(
        # 1B distance-masked finished at step-49999. Final eval loss=1.029.
        model_label="1b-step-49999",
        hf_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e/hf/step-49999",
    ),
    EvalRunEntry(
        # Finished at step-49999. Final eval loss=2.524.
        model_label="1b_unmasked-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-unmasked-8efbcb/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_1b_unmasked.protein_llama_1b",
    ),
    EvalRunEntry(
        # Run finished at step-50000. Unchanged.
        model_label="100m_unmasked-step-50000",
        levanter_checkpoint_path="gs://marin-us-east5/checkpoints/protein-contacts-100m-3.5e-4-unmasked-7c3ef7/checkpoints",
        levanter_model_spec="experiments.protein.train_protein_100m_unmasked.protein_llama_100m",
    ),
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
        run.model_path,
        "--pdb-id",
        target.pdb_id,
        "--prompt-contact-counts",
        *(str(n) for n in PROMPT_CONTACT_COUNTS),
        "--output-dir",
        output_dir,
    ]
    if run.levanter_model_spec is not None:
        inner.extend(["--levanter-model-spec", run.levanter_model_spec])
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
        "--enable-extra-resources",
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
