# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head PDF-corpus comparison (#7621): our OCR pipeline vs token-matched FinePDFs.

Trains grug-MoE compute-optimal rungs on two English PDF corpora and compares
``eval/paloma/macro_loss`` (primary) plus per-subset losses and throughput
(secondary), following the #6570 focus-vs-main WET side-by-side:

- ``pdf`` — the eng_Latn subset of our OCR pipeline's final NormalizedData
  output (:data:`PDF_FINAL_DIR`, filled in at launch time). A small select step
  materializes the ``language == "eng_Latn"`` rows to {id, text} parquet before
  tokenizing, since FinePDFs releases per-language and a fair comparison needs
  the same language cut.
- ``finepdfs`` — a 40-shard sample (~3.2B llama3 tokens) of the already
  normalized FinePDFs eng_Latn corpus on the same bucket, sized to clear the
  d768 rung with headroom while staying about the size of our corpus.

Token matching is by construction: each rung's (batch_size, steps) come from the
shared heuristic (:func:`~experiments.grug.moe.heuristic.build_from_heuristic`),
so both arms consume identical fixed token budgets. The only requirement is
that each arm's tokenized pool exceeds the rung budget (no repeats); the train
stage enforces this against the measured cache token counts and refuses to run
a rung the pool cannot cover (set ``PDFCMP_ALLOW_REPEATS=1`` to override and
train with repeated data, noting the repeat in the writeup).

Ladder: d512 (5.25e8 tokens) is cleared with certainty; d768 (1.81e9) is gated
on the measured post-filter eng_Latn token count that STAGE=data prints. (These
budgets come from the current May-recipe heuristic at seq 8192; the README's
8.37e8 / 2.71e9 figures are the older 4096-seq recipe.) Larger rungs need more
tokens than either pool holds and are deliberately absent.

The corpus lives on CoreWeave S3 us-east-02a, so training runs on CoreWeave
H100s in-region (8xH100 per run) — #6570 hit TransferBudgetExceeded streaming
training data cross-region; never stream training data across clouds. The
v5p-8 README baselines (paloma macro 3.8104 @ d512) are TPU numbers on a
different mix: numbers here are comparable only between the two arms, not to
the README table.

Launch checklist (DO NOT launch until the data pipeline output lands):

1. Fill in :data:`PDF_FINAL_DIR` with the final NormalizedData main output dir,
   ``s3://marin-us-east-02a/marin/data/datakit/final/common_crawl_focus_2026_22_pdf_ocr_all_<hash8>/outputs/main``
   (the hash8 comes from the merged pipeline's step key). ``main()`` refuses to
   run while it is ``None``. If it ever changes, bump :data:`_DATA_VERSION`.
2. Data stage — select + tokenize both arms, then print measured token counts
   and per-rung clearance (also written to ``token_counts.json``)::

       uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
           --job-name pdfcmp-data --cpu 2 --memory 8G --enable-extra-resources \\
           --extra cpu -e STAGE data \\
           -- python -m experiments.grug.moe.launch_pdf_compare --version 2026.08.06 --run

3. Read the clearance verdict from the job logs. d512 runs unconditionally;
   launch d768 only if BOTH arms clear its 1.81e9-token budget (the train stage
   re-checks and refuses otherwise).
4. Train stage — one CPU driver job per (arm, scale); the driver dispatches the
   actual 8xH100 training job via Fray inside cw-us-east-02a::

       uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
           --job-name pdfcmp-<arm>-<scale> --cpu 2 --memory 8G --enable-extra-resources \\
           --extra cpu -e WANDB_API_KEY "$WANDB_API_KEY" \\
           -e STAGE train -e ARM <arm> -e SCALE <scale> \\
           -- python -m experiments.grug.moe.launch_pdf_compare --version 2026.08.06 --run

   with ``<arm>`` in {pdf, finepdfs} and ``<scale>`` in {d512, d768}. One W&B
   run per (arm, scale): project ``marin_moe``, group ``pdf-vs-finepdfs``, seed 0.
5. Deliverable (per #7621): a W&B report over the ``pdf-vs-finepdfs`` group
   (final ``eval/paloma/macro_loss`` per rung, per-subset losses —
   arxiv_physics, arxiv_cs, m2d2_s2orc, mc4, ptb — and throughput/total_tokens;
   follow the wandb-reporting skill conventions), a logbook entry, and a
   summary comment on the issue linking both.

Env knobs: ``STAGE`` (data | train, default data), ``ARM`` (pdf | finepdfs),
``SCALE`` (d512 | d768), ``PDFCMP_ALLOW_REPEATS`` (1 to train past the pool
gate with repeated data).
"""

import dataclasses
import json
import logging
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from levanter.grug.attention import GrugAttentionImplementation
from levanter.tracker.wandb import WandbConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply
from marin.execution.remote import remote
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize import read_tokenized_cache_stats
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import StoragePath, prefix_join

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS, uncheatable_raw
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

PDF_FINAL_DIR: str | None = (
    "s3://marin-us-east-02a/marin/data/datakit/final/common_crawl_focus_2026_22_pdf_ocr_all_e4e8dda6/outputs/main"
)
"""The pipeline's final NormalizedData main output dir — filled in at launch time (step 1
of the module-docstring checklist). None until the pipeline output lands; ``main()``
refuses to run without it."""

# Identity version for the data-side steps (select, tokenize, report). These pin an
# explicit calendar version so a mutable --version can never silently rebuild the
# caches; bump it if PDF_FINAL_DIR or the FinePDFs sample ever changes.
_DATA_VERSION = "2026.08.06"

_ENGLISH_BUCKET = "eng_Latn"

# FinePDFs eng_Latn, already normalized onto the same CoreWeave bucket as our corpus
# (flat part-NNNNN-of-NNNNN.parquet, ~2.5 TB / ~7.45e11 llama3 tokens over 9,244
# shards ≈ 8.06e7 tokens per shard). 40 shards ≈ 3.2e9 tokens: clears the d768
# budget (1.81e9) with headroom while staying about the size of our corpus, without
# tokenizing the full 2.5 TB.
_FINEPDFS_NORMALIZED_DIR = "s3://marin-us-east-02a/marin/normalized/finepdfs_1ad51c52/outputs/main"
_FINEPDFS_NUM_SHARDS = 9244
_FINEPDFS_SAMPLE_SHARDS = 40

# Compute-optimal rungs (hidden_dim, budget) from experiments/grug/moe/README.md.
# Under the current heuristic these consume d512 = 5.25e8 tokens, d768 = 1.81e9;
# larger rungs exceed both pools.
LADDER: dict[str, tuple[int, float]] = {
    "d512": (512, 2.19e17),
    "d768": (768, 1.70e18),
}

ARMS = ("pdf", "finepdfs")

_ALLOW_REPEATS_ENV = "PDFCMP_ALLOW_REPEATS"

_SELECT_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="16g")

# One 8xH100 CoreWeave node per run; the CPU driver job dispatches training via
# Fray (regions inherit from the driver's target cluster). A run-arg, not part of
# the checkpoint's identity.
_TRAIN_RESOURCES = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g")

# GPU attention backend, as exercised daily by the CoreWeave canary
# (experiments/ferries/canary_ferry.py) on the same heuristic-built MoE.
_GPU_ATTENTION: GrugAttentionImplementation = "gpu_fa4_cute"


def _require_pdf_final_dir() -> str:
    if PDF_FINAL_DIR is None:
        raise ValueError(
            "PDF_FINAL_DIR is unset. Fill it with the pipeline's final NormalizedData main output dir "
            "(s3://marin-us-east-02a/marin/data/datakit/final/common_crawl_focus_2026_22_pdf_ocr_all_"
            "<hash8>/outputs/main) once the data pipeline lands — see the module-docstring checklist."
        )
    return PDF_FINAL_DIR


def _select_english_rows(source_dir: str, output_dir: str) -> None:
    """Materialize the ``language == eng_Latn`` rows of a NormalizedData dir as {id, text} parquet.

    Input basenames are preserved 1:1 and each input shard is sorted by id, which
    filtering preserves. Runs remotely next to the data (same bucket/region).
    """
    shards = sorted(str(p) for p in StoragePath(prefix_join(source_dir, "*.parquet")).glob())
    if not shards:
        raise RuntimeError(f"no parquet shards found under {source_dir}")
    docs_total = 0
    docs_kept = 0
    for shard in shards:
        with StoragePath(shard).open("rb") as f:
            table = pq.read_table(f, columns=["id", "text", "language"])
        selected = table.filter(pc.equal(table.column("language"), _ENGLISH_BUCKET)).select(["id", "text"])
        docs_total += table.num_rows
        docs_kept += selected.num_rows
        with StoragePath(prefix_join(output_dir, StoragePath(shard).name)).open("wb") as f:
            pq.write_table(selected, f)
    if docs_kept == 0:
        raise RuntimeError(f"no {_ENGLISH_BUCKET} rows in {docs_total} docs under {source_dir}")
    logger.info("kept %d/%d %s docs from %s", docs_kept, docs_total, _ENGLISH_BUCKET, source_dir)


def pdf_eng_latn_dataset() -> ArtifactStep[TokenizedCache]:
    """Our corpus's eng_Latn subset, selected to {id, text} parquet and llama3-tokenized."""
    selected = apply(
        "pdf_compare/pdf_eng_latn_text",
        remote(_select_english_rows, resources=_SELECT_RESOURCES),
        version=_DATA_VERSION,
        source_dir=_require_pdf_final_dir(),
        output_dir=OUT,
    )
    return tokenized(
        "pdf_compare/pdf_eng_latn-llama3",
        tokenizer=llama3_tokenizer,
        raw=selected,
        glob="*.parquet",
        version=_DATA_VERSION,
    )


def finepdfs_sample_dataset() -> ArtifactStep[TokenizedCache]:
    """A 40-shard (~3.2e9 token) sample of normalized FinePDFs eng_Latn, llama3-tokenized."""
    paths = [
        prefix_join(_FINEPDFS_NORMALIZED_DIR, f"part-{shard:05d}-of-{_FINEPDFS_NUM_SHARDS:05d}.parquet")
        for shard in range(_FINEPDFS_SAMPLE_SHARDS)
    ]
    return tokenized(
        "pdf_compare/finepdfs_eng_latn_sample-llama3",
        tokenizer=llama3_tokenizer,
        paths=paths,
        version=_DATA_VERSION,
    )


def _arm_dataset(arm: str) -> ArtifactStep[TokenizedCache]:
    if arm == "pdf":
        return pdf_eng_latn_dataset()
    if arm == "finepdfs":
        return finepdfs_sample_dataset()
    raise ValueError(f"ARM={arm!r} must be one of {ARMS}")


def _uncheatable_llama3_datasets() -> list[ArtifactStep[TokenizedCache]]:
    """The Uncheatable Eval subsets, retokenized with llama3 under this module's names.

    The shared ``uncheatable_eval/*-llama3`` caches record ``marin-community/marin-tokenizer``
    despite their names, while every paloma cache records llama3 -- so a mixture using both
    families can never satisfy mixture()'s single-tokenizer check. Rebuilding the seven small
    validation sets here (fresh names, this module's version) keeps the shared caches untouched
    and the whole mixture llama3.
    """
    raw = uncheatable_raw()
    return [
        tokenized(
            f"pdf_compare/uncheatable_{subset}-llama3",
            tokenizer=llama3_tokenizer,
            raw=raw,
            glob=glob,
            version=_DATA_VERSION,
            validation=True,
        )
        for subset, glob in UNCHEATABLE_SUBSETS.items()
    ]


def _rung_token_budget(scale: str) -> int:
    """The fixed token budget a rung consumes: batch_size * steps * seq_len from the heuristic."""
    hidden_dim, budget = LADDER[scale]
    model, _, batch_size, steps = build_from_heuristic(budget=budget, hidden_dim=hidden_dim)
    return batch_size * steps * model.max_seq_len


def _report_token_counts(pdf_cache: str, finepdfs_cache: str, output_dir: str) -> None:
    """Print each arm's measured train-token count and per-rung clearance; persist as JSON."""
    counts = {
        "pdf": read_tokenized_cache_stats(pdf_cache, "train").total_tokens,
        "finepdfs": read_tokenized_cache_stats(finepdfs_cache, "train").total_tokens,
    }
    budgets = {scale: _rung_token_budget(scale) for scale in LADDER}
    for arm, tokens in counts.items():
        print(f"{arm}: {tokens:,} train tokens")
    for scale, budget in budgets.items():
        verdicts = ", ".join(f"{arm} {'CLEARS' if counts[arm] >= budget else 'DOES NOT CLEAR'}" for arm in counts)
        print(f"{scale} (budget {budget:,} tokens): {verdicts}")
    report = {"train_tokens": counts, "rung_budgets": budgets}
    StoragePath(prefix_join(output_dir, "token_counts.json")).write_text(json.dumps(report, indent=2))


def token_report_step(
    pdf: ArtifactStep[TokenizedCache], finepdfs: ArtifactStep[TokenizedCache]
) -> ArtifactStep[Artifact]:
    return apply(
        "pdf_compare/token_report",
        _report_token_counts,
        version=_DATA_VERSION,
        pdf_cache=pdf,
        finepdfs_cache=finepdfs,
        output_dir=OUT,
    )


def _require_pool_covers_budget(*, arm: str, scale: str, pool_tokens: int, budget_tokens: int) -> None:
    """The no-repeats gate: refuse a rung whose fixed budget exceeds the arm's tokenized pool."""
    if pool_tokens >= budget_tokens:
        return
    if os.environ.get(_ALLOW_REPEATS_ENV) == "1":
        logger.warning(
            "%s/%s: pool %d < budget %d tokens; training WITH REPEATS (%s=1) — note this in the writeup",
            arm,
            scale,
            pool_tokens,
            budget_tokens,
            _ALLOW_REPEATS_ENV,
        )
        return
    raise ValueError(
        f"{arm}/{scale}: tokenized pool has {pool_tokens:,} tokens but the rung consumes "
        f"{budget_tokens:,} — the arm would repeat data. Pick a smaller rung, or set "
        f"{_ALLOW_REPEATS_ENV}=1 to train with repeats anyway."
    )


def build_train_step(arm: str, scale: str, *, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """One (arm, scale) training run as a lazy checkpoint, mirroring grug_moe_baseline."""
    if scale not in LADDER:
        raise ValueError(f"SCALE={scale!r} must be one of {sorted(LADDER)}")
    hidden_dim, budget = LADDER[scale]
    model, optimizer, batch_size, steps = build_from_heuristic(budget=budget, hidden_dim=hidden_dim)
    model = dataclasses.replace(model, attention_implementation=_GPU_ATTENTION)
    rung_tokens = batch_size * steps * model.max_seq_len

    tok = _arm_dataset(arm)
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *_uncheatable_llama3_datasets(),
    ]
    run_id = f"pdfcmp-{arm}-{scale}"
    name = f"grug/pdf_compare_{arm}_{scale}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if not ctx.is_fingerprint:
            _require_pool_covers_budget(
                arm=arm,
                scale=scale,
                pool_tokens=ctx.resolved(tok).num_train_tokens,
                budget_tokens=rung_tokens,
            )
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {tok: 1.0}, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "pdf-compare", arm, scale],
                group="pdf-vs-finepdfs",
                name=None,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(tok, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


def build() -> list[ArtifactStep]:
    _require_pdf_final_dir()
    stage = os.environ.get("STAGE", "data")
    if stage == "data":
        pdf = pdf_eng_latn_dataset()
        finepdfs = finepdfs_sample_dataset()
        return [pdf, finepdfs, token_report_step(pdf, finepdfs)]
    if stage == "train":
        arm = os.environ.get("ARM")
        scale = os.environ.get("SCALE")
        if arm is None or scale is None:
            raise ValueError(f"STAGE=train needs ARM (one of {ARMS}) and SCALE (one of {tuple(LADDER)}) in the env")
        return [build_train_step(arm, scale)]
    raise ValueError(f"STAGE={stage!r} must be 'data' or 'train'")


if __name__ == "__main__":
    experiment_main(build)()
