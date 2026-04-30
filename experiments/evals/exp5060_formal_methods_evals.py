# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
#5060: Formal-methods and hardware-RTL PPL eval slices (parent epic #5005).

Registers base-model PPL eval slices for formal-methods, proof, solver, and hardware-design
notation. These surface forms are close to code but have different grammar and symbol
distributions from ordinary Python/C++ source, so they live in their own families:

    formal_methods/smt_lib, formal_methods/tptp, formal_methods/coqgym, formal_methods/dimacs_cnf
    hardware_rtl/verilog_eval, hardware_rtl/rtl_repo, hardware_rtl/rtl_coder

The ``formal_methods/`` vs ``hardware_rtl/`` prefixes drive per-family grouping in
``levanter.analysis.perplexity_gap.register_dataset`` so one family does not mask the other
in the gap-report aggregation (issue #5060 DoD bullet 3).

AIGER/BTOR hardware model-checking benchmarks are binary with no canonical textual rendering,
so they are intentionally skipped (per issue discussion with @dlwh, 2026-04-22).

Per-source compressed output is capped at 20-40 MB per dlwh's guidance: each source downloads
its full upstream archive but only the first ~40 MB compressed JSONL of filtered files are
written. Pilot-scale downstream caps (``max_docs_per_dataset=256``, ``max_doc_bytes=32_768``)
match the existing ``exp1600_uncheatable_evals`` configuration.

This module intentionally does **not** modify ``default_raw_validation_sets()``; wire the
returned dicts into a gap-report runner explicitly (see ``MARIN_VS_LLAMA`` / ``MARIN_VS_QWEN3``
at the bottom of this file).

Upstream license notes and staging policy live in each source's shared
``IngestionSourceManifest``; the issue scopes this to eval use only.
"""

from __future__ import annotations

import logging
import os

from fray.types import ResourceConfig

from experiments.llama import llama3_tokenizer
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.datakit.download.formal_methods_evals import (
    DEFAULT_MAX_COMPRESSED_BYTES,
    JSONL_TEXT_COLUMN_CONTENT_MODE,
    RAW_FILE_CONTENT_MODE,
    ArchiveSourceConfig,
    archive_slice_step,
)
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
    raw_text_dataset,
)
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

EPIC_5005 = 5005
FORMAL_METHODS_ISSUE = 5060
Z3_MASTER_REV = "b9be33bb06b5e29ab65963e87c32bfa5c8a7f701"
COQGYM_MASTER_REV = "a739d99cdf5b0451dd8a362d3c541ca3b66112d3"
VERILOG_EVAL_MAIN_REV = "c498220d0a52248f8e3fdffe279075215bde2da6"
RTL_REPO_MAIN_REV = "7d10aa175afa56e500f58eacb7f5183b5f56ba25"
RTL_CODER_MAIN_REV = "b2847073be62d5f1d6d9b17bb247f0cfeb1ce642"


# --- Source configurations --------------------------------------------------------------------


def _archive_policy() -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only formal-methods and hardware-RTL probe slices.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: fixed held-out slice would directly contaminate evals if reused in training",
        provenance_notes="Materialization keeps raw file text or configured text columns verbatim.",
    )


def _archive_source_manifest(
    *,
    dataset_key: str,
    slice_key: str,
    archive_url: str,
    archive_format: str,
    include_globs: tuple[str, ...],
    source_license: str,
    source_format: str,
    exclude_globs: tuple[str, ...] = (),
    content_mode: str = RAW_FILE_CONTENT_MODE,
    jsonl_text_column: str | None = None,
    max_compressed_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
    max_files: int | None = None,
) -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key=dataset_key,
        slice_key=slice_key,
        source_label=slice_key,
        source_urls=(archive_url,),
        source_license=source_license,
        source_format=source_format,
        surface_form="raw_file_text" if content_mode == RAW_FILE_CONTENT_MODE else "jsonl_text_column",
        policy=_archive_policy(),
        staging=StagingMetadata(
            transform_name="download_archive_slice",
            serializer_name="archive_member_jsonl",
            metadata={
                "archive_format": archive_format,
                "include_globs": list(include_globs),
                "exclude_globs": list(exclude_globs),
                "content_mode": content_mode,
                "jsonl_text_column": jsonl_text_column,
                "output_filename": "data.jsonl.gz",
                "provenance_fields": ["id", "source", "filename"],
            },
        ),
        epic_issue=EPIC_5005,
        issue_numbers=(FORMAL_METHODS_ISSUE,),
        sample_caps=SampleCapConfig(
            max_bytes_per_source=max_compressed_bytes,
            max_files=max_files,
        ),
    )


FORMAL_METHODS_SOURCES: tuple[ArchiveSourceConfig, ...] = (
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="z3/smt_examples",
            slice_key="formal_methods/smt_lib",
            archive_url=f"https://github.com/Z3Prover/z3/archive/{Z3_MASTER_REV}.zip",
            archive_format="zip",
            include_globs=("*.smt2", "*.smt"),
            source_license="Z3 MIT license; files reused verbatim for PPL eval only.",
            source_format="zip archive of Z3 SMT examples",
        ),
    ),
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="tptp/archive",
            slice_key="formal_methods/tptp",
            archive_url="https://tptp.org/TPTP/Archive/TPTP-v8.2.0.tgz",
            archive_format="tar.gz",
            include_globs=("*.p", "*.ax"),
            source_license="TPTP: free for research; see https://www.tptp.org/.",
            source_format="tar.gz archive of TPTP theorem-proving problems",
        ),
    ),
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="coqgym/scripts",
            slice_key="formal_methods/coqgym",
            archive_url=f"https://github.com/princeton-vl/CoqGym/archive/{COQGYM_MASTER_REV}.zip",
            archive_format="zip",
            include_globs=("*.v",),
            exclude_globs=("*/node_modules/*", "*/.git/*"),
            source_license="CoqGym LGPL-2.1 license per upstream LICENSE.",
            source_format="zip archive of CoqGym proof scripts",
        ),
    ),
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="satlib/bmc",
            slice_key="formal_methods/dimacs_cnf",
            archive_url="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMC/bmc.tar.gz",
            archive_format="tar.gz",
            include_globs=("*.cnf",),
            source_license="SATLIB public benchmark collection; DIMACS CNF instances used for text PPL only.",
            source_format="tar.gz archive of SATLIB bounded-model-checking CNF files",
        ),
    ),
)

HARDWARE_RTL_SOURCES: tuple[ArchiveSourceConfig, ...] = (
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="verilog_eval/repo",
            slice_key="hardware_rtl/verilog_eval",
            archive_url=f"https://github.com/NVlabs/verilog-eval/archive/{VERILOG_EVAL_MAIN_REV}.zip",
            archive_format="zip",
            include_globs=("*.sv", "*.v"),
            source_license="VerilogEval MIT license per upstream LICENSE.",
            source_format="zip archive of VerilogEval sources",
        ),
    ),
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="rtl_repo/labels",
            slice_key="hardware_rtl/rtl_repo",
            archive_url=f"https://github.com/AUCOHL/RTL-Repo/archive/{RTL_REPO_MAIN_REV}.zip",
            archive_format="zip",
            include_globs=("predictions/*.jsonl",),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
            jsonl_text_column="label",
            source_license="RTL-Repo Apache-2.0 license per upstream LICENSE.",
            source_format="zip archive of RTL-Repo predictions",
        ),
    ),
    ArchiveSourceConfig(
        manifest=_archive_source_manifest(
            dataset_key="rtl_coder/responses",
            slice_key="hardware_rtl/rtl_coder",
            archive_url=f"https://github.com/hkust-zhiyao/RTL-Coder/archive/{RTL_CODER_MAIN_REV}.zip",
            archive_format="zip",
            include_globs=("dataset/*.json", "data_generation/data_sample.json"),
            content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
            jsonl_text_column="Response",
            source_license="RTL-Coder repo has no top-level LICENSE; README describes the dataset as open-source.",
            source_format="zip archive of RTL-Coder JSON datasets",
        ),
    ),
)

# AIGER / BTOR hardware model-checking benchmarks (https://fmv.jku.at/hwmcc11/benchmarks.html)
# are intentionally omitted: the primary distribution is binary AIGER. Adding a Marin-internal
# text rendering would be a bespoke serializer nobody else uses. Per @dlwh 2026-04-22, skip.


# --- StepSpec registration -------------------------------------------------------------------


def _build_steps(sources: tuple[ArchiveSourceConfig, ...]) -> dict[str, StepSpec]:
    return {source.slice_key: archive_slice_step(source) for source in sources}


FORMAL_METHODS_STEPS: dict[str, StepSpec] = _build_steps(FORMAL_METHODS_SOURCES)
HARDWARE_RTL_STEPS: dict[str, StepSpec] = _build_steps(HARDWARE_RTL_SOURCES)


def _raw_validation_sets(
    steps: dict[str, StepSpec],
    *,
    family_tag: str,
) -> dict[str, RawTextEvaluationDataset]:
    """Turn download steps into raw-text validation datasets keyed by ``family/slice``.

    Tags propagate to the levanter gap-report so family-level aggregations stay separable.
    """

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_key, step in steps.items():
        tags = (family_tag, f"issue:{5060}", slice_key)
        datasets[slice_key] = raw_text_dataset(step.as_executor_step().cd("data.jsonl.gz"), tags=tags)
    return datasets


def formal_methods_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw-text eval slices for formal-methods / proof / solver sources."""
    return _raw_validation_sets(FORMAL_METHODS_STEPS, family_tag="formal_methods")


def hardware_rtl_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw-text eval slices for hardware-RTL / Verilog / SystemVerilog sources."""
    return _raw_validation_sets(HARDWARE_RTL_STEPS, family_tag="hardware_rtl")


def exp5060_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Merged dict of both families, ready for a gap-report runner."""
    datasets = dict(formal_methods_raw_validation_sets())
    datasets.update(hardware_rtl_raw_validation_sets())
    return datasets


# --- Pilot gap-report runner (issue #5060 DoD bullet 4) --------------------------------------

_PILOT_RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
_PILOT_MAX_DOCS_PER_DATASET = 256
_PILOT_MAX_DOC_BYTES = 32_768
_PILOT_PER_DEVICE_BATCH_SIZE = 4
_PILOT_MAX_EVAL_LENGTH = 4096

_MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer=llama3_tokenizer,
)


def _pilot_gap_report(
    *,
    name: str,
    model_b: GapFinderModelConfig,
    model_b_label: str,
) -> tuple[ExecutorStep, ExecutorStep, ExecutorStep]:
    datasets = exp5060_raw_validation_sets()
    common_tags = [
        "rerun=exp5060-formal-hardware-first-pass",
        "dataset_bundle=exp5060_formal_methods_hardware_rtl",
        "source_split=http_archive",
        f"max_docs_per_dataset={_PILOT_MAX_DOCS_PER_DATASET}",
    ]
    marin_scores = model_perplexity_scores(
        name=f"{name}/marin_8b_scores",
        model=_MARIN_MODEL,
        datasets=datasets,
        resource_config=_PILOT_RESOURCE_CONFIG,
        per_device_batch_size=_PILOT_PER_DEVICE_BATCH_SIZE,
        max_eval_length=_PILOT_MAX_EVAL_LENGTH,
        max_docs_per_dataset=_PILOT_MAX_DOCS_PER_DATASET,
        max_doc_bytes=_PILOT_MAX_DOC_BYTES,
        wandb_tags=[
            "eval=model-perplexity",
            "model=marin-community/marin-8b-base",
            *common_tags,
        ],
    )
    model_b_scores = model_perplexity_scores(
        name=f"{name}/{model_b_label.replace('/', '_')}_scores",
        model=model_b,
        datasets=datasets,
        resource_config=_PILOT_RESOURCE_CONFIG,
        per_device_batch_size=_PILOT_PER_DEVICE_BATCH_SIZE,
        max_eval_length=_PILOT_MAX_EVAL_LENGTH,
        max_docs_per_dataset=_PILOT_MAX_DOCS_PER_DATASET,
        max_doc_bytes=_PILOT_MAX_DOC_BYTES,
        wandb_tags=[
            "eval=model-perplexity",
            f"model={model_b_label}",
            *common_tags,
        ],
    )
    gap_report = model_perplexity_gap_from_scores(
        name=name,
        model_a_name="marin-community/marin-8b-base",
        model_b_name=model_b_label,
        model_a_scores_path=marin_scores.as_input_name(),
        model_b_scores_path=model_b_scores.as_input_name(),
        wandb_tags=[
            "eval=perplexity-gap",
            "model_a=marin-community/marin-8b-base",
            f"model_b={model_b_label}",
            *common_tags,
        ],
    )
    return marin_scores, model_b_scores, gap_report


MARIN_VS_LLAMA_SCORES, LLAMA_SCORES, MARIN_VS_LLAMA = _pilot_gap_report(
    name="exp5060-marin-8b-base-vs-llama-3.1-8b-doccap256",
    model_b=GapFinderModelConfig(
        checkpoint_path="meta-llama/Llama-3.1-8B",
        checkpoint_is_hf=True,
        tokenizer=llama3_tokenizer,
    ),
    model_b_label="meta-llama/Llama-3.1-8B",
)

MARIN_VS_QWEN3_SCORES, QWEN3_SCORES, MARIN_VS_QWEN3 = _pilot_gap_report(
    name="exp5060-marin-8b-base-vs-qwen3-8b-base-doccap256",
    model_b=GapFinderModelConfig(
        checkpoint_path="Qwen/Qwen3-8B-Base",
        checkpoint_is_hf=True,
        tokenizer="Qwen/Qwen3-8B",
    ),
    model_b_label="Qwen/Qwen3-8B-Base",
)


def main() -> None:
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment; needs network + HF access.")
        return
    # Downloads materialize the per-source JSONL.gz; gap-report steps consume them.
    download_steps = [step.as_executor_step() for step in FORMAL_METHODS_STEPS.values()]
    download_steps.extend(step.as_executor_step() for step in HARDWARE_RTL_STEPS.values())
    executor_main(
        steps=[
            *download_steps,
            MARIN_VS_LLAMA_SCORES,
            LLAMA_SCORES,
            MARIN_VS_LLAMA,
            MARIN_VS_QWEN3_SCORES,
            QWEN3_SCORES,
            MARIN_VS_QWEN3,
        ],
        description=(
            "Issue #5060: formal-methods and hardware-RTL PPL slices, plus pilot gap-report "
            "against Llama-3.1-8B and Qwen3-8B-Base (parent epic #5005)."
        ),
    )


if __name__ == "__main__":
    main()
