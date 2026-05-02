# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Formal-methods and hardware-RTL raw PPL slices for issue #5060."""

from __future__ import annotations

from marin.datakit.download.formal_methods_evals import (
    DEFAULT_MAX_COMPRESSED_BYTES,
    JSONL_TEXT_COLUMN_CONTENT_MODE,
    RAW_FILE_CONTENT_MODE,
    ArchiveSourceConfig,
    archive_slice_step,
)
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.step_spec import StepSpec

EPIC_5005 = 5005
FORMAL_METHODS_ISSUE = 5060
Z3_MASTER_REV = "b9be33bb06b5e29ab65963e87c32bfa5c8a7f701"
COQGYM_MASTER_REV = "a739d99cdf5b0451dd8a362d3c541ca3b66112d3"
VERILOG_EVAL_MAIN_REV = "c498220d0a52248f8e3fdffe279075215bde2da6"
RTL_REPO_MAIN_REV = "7d10aa175afa56e500f58eacb7f5183b5f56ba25"
RTL_CODER_MAIN_REV = "b2847073be62d5f1d6d9b17bb247f0cfeb1ce642"


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


def _build_steps(sources: tuple[ArchiveSourceConfig, ...]) -> dict[str, StepSpec]:
    return {source.slice_key: archive_slice_step(source) for source in sources}


FORMAL_METHODS_STEPS: dict[str, StepSpec] = _build_steps(FORMAL_METHODS_SOURCES)
HARDWARE_RTL_STEPS: dict[str, StepSpec] = _build_steps(HARDWARE_RTL_SOURCES)


def _raw_validation_sets(
    steps: dict[str, StepSpec],
    *,
    family_tag: str,
) -> dict[str, RawTextEvaluationDataset]:
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_key, step in steps.items():
        tags = (family_tag, f"issue:{FORMAL_METHODS_ISSUE}", slice_key)
        datasets[slice_key] = raw_text_dataset(step.as_executor_step().cd("data.jsonl.gz"), tags=tags)
    return datasets


def formal_methods_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return _raw_validation_sets(FORMAL_METHODS_STEPS, family_tag="formal_methods")


def hardware_rtl_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return _raw_validation_sets(HARDWARE_RTL_STEPS, family_tag="hardware_rtl")


def formal_methods_hardware_rtl_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    datasets = dict(formal_methods_raw_validation_sets())
    datasets.update(hardware_rtl_raw_validation_sets())
    return datasets
