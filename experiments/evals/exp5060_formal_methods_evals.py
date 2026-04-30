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

Upstream license notes are attached to each ``ArchiveSourceConfig``.license_note for
auditability; the issue scopes this to eval use only.
"""

from __future__ import annotations

import logging
import os

from fray.v2.types import ResourceConfig

from experiments.llama import llama3_tokenizer
from marin.datakit.download.formal_methods_evals import (
    DEFAULT_MAX_COMPRESSED_BYTES,
    JSONL_TEXT_COLUMN_CONTENT_MODE,
    ArchiveSourceConfig,
    archive_slice_step,
)
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    default_model_perplexity_gap,
    raw_text_dataset,
)
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)


# --- Source configurations --------------------------------------------------------------------

FORMAL_METHODS_SOURCES: tuple[ArchiveSourceConfig, ...] = (
    ArchiveSourceConfig(
        slice_key="formal_methods/smt_lib",
        # Z3 ships a large corpus of SMT-LIB2 example/regression files we can mirror as a
        # representative SMT-LIB sample. For full SMT-LIB benchmark logics, repoint to a
        # specific Zenodo tar.zst (see https://smt-lib.org/benchmarks.shtml).
        url="https://github.com/Z3Prover/z3/archive/refs/heads/master.zip",
        archive_format="zip",
        include_globs=("*.smt2", "*.smt"),
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="Z3 MIT license; files reused verbatim for PPL eval only.",
    ),
    ArchiveSourceConfig(
        slice_key="formal_methods/tptp",
        # TPTP canonical distribution. The tarball is large (~1 GB); the byte budget caps
        # what is materialized downstream.
        url="https://tptp.org/TPTP/Archive/TPTP-v8.2.0.tgz",
        archive_format="tar.gz",
        include_globs=("*.p", "*.ax"),
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="TPTP: free for research; see https://www.tptp.org/.",
    ),
    ArchiveSourceConfig(
        slice_key="formal_methods/coqgym",
        url="https://github.com/princeton-vl/CoqGym/archive/refs/heads/master.zip",
        archive_format="zip",
        include_globs=("*.v",),
        # CoqGym vendors proof-state JSON under data/; we only want the Coq script text.
        exclude_globs=("*/node_modules/*", "*/.git/*"),
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="CoqGym LGPL-2.1 license per upstream LICENSE.",
    ),
    ArchiveSourceConfig(
        slice_key="formal_methods/dimacs_cnf",
        # SATLIB's bounded-model-checking suite is a compact DIMACS CNF sample. Larger SAT
        # Competition archives can be added once the downloader streams zip files instead of
        # materializing them in memory.
        url="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMC/bmc.tar.gz",
        archive_format="tar.gz",
        include_globs=("*.cnf",),
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="SATLIB public benchmark collection; DIMACS CNF instances used for text PPL only.",
    ),
)

HARDWARE_RTL_SOURCES: tuple[ArchiveSourceConfig, ...] = (
    ArchiveSourceConfig(
        slice_key="hardware_rtl/verilog_eval",
        url="https://github.com/NVlabs/verilog-eval/archive/refs/heads/main.zip",
        archive_format="zip",
        include_globs=("*.sv", "*.v"),
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="VerilogEval MIT license per upstream LICENSE.",
    ),
    ArchiveSourceConfig(
        slice_key="hardware_rtl/rtl_repo",
        url="https://github.com/AUCOHL/RTL-Repo/archive/refs/heads/main.zip",
        archive_format="zip",
        include_globs=("predictions/*.jsonl",),
        content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
        jsonl_text_column="label",
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="RTL-Repo Apache-2.0 license per upstream LICENSE.",
    ),
    ArchiveSourceConfig(
        slice_key="hardware_rtl/rtl_coder",
        url="https://github.com/hkust-zhiyao/RTL-Coder/archive/refs/heads/main.zip",
        archive_format="zip",
        include_globs=("dataset/*.json", "data_generation/data_sample.json"),
        content_mode=JSONL_TEXT_COLUMN_CONTENT_MODE,
        jsonl_text_column="Response",
        max_compressed_bytes=DEFAULT_MAX_COMPRESSED_BYTES,
        license_note="RTL-Coder repo has no top-level LICENSE; README describes the dataset as open-source.",
    ),
)

# AIGER / BTOR hardware model-checking benchmarks (https://fmv.jku.at/hwmcc11/benchmarks.html)
# are intentionally omitted: the primary distribution is binary AIGER. Adding a Marin-internal
# text rendering would be a bespoke serializer nobody else uses. Per @dlwh 2026-04-22, skip.
SKIPPED_BINARY_SOURCES: tuple[str, ...] = ("hardware_rtl/aiger_hwmcc",)


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
) -> ExecutorStep:
    return default_model_perplexity_gap(
        name=name,
        model_a=_MARIN_MODEL,
        model_b=model_b,
        datasets=exp5060_raw_validation_sets(),
        resource_config=_PILOT_RESOURCE_CONFIG,
        per_device_batch_size=_PILOT_PER_DEVICE_BATCH_SIZE,
        max_eval_length=_PILOT_MAX_EVAL_LENGTH,
        max_docs_per_dataset=_PILOT_MAX_DOCS_PER_DATASET,
        max_doc_bytes=_PILOT_MAX_DOC_BYTES,
        wandb_tags=[
            "eval=perplexity-gap",
            "rerun=exp5060-formal-hardware-first-pass",
            "model_a=marin-community/marin-8b-base",
            f"model_b={model_b_label}",
            "dataset_bundle=exp5060_formal_methods_hardware_rtl",
            "source_split=http_archive",
            f"max_docs_per_dataset={_PILOT_MAX_DOCS_PER_DATASET}",
        ],
    )


MARIN_VS_LLAMA = _pilot_gap_report(
    name="exp5060-marin-8b-base-vs-llama-3.1-8b-doccap256",
    model_b=GapFinderModelConfig(
        checkpoint_path="meta-llama/Llama-3.1-8B",
        checkpoint_is_hf=True,
        tokenizer=llama3_tokenizer,
    ),
    model_b_label="meta-llama/Llama-3.1-8B",
)

MARIN_VS_QWEN3 = _pilot_gap_report(
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
        steps=[*download_steps, MARIN_VS_LLAMA, MARIN_VS_QWEN3],
        description=(
            "Issue #5060: formal-methods and hardware-RTL PPL slices, plus pilot gap-report "
            "against Llama-3.1-8B and Qwen3-8B-Base (parent epic #5005)."
        ),
    )


if __name__ == "__main__":
    main()
