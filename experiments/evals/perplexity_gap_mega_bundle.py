# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merged raw PPL bundle for all self-contained eval sources currently runnable.

This intentionally includes only sources whose branches now contain enough code
to materialize the dataset from source and register raw-text eval datasets
without additional manual staging.

Notably excluded for now:
- diff/patch (#5095): current branch has row builders/registry, but not a
  self-contained upstream materializer
- diagnostic-log eval builders (#5093/#5104): current branch expects pre-staged
  source inputs rather than fetching its own held-out slices
- in-progress follow-up branches for #5053/#5056 WARC-WAT/#5059 wave 2/#5061
  basic package metadata/#5062 first-pass game-music
- unstable slices on this branch:
  - ``agent_traces/openhands_swe_rebench``: HF loader startup is still hanging
    locally
  - ``structured_text/*``: staged parquet discovery does not match the current
    raw HF snapshot layout
  - ``bio_chem/rcsb/rcsb_mmcif``: upstream sample URL currently 404s
  - raw OCR/TextOCR/OCR-VQA surfaces: the shared materializer is hitting HF
    dataset-server 429s; keep SVG only for now
  - FLORES translation slices: upstream HF reader is currently throwing
    ``UnicodeDecodeError`` on this pinned config
"""

from __future__ import annotations

import posixpath

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.transform.huggingface.raw_text import HfRawTextMaterializationConfig, materialize_hf_raw_text

from experiments.bio_chem_notation import bio_chem_raw_validation_sets
from experiments.defaults import default_raw_validation_sets
from experiments.exp5056_raw_web_markup_ppl import (
    RAW_WEB_MARKUP_HF_SURFACES,
    RAW_WEB_MARKUP_ISSUE_TAG,
    RAW_WEB_MARKUP_PREFIX,
    SVG_STACK_DATASET_ID,
)
from experiments.evals.asr_ocr_noisy_ppl import noisy_asr_ocr_raw_validation_sets
from experiments.evals.exp5057_binary_network_security_evals import binary_network_security_raw_validation_sets
from experiments.evals.exp5060_formal_methods_evals import exp5060_raw_validation_sets
from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets
from experiments.evals.gh_archive_structured_output import gh_archive_structured_output_raw_validation_sets
from experiments.evals.paired_robustness_ppl import (
    PairedRobustnessFamily,
    paired_robustness_raw_validation_sets,
    paired_robustness_slices,
)
from experiments.evals.synthetic_reasoning_ppl import synthetic_reasoning_raw_validation_sets

EXCLUDED_DATASET_KEYS = frozenset(
    {
        "agent_traces/openhands_swe_rebench",
        "bio_chem/rcsb/rcsb_mmcif",
    }
)

SVG_ONLY_RAW_WEB_MARKUP_SURFACES = tuple(
    surface for surface in RAW_WEB_MARKUP_HF_SURFACES if surface.dataset_id == SVG_STACK_DATASET_ID
)

svg_only_raw_web_markup_hf = ExecutorStep(
    name="raw/raw_web_markup/svg_only",
    fn=materialize_hf_raw_text,
    config=HfRawTextMaterializationConfig(
        surfaces=SVG_ONLY_RAW_WEB_MARKUP_SURFACES,
        output_path=this_output_path(),
    ),
)


def _without_excluded(dataset_map: dict[str, RawTextEvaluationDataset]) -> dict[str, RawTextEvaluationDataset]:
    return {key: dataset for key, dataset in dataset_map.items() if key not in EXCLUDED_DATASET_KEYS}


def _svg_only_raw_web_markup_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    return {
        posixpath.join(RAW_WEB_MARKUP_PREFIX, "svg_stack", "svg_xml_val"): raw_text_dataset(
            svg_only_raw_web_markup_hf.cd("svg_stack/svg_xml_val.jsonl.gz"),
            tags=(
                RAW_WEB_MARKUP_PREFIX,
                RAW_WEB_MARKUP_ISSUE_TAG,
                "source:svg_stack",
                "surface:svg_xml",
                "split:val",
            ),
        ),
        posixpath.join(RAW_WEB_MARKUP_PREFIX, "svg_stack", "svg_xml_test"): raw_text_dataset(
            svg_only_raw_web_markup_hf.cd("svg_stack/svg_xml_test.jsonl.gz"),
            tags=(
                RAW_WEB_MARKUP_PREFIX,
                RAW_WEB_MARKUP_ISSUE_TAG,
                "source:svg_stack",
                "surface:svg_xml",
                "split:test",
            ),
        ),
    }


def _paraphrase_only_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    paraphrase_slices = paired_robustness_slices(family=PairedRobustnessFamily.PARAPHRASE)
    return paired_robustness_raw_validation_sets(slices=paraphrase_slices)


def mega_available_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return the merged raw eval bundle for all currently runnable sources."""
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for dataset_map in (
        _without_excluded(default_raw_validation_sets()),
        fineweb2_multilingual_raw_validation_sets(),
        _svg_only_raw_web_markup_validation_sets(),
        binary_network_security_raw_validation_sets(),
        _without_excluded(bio_chem_raw_validation_sets()),
        exp5060_raw_validation_sets(),
        synthetic_reasoning_raw_validation_sets(),
        _paraphrase_only_validation_sets(),
        noisy_asr_ocr_raw_validation_sets(),
        gh_archive_structured_output_raw_validation_sets(),
    ):
        overlap = set(datasets).intersection(dataset_map)
        if overlap:
            raise ValueError(f"Duplicate dataset keys in mega bundle: {sorted(overlap)}")
        datasets.update(dataset_map)
    return datasets
