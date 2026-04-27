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
"""

from __future__ import annotations

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset

from experiments.bio_chem_notation import bio_chem_raw_validation_sets
from experiments.defaults import default_raw_validation_sets
from experiments.exp5056_raw_web_markup_ppl import raw_web_markup_raw_validation_sets
from experiments.structured_evals import structured_evals_raw_validation_sets
from experiments.evals.asr_ocr_noisy_ppl import noisy_asr_ocr_raw_validation_sets
from experiments.evals.exp5057_binary_network_security_evals import binary_network_security_raw_validation_sets
from experiments.evals.exp5060_formal_methods_evals import exp5060_raw_validation_sets
from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets
from experiments.evals.gh_archive_structured_output import gh_archive_structured_output_raw_validation_sets
from experiments.evals.paired_robustness_ppl import paired_robustness_raw_validation_sets
from experiments.evals.synthetic_reasoning_ppl import synthetic_reasoning_raw_validation_sets


def mega_available_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return the merged raw eval bundle for all currently runnable sources."""
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for dataset_map in (
        default_raw_validation_sets(),
        fineweb2_multilingual_raw_validation_sets(),
        raw_web_markup_raw_validation_sets(),
        structured_evals_raw_validation_sets(),
        binary_network_security_raw_validation_sets(),
        bio_chem_raw_validation_sets(),
        exp5060_raw_validation_sets(),
        synthetic_reasoning_raw_validation_sets(),
        paired_robustness_raw_validation_sets(),
        noisy_asr_ocr_raw_validation_sets(),
        gh_archive_structured_output_raw_validation_sets(),
    ):
        overlap = set(datasets).intersection(dataset_map)
        if overlap:
            raise ValueError(f"Duplicate dataset keys in mega bundle: {sorted(overlap)}")
        datasets.update(dataset_map)
    return datasets
