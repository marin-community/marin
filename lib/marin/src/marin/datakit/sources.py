# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical source registry for the Datakit.

Each :class:`DatakitSource` is the canonical recipe for a normalized dataset:
a stable ``name``, the ordered ``(download, ..., normalize)`` :class:`StepSpec`
chain that materializes it, and a rough per-source token count for mixture
weighting.

The chains themselves live in the family-specific modules under
``lib/marin/src/marin/datakit/download/``; this file is just the catalog that
ties them to a ``name`` and a token count.
"""

from dataclasses import dataclass
from functools import cache

from marin.datakit.download.common_pile import common_pile_normalize_steps
from marin.datakit.download.finepdfs import finepdfs_normalize_steps
from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.datakit.download.nemotron_v2 import NEMOTRON_V2_DATASETS, nemotron_v2_normalize_steps
from marin.datakit.download.starcoder2_extras import starcoder2_extras_normalize_steps
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class DatakitSource:
    """One mixture component: name + the StepSpec chain that produces its normalized output."""

    name: str
    """Mixture-component key, e.g. ``"nemotron_cc_v2_1/high_quality"``."""

    normalize_steps: tuple[StepSpec, ...]
    """Ordered step chain. Always starts with a download and ends with
    ``normalize``; may contain preprocessing steps in between for sources
    that need filtering or transforms."""

    rough_token_count_b: float | None = None
    """Approximate token count in billions (Llama-3 tokenizer). Used as the
    initial per-source mixing weight; ``None`` means unknown and callers
    should supply their own fallback."""

    @property
    def normalized(self) -> StepSpec:
        """The terminal step (normalize). This is the canonical artifact
        downstream consumers sample, dedup, or tokenize off of."""
        return self.normalize_steps[-1]


# ---- Plain-HF single-source entries -----------------------------------------
# Sources that are just download + normalize, no custom preprocessing, no
# shared family download. One :func:`hf_normalize_steps` call per entry.
#
# (marin_name, hf_dataset_id, revision, staged_path, rough_token_count_b)
_SIMPLE_HF_SOURCES: tuple[tuple[str, str, str, str, float], ...] = (
    ("coderforge", "togethercomputer/CoderForge-Preview", "060fca9", "raw/coderforge-preview_ad26b119", 10.29),
    ("gpt-oss-rollouts", "andyrdt/gpt-oss-20b-rollouts", "f47b4a2", "raw/gpt-oss-20b-rollouts_58b022a7", 3.20),
    (
        "institutional_books",
        "institutional/institutional-books-1.0",
        "d2f504a",
        "raw/institutional-books-d2f504a",
        203.63,
    ),
    ("nemotron-terminal", "nvidia/Nemotron-Terminal-Corpus", "a1667c4", "raw/nemotron-terminal-corpus_c68d0061", 6.08),
    (
        "superior-reasoning",
        "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        "21b55a6",
        "raw/superior-reasoning-sft_b42ea7b3",
        7.08,
    ),
    (
        "swe-rebench-openhands",
        "nebius/SWE-rebench-openhands-trajectories",
        "3545538",
        "raw/swe-rebench-openhands-trajectories_e1e457c7",
        2.47,
    ),
    ("synthetic-1", "PrimeIntellect/SYNTHETIC-1", "f08fe8c", "raw/synthetic-1_1b24a14b", 7.32),
)


# ---- StarCoder2-Extras token counts ----------------------------------------
# Chains live in starcoder2_extras.starcoder2_extras_normalize_steps(); this
# registry advertises 5 of the 6 subsets (ir_low_resource isn't in the
# token-count-viewer set, so it's filtered out here).
_STARCODER2_EXTRAS_TOKEN_COUNTS: dict[str, float] = {
    "starcoder2/documentation": 1.40,
    "starcoder2/ir_cpp": 39.01,
    "starcoder2/ir_python": 4.64,
    "starcoder2/ir_rust": 1.84,
    "starcoder2/kaggle": 1.38,
}


# ---- Nemotron v2 families --------------------------------------------------
# One family download shared across all subsets; each subset has its own
# normalize with a data_subdir. See ``NEMOTRON_V2_DATASETS``.
#
# Token counts per marin_name. Families not listed here are excluded from the
# active registry (e.g. nemotron_pretraining_code_v1 and v2's code_metadata
# subset aren't in the token-count-viewer set).
_NEMOTRON_V2_TOKEN_COUNTS: dict[str, dict[str, float]] = {
    "nemotron_cc_v2": {
        "nemotron_cc_v2/diverse_qa": 676.57,
        "nemotron_cc_v2/high_quality": 608.96,
        "nemotron_cc_v2/high_quality_synthetic": 1223.46,
        "nemotron_cc_v2/medium_high_quality": 535.45,
        "nemotron_cc_v2/medium_quality": 2114.33,
        "nemotron_cc_v2/translated_diverse_qa": 592.85,
    },
    "nemotron_cc_v2_1": {
        "nemotron_cc_v2_1/high_quality": 25.15,
        "nemotron_cc_v2_1/high_quality_dqa": 7.81,
        "nemotron_cc_v2_1/high_quality_synthetic": 90.86,
        "nemotron_cc_v2_1/high_quality_translated": 38.65,
        "nemotron_cc_v2_1/high_quality_translated_synthetic": 153.41,
        "nemotron_cc_v2_1/medium_high_quality": 16.35,
        "nemotron_cc_v2_1/medium_high_quality_synthetic": 2065.38,
        "nemotron_cc_v2_1/medium_high_quality_translated": 26.03,
        "nemotron_cc_v2_1/medium_quality": 51.67,
    },
    "nemotron_cc_code_v1": {
        "nemotron_cc_code_v1/all": 399.41,
    },
    "nemotron_cc_math_v1": {
        "nemotron_cc_math_v1/3": 78.90,
        "nemotron_cc_math_v1/4plus_mind": 72.20,
    },
    "nemotron_pretraining_code_v2": {
        # Token-count-viewer uses the short "nemotron_code_v2" name.
        "nemotron_code_v2/synthetic_code_review": 74.24,
        "nemotron_code_v2/synthetic_rewriting": 73.73,
        "nemotron_code_v2/synthetic_student_teacher": 25.20,
        "nemotron_code_v2/synthetic_question_answering": 233.03,
        "nemotron_code_v2/synthetic_transpilation": 27.78,
    },
    "nemotron_pretraining_sft_v1": {
        "nemotron_sft/sft_code": 56.65,
        "nemotron_sft/sft_general": 85.20,
        "nemotron_sft/sft_math": 199.94,
    },
    "nemotron_pretraining_specialized_v1": {
        "nemotron_specialized/infinibyte_reasoning": 18.69,
        "nemotron_specialized/math_textbooks": 25.59,
        "nemotron_specialized/rqa": 135.17,
        "nemotron_specialized/scientific_coding": 1.18,
        "nemotron_specialized/stem_sft": 81.20,
        "nemotron_specialized/wiki_rewrite": 7.26,
    },
    "nemotron_pretraining_specialized_v1_1": {
        "nemotron_specialized_v1_1/code_concepts": 7.03,
        "nemotron_specialized_v1_1/economics": 0.07,
        "nemotron_specialized_v1_1/formal_logic": 0.13,
        "nemotron_specialized_v1_1/multiple_choice": 1.56,
        "nemotron_specialized_v1_1/unconditional_algorithmic": 0.19,
    },
}

# The registry's marin_names differ slightly from ``nemotron_v2_normalize_steps``
# output (which uses ``nemotron_pretraining_*`` as the family key). Map the
# library-name → registry-name for the renamed families.
_NEMOTRON_REGISTRY_RENAMES: dict[str, str] = {
    "nemotron_pretraining_code_v2": "nemotron_code_v2",
    "nemotron_pretraining_sft_v1": "nemotron_sft",
    "nemotron_pretraining_specialized_v1": "nemotron_specialized",
    "nemotron_pretraining_specialized_v1_1": "nemotron_specialized_v1_1",
}


# ---- FinePDFs --------------------------------------------------------------
# 19 language subsets share one HF repo but stage per-language. Token counts:
_FINEPDFS_TOKEN_COUNTS: dict[str, float] = {
    "finepdfs": 1186.47,
    "finepdfs/arb_Arab": 29.72,
    "finepdfs/ces_Latn": 29.83,
    "finepdfs/cmn_Hani": 32.97,
    "finepdfs/deu_Latn": 177.10,
    "finepdfs/fra_Latn": 164.75,
    "finepdfs/hun_Latn": 37.44,
    "finepdfs/ind_Latn": 20.32,
    "finepdfs/ita_Latn": 94.79,
    "finepdfs/jpn_Jpan": 115.87,
    "finepdfs/nld_Latn": 46.97,
    "finepdfs/pol_Latn": 54.40,
    "finepdfs/por_Latn": 94.69,
    "finepdfs/ron_Latn": 22.61,
    "finepdfs/rus_Cyrl": 146.95,
    "finepdfs/spa_Latn": 216.74,
    "finepdfs/swe_Latn": 25.34,
    "finepdfs/tha_Thai": 17.40,
    "finepdfs/ukr_Cyrl": 25.53,
}


# ---- common-pile -----------------------------------------------------------
# 27 entries, each its own HF repo. Token counts keyed by registry marin_name.
_COMMON_PILE_TOKEN_COUNTS: dict[str, float] = {
    "cp/arxiv_abstracts": 0.54,
    "cp/arxiv_papers": 6.63,
    "cp/biodiversity": 8.60,
    "cp/caselaw": 17.55,
    "cp/data_provenance": 0.82,
    "cp/doab": 2.93,
    "cp/foodista": 0.02,
    "cp/github_archive": 10.26,
    "cp/library_of_congress": 8.06,
    "cp/libretexts": 0.08,
    "cp/news": 0.05,
    "cp/oercommons": 0.01,
    "cp/peS2o": 40.74,
    "cp/peps": 0.003,
    "cp/pre_1929_books": 10.57,
    "cp/pressbooks": 0.13,
    "cp/project_gutenberg": 4.91,
    "cp/public_domain_review": 0.002,
    "cp/pubmed": 38.08,
    "cp/regulations": 1.28,
    "cp/stackexchange": 21.89,
    "cp/stackv2_code": 352.76,
    "cp/ubuntu_irc": 1.76,
    "cp/uk_hansard": 2.13,
    "cp/usgpo": 7.78,
    "cp/uspto": 142.41,
    "cp/wikiteam": 2.97,
    "cp/youtube": 4.07,
}


# ---- Disabled sources (tracked in the token-count-viewer but can't ferry today) ----
#
# TODO: confirm there's a download module for PleIAs/common_corpus.
# Staged dir ``raw/common_corpus_english-b78a5c1`` is missing its
# .executor_status marker, so we can't confirm the staging run completed
# cleanly. Re-enable once the staging is re-verified.
#
# TODO: confirm there's a download module for HuggingFaceFW/finetranslations.
# Staging at ``raw/finetranslations_d17a789b`` is still in progress — no
# provenance.json, no .executor_status=SUCCESS yet. The upstream is a parallel
# corpus of original multilingual text + machine-translated English; splitting
# into /multilingual and /web needs different text_field, hf_urls_glob, or
# data_subdir so the two accounting slices don't normalize to identical rows.
#
# TODO: confirm there's a download module for HPLT/HPLT3.0. The previous
# download_hplt_v3_step was removed from the tree and the staged dir has no
# provenance.json to recover the revision.
#
# TODO: confirm there's a download module for AI-MO/NuminaMath-1.5. Today
# the dataset is only referenced through gpt-oss-rollouts' NuminaMath-CoT
# subset; there's no standalone download helper.
#
# TODO: confirm there's a download module for NSF Awards (API-sourced via
# nsf.gov, no HF repo).


@cache
def all_sources() -> dict[str, DatakitSource]:
    """Return the canonical active source set as ``{name: DatakitSource}``.

    Every entry is materializable — has a full :attr:`DatakitSource.normalize_steps`
    chain ready to run. Disabled entries (see TODOs above) are commented out of
    the module.
    """
    entries: dict[str, DatakitSource] = {}

    def _add(name: str, steps: tuple[StepSpec, ...], rough: float | None) -> None:
        entries[name] = DatakitSource(name=name, normalize_steps=steps, rough_token_count_b=rough)

    # Plain-HF single-source entries
    for name, hf_id, rev, staged, rough in _SIMPLE_HF_SOURCES:
        _add(name, hf_normalize_steps(marin_name=name, hf_dataset_id=hf_id, revision=rev, staged_path=staged), rough)

    # common-pile: 27 flat HF entries
    for marin_name, chain in common_pile_normalize_steps().items():
        _add(marin_name, chain, _COMMON_PILE_TOKEN_COUNTS[marin_name])

    # FinePDFs: 19 per-language chains
    for marin_name, chain in finepdfs_normalize_steps().items():
        _add(marin_name, chain, _FINEPDFS_TOKEN_COUNTS[marin_name])

    # StarCoder2-Extras: 5 per-subset chains (out of 6 in the download module;
    # ir_low_resource isn't in the token-count-viewer set)
    for marin_name, chain in starcoder2_extras_normalize_steps().items():
        if marin_name in _STARCODER2_EXTRAS_TOKEN_COUNTS:
            _add(marin_name, chain, _STARCODER2_EXTRAS_TOKEN_COUNTS[marin_name])

    # Nemotron v2 family dumps — one shared download per family
    for library_family, counts in _NEMOTRON_V2_TOKEN_COUNTS.items():
        registry_family = _NEMOTRON_REGISTRY_RENAMES.get(library_family, library_family)
        for library_name, chain in nemotron_v2_normalize_steps(library_family).items():
            # Library keys look like "nemotron_pretraining_code_v2/synthetic_code_review";
            # registry keys are "nemotron_code_v2/synthetic_code_review". Apply the rename.
            registry_name = library_name.replace(library_family, registry_family, 1)
            if registry_name not in counts:
                # Skip subsets the registry doesn't advertise (e.g. code_metadata
                # subsets that aren't in the token-count-viewer set).
                continue
            _add(registry_name, chain, counts[registry_name])

    assert len(entries) > 0
    return entries


# Sanity guard: every token count key in the multi-subset families must match
# a subset name in NEMOTRON_V2_DATASETS. Runs at import to catch typos cheaply.
for _family in _NEMOTRON_V2_TOKEN_COUNTS:
    assert _family in NEMOTRON_V2_DATASETS, f"unknown Nemotron family in token counts: {_family}"
