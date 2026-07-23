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

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from marin.datakit.canonical.safety_pretraining import safety_pretraining_normalize_steps
from marin.datakit.download.biocollection import biocollection_normalize_steps
from marin.datakit.download.biodiversity import biodiversity_normalize_steps
from marin.datakit.download.climblab_ja import climblab_ja_normalize_steps
from marin.datakit.download.coderforge import coderforge_normalize_steps
from marin.datakit.download.common_crawl_focus import common_crawl_focus_normalize_steps
from marin.datakit.download.common_pile import common_pile_normalize_steps
from marin.datakit.download.davinci_dev import (
    davinci_dev_ctx_native_normalize_steps,
    davinci_dev_env_native_normalize_steps,
)
from marin.datakit.download.diagnostic_logs import GHALOGS_ROUGH_TOKENS_B, ghalogs_public_normalize_steps
from marin.datakit.download.dolma3_5_code import dolma3_5_code_normalize_steps
from marin.datakit.download.dolma4pdfs import dolma4pdfs_normalize_steps
from marin.datakit.download.eai_taxonomy_code import eai_taxonomy_code_normalize_steps
from marin.datakit.download.finepdfs import finepdfs_normalize_steps
from marin.datakit.download.finetranslations import finetranslations_normalize_steps
from marin.datakit.download.gpt_oss_rollouts import gpt_oss_rollouts_normalize_steps
from marin.datakit.download.hplt import hplt_v3_normalize_steps
from marin.datakit.download.institutional_books import institutional_books_normalize_steps
from marin.datakit.download.massive import massive_normalize_steps
from marin.datakit.download.molmo2_cap import molmo2_cap_normalize_steps
from marin.datakit.download.nemotron_code_v2_content import nemotron_code_v2_content_normalize_steps
from marin.datakit.download.nemotron_terminal import nemotron_terminal_normalize_steps
from marin.datakit.download.nemotron_v2 import (
    NEMOTRON_PRETRAINING_LEGAL_V1,
    NEMOTRON_PRETRAINING_SPECIALIZED_V1_2,
    nemotron_v2_normalize_steps,
)
from marin.datakit.download.nsf_awards import nsf_awards_normalize_steps
from marin.datakit.download.numinamath_tir import numinamath_tir_normalize_steps
from marin.datakit.download.numinamath_v1_5 import numinamath_v1_5_normalize_steps
from marin.datakit.download.sec_edgar import sec_edgar_normalize_steps
from marin.datakit.download.starcoder2_extras import starcoder2_extras_normalize_steps
from marin.datakit.download.superior_reasoning import superior_reasoning_normalize_steps
from marin.datakit.download.svgfind import svgfind_creativecommons_normalize_steps
from marin.datakit.download.swe_rebench_contree import swe_rebench_contree_normalize_steps
from marin.datakit.download.swe_rebench_openhands import swe_rebench_openhands_normalize_steps
from marin.datakit.download.swe_zero_12m import swe_zero_12m_normalize_steps
from marin.datakit.download.synthetic1 import synthetic1_normalize_steps
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

    rough_token_count_b: float
    """Approximate token count in billions (Llama-3 tokenizer). Used as the
    initial per-source mixing weight — required so callers never have to
    fall back to a made-up default."""

    @property
    def normalized(self) -> StepSpec:
        """The terminal step (normalize). This is the canonical artifact
        downstream consumers sample, dedup, or tokenize off of."""
        return self.normalize_steps[-1]


# Every registry row is a ``(marin_name, chain_factory, rough_token_count_b)``
# triple. The chain factory, called with no args, returns the ordered
# ``(download, ..., normalize)`` StepSpec tuple for that source.
_SourceRow = tuple[str, Callable[[], tuple[StepSpec, ...]], float]


def _rows_flat(
    factory: Callable[[], dict[str, tuple[StepSpec, ...]]],
    counts: dict[str, float],
) -> tuple[_SourceRow, ...]:
    """Project a multi-subset family factory into per-subset rows.

    The registry names in ``counts`` must match the keys returned by
    ``factory()``. Rows whose registry name isn't in ``counts`` are skipped.
    """
    return tuple((name, lambda f=factory, n=name: f()[n], count) for name, count in counts.items())


def _rows_nemotron(
    library_family: str,
    registry_family: str,
    counts: dict[str, float],
) -> tuple[_SourceRow, ...]:
    """Project a Nemotron v2 family into per-subset rows.

    Nemotron library names (``nemotron_pretraining_code_v2/...``) differ from
    the registry's shorter marin_names (``nemotron_code_v2/...``). The
    ``registry_family`` → ``library_family`` prefix swap recovers the library
    key used to look up the chain. All subsets share the family download
    thanks to ``@cache`` on ``download_nemotron_v2_step``.
    """
    rows: list[_SourceRow] = []
    for registry_name, count in counts.items():
        library_key = registry_name.replace(registry_family, library_family, 1)
        rows.append(
            (
                registry_name,
                lambda lf=library_family, lib=library_key: nemotron_v2_normalize_steps(lf)[lib],
                count,
            )
        )
    return tuple(rows)


# ---- Disabled sources (tracked in the token-count-viewer but can't ferry today) ----
#
# TODO: confirm there's a download module for PleIAs/common_corpus.
# Staged dir ``raw/common_corpus_english-b78a5c1`` is missing its
# .executor_status marker, so we can't confirm the staging run completed
# cleanly. Re-enable once the staging is re-verified.
#
@cache
def all_sources() -> dict[str, DatakitSource]:
    """Return the canonical active source set as ``{name: DatakitSource}``.

    Every entry is materializable — has a full :attr:`DatakitSource.normalize_steps`
    chain ready to run. Disabled entries (see TODOs above) are commented out of
    the module.
    """
    # Single-source families. Each exposes a ``<family>_normalize_steps()``
    # returning ``tuple[StepSpec, ...]``; the registry pairs the chain with
    # a rough token count.
    single_sources: tuple[_SourceRow, ...] = (
        # cp/biodiversity is carved out of common_pile (see common_pile.py)
        # because it needs page-stitching before normalize.
        ("cp/biodiversity", biodiversity_normalize_steps, 8.60),
        ("climblab-ja", climblab_ja_normalize_steps, 371.92),
        ("coderforge", coderforge_normalize_steps, 10.29),
        ("common-crawl-focus-2026-22", common_crawl_focus_normalize_steps, 49.702569456),
        ("davinci-dev/ctx-native", davinci_dev_ctx_native_normalize_steps, 57.57),
        ("davinci-dev/env-native", davinci_dev_env_native_normalize_steps, 2.58),
        ("eai-taxonomy-code-w-dclm", eai_taxonomy_code_normalize_steps, 591.90),
        ("finetranslations", finetranslations_normalize_steps, 3040.0),
        ("ghalogs/public", ghalogs_public_normalize_steps, GHALOGS_ROUGH_TOKENS_B),
        ("gpt-oss-rollouts", gpt_oss_rollouts_normalize_steps, 3.20),
        ("hplt_v3", hplt_v3_normalize_steps, 612.7),
        ("institutional_books", institutional_books_normalize_steps, 203.63),
        ("massive_function_calling", massive_normalize_steps, 11.39),
        ("molmo2-cap", molmo2_cap_normalize_steps, 0.36),
        (
            "nemotron_code_v2/content",
            nemotron_code_v2_content_normalize_steps,
            120.254379519,
        ),
        ("nemotron-terminal", nemotron_terminal_normalize_steps, 6.08),
        ("nsf_awards", nsf_awards_normalize_steps, 0.17),
        ("numinamath-1.5", numinamath_v1_5_normalize_steps, 0.40),
        ("numinamath-tir", numinamath_tir_normalize_steps, 0.08),
        ("sec-edgar", sec_edgar_normalize_steps, 334.90),
        ("superior-reasoning", superior_reasoning_normalize_steps, 7.08),
        ("svg", svgfind_creativecommons_normalize_steps, 8.95),
        ("swe-rebench-contree", swe_rebench_contree_normalize_steps, 182.60),
        ("swe-rebench-openhands", swe_rebench_openhands_normalize_steps, 2.47),
        ("swe-zero-12m", swe_zero_12m_normalize_steps, 106.91),
        ("synthetic-1", synthetic1_normalize_steps, 7.32),
    )

    # StarCoder2-Extras: 5 of 6 subsets advertised (ir_low_resource isn't in
    # the token-count-viewer set).
    starcoder2_extras = _rows_flat(
        starcoder2_extras_normalize_steps,
        {
            "starcoder2/documentation": 1.40,
            "starcoder2/ir_cpp": 39.01,
            "starcoder2/ir_python": 4.64,
            "starcoder2/ir_rust": 1.84,
            "starcoder2/kaggle": 1.38,
        },
    )

    # TheBioCollection: two synthetic bio/chem streams from one HF repo, each
    # staged and downloaded per-stream (see biocollection.py). Token counts are
    # the exact Marin-tokenizer (Llama-3) totals from each stream's tokenized
    # cache .stats.json.
    biocollection = _rows_flat(
        biocollection_normalize_steps,
        {
            "biocollection/free_text_stream": 33.186704843,
            "biocollection/instruction_stream": 18.123700603,
        },
    )

    # common-pile: 27 entries, each its own HF repo.
    common_pile = _rows_flat(
        common_pile_normalize_steps,
        {
            "cp/arxiv_abstracts": 0.54,
            "cp/arxiv_papers": 6.63,
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
        },
    )

    # FinePDFs: 19 language subsets, each staged per-language (no shared
    # family download).
    finepdfs = _rows_flat(
        finepdfs_normalize_steps,
        {
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
        },
    )

    # Exact counts, measured with marin-community/marin-tokenizer over the
    # normalized data: 1,221,318,442,892 tokens / 635,390,613 docs and
    # 65,538,632,427 / 31,179,056. Both came in above their chars/4 estimates
    # (by 11.7% and 6.0%), which is the bias to expect from the estimates below.
    dolma3_5_code = _rows_flat(
        dolma3_5_code_normalize_steps,
        {
            "dolma_code": 1221.32,
            "dolma_code_prose": 65.54,
        },
    )

    # dolma3.5_pool PDF subset, minus the finepdfs component we already ingest
    # separately (see dolma4pdfs.py). Exact count measured over the normalized
    # data with marin-community/marin-tokenizer: 1,804,002,448,556 tokens over
    # 137,132,279 documents.
    dolma4pdfs = _rows_flat(dolma4pdfs_normalize_steps, {"dolma4pdfs": 1804.002448556})

    # Nemotron v2 families: one family download shared across all subsets
    # (via ``@cache`` on ``download_nemotron_v2_step``); each subset has its
    # own normalize.
    nemotron_cc_v2 = _rows_nemotron(
        "nemotron_cc_v2",
        "nemotron_cc_v2",
        {
            "nemotron_cc_v2/diverse_qa": 676.57,
            "nemotron_cc_v2/high_quality": 608.96,
            "nemotron_cc_v2/high_quality_synthetic": 1223.46,
            "nemotron_cc_v2/medium_high_quality": 535.45,
            "nemotron_cc_v2/medium_quality": 2114.33,
            "nemotron_cc_v2/translated_diverse_qa": 592.85,
        },
    )
    nemotron_cc_v2_1 = _rows_nemotron(
        "nemotron_cc_v2_1",
        "nemotron_cc_v2_1",
        {
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
    )
    nemotron_cc_code_v1 = _rows_nemotron(
        "nemotron_cc_code_v1",
        "nemotron_cc_code_v1",
        {"nemotron_cc_code_v1/all": 399.41},
    )
    nemotron_cc_math_v1 = _rows_nemotron(
        "nemotron_cc_math_v1",
        "nemotron_cc_math_v1",
        {
            "nemotron_cc_math_v1/3": 78.90,
            "nemotron_cc_math_v1/4plus_mind": 72.20,
        },
    )
    nemotron_code_v2 = _rows_nemotron(
        "nemotron_pretraining_code_v2",
        "nemotron_code_v2",
        {
            "nemotron_code_v2/synthetic_code_review": 74.24,
            "nemotron_code_v2/synthetic_rewriting": 73.73,
            "nemotron_code_v2/synthetic_student_teacher": 25.20,
            "nemotron_code_v2/synthetic_question_answering": 233.03,
            "nemotron_code_v2/synthetic_transpilation": 27.78,
        },
    )
    nemotron_sft = _rows_nemotron(
        "nemotron_pretraining_sft_v1",
        "nemotron_sft",
        {
            "nemotron_sft/sft_code": 56.65,
            "nemotron_sft/sft_general": 85.20,
            "nemotron_sft/sft_math": 199.94,
        },
    )
    nemotron_specialized = _rows_nemotron(
        "nemotron_pretraining_specialized_v1",
        "nemotron_specialized",
        {
            "nemotron_specialized/infinibyte_reasoning": 18.69,
            "nemotron_specialized/math_textbooks": 25.59,
            "nemotron_specialized/rqa": 135.17,
            "nemotron_specialized/scientific_coding": 1.18,
            "nemotron_specialized/stem_sft": 81.20,
            "nemotron_specialized/wiki_rewrite": 7.26,
        },
    )
    nemotron_specialized_v1_1 = _rows_nemotron(
        "nemotron_pretraining_specialized_v1_1",
        "nemotron_specialized_v1_1",
        {
            "nemotron_specialized_v1_1/code_concepts": 7.03,
            "nemotron_specialized_v1_1/economics": 0.07,
            "nemotron_specialized_v1_1/formal_logic": 0.13,
            "nemotron_specialized_v1_1/multiple_choice": 1.56,
            "nemotron_specialized_v1_1/unconditional_algorithmic": 0.19,
        },
    )
    # v1.2 supersedes neither v1 nor v1.1 — it adds four new synthetic subsets
    # (fact-seeking, moral scenarios, generative and multiple-choice questions).
    # Its multiple_choice is a distinct, larger regeneration of the v1.1 subset
    # of the same name, so both are carried.
    nemotron_specialized_v1_2 = _rows_nemotron(
        NEMOTRON_PRETRAINING_SPECIALIZED_V1_2,
        "nemotron_specialized_v1_2",
        {
            "nemotron_specialized_v1_2/fact_seeking": 34.264249298,
            "nemotron_specialized_v1_2/generative": 0.657347056,
            "nemotron_specialized_v1_2/moral_scenarios": 0.014813270,
            "nemotron_specialized_v1_2/multiple_choice": 6.826340523,
        },
    )
    nemotron_legal = _rows_nemotron(
        NEMOTRON_PRETRAINING_LEGAL_V1,
        "nemotron_legal",
        {
            "nemotron_legal/california_code_of_regulations": 0.033064243,
            "nemotron_legal/case_law_summary": 0.027076493,
            "nemotron_legal/casehold": 3.839242351,
            "nemotron_legal/definition_classification": 0.001354030,
            "nemotron_legal/diversity_jurisdiction": 0.000837717,
            "nemotron_legal/ecfr": 0.122605305,
            "nemotron_legal/ecfr_qa": 0.549815436,
            "nemotron_legal/function_of_decision": 0.023167261,
            "nemotron_legal/globalcit": 0.007366386,
            "nemotron_legal/legalbench_cuad_v2": 0.047987198,
            "nemotron_legal/nycourts_judicial_ethics_opinions": 0.004178264,
        },
    )

    # locuslab Safety Pretraining: moral_education, safeweb, and refuseweb
    # (fineweb_annotated is a score-annotated copy of FineWeb itself and is
    # excluded to avoid double-counting that corpus). Token counts measured
    # by tokenizing every subset with the marin-community tokenizer (see
    # ``scripts/datakit/tokenize_safety_pt.py``).
    safety_pretraining = _rows_flat(
        safety_pretraining_normalize_steps,
        {
            "safety_pt/moral_education/score_4_morals": 4.41,
            "safety_pt/moral_education/score_5_morals": 1.80,
            "safety_pt/safeweb/score_1_rephrased": 6.04,
            "safety_pt/safeweb/score_3_rephrased": 2.96,
            "safety_pt/safeweb/score_4_rephrased": 4.20,
            "safety_pt/safeweb/score_5_rephrased": 3.54,
            "safety_pt/refuseweb/score_4_refusal": 3.03,
            "safety_pt/refuseweb/score_5_refusal": 1.12,
        },
    )

    all_rows: tuple[_SourceRow, ...] = (
        *single_sources,
        *starcoder2_extras,
        *biocollection,
        *common_pile,
        *finepdfs,
        *dolma3_5_code,
        *dolma4pdfs,
        *nemotron_cc_v2,
        *nemotron_cc_v2_1,
        *nemotron_cc_code_v1,
        *nemotron_cc_math_v1,
        *nemotron_code_v2,
        *nemotron_sft,
        *nemotron_specialized,
        *nemotron_specialized_v1_1,
        *nemotron_specialized_v1_2,
        *nemotron_legal,
        *safety_pretraining,
    )

    entries = {
        name: DatakitSource(name=name, normalize_steps=factory(), rough_token_count_b=count)
        for name, factory, count in all_rows
    }
    assert len(entries) == len(all_rows), "duplicate marin_name across families"
    return entries
