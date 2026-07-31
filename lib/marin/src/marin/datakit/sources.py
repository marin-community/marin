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
from marin.datakit.download.agenttrove import agenttrove_normalize_steps
from marin.datakit.download.biocollection import biocollection_normalize_steps
from marin.datakit.download.biocorpus import biocorpus_normalize_steps
from marin.datakit.download.biodiversity import biodiversity_normalize_steps
from marin.datakit.download.climblab_ja import climblab_ja_normalize_steps
from marin.datakit.download.coderforge import coderforge_normalize_steps
from marin.datakit.download.common_crawl_focus import common_crawl_focus_normalize_steps
from marin.datakit.download.common_pile import common_pile_normalize_steps
from marin.datakit.download.davinci_dev import (
    davinci_dev_ctx_native_normalize_steps,
    davinci_dev_env_native_normalize_steps,
)
from marin.datakit.download.diagnostic_logs import ghalogs_public_normalize_steps
from marin.datakit.download.docx_corpus import docx_corpus_normalize_steps
from marin.datakit.download.dolma3_5_code import dolma3_5_code_prose_normalize_steps
from marin.datakit.download.dolma4pdfs import dolma4pdfs_normalize_steps
from marin.datakit.download.eai_taxonomy_code import eai_taxonomy_code_normalize_steps
from marin.datakit.download.finepdfs import finepdfs_normalize_steps
from marin.datakit.download.finetranslations import finetranslations_normalize_steps
from marin.datakit.download.glm_kernelgym_rollouts import glm_kernelgym_rollouts_normalize_steps
from marin.datakit.download.gpt_oss_rollouts import gpt_oss_rollouts_normalize_steps
from marin.datakit.download.hplt import hplt_v3_normalize_steps
from marin.datakit.download.identity_data import identity_data_content_normalize_steps
from marin.datakit.download.institutional_books import institutional_books_normalize_steps
from marin.datakit.download.massive import massive_normalize_steps
from marin.datakit.download.molmo2_cap import molmo2_cap_normalize_steps
from marin.datakit.download.nemotron_code_v1_content import nemotron_code_v1_content_normalize_steps
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
from marin.datakit.download.penfever_rollouts import penfever_rollouts_normalize_steps
from marin.datakit.download.sec_edgar import sec_edgar_normalize_steps
from marin.datakit.download.stack_v3 import stack_v3_normalize_steps
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
    """Ordered step chain ending with ``normalize``. Earlier steps include a
    download or depend on one transitively, and may preprocess the source."""

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
        # Exact count from the tokenized cache .stats.json, measured with
        # marin-community/marin-tokenizer over the normalized artifact:
        # 8,957,298,636 tokens / 781,076 docs. The row chain behind that doc count
        # is 1,696,847 → 997,026 past the proprietary-teacher filter → 869,901
        # with a non-null transcript → 781,076 after exact dedup.
        ("agenttrove", agenttrove_normalize_steps, 8.957298636),
        # Measured with marin-community/marin-tokenizer:
        # 9,138,977,526 tokens / 21,138,120 documents.
        ("biocorpus", biocorpus_normalize_steps, 9.138977526),
        # cp/biodiversity is carved out of common_pile (see common_pile.py)
        # because it needs page-stitching before normalize.
        ("cp/biodiversity", biodiversity_normalize_steps, 8.60),
        ("climblab-ja", climblab_ja_normalize_steps, 371.92),
        ("coderforge", coderforge_normalize_steps, 10.29),
        ("common-crawl-focus-2026-22", common_crawl_focus_normalize_steps, 49.702569456),
        ("davinci-dev/ctx-native", davinci_dev_ctx_native_normalize_steps, 57.57),
        ("davinci-dev/env-native", davinci_dev_env_native_normalize_steps, 2.58),
        # Exact count measured with marin-community/marin-tokenizer:
        # 1,497,301,429 tokens / 242,086 docs.
        ("docx-corpus/en", docx_corpus_normalize_steps, 1.497301429),
        # Exact count measured with marin-community/marin-tokenizer:
        # 65,538,632,427 tokens / 31,179,056 docs.
        ("dolma_code_prose", dolma3_5_code_prose_normalize_steps, 65.54),
        ("eai-taxonomy-code-w-dclm", eai_taxonomy_code_normalize_steps, 591.90),
        ("finetranslations", finetranslations_normalize_steps, 3040.0),
        # Exact count from the tokenized cache .stats.json, measured with
        # marin-community/marin-tokenizer over the normalized artifact:
        # 253,343,866,746 tokens / 2,613,421 documents.
        ("ghalogs/public", ghalogs_public_normalize_steps, 253.343866746),
        # Exact count from the tokenized cache .stats.json, measured with
        # marin-community/marin-tokenizer over the normalized artifact under the
        # final-turn truncation filter: 64,054,289 tokens / 2,835 docs.
        ("glm-5.2-kernelgym-rollouts", glm_kernelgym_rollouts_normalize_steps, 0.064054289),
        ("gpt-oss-rollouts", gpt_oss_rollouts_normalize_steps, 3.20),
        ("hplt_v3", hplt_v3_normalize_steps, 612.7),
        ("identity-data/content", identity_data_content_normalize_steps, 0.061711380),
        ("institutional_books", institutional_books_normalize_steps, 203.63),
        ("massive_function_calling", massive_normalize_steps, 11.39),
        ("molmo2-cap", molmo2_cap_normalize_steps, 0.36),
        # Rough count: no tokenized cache exists yet, so this scales v2's measured
        # rate (120.254379519 B tokens / 132,666,330 present docs ~= 906 tokens/doc)
        # to v1's 513,109,851 present docs. Replace with the measured
        # marin-community/marin-tokenizer total once the normalized cache is built.
        (
            "nemotron_code_v1/content",
            nemotron_code_v1_content_normalize_steps,
            465.0,
        ),
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
        # Exact count measured with marin-community/marin-tokenizer:
        # 3,363,007,313,642 tokens / 172,898,790 docs.
        ("stack-v3", stack_v3_normalize_steps, 3363.007313642),
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

    # common-pile: 26 entries, each its own HF repo.
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

    # Public, verifier-valid Penfever rollouts generated by OT-Agent. Rough
    # estimates multiply pinned HF row counts by observed mean trajectory
    # lengths: 9,996 tokens for MiniMax, 8,377 for Qwen 32k and GLM Terminus-2,
    # and 3,294 for Qwen 131k OpenCode.
    penfever_rollouts = _rows_flat(
        penfever_rollouts_normalize_steps,
        {
            "penfever-traces/minimax-m27-131k/code-contests-noblock": 0.060932514,
            "penfever-traces/minimax-m27-131k/exp_rle_minimal_instructions-v3": 0.002318954,
            "penfever-traces/minimax-m27-131k/exp_rpt_codenet-python-v2": 0.094027585,
            "penfever-traces/minimax-m27-131k/exp_rpt_crosscodeeval-csharp-v4": 0.016932362,
            "penfever-traces/minimax-m27-131k/exp_rpt_curriculum-easy": 0.005137682,
            "penfever-traces/minimax-m27-131k/exp_rpt_curriculum-medium": 0.005087705,
            "penfever-traces/minimax-m27-131k/exp_rpt_e2egit-large": 0.049727568,
            "penfever-traces/minimax-m27-131k/exp_rpt_e2egit-v2": 0.004967759,
            "penfever-traces/minimax-m27-131k/exp_rpt_ghactions-v3": 0.098305655,
            "penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v2": 0.044749814,
            "penfever-traces/minimax-m27-131k/exp_rpt_methods2test-large-v3": 0.044509922,
            "penfever-traces/minimax-m27-131k/exp_rpt_nemotron-cpp": 0.049977455,
            "penfever-traces/minimax-m27-131k/exp_rpt_nemotron-junit": 0.039672104,
            "penfever-traces/minimax-m27-131k/exp_rpt_pr": 0.044459944,
            "penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-large": 0.049017888,
            "penfever-traces/minimax-m27-131k/exp_rpt_pymethods2test-v3": 0.004647903,
            "penfever-traces/minimax-m27-131k/exp_rpt_stack-bash-v3": 0.093467837,
            "penfever-traces/minimax-m27-131k/exp_rpt_stack-junit-v6": 0.008716068,
            "penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-large": 0.049967460,
            "penfever-traces/minimax-m27-131k/exp_rpt_stack-pytest-v2": 0.004997746,
            "penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-large": 0.048098303,
            "penfever-traces/minimax-m27-131k/exp_rpt_unitsyn-python-v3": 0.004787840,
            "penfever-traces/minimax-m27-131k/inferredbugs-sandboxes-verifier": 0.069158803,
            "penfever-traces/minimax-m27-131k/llm-verifier-freelancer": 0.073926652,
            "penfever-traces/minimax-m27-131k/mix_h10_reward_binary-v2": 0.028437172,
            "penfever-traces/minimax-m27-131k/mix_h10_reward_proportional-v2": 0.028477154,
            "penfever-traces/minimax-m27-131k/mix_h10_reward_staged-v2": 0.037233204,
            "penfever-traces/minimax-m27-131k/mix_h11_single_skill_only-v2": 0.028467159,
            "penfever-traces/minimax-m27-131k/mix_h1_struggle_zone-v2": 0.030596198,
            "penfever-traces/minimax-m27-131k/mix_h2_language_balanced-v2": 0.045029687,
            "penfever-traces/minimax-m27-131k/mix_h2_language_proportional": 0.041301369,
            "penfever-traces/minimax-m27-131k/mix_h4_binary_easy": 0.019960996,
            "penfever-traces/minimax-m27-131k/mix_h8_original_tests-v2": 0.028397190,
            "penfever-traces/minimax-m27-131k/nemotron-code-oracle-filtered": 0.108561029,
            "penfever-traces/minimax-m27-131k/nemotron-gym-agent-calendar": 0.032645274,
            "penfever-traces/minimax-m27-131k/nemotron-gym-agent-workplace-v2": 0.002968661,
            "penfever-traces/minimax-m27-131k/nemotron-gym-competitive-coding": 0.082352851,
            "penfever-traces/minimax-m27-131k/nemotron-gym-identity-following-v2": 0.216482346,
            "penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-calendar": 0.082892608,
            "penfever-traces/minimax-m27-131k/nemotron-gym-instruction-following-structured": 0.094197508,
            "penfever-traces/minimax-m27-131k/nemotron-gym-knowledge-web-search-mcqa": 0.029086879,
            "penfever-traces/minimax-m27-131k/nemotron-gym-math-advanced-calculations-v3": 0.052416355,
            "penfever-traces/minimax-m27-131k/nl2bash-tasks-cleaned-oracle": 0.015692921,
            "penfever-traces/minimax-m27-131k/selfinstruct-naive-sandboxes-2-verified": 0.095576886,
            "penfever-traces/minimax-m27-131k/swegym-tasks-patched-validated-v5": 0.021630243,
            "penfever-traces/qwen35-122b-32k/code-contests-noblock": 0.058066906,
            "penfever-traces/qwen35-122b-32k/exp_rle_minimal_instructions-v3": 0.002596760,
            "penfever-traces/qwen35-122b-32k/exp_rpt_codenet-python-v2": 0.077810660,
            "penfever-traces/qwen35-122b-32k/exp_rpt_crosscodeeval-csharp-v4": 0.013402633,
            "penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-easy": 0.004272089,
            "penfever-traces/qwen35-122b-32k/exp_rpt_curriculum-medium": 0.004414492,
            "penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-large": 0.031814499,
            "penfever-traces/qwen35-122b-32k/exp_rpt_e2egit-v2": 0.004179946,
            "penfever-traces/qwen35-122b-32k/exp_rpt_ghactions-v3": 0.084679509,
            "penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v2": 0.041028809,
            "penfever-traces/qwen35-122b-32k/exp_rpt_methods2test-large-v3": 0.022415903,
            "penfever-traces/qwen35-122b-32k/exp_rpt_nemotron-junit": 0.016677901,
            "penfever-traces/qwen35-122b-32k/exp_rpt_pr": 0.052965529,
            "penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-large": 0.039102181,
            "penfever-traces/qwen35-122b-32k/exp_rpt_pymethods2test-v3": 0.003987283,
            "penfever-traces/qwen35-122b-32k/exp_rpt_stack-bash-v3": 0.087963154,
            "penfever-traces/qwen35-122b-32k/exp_rpt_stack-junit-v6": 0.002781046,
            "penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-large": 0.045376288,
            "penfever-traces/qwen35-122b-32k/exp_rpt_stack-pytest-v2": 0.004732805,
            "penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-large": 0.039479130,
            "penfever-traces/qwen35-122b-32k/exp_rpt_unitsyn-python-v3": 0.003978907,
            "penfever-traces/qwen35-122b-32k/inferredbugs-sandboxes-verifier": 0.066996410,
            "penfever-traces/qwen35-122b-32k/llm-verifier-freelancer": 0.023303828,
            "penfever-traces/qwen35-122b-32k/mix_h10_reward_binary-v2": 0.012539838,
            "penfever-traces/qwen35-122b-32k/mix_h10_reward_proportional-v2": 0.024359285,
            "penfever-traces/qwen35-122b-32k/mix_h10_reward_staged-v2": 0.030951705,
            "penfever-traces/qwen35-122b-32k/mix_h11_single_skill_only-v2": 0.023488114,
            "penfever-traces/qwen35-122b-32k/mix_h1_struggle_zone-v2": 0.028313062,
            "penfever-traces/qwen35-122b-32k/mix_h2_language_balanced-v2": 0.047085124,
            "penfever-traces/qwen35-122b-32k/mix_h2_language_proportional": 0.042477969,
            "penfever-traces/qwen35-122b-32k/mix_h4_binary_easy": 0.017331279,
            "penfever-traces/qwen35-122b-32k/mix_h8_original_tests-v2": 0.024685974,
            "penfever-traces/qwen35-122b-32k/nemotron-code-oracle-filtered": 0.091883424,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-agent-calendar": 0.027517280,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-agent-workplace-v2": 0.002395721,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-competitive-coding": 0.093441480,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-identity-following-v2": 0.008368269,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-calendar": 0.065999590,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-structured": 0.074979354,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-instruction-following-v2": 0.210982569,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-mcqa": 0.067482256,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-openqa-v2": 0.053602154,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-knowledge-web-search-mcqa": 0.024376038,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-math-advanced-calculations-v3": 0.044320831,
            "penfever-traces/qwen35-122b-32k/nemotron-gym-safety-v2": 0.162816858,
            "penfever-traces/qwen35-122b-32k/nemotron-math-oracle-filtered": 0.115438551,
            "penfever-traces/qwen35-122b-32k/nl2bash-tasks-cleaned-oracle": 0.013117827,
            "penfever-traces/qwen35-122b-32k/selfinstruct-naive-sandboxes-2-verified": 0.011224705,
            "penfever-traces/qwen35-122b-32k/swesmith-oracle-filtered": 0.007237422,
            "penfever-traces/qwen35-122b-131k-opencode/code-contests-noblock": 0.016186494,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rle_adversarial": 0.016015208,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-csharp-v4": 0.005823712,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_crosscodeeval-java": 0.006979890,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-easy": 0.001646977,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-hard": 0.000072467,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_curriculum-medium": 0.001511925,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-large": 0.013488745,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_e2egit-v2": 0.001643683,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_ghactions-v3": 0.026980784,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_issue": 0.000988186,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_methods2test-large-v3": 0.003422419,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_multifile": 0.000335983,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-cpp-v2": 0.002628576,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_nemotron-junit": 0.001254997,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pr": 0.015389357,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-large": 0.016150260,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_pymethods2test-v3": 0.001607450,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-junit-v6": 0.002826213,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-large": 0.004525894,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_stack-pytest-v2": 0.001623920,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-large": 0.007279640,
            "penfever-traces/qwen35-122b-131k-opencode/exp_rpt_unitsyn-python-v3": 0.000428214,
            "penfever-traces/qwen35-122b-131k-opencode/inferredbugs-sandboxes-verifier": 0.019193875,
            "penfever-traces/qwen35-122b-131k-opencode/llm-verifier-freelancer": 0.029530305,
            "penfever-traces/qwen35-122b-131k-opencode/mix_h4_binary_easy": 0.006538500,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-code-oracle-filtered": 0.018736015,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-calendar": 0.010758056,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-workplace-v2": 0.000975011,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-identity-following-v2": 0.028584940,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-instruction-following-structured": 0.030024398,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-knowledge-web-search-mcqa": 0.009595290,
            "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-math-advanced-calculations-v3": 0.017369024,
            "penfever-traces/qwen35-122b-131k-opencode/nl2bash-tasks-cleaned-oracle": 0.002305768,
            "penfever-traces/qwen35-122b-131k-opencode/selfinstruct-naive-sandboxes-2-verified": 0.030126510,
            "penfever-traces/glm52-terminus2/exp_rpt_crosscodeeval-csharp-v4": 0.014809909,
            "penfever-traces/glm52-terminus2/exp_rpt_curriculum-easy": 0.004263713,
            "penfever-traces/glm52-terminus2/exp_rpt_curriculum-medium": 0.004205076,
            "penfever-traces/glm52-terminus2/exp_rpt_e2egit-large": 0.041874851,
            "penfever-traces/glm52-terminus2/exp_rpt_e2egit-v2": 0.004188323,
            "penfever-traces/glm52-terminus2/exp_rpt_nemotron-cpp-v2": 0.006701316,
            "penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-large-v2": 0.021360446,
            "penfever-traces/glm52-terminus2/exp_rpt_stack-pytest-v2": 0.004188323,
            "penfever-traces/glm52-terminus2/exp_rpt_unitsyn-python-v3": 0.004179946,
            "penfever-traces/glm52-terminus2/nemotron-gym-agent-calendar": 0.020480898,
            "penfever-traces/glm52-terminus2/nemotron-gym-instruction-following-structured": 0.078966637,
            "penfever-traces/glm52-terminus2/nemotron-gym-knowledge-web-search-mcqa": 0.024317402,
            "penfever-traces/glm52-terminus2/nl2bash-tasks-cleaned-oracle": 0.013142957,
        },
    )

    # locuslab Safety Pretraining: moral_education, safeweb, and refuseweb
    # (fineweb_annotated is a score-annotated copy of FineWeb itself and is
    # excluded to avoid double-counting that corpus). Token counts were measured
    # by tokenizing every subset with the marin-community tokenizer.
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
        *penfever_rollouts,
        *safety_pretraining,
    )

    entries = {
        name: DatakitSource(name=name, normalize_steps=factory(), rough_token_count_b=count)
        for name, factory, count in all_rows
    }
    assert len(entries) == len(all_rows), "duplicate marin_name across families"
    return entries
