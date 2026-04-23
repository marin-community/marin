# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical source registry for the Datakit.

Each ``DatakitSource`` is one mixture component: a (hf_dataset_id, revision)
pinned at the revision, plus the schema hints (id_field, text_field) the
ferry's normalize step needs to preserve provenance.
"""

from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class DatakitSource:
    """One mixture component in the testbed.

    The pair (``hf_dataset_id``, ``revision``) uniquely identifies the upstream
    bytes; ``staged_path`` — if set — points at an existing copy under
    ``MARIN_PREFIX`` so the ferry's download step becomes verify-only.
    """

    name: str
    """Mixture-component key, e.g. ``"nemotron_cc_v2_1/high_quality"``. Stable."""

    hf_dataset_id: str
    """HuggingFace repo ID, e.g. ``"HuggingFaceFW/fineweb-edu"``.

    Empty string for API-sourced datasets (e.g. ``nsf_awards``) — those are
    carried in the registry for completeness but require a bespoke download
    step and will not flow through the generic ferry DAG.
    """

    revision: str | None = None
    """Pinned HF commit SHA, or ``None`` if the source isn't yet pinned in code.

    ``None`` entries are included because the token-count-viewer lists them, but
    the ferry cannot materialize them until someone adds a download module
    that fixes a revision. Filter with :func:`pinned_sources` for the subset
    actually runnable today.
    """

    hf_urls_glob: tuple[str, ...] | None = None
    """If set, restrict the download to matching file paths within the repo."""

    staged_path: str | None = None
    """Relative path under ``MARIN_PREFIX`` if the raw dump is pre-staged.

    Mirrors ``override_output_path`` in the corresponding ``download_hf_step``
    call so verify-only steps point at the same bytes the staging pipeline wrote.
    Multiple sources may share one ``staged_path`` (e.g. every Nemotron-CC v2.1
    subset lives under the same family download); the ferry groups them into a
    single download step.
    """

    data_subdir: str = ""
    """Subdirectory within the staged download containing the actual data files.

    Empty string means the download directory itself holds the parquet shards.
    Used as the relative ``input_path`` override for the per-source normalize
    step — e.g. ``"High-Quality"`` for a Nemotron-CC v2.1 subset.
    """

    id_field: str = "id"
    """Raw-record field copied into ``source_id`` during normalize"""

    text_field: str = "text"

    file_extensions: tuple[str, ...] = (".parquet",)

    rough_token_count_b: float | None = None
    """Approximate token count in billions (Llama-3 tokenizer).

    Used as the initial per-source mixing weight. ``None`` means unknown —
    replace with a measured value from the tokenize step's stats later.
    """


@cache
def all_sources() -> dict[str, DatakitSource]:
    """Return the canonical source set as ``{name: DatakitSource}``"""
    entries: tuple[DatakitSource, ...] = (
        # ---- Code ----
        DatakitSource(
            name="coderforge",
            hf_dataset_id="togethercomputer/CoderForge-Preview",
            revision="060fca9",
            rough_token_count_b=10.29,
            staged_path="raw/coderforge-preview_ad26b119",
        ),
        # ---- Common Corpus (English filter) ----
        # TODO: staged dir `raw/common_corpus_english-b78a5c1` is missing its
        # .executor_status marker — we can't confirm the staging run completed
        # cleanly. Re-enable once the staging is re-verified.
        # DatakitSource(
        #     name="common_corpus/english",
        #     hf_dataset_id="PleIAs/common_corpus",
        #     revision="b78a5c1",
        #     rough_token_count_b=1015.39,
        #     staged_path="raw/common_corpus_english-b78a5c1",
        # ),
        # ---- common-pile: 28 filtered subsets ----
        DatakitSource(
            name="cp/arxiv_abstracts",
            hf_dataset_id="common-pile/arxiv_abstracts_filtered",
            revision="f1d7a9a",
            staged_path="raw/common_pile/arxiv_abstracts_filtered-f1d7a9a",
            rough_token_count_b=0.54,
        ),
        DatakitSource(
            name="cp/arxiv_papers",
            hf_dataset_id="common-pile/arxiv_papers_filtered",
            revision="033cf7f",
            staged_path="raw/common_pile/arxiv_papers_filtered-033cf7f",
            rough_token_count_b=6.63,
        ),
        DatakitSource(
            name="cp/biodiversity",
            hf_dataset_id="common-pile/biodiversity_heritage_library_filtered",
            revision="0486ed6",
            staged_path="raw/common_pile/biodiversity_heritage_library_filtered-0486ed6",
            rough_token_count_b=8.60,
        ),
        DatakitSource(
            name="cp/caselaw",
            hf_dataset_id="common-pile/caselaw_access_project_filtered",
            revision="50e1961",
            staged_path="raw/common_pile/caselaw_access_project_filtered-50e1961",
            rough_token_count_b=17.55,
        ),
        DatakitSource(
            name="cp/data_provenance",
            hf_dataset_id="common-pile/data_provenance_initiative_filtered",
            revision="8f5afcf",
            staged_path="raw/common_pile/data_provenance_initiative_filtered-8f5afcf",
            rough_token_count_b=0.82,
        ),
        DatakitSource(
            name="cp/doab",
            hf_dataset_id="common-pile/doab_filtered",
            revision="defb24c",
            staged_path="raw/common_pile/doab_filtered-defb24c",
            rough_token_count_b=2.93,
        ),
        DatakitSource(
            name="cp/foodista",
            hf_dataset_id="common-pile/foodista_filtered",
            revision="bf2c7aa",
            staged_path="raw/common_pile/foodista_filtered-bf2c7aa",
            rough_token_count_b=0.02,
        ),
        DatakitSource(
            name="cp/github_archive",
            hf_dataset_id="common-pile/github_archive_filtered",
            revision="52282fe",
            staged_path="raw/common_pile/github_archive_filtered-52282fe",
            rough_token_count_b=10.26,
        ),
        DatakitSource(
            name="cp/library_of_congress",
            hf_dataset_id="common-pile/library_of_congress_filtered",
            revision="56725c7",
            staged_path="raw/common_pile/library_of_congress_filtered-56725c7",
            rough_token_count_b=8.06,
        ),
        DatakitSource(
            name="cp/libretexts",
            hf_dataset_id="common-pile/libretexts_filtered",
            revision="70388bc",
            staged_path="raw/common_pile/libretexts_filtered-70388bc",
            rough_token_count_b=0.08,
        ),
        DatakitSource(
            name="cp/news",
            hf_dataset_id="common-pile/news_filtered",
            revision="59aaa8f",
            staged_path="raw/common_pile/news_filtered-59aaa8f",
            rough_token_count_b=0.05,
        ),
        DatakitSource(
            name="cp/oercommons",
            hf_dataset_id="common-pile/oercommons_filtered",
            revision="506b615",
            staged_path="raw/common_pile/oercommons_filtered-506b615",
            rough_token_count_b=0.01,
        ),
        DatakitSource(
            name="cp/peS2o",
            hf_dataset_id="common-pile/peS2o_filtered",
            revision="2977475",
            staged_path="raw/common_pile/peS2o_filtered-2977475",
            rough_token_count_b=40.74,
        ),
        DatakitSource(
            name="cp/peps",
            hf_dataset_id="common-pile/python_enhancement_proposals_filtered",
            revision="5821709",
            staged_path="raw/common_pile/python_enhancement_proposals_filtered-5821709",
            rough_token_count_b=0.003,
        ),
        DatakitSource(
            name="cp/pre_1929_books",
            hf_dataset_id="common-pile/pre_1929_books_filtered",
            revision="23f9d96",
            staged_path="raw/common_pile/pre_1929_books_filtered-23f9d96",
            rough_token_count_b=10.57,
        ),
        DatakitSource(
            name="cp/pressbooks",
            hf_dataset_id="common-pile/pressbooks_filtered",
            revision="1a1d3b5",
            staged_path="raw/common_pile/pressbooks_filtered-1a1d3b5",
            rough_token_count_b=0.13,
        ),
        DatakitSource(
            name="cp/project_gutenberg",
            hf_dataset_id="common-pile/project_gutenberg_filtered",
            revision="3cdf687",
            staged_path="raw/common_pile/project_gutenberg_filtered-3cdf687",
            rough_token_count_b=4.91,
        ),
        DatakitSource(
            name="cp/public_domain_review",
            hf_dataset_id="common-pile/public_domain_review_filtered",
            revision="efc7f21",
            staged_path="raw/common_pile/public_domain_review_filtered-efc7f21",
            rough_token_count_b=0.002,
        ),
        DatakitSource(
            name="cp/pubmed",
            hf_dataset_id="common-pile/pubmed_filtered",
            revision="c156f05",
            staged_path="raw/common_pile/pubmed_filtered-c156f05",
            rough_token_count_b=38.08,
        ),
        DatakitSource(
            name="cp/regulations",
            hf_dataset_id="common-pile/regulations_filtered",
            revision="3327364",
            staged_path="raw/common_pile/regulations_filtered-3327364",
            rough_token_count_b=1.28,
        ),
        DatakitSource(
            name="cp/stackexchange",
            hf_dataset_id="common-pile/stackexchange_filtered",
            revision="c0ac737",
            staged_path="raw/common_pile/stackexchange_filtered-c0ac737",
            rough_token_count_b=21.89,
        ),
        DatakitSource(
            name="cp/stackv2_code",
            hf_dataset_id="common-pile/stackv2",
            revision="d0e3266",
            staged_path="raw/common_pile/stackv2-d0e3266",
            rough_token_count_b=352.76,
        ),
        DatakitSource(
            name="cp/ubuntu_irc",
            hf_dataset_id="common-pile/ubuntu_irc_filtered",
            revision="84f88c9",
            staged_path="raw/common_pile/ubuntu_irc_filtered-84f88c9",
            rough_token_count_b=1.76,
        ),
        DatakitSource(
            name="cp/uk_hansard",
            hf_dataset_id="common-pile/uk_hansard_filtered",
            revision="c88adc4",
            staged_path="raw/common_pile/uk_hansard_filtered-c88adc4",
            rough_token_count_b=2.13,
        ),
        DatakitSource(
            name="cp/usgpo",
            hf_dataset_id="common-pile/usgpo_filtered",
            revision="b150cc2",
            staged_path="raw/common_pile/usgpo_filtered-b150cc2",
            rough_token_count_b=7.78,
        ),
        DatakitSource(
            name="cp/uspto",
            hf_dataset_id="common-pile/uspto_filtered",
            revision="13894c5",
            staged_path="raw/common_pile/uspto_filtered-13894c5",
            rough_token_count_b=142.41,
        ),
        DatakitSource(
            name="cp/wikiteam",
            hf_dataset_id="common-pile/wikiteam_filtered",
            revision="f4ed055",
            staged_path="raw/common_pile/wikiteam_filtered-f4ed055",
            rough_token_count_b=2.97,
        ),
        DatakitSource(
            name="cp/youtube",
            hf_dataset_id="common-pile/youtube_filtered",
            revision="dff8c8a",
            staged_path="raw/common_pile/youtube_filtered-dff8c8a",
            rough_token_count_b=4.07,
        ),
        # ---- FinePDFs (19 subsets: English + 18 multilingual) ----
        DatakitSource(
            name="finepdfs",
            hf_dataset_id="HuggingFaceFW/finepdfs",
            revision="89f5411",
            rough_token_count_b=1186.47,
            staged_path="raw/finepdfs_eng_Latn_1a6e7def",
        ),
        *(
            DatakitSource(
                name=f"finepdfs/{lang}",
                hf_dataset_id="HuggingFaceFW/finepdfs",
                revision="89f5411",
                staged_path=f"raw/finepdfs_{lang}_{sha}",
                rough_token_count_b=tok,
            )
            for lang, sha, tok in (
                ("arb_Arab", "d45e1edc", 29.72),
                ("ces_Latn", "b3371d5c", 29.83),
                ("cmn_Hani", "07be0dc4", 32.97),
                ("deu_Latn", "ce5aaacd", 177.10),
                ("fra_Latn", "35a75de8", 164.75),
                ("hun_Latn", "e906b5de", 37.44),
                ("ind_Latn", "8ba9e288", 20.32),
                ("ita_Latn", "c8fa2fa7", 94.79),
                ("jpn_Jpan", "7b65dbec", 115.87),
                ("nld_Latn", "a60bc417", 46.97),
                ("pol_Latn", "2558940c", 54.40),
                ("por_Latn", "cdf5ff50", 94.69),
                ("ron_Latn", "c41b1d50", 22.61),
                ("rus_Cyrl", "6e14b64d", 146.95),
                ("spa_Latn", "89be7172", 216.74),
                ("swe_Latn", "eac6cc36", 25.34),
                ("tha_Thai", "2921d58a", 17.40),
                ("ukr_Cyrl", "be1fb148", 25.53),
            )
        ),
        # ---- FineTranslations ----
        # TODO: staging at `raw/finetranslations_d17a789b` is still in progress;
        # no provenance.json and no .executor_status=SUCCESS yet. Re-enable once
        # staging lands and add a download module for HuggingFaceFW/finetranslations
        # so pinned_sources() can include these entries.
        # TODO: both entries point at the same physical dump — the upstream is a
        # parallel corpus of original multilingual text + machine-translated
        # English. Splitting into /multilingual and /web needs different
        # text_field, hf_urls_glob, or data_subdir so the two accounting slices
        # don't normalize to identical rows and double-count the mixture.
        # DatakitSource(
        #     name="finetranslations/multilingual",
        #     hf_dataset_id="HuggingFaceFW/finetranslations",
        #     revision=None,
        #     rough_token_count_b=1520.07,
        #     staged_path="raw/finetranslations_d17a789b",
        # ),
        # DatakitSource(
        #     name="finetranslations/web",
        #     hf_dataset_id="HuggingFaceFW/finetranslations",
        #     revision=None,
        #     rough_token_count_b=1520.07,
        #     staged_path="raw/finetranslations_d17a789b",
        # ),
        # ---- gpt-oss rollouts ----
        DatakitSource(
            name="gpt-oss-rollouts",
            hf_dataset_id="andyrdt/gpt-oss-20b-rollouts",
            revision="f47b4a2",
            rough_token_count_b=3.20,
            staged_path="raw/gpt-oss-20b-rollouts_58b022a7",
        ),
        # ---- HPLT v3 ----
        # TODO: add a download module for HPLT/HPLT3.0 and pin the revision.
        # The staged dir has no provenance.json and the download_hplt_v3_step
        # function that produced it has been removed from the tree.
        DatakitSource(
            name="hplt_v3",
            hf_dataset_id="HPLT/HPLT3.0",
            revision=None,
            rough_token_count_b=612.70,
            staged_path="raw/hplt_v3_2a08d6f3",
        ),
        # ---- Institutional Books ----
        DatakitSource(
            name="institutional_books",
            hf_dataset_id="institutional/institutional-books-1.0",
            revision="d2f504a",
            staged_path="raw/institutional-books-d2f504a",
            rough_token_count_b=203.63,
        ),
        # ---- Nemotron Terminal Corpus ----
        DatakitSource(
            name="nemotron-terminal",
            hf_dataset_id="nvidia/Nemotron-Terminal-Corpus",
            revision="a1667c4",
            rough_token_count_b=6.08,
            staged_path="raw/nemotron-terminal-corpus_c68d0061",
        ),
        # ---- Nemotron-CC Code v1 ----
        DatakitSource(
            name="nemotron_cc_code_v1/all",
            hf_dataset_id="nvidia/Nemotron-CC-Code-v1",
            revision="5c5bebc",
            staged_path="raw/nemotron_cc_code_v1-c55cd9",
            data_subdir="data",
            rough_token_count_b=399.41,
        ),
        # ---- Nemotron-CC Math v1 (2 of 3 available subsets in CSV) ----
        DatakitSource(
            name="nemotron_cc_math_v1/3",
            hf_dataset_id="nvidia/Nemotron-CC-Math-v1",
            revision="397a250",
            staged_path="raw/nemotron_cc_math_v1-322fe4",
            data_subdir="3",
            rough_token_count_b=78.90,
        ),
        DatakitSource(
            name="nemotron_cc_math_v1/4plus_mind",
            hf_dataset_id="nvidia/Nemotron-CC-Math-v1",
            revision="397a250",
            staged_path="raw/nemotron_cc_math_v1-322fe4",
            data_subdir="4plus_MIND",
            rough_token_count_b=72.20,
        ),
        # ---- Nemotron-CC v2 (6 subsets) ----
        *(
            DatakitSource(
                name=f"nemotron_cc_v2/{key}",
                hf_dataset_id="nvidia/Nemotron-CC-v2",
                revision="229a2e7",
                staged_path="raw/nemotron_cc_v2-674913",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, cat, tok in (
                ("diverse_qa", "Diverse-QA", "web", 676.57),
                ("high_quality", "High-Quality", "web", 608.96),
                ("high_quality_synthetic", "High-Quality-Synthetic", "web", 1223.46),
                ("medium_high_quality", "Medium-High-Quality", "web", 535.45),
                ("medium_quality", "Medium-Quality", "web", 2114.33),
                ("translated_diverse_qa", "Translated-Diverse-QA", "multilingual", 592.85),
            )
        ),
        # ---- Nemotron-CC v2.1 (9 subsets) ----
        *(
            DatakitSource(
                name=f"nemotron_cc_v2_1/{key}",
                hf_dataset_id="nvidia/Nemotron-CC-v2.1",
                revision="ba6f2aa",
                staged_path="raw/nemotron_cc_v2_1-a7afb6",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, cat, tok in (
                ("high_quality", "High-Quality", "web", 25.15),
                ("high_quality_dqa", "High-Quality-DQA", "web", 7.81),
                ("high_quality_synthetic", "High-Quality-Synthetic", "web", 90.86),
                ("high_quality_translated", "High-Quality-Translated-To-English", "multilingual", 38.65),
                (
                    "high_quality_translated_synthetic",
                    "High-Quality-Translated-To-English-Synthetic",
                    "multilingual",
                    153.41,
                ),
                ("medium_high_quality", "Medium-High-Quality", "web", 16.35),
                ("medium_high_quality_synthetic", "Medium-High-Quality-Synthetic", "web", 2065.38),
                (
                    "medium_high_quality_translated",
                    "Medium-High-Quality-Translated-To-English",
                    "multilingual",
                    26.03,
                ),
                ("medium_quality", "Medium-Quality", "web", 51.67),
            )
        ),
        # ---- Nemotron Pretraining Code v2 (5 subsets in CSV; 1 more in code registry) ----
        *(
            DatakitSource(
                name=f"nemotron_code_v2/{key}",
                hf_dataset_id="nvidia/Nemotron-Pretraining-Code-v2",
                revision="7b1a453",
                staged_path="raw/nemotron_pretraining_code_v2-d15a24",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, tok in (
                ("code_review", "Synthetic-Code-Review", 74.24),
                ("rewriting", "Synthetic-Rewriting", 73.73),
                ("student_teacher", "Synthetic-Student-Teacher", 25.20),
                ("synthetic_qa", "Synthetic-Question-Answering", 233.03),
                ("transpilation", "Synthetic-Transpilation", 27.78),
            )
        ),
        # ---- Nemotron Pretraining SFT v1 (3 subsets) ----
        *(
            DatakitSource(
                name=f"nemotron_sft/{key}",
                hf_dataset_id="nvidia/Nemotron-Pretraining-SFT-v1",
                revision="3f1a5b8",
                staged_path="raw/nemotron_pretraining_sft_v1-10f77e",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, cat, tok in (
                ("code", "Nemotron-SFT-Code", "code", 56.65),
                ("general", "Nemotron-SFT-General", "specialized", 85.20),
                ("math", "Nemotron-SFT-MATH", "math", 199.94),
            )
        ),
        # ---- Nemotron Pretraining Specialized v1 (6 subsets) ----
        *(
            DatakitSource(
                name=f"nemotron_specialized/{key}",
                hf_dataset_id="nvidia/Nemotron-Pretraining-Specialized-v1",
                revision="9ed3718",
                staged_path="raw/nemotron_pretraining_specialized_v1-a31fae",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, cat, tok in (
                ("infinibyte_reasoning", "Nemotron-Pretraining-InfiniByte-Reasoning", "specialized", 18.69),
                ("math_textbooks", "Nemotron-Pretraining-Math-Textbooks", "math", 25.59),
                ("rqa", "Nemotron-Pretraining-RQA", "specialized", 135.17),
                ("scientific_coding", "Nemotron-Pretraining-Scientific-Coding", "code", 1.18),
                ("stem_sft", "Nemotron-Pretraining-STEM-SFT", "specialized", 81.20),
                ("wiki_rewrite", "Nemotron-Pretraining-Wiki-Rewrite", "specialized", 7.26),
            )
        ),
        # ---- Nemotron Pretraining Specialized v1.1 (5 subsets; revision from staged provenance.json) ----
        *(
            DatakitSource(
                name=f"nemotron_specialized_v1_1/{key}",
                hf_dataset_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
                revision="13fa979",
                staged_path="raw/nemotron_pretraining_specialized_v1_1-b12f71",
                data_subdir=subdir,
                rough_token_count_b=tok,
            )
            for key, subdir, tok in (
                ("code_concepts", "Nemotron-Pretraining-Code-Concepts", 7.03),
                ("economics", "Nemotron-Pretraining-Economics", 0.07),
                ("formal_logic", "Nemotron-Pretraining-Formal-Logic", 0.13),
                ("multiple_choice", "Nemotron-Pretraining-Multiple-Choice", 1.56),
                ("unconditional_algorithmic", "Nemotron-Pretraining-Unconditional-Algorithmic", 0.19),
            )
        ),
        # ---- NSF Awards ----
        # TODO: hf_dataset_id="" because NSF Awards is API-sourced (no HF repo);
        # it needs a bespoke download step before pinned_sources() can include it.
        DatakitSource(
            name="nsf_awards",
            hf_dataset_id="",
            revision=None,
            rough_token_count_b=0.17,
            staged_path="raw/nsf-awards_6d7f6004",
        ),
        # ---- NuminaMath ----
        DatakitSource(
            name="numinamath",
            hf_dataset_id="AI-MO/NuminaMath-1.5",
            revision="1b05109",
            rough_token_count_b=0.38,
            staged_path="raw/numinamath_1_5_4911a6eb",
        ),
        # ---- StarCoder2-Extras (5 subsets) ----
        *(
            DatakitSource(
                name=f"starcoder2/{key}",
                hf_dataset_id="bigcode/StarCoder2-Extras",
                revision="1ba0d4f",
                staged_path=f"raw/starcoder2_extras-1ba0d4f/{key}",
                rough_token_count_b=tok,
            )
            for key, tok in (
                ("documentation", 1.40),
                ("ir_cpp", 39.01),
                ("ir_python", 4.64),
                ("ir_rust", 1.84),
                ("kaggle", 1.38),
            )
        ),
        # ---- Single-repo specialized / code sources ----
        DatakitSource(
            name="superior-reasoning",
            hf_dataset_id="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
            revision="21b55a6",
            rough_token_count_b=7.08,
            staged_path="raw/superior-reasoning-sft_b42ea7b3",
        ),
        DatakitSource(
            name="swe-rebench-openhands",
            hf_dataset_id="nebius/SWE-rebench-openhands-trajectories",
            revision="3545538",
            rough_token_count_b=2.47,
            staged_path="raw/swe-rebench-openhands-trajectories_e1e457c7",
        ),
        DatakitSource(
            name="synthetic-1",
            hf_dataset_id="PrimeIntellect/SYNTHETIC-1",
            revision="f08fe8c",
            rough_token_count_b=7.32,
            staged_path="raw/synthetic-1_1b24a14b",
        ),
    )
    return {e.name: e for e in entries}


@cache
def pinned_sources() -> dict[str, DatakitSource]:
    """Subset of :func:`all_sources` that the ferry can materialize.

    Includes only entries with a non-empty ``hf_dataset_id`` and a pinned
    ``revision``. Others (e.g. ``nsf_awards``, ``hplt_v3``) are carried
    for completeness but need custom wiring before they'll ferry.
    """
    return {name: src for name, src in all_sources().items() if src.revision and src.hf_dataset_id}
