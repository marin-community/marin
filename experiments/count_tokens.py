# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count tokens in all completed tokenized datasets and push to HuggingFace.

Reads train/.stats.json from each tokenized dataset on GCS, then writes a CSV
with columns (dataset, marin_tokens, category, synthetic) and pushes it to:
https://huggingface.co/datasets/marin-community/token-counts

Requires HF_TOKEN env var with write access to the dataset repo.
"""

import csv
import dataclasses
import io
import json
import logging
import tempfile

import fsspec
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

BUCKET = "gs://marin-us-central1"
HF_REPO = "marin-community/token-counts"


@dataclasses.dataclass(frozen=True)
class Dataset:
    gcs_prefix: str
    category: str  # web, code, math, multilingual, specialized
    synthetic: bool
    machine_translated: bool = False
    pdf: bool = False
    hf_repo: str = ""
    hf_subset: str = ""
    token_fraction: float = 1.0  # fraction of total_tokens to attribute to this row
    transform_tldr: str = ""
    license: str = ""  # manual override; API-fetched license used if empty


def _is_matching_output_dir(entry: str, basename: str) -> bool:
    entry_basename = entry.split("/")[-1]
    return entry_basename.startswith(f"{basename}-") or entry_basename.startswith(f"{basename}_")


def _matching_output_dirs(entries: list[str], basename: str) -> list[str]:
    return [entry for entry in entries if _is_matching_output_dir(entry, basename)]


NVIDIA_LICENSE = "NVIDIA Data Agreement for Model Training"

DATASETS: dict[str, Dataset] = {
    # Nemotron CC v2 — quality-classified Common Crawl web text
    "nemotron_cc_v2/diverse_qa": Dataset(
        "nemotron_cc_v2/diverse_qa",
        "web",
        True,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="diverse_qa",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2/high_quality": Dataset(
        "nemotron_cc_v2/high_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="high_quality",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2/high_quality_synthetic": Dataset(
        "nemotron_cc_v2/high_quality_synthetic",
        "web",
        True,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="high_quality_synthetic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2/medium_high_quality": Dataset(
        "nemotron_cc_v2/medium_high_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="medium_high_quality",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2/medium_quality": Dataset(
        "nemotron_cc_v2/medium_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="medium_quality",
        license=NVIDIA_LICENSE,
    ),
    # Nemotron CC v2.1
    "nemotron_cc_v2_1/high_quality": Dataset(
        "nemotron_cc_v2_1/high_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="high_quality",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/high_quality_dqa": Dataset(
        "nemotron_cc_v2_1/high_quality_dqa",
        "web",
        True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="high_quality_dqa",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/high_quality_synthetic": Dataset(
        "nemotron_cc_v2_1/high_quality_synthetic",
        "web",
        True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="high_quality_synthetic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/medium_high_quality": Dataset(
        "nemotron_cc_v2_1/medium_high_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="medium_high_quality",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/medium_high_quality_synthetic": Dataset(
        "nemotron_cc_v2_1/medium_high_quality_synthetic",
        "web",
        True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="medium_high_quality_synthetic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/medium_quality": Dataset(
        "nemotron_cc_v2_1/medium_quality",
        "web",
        False,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="medium_quality",
        license=NVIDIA_LICENSE,
    ),
    # Nemotron CC v2/v2.1 — multilingual variants
    "nemotron_cc_v2/translated_diverse_qa": Dataset(
        "nemotron_cc_v2/translated_diverse_qa",
        "multilingual",
        True,
        machine_translated=True,
        hf_repo="nvidia/Nemotron-CC-v2",
        hf_subset="translated_diverse_qa",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/high_quality_translated": Dataset(
        "nemotron_cc_v2_1/high_quality_translated",
        "multilingual",
        True,
        machine_translated=True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="high_quality_translated",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/high_quality_translated_synthetic": Dataset(
        "nemotron_cc_v2_1/high_quality_translated_synthetic",
        "multilingual",
        True,
        machine_translated=True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="high_quality_translated_synthetic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_v2_1/medium_high_quality_translated": Dataset(
        "nemotron_cc_v2_1/medium_high_quality_translated",
        "multilingual",
        True,
        machine_translated=True,
        hf_repo="nvidia/Nemotron-CC-v2.1",
        hf_subset="medium_high_quality_translated",
        license=NVIDIA_LICENSE,
    ),
    # Nemotron CC code & math
    "nemotron_cc_code_v1/all": Dataset(
        "nemotron_cc_code_v1/all", "code", False, hf_repo="nvidia/Nemotron-CC-Code-v1", license=NVIDIA_LICENSE
    ),
    "nemotron_cc_math_v1/3": Dataset(
        "nemotron_cc_math_v1/3",
        "math",
        False,
        hf_repo="nvidia/Nemotron-CC-Math-v1",
        hf_subset="3",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_cc_math_v1/4plus_mind": Dataset(
        "nemotron_cc_math_v1/4plus_mind",
        "math",
        False,
        hf_repo="nvidia/Nemotron-CC-Math-v1",
        hf_subset="4plus_mind",
        license=NVIDIA_LICENSE,
    ),
    # Nemotron synthetic code v2
    "nemotron_code_v2/synthetic_qa": Dataset(
        "nemotron_pretraining_code_v2/synthetic_question_answering",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Code-v2",
        hf_subset="synthetic_question_answering",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_code_v2/student_teacher": Dataset(
        "nemotron_pretraining_code_v2/synthetic_student_teacher",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Code-v2",
        hf_subset="synthetic_student_teacher",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_code_v2/code_review": Dataset(
        "nemotron_pretraining_code_v2/synthetic_code_review",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Code-v2",
        hf_subset="synthetic_code_review",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_code_v2/rewriting": Dataset(
        "nemotron_pretraining_code_v2/synthetic_rewriting",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Code-v2",
        hf_subset="synthetic_rewriting",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_code_v2/transpilation": Dataset(
        "nemotron_pretraining_code_v2/synthetic_transpilation",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Code-v2",
        hf_subset="synthetic_transpilation",
        license=NVIDIA_LICENSE,
    ),
    # Nemotron specialized synthetic
    "nemotron_specialized/rqa": Dataset(
        "nemotron_pretraining_specialized_v1/rqa",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="rqa",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized/infinibyte_reasoning": Dataset(
        "nemotron_pretraining_specialized_v1/infinibyte_reasoning",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="infinibyte_reasoning",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized/wiki_rewrite": Dataset(
        "nemotron_pretraining_specialized_v1/wiki_rewrite",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="wiki_rewrite",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized/scientific_coding": Dataset(
        "nemotron_pretraining_specialized_v1/scientific_coding",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="scientific_coding",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized/math_textbooks": Dataset(
        "nemotron_pretraining_specialized_v1/math_textbooks",
        "math",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="math_textbooks",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized/stem_sft": Dataset(
        "nemotron_pretraining_specialized_v1/stem_sft",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1",
        hf_subset="stem_sft",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized_v1_1/code_concepts": Dataset(
        "nemotron_pretraining_specialized_v1_1/code_concepts",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        hf_subset="code_concepts",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized_v1_1/unconditional_algorithmic": Dataset(
        "nemotron_pretraining_specialized_v1_1/unconditional_algorithmic",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        hf_subset="unconditional_algorithmic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized_v1_1/formal_logic": Dataset(
        "nemotron_pretraining_specialized_v1_1/formal_logic",
        "math",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        hf_subset="formal_logic",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized_v1_1/economics": Dataset(
        "nemotron_pretraining_specialized_v1_1/economics",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        hf_subset="economics",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_specialized_v1_1/multiple_choice": Dataset(
        "nemotron_pretraining_specialized_v1_1/multiple_choice",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        hf_subset="multiple_choice",
        license=NVIDIA_LICENSE,
    ),
    # SFT
    "nemotron_sft/code": Dataset(
        "nemotron_pretraining_sft_v1/sft_code",
        "code",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-SFT-v1",
        hf_subset="sft_code",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_sft/general": Dataset(
        "nemotron_pretraining_sft_v1/sft_general",
        "specialized",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-SFT-v1",
        hf_subset="sft_general",
        license=NVIDIA_LICENSE,
    ),
    "nemotron_sft/math": Dataset(
        "nemotron_pretraining_sft_v1/sft_math",
        "math",
        True,
        hf_repo="nvidia/Nemotron-Pretraining-SFT-v1",
        hf_subset="sft_math",
        license=NVIDIA_LICENSE,
    ),
    # Synthetic rollout datasets
    "synthetic-1": Dataset("synthetic-1", "specialized", True, hf_repo="PrimeIntellect/SYNTHETIC-1"),
    "coderforge": Dataset(
        "coderforge-preview",
        "code",
        True,
        hf_repo="togethercomputer/CoderForge-Preview",
        license=(
            "Per-file: MIT, BSD-3, Apache-2.0, BSD, BSD-2, BSD-4, ISC, PostgreSQL, PSF, CC0-1.0, "
            "MIT-NA, MIT-CMU, HPND, Dual MIT/Apache-2.0, Dual Apache-2.0/BSD-3, Dual BSD-3/MIT"
        ),
    ),
    "gpt-oss-rollouts": Dataset("gpt-oss-20b-rollouts", "specialized", True, hf_repo="andyrdt/gpt-oss-20b-rollouts"),
    "nemotron-terminal": Dataset("nemotron-terminal-corpus", "code", True, hf_repo="nvidia/Nemotron-Terminal-Corpus"),
    "superior-reasoning": Dataset(
        "superior-reasoning-sft", "specialized", True, hf_repo="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b"
    ),
    "swe-rebench-openhands": Dataset(
        "swe-rebench-openhands-trajectories", "code", True, hf_repo="nebius/SWE-rebench-openhands-trajectories"
    ),
    # Web corpora
    "hplt_v3": Dataset("hplt_v3/all", "web", False, hf_repo="HPLT/HPLT3.0"),
    "nsf_awards": Dataset("nsf_awards", "web", False, license="Public Domain"),
    "common_corpus/english": Dataset("common_corpus_english", "web", False, hf_repo="PleIAs/common_corpus"),
    # StarCoder2 extras
    "starcoder2/ir_cpp": Dataset("starcoder2_extras/ir_cpp", "code", False, hf_repo="bigcode/StarCoder2-Extras"),
    "starcoder2/ir_python": Dataset("starcoder2_extras/ir_python", "code", False, hf_repo="bigcode/StarCoder2-Extras"),
    "starcoder2/ir_rust": Dataset("starcoder2_extras/ir_rust", "code", False, hf_repo="bigcode/StarCoder2-Extras"),
    "starcoder2/documentation": Dataset(
        "starcoder2_extras/documentation", "code", False, hf_repo="bigcode/StarCoder2-Extras"
    ),
    "starcoder2/kaggle": Dataset("starcoder2_extras/kaggle", "code", False, hf_repo="bigcode/StarCoder2-Extras"),
    # Standalone datasets
    "finepdfs": Dataset(
        "finepdfs_eng_Latn", "web", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="eng_Latn"
    ),
    "finepdfs/spa_Latn": Dataset(
        "finepdfs_spa_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="spa_Latn"
    ),
    "finepdfs/deu_Latn": Dataset(
        "finepdfs_deu_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="deu_Latn"
    ),
    "finepdfs/fra_Latn": Dataset(
        "finepdfs_fra_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="fra_Latn"
    ),
    "finepdfs/rus_Cyrl": Dataset(
        "finepdfs_rus_Cyrl", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="rus_Cyrl"
    ),
    "finepdfs/jpn_Jpan": Dataset(
        "finepdfs_jpn_Jpan", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="jpn_Jpan"
    ),
    "finepdfs/ita_Latn": Dataset(
        "finepdfs_ita_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="ita_Latn"
    ),
    "finepdfs/por_Latn": Dataset(
        "finepdfs_por_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="por_Latn"
    ),
    "finepdfs/pol_Latn": Dataset(
        "finepdfs_pol_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="pol_Latn"
    ),
    "finepdfs/nld_Latn": Dataset(
        "finepdfs_nld_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="nld_Latn"
    ),
    "finepdfs/hun_Latn": Dataset(
        "finepdfs_hun_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="hun_Latn"
    ),
    "finepdfs/cmn_Hani": Dataset(
        "finepdfs_cmn_Hani", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="cmn_Hani"
    ),
    "finepdfs/ces_Latn": Dataset(
        "finepdfs_ces_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="ces_Latn"
    ),
    "finepdfs/arb_Arab": Dataset(
        "finepdfs_arb_Arab", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="arb_Arab"
    ),
    "finepdfs/ukr_Cyrl": Dataset(
        "finepdfs_ukr_Cyrl", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="ukr_Cyrl"
    ),
    "finepdfs/swe_Latn": Dataset(
        "finepdfs_swe_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="swe_Latn"
    ),
    "finepdfs/ron_Latn": Dataset(
        "finepdfs_ron_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="ron_Latn"
    ),
    "finepdfs/ind_Latn": Dataset(
        "finepdfs_ind_Latn", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="ind_Latn"
    ),
    "finepdfs/tha_Thai": Dataset(
        "finepdfs_tha_Thai", "multilingual", False, pdf=True, hf_repo="HuggingFaceFW/finepdfs", hf_subset="tha_Thai"
    ),
    # FineTranslations concatenates original multilingual text + machine-translated English.
    # Split into two halves for accounting: multilingual original + web-quality English MT.
    "finetranslations/multilingual": Dataset(
        "finetranslations_parallel",
        "multilingual",
        False,
        token_fraction=0.5,
        hf_repo="HuggingFaceFW/finetranslations",
        transform_tldr="50% of tokens: original multilingual text",
    ),
    "finetranslations/web": Dataset(
        "finetranslations_parallel",
        "web",
        False,
        machine_translated=True,
        token_fraction=0.5,
        hf_repo="HuggingFaceFW/finetranslations",
        transform_tldr="50% of tokens: machine-translated English",
    ),
    "numinamath": Dataset("numinamath_1_5", "math", False, hf_repo="AI-MO/NuminaMath-1.5"),
    "institutional_books": Dataset(
        "institutional_books",
        "web",
        False,
        hf_repo="institutional/institutional-books-1.0",
        transform_tldr="Pages concatenated from text_by_page_gen into full books",
    ),
    # Common Pile — licenses from arXiv:2506.05209
    "cp/peS2o": Dataset("common_pile/peS2o", "web", False, hf_repo="common-pile/peS2o", license="CC BY (per-paper)"),
    "cp/pubmed": Dataset(
        "common_pile/pubmed", "web", False, hf_repo="common-pile/pubmed", license="CC BY (per-article)"
    ),
    "cp/arxiv_papers": Dataset(
        "common_pile/arxiv_papers",
        "web",
        False,
        pdf=True,
        hf_repo="common-pile/arxiv_papers",
        license="CC BY (per-paper)",
    ),
    "cp/arxiv_abstracts": Dataset(
        "common_pile/arxiv_abstracts", "web", False, hf_repo="common-pile/arxiv_abstracts", license="CC0 1.0"
    ),
    "cp/caselaw": Dataset(
        "common_pile/caselaw_access_project",
        "web",
        False,
        hf_repo="common-pile/caselaw_access_project",
        license="Public Domain",
    ),
    "cp/regulations": Dataset(
        "common_pile/regulations", "web", False, hf_repo="common-pile/regulations", license="Public Domain"
    ),
    "cp/uspto": Dataset("common_pile/uspto", "web", False, hf_repo="common-pile/uspto", license="Public Domain"),
    "cp/uk_hansard": Dataset(
        "common_pile/uk_hansard", "web", False, hf_repo="common-pile/uk_hansard", license="Open Parliament License"
    ),
    "cp/usgpo": Dataset("common_pile/usgpo", "web", False, hf_repo="common-pile/usgpo", license="Public Domain"),
    "cp/doab": Dataset("common_pile/doab", "web", False, hf_repo="common-pile/doab", license="CC BY (per-book)"),
    "cp/pre_1929_books": Dataset(
        "common_pile/pre_1929_books", "web", False, hf_repo="common-pile/pre_1929_books", license="Public Domain"
    ),
    "cp/project_gutenberg": Dataset(
        "common_pile/project_gutenberg", "web", False, hf_repo="common-pile/project_gutenberg", license="Public Domain"
    ),
    "cp/pressbooks": Dataset(
        "common_pile/pressbooks", "web", False, hf_repo="common-pile/pressbooks", license="CC BY (per-book)"
    ),
    "cp/library_of_congress": Dataset(
        "common_pile/library_of_congress",
        "web",
        False,
        hf_repo="common-pile/library_of_congress",
        license="Public Domain",
    ),
    "cp/biodiversity": Dataset(
        "common_pile/biodiversity_heritage_library_books",
        "web",
        False,
        hf_repo="common-pile/biodiversity_heritage_library_books",
        license="Public Domain",
        transform_tldr="OCR pages stitched into full books by item_id",
    ),
    "cp/libretexts": Dataset(
        "common_pile/libretexts", "web", False, hf_repo="common-pile/libretexts", license="CC BY (per-text)"
    ),
    "cp/oercommons": Dataset(
        "common_pile/oercommons", "web", False, hf_repo="common-pile/oercommons", license="CC BY (per-resource)"
    ),
    "cp/peps": Dataset(
        "common_pile/python_enhancement_proposals",
        "code",
        False,
        hf_repo="common-pile/python_enhancement_proposals",
        license="Public Domain",
    ),
    "cp/github_archive": Dataset(
        "common_pile/github_archive",
        "code",
        False,
        hf_repo="common-pile/github_archive",
        license="Blue Oak Council (per-repo)",
    ),
    "cp/stackv2_code": Dataset(
        "common_pile/stackv2_code",
        "code",
        False,
        hf_repo="common-pile/stackv2_code",
        license="Blue Oak Council (per-file)",
        transform_tldr="Filtered to 40+ programming language extensions",
    ),
    "cp/stackexchange": Dataset(
        "common_pile/stackexchange", "web", False, hf_repo="common-pile/stackexchange", license="CC BY-SA"
    ),
    "cp/ubuntu_irc": Dataset(
        "common_pile/ubuntu_irc", "web", False, hf_repo="common-pile/ubuntu_irc", license="Public Domain"
    ),
    "cp/wikiteam": Dataset(
        "common_pile/wikiteam", "web", False, hf_repo="common-pile/wikiteam", license="CC BY-SA (per-wiki)"
    ),
    "cp/public_domain_review": Dataset(
        "common_pile/public_domain_review",
        "web",
        False,
        hf_repo="common-pile/public_domain_review",
        license="Public Domain",
    ),
    "cp/data_provenance": Dataset(
        "common_pile/data_provenance_initiative",
        "specialized",
        False,
        hf_repo="common-pile/data_provenance_initiative",
        license="Open (per-dataset)",
    ),
    "cp/youtube": Dataset("common_pile/youtube", "web", False, hf_repo="common-pile/youtube", license="CC BY"),
    "cp/news": Dataset("common_pile/news", "web", False, hf_repo="common-pile/news", license="CC BY / CC BY-SA"),
    "cp/foodista": Dataset("common_pile/foodista", "web", False, hf_repo="common-pile/foodista", license="CC BY"),
}


def fetch_licenses(datasets: dict[str, Dataset]) -> dict[str, str]:
    api = HfApi()
    seen: dict[str, str] = {}
    licenses: dict[str, str] = {}
    for name, ds in datasets.items():
        if ds.license:
            licenses[name] = ds.license
            continue
        if not ds.hf_repo:
            licenses[name] = ""
            continue
        if ds.hf_repo in seen:
            licenses[name] = seen[ds.hf_repo]
            continue
        try:
            info = api.dataset_info(ds.hf_repo)
            license_val = info.card_data.get("license", "") if info.card_data else ""
            if isinstance(license_val, list):
                license_val = ", ".join(license_val)
            seen[ds.hf_repo] = license_val
            licenses[name] = license_val
            logger.info(f"{ds.hf_repo}: {license_val or '(no license)'}")
        except Exception as e:
            logger.warning(f"{ds.hf_repo}: failed to fetch license: {e}")
            seen[ds.hf_repo] = ""
            licenses[name] = ""
    return licenses


def upload_to_hf(csv_content: str):
    api = HfApi()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="data/token_counts.csv",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message="Update token counts",
        )
    logger.info("Dataset updated: https://huggingface.co/datasets/%s", HF_REPO)


def count_tokens():
    logging.basicConfig(level=logging.INFO)
    fs = fsspec.filesystem("gcs")
    bucket = BUCKET.replace("gs://", "")

    # Fetch license info from HuggingFace dataset cards
    logger.info("Fetching license info from HuggingFace...")
    licenses = fetch_licenses(DATASETS)

    total_tokens = 0
    results: list[tuple[str, Dataset, int | None, str]] = []

    for name, ds in sorted(DATASETS.items()):
        try:
            prefix = ds.gcs_prefix
            parent_dir = f"{bucket}/tokenized/{prefix.rsplit('/', 1)[0]}" if "/" in prefix else f"{bucket}/tokenized"
            basename = prefix.rsplit("/", 1)[-1] if "/" in prefix else prefix

            try:
                entries = fs.ls(parent_dir, detail=False)
            except FileNotFoundError:
                results.append((name, ds, None, "DIR NOT FOUND"))
                continue

            stats_data = None
            for entry in _matching_output_dirs(entries, basename):
                stats_path = f"{entry}/train/.stats.json"
                try:
                    with fs.open(stats_path, "r") as f:
                        stats_data = json.load(f)
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.warning(f"{name}: error reading {stats_path}: {e}")
                    continue

            if stats_data is None:
                results.append((name, ds, None, "NO STATS"))
                continue

            logger.info(f"{name}: stats keys = {list(stats_data.keys())}")
            tokens = stats_data.get("total_tokens", stats_data.get("num_tokens", stats_data.get("total_num_tokens", 0)))
            if tokens == 0:
                logger.warning(f"{name}: no token count found in stats keys {list(stats_data.keys())}")
                results.append((name, ds, None, "NO TOKEN COUNT"))
                continue

            tokens = int(tokens * ds.token_fraction)
            results.append((name, ds, tokens, "OK"))
            total_tokens += tokens
            logger.info(f"{name}: {tokens / 1e9:.2f}B tokens")

        except Exception as e:
            results.append((name, ds, None, f"ERROR: {e}"))
            logger.warning(f"{name}: {e}")

    # Build CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "marin_name",
            "marin_tokens",
            "category",
            "synthetic",
            "machine_translated",
            "pdf",
            "hf_repo",
            "hf_subset",
            "license",
            "transform_tldr",
        ]
    )
    for name, ds, tokens, status in results:
        if tokens is not None and tokens > 0:
            writer.writerow(
                [
                    name,
                    tokens,
                    ds.category,
                    ds.synthetic,
                    ds.machine_translated,
                    ds.pdf,
                    ds.hf_repo,
                    ds.hf_subset,
                    licenses.get(name, ""),
                    ds.transform_tldr,
                ]
            )
        else:
            logger.warning(f"Skipping {name}: {status}")
    csv_content = buf.getvalue()

    # Print summary table
    print(f"\n{'Dataset':<45} {'Tokens':>12} {'Category':>16} {'Synth':>6} {'Status':>12}")
    print("-" * 94)
    for name, ds, tokens, status in results:
        syn = "Y" if ds.synthetic else ""
        if tokens is not None and tokens > 0:
            print(f"{name:<45} {tokens / 1e9:>11.2f}B {ds.category:>16} {syn:>6} {'OK':>12}")
        else:
            print(f"{name:<45} {'':>12} {ds.category:>16} {syn:>6} {status:>12}")
    print("-" * 94)
    print(f"{'TOTAL':<45} {total_tokens / 1e9:>11.2f}B")

    # Upload to HuggingFace
    upload_to_hf(csv_content)


if __name__ == "__main__":
    count_tokens()
