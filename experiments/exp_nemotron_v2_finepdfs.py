# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize Nemotron v2, FinePDFs, FineTranslations, NuminaMath, and Common Pile.

Each dataset is tokenized individually for reusability. Run all datasets:

    uv run python experiments/exp_nemotron_v2_finepdfs.py

Or a single dataset:

    uv run python experiments/tokenize_single.py --dataset <name>

NOTE on CC-Math overlap: "3" is nemotron-cc-math-3plus (scores 3,4,5 = 133B) and
"4plus" is a subset (scores 4,5 = 52B). Using both would double-count 52B tokens.
We use "3" (3plus) + "4plus_MIND" for 206B unique tokens.

NOTE on Common Pile: datasets where quality filtering removes <10% are used raw.
Code datasets use raw + extension filter (edu filter too aggressive for pretraining).
"""

import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.common_pile.filter_stackv2_code import stackv2_code_filtered
from experiments.common_pile.stitch_bhl_books import bhl_full_books
from experiments.common_pile.tokenize_common_pile import (
    data_provenance_initiative_filtered,
    foodista_filtered,
    github_archive_filtered,
    library_of_congress_filtered,
    libretexts_filtered,
    news_filtered,
    oercommons_filtered,
    pre_1929_books_filtered,
    pressbooks_filtered,
    project_gutenberg_filtered,
    regulations_filtered,
    stackexchange_filtered,
    ubuntu_irc_filtered,
    usgpo_filtered,
    uspto_filtered,
    wikiteam_filtered,
    youtube_filtered,
)
from experiments.defaults import default_download, default_tokenize
from experiments.finetranslations.prepare_finetranslations import finetranslations_prepared
from experiments.long_context_datasets import finepdfs_by_language, institutional_books_raw
from experiments.long_context_datasets.finepdfs import finepdfs_extra_by_language
from experiments.pretraining_datasets.nemotron_v2 import NEMOTRON_V2_DATASETS, downloads
from experiments.reshard_parquet import ReshardConfig, reshard_parquet
from fray.cluster import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from zephyr import Dataset, ZephyrContext, load_parquet

TOKENIZER = "stanford-crfm/marin-tokenizer"

# ============================================================================
# Helpers
# ============================================================================


def _tokenize_step(
    name: str,
    train_paths: list,
    *,
    worker_ram: str = "10g",
    text_key: str = "text",
) -> ExecutorStep:
    """Create a tokenization ExecutorStep with explicit worker resources.

    We use this instead of default_tokenize because default_tokenize doesn't
    expose worker_resources, and many datasets have compressed shards that
    decompress to 10-30GB, OOMing the default 10g zephyr workers.
    """
    kwargs = {}
    if worker_ram != "10g":
        kwargs["worker_resources"] = ResourceConfig(ram=worker_ram, disk="10g")
    fmt = TextLmDatasetFormat(text_key=text_key)
    return ExecutorStep(
        name=f"tokenized/{name}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(TOKENIZER),
            format=fmt,
            **kwargs,
        ),
    )


def _nemotron_tokenize(
    family: str,
    subset: str,
    *,
    worker_ram: str = "10g",
    text_key: str = "text",
    reshard: bool = False,
) -> ExecutorStep:
    """Tokenize a Nemotron v2 subset. Optionally reshard large parquets first.

    reshard=True converts parquet files to smaller JSONL shards before tokenizing.
    Needed when individual parquet files are multi-GB (e.g. SFT-Math has 8.7GB
    parquets) and would OOM zephyr tokenization workers that load one file at a time.
    """
    dl = downloads[family]
    glob = NEMOTRON_V2_DATASETS[family].subsets[subset]

    if reshard:
        reshard_step = ExecutorStep(
            name=f"resharded/{family}/{subset}",
            fn=reshard_parquet,
            config=ReshardConfig(input_path=dl, output_path=this_output_path(), input_glob=glob),
        )
        train_paths = [reshard_step / "**/*.jsonl.gz"]
    else:
        train_paths = [dl / glob]

    return _tokenize_step(f"{family}/{subset}", train_paths, worker_ram=worker_ram, text_key=text_key)


def _cp_raw_tokenize(name: str, hf_id: str, revision: str, glob: str, *, worker_ram: str = "10g") -> ExecutorStep:
    """Download + tokenize a raw Common Pile dataset."""
    dl = ExecutorStep(
        name=f"raw/common_pile/{name}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=hf_id,
            revision=versioned(revision),
            hf_urls_glob=[glob],  # explicit glob to avoid HfFileSystem.find() truncation (#4170)
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )
    return _tokenize_step(f"common_pile/{name}", [dl / glob], worker_ram=worker_ram)


def _cp_filtered_tokenize(name: str, filtered_step, *, worker_ram: str = "10g") -> ExecutorStep:
    """Tokenize a pre-downloaded Common Pile filtered dataset."""
    return _tokenize_step(f"common_pile/{name}", [filtered_step / "**/*.json*"], worker_ram=worker_ram)


# ============================================================================
# Nemotron CC-v2
# ============================================================================

nemotron_cc_v2_diverse_qa = _nemotron_tokenize("nemotron_cc_v2", "diverse_qa")
nemotron_cc_v2_high_quality = _nemotron_tokenize("nemotron_cc_v2", "high_quality")
nemotron_cc_v2_high_quality_synthetic = _nemotron_tokenize("nemotron_cc_v2", "high_quality_synthetic")
nemotron_cc_v2_medium_high_quality = _nemotron_tokenize("nemotron_cc_v2", "medium_high_quality")
nemotron_cc_v2_medium_quality = _nemotron_tokenize("nemotron_cc_v2", "medium_quality")
# translated_diverse_qa has 2GB parquets; reshard + 20g workers to avoid OOM
nemotron_cc_v2_translated_diverse_qa = _nemotron_tokenize(
    "nemotron_cc_v2", "translated_diverse_qa", worker_ram="20g", reshard=True
)

# ============================================================================
# Nemotron CC-v2.1
# ============================================================================

# CC-v2.1 parquets are 1-2GB each; 20g workers needed to decompress without OOM
nemotron_cc_v2_1_high_quality = _nemotron_tokenize("nemotron_cc_v2_1", "high_quality", worker_ram="20g")
nemotron_cc_v2_1_high_quality_dqa = _nemotron_tokenize("nemotron_cc_v2_1", "high_quality_dqa", worker_ram="20g")
nemotron_cc_v2_1_high_quality_synthetic = _nemotron_tokenize(
    "nemotron_cc_v2_1", "high_quality_synthetic", worker_ram="20g"
)
nemotron_cc_v2_1_high_quality_translated = _nemotron_tokenize(
    "nemotron_cc_v2_1", "high_quality_translated", worker_ram="20g"
)
nemotron_cc_v2_1_high_quality_translated_synthetic = _nemotron_tokenize(
    "nemotron_cc_v2_1", "high_quality_translated_synthetic", worker_ram="20g"
)
nemotron_cc_v2_1_medium_high_quality = _nemotron_tokenize("nemotron_cc_v2_1", "medium_high_quality", worker_ram="20g")
nemotron_cc_v2_1_medium_high_quality_synthetic = _nemotron_tokenize(
    "nemotron_cc_v2_1", "medium_high_quality_synthetic", worker_ram="20g"
)
nemotron_cc_v2_1_medium_high_quality_translated = _nemotron_tokenize(
    "nemotron_cc_v2_1", "medium_high_quality_translated", worker_ram="20g"
)
nemotron_cc_v2_1_medium_quality = _nemotron_tokenize("nemotron_cc_v2_1", "medium_quality", worker_ram="20g")

# ============================================================================
# Nemotron CC-Code, CC-Math, Code-v2, Specialized, SFT
# ============================================================================

# CC-Code-v1 uses default text_key="text" (NOT "content" — despite being code, the
# HF dataset column is "text"). Code-v2 synthetic subsets use "content".
nemotron_cc_code_v1 = _nemotron_tokenize("nemotron_cc_code_v1", "all", reshard=True)

nemotron_cc_math_3 = _nemotron_tokenize("nemotron_cc_math_v1", "3", reshard=True)
nemotron_cc_math_mind = _nemotron_tokenize("nemotron_cc_math_v1", "4plus_mind")

# Code-v2 synthetic subsets use "content" column, not "text"
nemotron_code_v2_qa = _nemotron_tokenize(
    "nemotron_pretraining_code_v2", "synthetic_question_answering", text_key="content", reshard=True
)
nemotron_code_v2_student_teacher = _nemotron_tokenize(
    "nemotron_pretraining_code_v2", "synthetic_student_teacher", text_key="content"
)
nemotron_code_v2_code_review = _nemotron_tokenize(
    "nemotron_pretraining_code_v2", "synthetic_code_review", text_key="content"
)
nemotron_code_v2_rewriting = _nemotron_tokenize(
    "nemotron_pretraining_code_v2", "synthetic_rewriting", text_key="content"
)
nemotron_code_v2_transpilation = _nemotron_tokenize(
    "nemotron_pretraining_code_v2", "synthetic_transpilation", text_key="content"
)

# Specialized-v1 rqa/infinibyte/stem_sft have large parquets; reshard + extra RAM
nemotron_specialized_rqa = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1", "rqa", worker_ram="20g", reshard=True
)
nemotron_specialized_infinibyte = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1", "infinibyte_reasoning", worker_ram="20g", reshard=True
)
nemotron_specialized_wiki_rewrite = _nemotron_tokenize("nemotron_pretraining_specialized_v1", "wiki_rewrite")
nemotron_specialized_scientific_coding = _nemotron_tokenize("nemotron_pretraining_specialized_v1", "scientific_coding")
nemotron_specialized_math_textbooks = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1", "math_textbooks", reshard=True
)
nemotron_specialized_stem_sft = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1", "stem_sft", worker_ram="20g", reshard=True
)

nemotron_specialized_v1_1_code_concepts = _nemotron_tokenize("nemotron_pretraining_specialized_v1_1", "code_concepts")
nemotron_specialized_v1_1_algorithmic = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1_1", "unconditional_algorithmic"
)
nemotron_specialized_v1_1_formal_logic = _nemotron_tokenize("nemotron_pretraining_specialized_v1_1", "formal_logic")
nemotron_specialized_v1_1_economics = _nemotron_tokenize("nemotron_pretraining_specialized_v1_1", "economics")
nemotron_specialized_v1_1_multiple_choice = _nemotron_tokenize(
    "nemotron_pretraining_specialized_v1_1", "multiple_choice"
)

nemotron_sft_code = _nemotron_tokenize("nemotron_pretraining_sft_v1", "sft_code", reshard=True)
nemotron_sft_general = _nemotron_tokenize("nemotron_pretraining_sft_v1", "sft_general", reshard=True)
# SFT-Math has an 8.7GB parquet tail shard; needs 120g reshard workers (set in
# reshard_parquet.py) and 40g tokenize workers
nemotron_sft_math = _nemotron_tokenize("nemotron_pretraining_sft_v1", "sft_math", worker_ram="40g", reshard=True)

# ============================================================================
# FinePDFs (null text rows filtered during reshard)
# ============================================================================

# FinePDFs has two problems:
# 1. Some parquet rows have text=None, which crashes the tokenizer (TypeError in
#    BatchTokenizer). We filter these out during reshard (filter_null_text=True).
# 2. The 4.9GB parquets reshard into ~3GB JSONL.gz shards (zephyr's default shard
#    size), which decompress to ~15GB+. Tokenize workers need 80g to handle these.
# input_glob="" because finepdfs_by_language already includes the *.parquet glob.
finepdfs_resharded = ExecutorStep(
    name="resharded/finepdfs_eng_Latn",
    fn=reshard_parquet,
    config=ReshardConfig(
        input_path=finepdfs_by_language["eng_Latn"],
        output_path=this_output_path(),
        input_glob="",
        filter_null_text=True,
    ),
)
finepdfs = _tokenize_step("finepdfs_eng_Latn", [finepdfs_resharded / "**/*.jsonl.gz"], worker_ram="80g")

# Same null-text + large-shard treatment for non-English languages
finepdfs_extra = {}
for _lang, _path in finepdfs_extra_by_language.items():
    _resharded = ExecutorStep(
        name=f"resharded/finepdfs_{_lang}",
        fn=reshard_parquet,
        config=ReshardConfig(
            input_path=_path,
            output_path=this_output_path(),
            input_glob="",
            filter_null_text=True,
        ),
    )
    finepdfs_extra[_lang] = _tokenize_step(f"finepdfs_{_lang}", [_resharded / "**/*.jsonl.gz"], worker_ram="80g")

# ============================================================================
# FineTranslations
# ============================================================================

# FineTranslations prepare step produces 7149 shards at ~760MB compressed each;
# 20g workers needed for decompression overhead
finetranslations = _tokenize_step(
    "finetranslations_parallel", [finetranslations_prepared / "**/*.jsonl.gz"], worker_ram="20g"
)

# ============================================================================
# NuminaMath-1.5
# ============================================================================

numinamath_raw = default_download(
    name="numinamath_1_5",
    hf_dataset_id="AI-MO/NuminaMath-1.5",
    revision="1b05109",
)


@dataclasses.dataclass(frozen=True)
class PrepareNuminaMathConfig:
    input_path: str
    output_path: str


def prepare_numinamath(config: PrepareNuminaMathConfig):
    def format_record(record: dict) -> dict | None:
        problem = record.get("problem", "")
        solution = record.get("solution", "")
        if not problem:
            return None
        return {"text": f"{problem}\n\n{solution}"}

    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/*.parquet")
        .flat_map(load_parquet)
        .map(format_record)
        .filter(lambda r: r is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="prepare-numinamath", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


numinamath_prepared = ExecutorStep(
    name="documents/numinamath_1_5",
    fn=prepare_numinamath,
    config=PrepareNuminaMathConfig(input_path=numinamath_raw, output_path=this_output_path()),
)

numinamath = default_tokenize(
    name="numinamath_1_5",
    dataset=numinamath_prepared / "**/*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# ============================================================================
# Institutional Books 1.0
# ============================================================================


@dataclasses.dataclass(frozen=True)
class PrepareInstitutionalBooksConfig:
    input_path: str
    output_path: str


def prepare_institutional_books(config: PrepareInstitutionalBooksConfig):
    """Concat text_by_page_gen (list of page strings) into a single text field per book."""

    def concat_pages(record: dict) -> dict | None:
        pages = record.get("text_by_page_gen")
        if not pages:
            return None
        text = "\n\n".join(p for p in pages if p)
        if not text.strip():
            return None
        return {"text": text}

    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/**/*.parquet")
        .flat_map(load_parquet)
        .map(concat_pages)
        .filter(lambda r: r is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    # Books are large; need enough RAM for parquet decompression
    ctx = ZephyrContext(name="prepare-institutional-books", resources=ResourceConfig(cpu=2, ram="20g"))
    ctx.execute(pipeline)


institutional_books_prepared = ExecutorStep(
    name="documents/institutional_books",
    fn=prepare_institutional_books,
    config=PrepareInstitutionalBooksConfig(
        input_path=institutional_books_raw,
        output_path=this_output_path(),
    ),
)

# Some books are very large (hundreds of pages); 40g workers needed for the tail shards
institutional_books = _tokenize_step(
    "institutional_books", [institutional_books_prepared / "**/*.jsonl.gz"], worker_ram="40g"
)

# ============================================================================
# Common Pile: raw datasets (>=90% kept after filtering)
# ============================================================================

cp_peS2o = _cp_raw_tokenize("peS2o", "common-pile/peS2o", "2caeba1", "v0/documents/*.json.gz")
# pubmed/arxiv/caselaw/hansard have 1-3GB compressed shards; 20g workers for decompression
cp_pubmed = _cp_raw_tokenize("pubmed", "common-pile/pubmed", "648b8cf", "data/*.jsonl.gz", worker_ram="20g")
cp_arxiv_papers = _cp_raw_tokenize("arxiv_papers", "common-pile/arxiv_papers", "963fe98", "*.jsonl.gz", worker_ram="20g")
cp_arxiv_abstracts = _cp_raw_tokenize("arxiv_abstracts", "common-pile/arxiv_abstracts", "828e35d", "*.jsonl.gz")
cp_caselaw = _cp_raw_tokenize(
    "caselaw_access_project", "common-pile/caselaw_access_project", "3c2cb50", "*.jsonl.gz", worker_ram="20g"
)
cp_doab = _cp_raw_tokenize("doab", "common-pile/doab", "89e7a35", "*.json.gz")
cp_uk_hansard = _cp_raw_tokenize(
    "uk_hansard", "common-pile/uk_hansard", "05eeb43", "uk_hansard/*.jsonl.gz", worker_ram="20g"
)
cp_peps = _cp_raw_tokenize(
    "python_enhancement_proposals", "common-pile/python_enhancement_proposals", "f932757", "raw/documents/*.jsonl.gz"
)
cp_public_domain_review = _cp_raw_tokenize(
    "public_domain_review", "common-pile/public_domain_review", "e9c7669", "v0/*.jsonl.gz"
)

# ============================================================================
# Common Pile: filtered datasets (<90% kept)
# ============================================================================

cp_wikiteam = _cp_filtered_tokenize("wikiteam", wikiteam_filtered)
# pre_1929_books/regulations/project_gutenberg/library_of_congress/usgpo OOM at 20g;
# bumped to 40g (filtered output shards are large due to long documents)
cp_pre_1929_books = _cp_filtered_tokenize("pre_1929_books", pre_1929_books_filtered, worker_ram="40g")
cp_ubuntu_irc = _cp_filtered_tokenize("ubuntu_irc", ubuntu_irc_filtered, worker_ram="20g")
cp_regulations = _cp_filtered_tokenize("regulations", regulations_filtered, worker_ram="40g")
cp_project_gutenberg = _cp_filtered_tokenize("project_gutenberg", project_gutenberg_filtered, worker_ram="40g")
cp_data_provenance = _cp_filtered_tokenize("data_provenance_initiative", data_provenance_initiative_filtered)
cp_youtube = _cp_filtered_tokenize("youtube", youtube_filtered)
cp_biodiversity = _tokenize_step(
    "common_pile/biodiversity_heritage_library_books", [bhl_full_books / "**/*.jsonl.gz"], worker_ram="20g"
)
cp_library_of_congress = _cp_filtered_tokenize("library_of_congress", library_of_congress_filtered, worker_ram="40g")
cp_usgpo = _cp_filtered_tokenize("usgpo", usgpo_filtered, worker_ram="40g")
cp_pressbooks = _cp_filtered_tokenize("pressbooks", pressbooks_filtered)
cp_libretexts = _cp_filtered_tokenize("libretexts", libretexts_filtered)
cp_news = _cp_filtered_tokenize("news", news_filtered)
cp_foodista = _cp_filtered_tokenize("foodista", foodista_filtered)
cp_oercommons = _cp_filtered_tokenize("oercommons", oercommons_filtered)
cp_uspto = _cp_filtered_tokenize("uspto", uspto_filtered, worker_ram="20g")

# ============================================================================
# Common Pile: code datasets
# ============================================================================

cp_stackexchange = _cp_filtered_tokenize("stackexchange", stackexchange_filtered)
cp_github_archive = _cp_filtered_tokenize("github_archive", github_archive_filtered)
cp_stackv2_code = _tokenize_step("common_pile/stackv2_code", [stackv2_code_filtered / "**/*.jsonl.gz"], worker_ram="20g")

# ============================================================================
# ALL_COMPONENTS — used by tokenize_single.py
# ============================================================================

ALL_COMPONENTS: dict[str, ExecutorStep] = {
    # CC-v2
    "nemotron_cc_v2/diverse_qa": nemotron_cc_v2_diverse_qa,
    "nemotron_cc_v2/high_quality": nemotron_cc_v2_high_quality,
    "nemotron_cc_v2/high_quality_synthetic": nemotron_cc_v2_high_quality_synthetic,
    "nemotron_cc_v2/medium_high_quality": nemotron_cc_v2_medium_high_quality,
    "nemotron_cc_v2/medium_quality": nemotron_cc_v2_medium_quality,
    "nemotron_cc_v2/translated_diverse_qa": nemotron_cc_v2_translated_diverse_qa,
    # CC-v2.1
    "nemotron_cc_v2_1/high_quality": nemotron_cc_v2_1_high_quality,
    "nemotron_cc_v2_1/high_quality_dqa": nemotron_cc_v2_1_high_quality_dqa,
    "nemotron_cc_v2_1/high_quality_synthetic": nemotron_cc_v2_1_high_quality_synthetic,
    "nemotron_cc_v2_1/high_quality_translated": nemotron_cc_v2_1_high_quality_translated,
    "nemotron_cc_v2_1/high_quality_translated_synthetic": nemotron_cc_v2_1_high_quality_translated_synthetic,
    "nemotron_cc_v2_1/medium_high_quality": nemotron_cc_v2_1_medium_high_quality,
    "nemotron_cc_v2_1/medium_high_quality_synthetic": nemotron_cc_v2_1_medium_high_quality_synthetic,
    "nemotron_cc_v2_1/medium_high_quality_translated": nemotron_cc_v2_1_medium_high_quality_translated,
    "nemotron_cc_v2_1/medium_quality": nemotron_cc_v2_1_medium_quality,
    # CC-Code
    "nemotron_cc_code_v1/all": nemotron_cc_code_v1,
    # CC-Math
    "nemotron_cc_math_v1/3": nemotron_cc_math_3,
    "nemotron_cc_math_v1/4plus_mind": nemotron_cc_math_mind,
    # Code-v2 synthetic
    "nemotron_code_v2/synthetic_qa": nemotron_code_v2_qa,
    "nemotron_code_v2/student_teacher": nemotron_code_v2_student_teacher,
    "nemotron_code_v2/code_review": nemotron_code_v2_code_review,
    "nemotron_code_v2/rewriting": nemotron_code_v2_rewriting,
    "nemotron_code_v2/transpilation": nemotron_code_v2_transpilation,
    # Specialized-v1
    "nemotron_specialized/rqa": nemotron_specialized_rqa,
    "nemotron_specialized/infinibyte_reasoning": nemotron_specialized_infinibyte,
    "nemotron_specialized/wiki_rewrite": nemotron_specialized_wiki_rewrite,
    "nemotron_specialized/scientific_coding": nemotron_specialized_scientific_coding,
    "nemotron_specialized/math_textbooks": nemotron_specialized_math_textbooks,
    "nemotron_specialized/stem_sft": nemotron_specialized_stem_sft,
    # Specialized-v1.1
    "nemotron_specialized_v1_1/code_concepts": nemotron_specialized_v1_1_code_concepts,
    "nemotron_specialized_v1_1/unconditional_algorithmic": nemotron_specialized_v1_1_algorithmic,
    "nemotron_specialized_v1_1/formal_logic": nemotron_specialized_v1_1_formal_logic,
    "nemotron_specialized_v1_1/economics": nemotron_specialized_v1_1_economics,
    "nemotron_specialized_v1_1/multiple_choice": nemotron_specialized_v1_1_multiple_choice,
    # SFT-v1
    "nemotron_sft/code": nemotron_sft_code,
    "nemotron_sft/general": nemotron_sft_general,
    "nemotron_sft/math": nemotron_sft_math,
    # FinePDFs
    "finepdfs": finepdfs,
    **{f"finepdfs/{lang}": step for lang, step in finepdfs_extra.items()},
    # FineTranslations
    "finetranslations": finetranslations,
    # NuminaMath
    "numinamath": numinamath,
    # Institutional Books
    "institutional_books": institutional_books,
    # Common Pile NL (raw)
    "cp/peS2o": cp_peS2o,
    "cp/pubmed": cp_pubmed,
    "cp/arxiv_papers": cp_arxiv_papers,
    "cp/arxiv_abstracts": cp_arxiv_abstracts,
    "cp/caselaw": cp_caselaw,
    "cp/doab": cp_doab,
    "cp/uk_hansard": cp_uk_hansard,
    "cp/peps": cp_peps,
    "cp/public_domain_review": cp_public_domain_review,
    # Common Pile NL (filtered)
    "cp/wikiteam": cp_wikiteam,
    "cp/pre_1929_books": cp_pre_1929_books,
    "cp/ubuntu_irc": cp_ubuntu_irc,
    "cp/regulations": cp_regulations,
    "cp/project_gutenberg": cp_project_gutenberg,
    "cp/data_provenance": cp_data_provenance,
    "cp/youtube": cp_youtube,
    "cp/biodiversity": cp_biodiversity,
    "cp/library_of_congress": cp_library_of_congress,
    "cp/usgpo": cp_usgpo,
    "cp/pressbooks": cp_pressbooks,
    "cp/libretexts": cp_libretexts,
    "cp/news": cp_news,
    "cp/foodista": cp_foodista,
    "cp/oercommons": cp_oercommons,
    "cp/uspto": cp_uspto,
    # Common Pile code
    "cp/stackexchange": cp_stackexchange,
    "cp/github_archive": cp_github_archive,
    "cp/stackv2_code": cp_stackv2_code,
}

if __name__ == "__main__":
    executor_main(
        steps=[
            stackv2_code_filtered,
            finetranslations_prepared,
            numinamath_prepared,
            institutional_books_prepared,
            bhl_full_books,
            *ALL_COMPONENTS.values(),
        ]
    )
