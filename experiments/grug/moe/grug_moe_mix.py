# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE on the Dolma 3 + Dolmino top-level mixture, five compute scales.

The mixture has 39 top-level domains that resolve in three ways depending
on how they were tokenized:

1. 28 cache-backed domains (26 Dolma 3 CC quality splits + Dolma 3 stack-edu
   + Dolmino stem-heavy-crawl) — `DatasetComponent`s pointing at merged
   caches under `tokenized/merged/dolma3_dolmino_top_level/...`.
2. 3 singletons (arxiv, finemath_3plus, wikipedia) — `DatasetComponent`s
   pointing at their single-partition tokenized caches directly.
3. 8 multi-partition Dolmino groups (common_crawl_hq, olmocr_pdfs_hq,
   stack_edu_fim, synth_*) tokenized as one cache per partition. Each
   group is a `ConcatDatasetComponent` whose children are the
   per-partition `DatasetComponent`s; Levanter concatenates them at
   training time via `ConcatDataset` and the parent mixture shuffles
   globally.

All caches live under `gs://marin-us-east5/...`; every path is region-
relative via `InputName.hardcoded`, so launches outside us-east5 hard-fail
on missing-cache rather than triggering a cross-region copy.

Emits one `ExecutorStep` per compute-optimal scale in `_SCALES` (d512 -
d1536). Simulated epoching is enabled by anchoring `target_budget` to the
natural Dolma 3 common-crawl pool (~6.33T tokens) and filling in
`experiment_budget = batch * steps * seq_len` per scale.
"""

import dataclasses
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_models import marin_tokenizer

# ---------------------------------------------------------------------------
# Mixture: 39 hardcoded merged caches, proportional weights.
# ---------------------------------------------------------------------------

_MERGED_PREFIX = "tokenized/merged/dolma3_dolmino_top_level"

# Region-relative paths: `InputName.hardcoded("foo")` resolves to
# `gs://marin-{REGION}/foo` at execution time. The merged caches only exist in
# `gs://marin-us-east5/`, so launching in any other region intentionally hard-
# fails on missing-cache rather than triggering a cross-region copy.
DOMAIN_CACHE_PATHS: dict[str, str] = {
    # Dolma 3 CC quality-split domains (13 topics x {high, low}).
    "dolma3_cc/art_and_design_high": f"{_MERGED_PREFIX}/dolma3_cc/art_and_design_high-8fd3f7",
    "dolma3_cc/art_and_design_low": f"{_MERGED_PREFIX}/dolma3_cc/art_and_design_low-f75d57",
    "dolma3_cc/crime_and_law_high": f"{_MERGED_PREFIX}/dolma3_cc/crime_and_law_high-d1e71b",
    "dolma3_cc/crime_and_law_low": f"{_MERGED_PREFIX}/dolma3_cc/crime_and_law_low-c5231a",
    "dolma3_cc/education_and_jobs_high": f"{_MERGED_PREFIX}/dolma3_cc/education_and_jobs_high-fd5aa1",
    "dolma3_cc/education_and_jobs_low": f"{_MERGED_PREFIX}/dolma3_cc/education_and_jobs_low-6d3fc9",
    "dolma3_cc/electronics_and_hardware_high": f"{_MERGED_PREFIX}/dolma3_cc/electronics_and_hardware_high-c901ee",
    "dolma3_cc/electronics_and_hardware_low": f"{_MERGED_PREFIX}/dolma3_cc/electronics_and_hardware_low-35ef64",
    "dolma3_cc/entertainment_high": f"{_MERGED_PREFIX}/dolma3_cc/entertainment_high-a099bb",
    "dolma3_cc/entertainment_low": f"{_MERGED_PREFIX}/dolma3_cc/entertainment_low-952a40",
    "dolma3_cc/finance_and_business_high": f"{_MERGED_PREFIX}/dolma3_cc/finance_and_business_high-71c684",
    "dolma3_cc/finance_and_business_low": f"{_MERGED_PREFIX}/dolma3_cc/finance_and_business_low-fe69a6",
    "dolma3_cc/food_and_dining_high": f"{_MERGED_PREFIX}/dolma3_cc/food_and_dining_high-103e51",
    "dolma3_cc/food_and_dining_low": f"{_MERGED_PREFIX}/dolma3_cc/food_and_dining_low-f491f1",
    "dolma3_cc/games_high": f"{_MERGED_PREFIX}/dolma3_cc/games_high-4762ec",
    "dolma3_cc/games_low": f"{_MERGED_PREFIX}/dolma3_cc/games_low-a8fdde",
    "dolma3_cc/health_high": f"{_MERGED_PREFIX}/dolma3_cc/health_high-d28699",
    "dolma3_cc/health_low": f"{_MERGED_PREFIX}/dolma3_cc/health_low-d93c8a",
    "dolma3_cc/history_and_geography_high": f"{_MERGED_PREFIX}/dolma3_cc/history_and_geography_high-b49626",
    "dolma3_cc/history_and_geography_low": f"{_MERGED_PREFIX}/dolma3_cc/history_and_geography_low-979799",
    "dolma3_cc/industrial_high": f"{_MERGED_PREFIX}/dolma3_cc/industrial_high-818b37",
    "dolma3_cc/industrial_low": f"{_MERGED_PREFIX}/dolma3_cc/industrial_low-dbd530",
    "dolma3_cc/literature_high": f"{_MERGED_PREFIX}/dolma3_cc/literature_high-bddf01",
    "dolma3_cc/literature_low": f"{_MERGED_PREFIX}/dolma3_cc/literature_low-a7a06c",
    "dolma3_cc/science_math_and_technology_high": f"{_MERGED_PREFIX}/dolma3_cc/science_math_and_technology_high-8c6157",
    "dolma3_cc/science_math_and_technology_low": f"{_MERGED_PREFIX}/dolma3_cc/science_math_and_technology_low-f3e030",
    # Dolma 3 stack-edu (prebuilt merged cache).
    "dolma3_stack_edu": f"{_MERGED_PREFIX}/dolma3_stack_edu-a7297b",
    # Dolmino stem-heavy-crawl (prebuilt merged cache).
    "dolmino_stem_heavy_crawl": f"{_MERGED_PREFIX}/dolmino_stem_heavy_crawl-e1ec3b",
    # Singletons: no merged cache under the `dolma3_dolmino_top_level` prefix,
    # so point at the underlying single-partition tokenized caches directly.
    "dolma3_arxiv": "tokenized/dolma/arxiv-07a51f",
    "dolma3_finemath_3plus": "tokenized/finemath_3_plus-a26b0f",
    "dolma3_wikipedia": "tokenized/dolma/wiki-212315",
}

# Per-partition cache paths for the 8 multi-partition Dolmino groups. Each
# entry maps a domain to a list of (partition_name, relative_cache_path)
# pairs; paths live under
# `gs://marin-{region}/tokenized/dolma3_dolmino_pool/...`. Each group is
# materialized at runtime as a `ConcatDatasetComponent` whose children are
# the per-partition cache-backed `DatasetComponent`s.
HIERARCHICAL_PARTITION_PATHS: dict[str, list[tuple[str, str]]] = {
    "dolmino_common_crawl_hq": [
        ("19_adult_content", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_adult_content-986941"),
        ("19_art_and_design", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_art_and_design-140701"),
        ("19_crime_and_law", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_crime_and_law-7d7fdc"),
        ("19_education_and_jobs", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_education_and_jobs-90ee22"),
        (
            "19_electronics_and_hardware",
            "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_electronics_and_hardware-39f802",
        ),
        ("19_entertainment", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_entertainment-ae89d3"),
        ("19_fashion_and_beauty", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_fashion_and_beauty-b5d2e1"),
        ("19_finance_and_business", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_finance_and_business-20e87b"),
        ("19_food_and_dining", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_food_and_dining-d57db7"),
        ("19_games", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_games-9e426a"),
        ("19_health", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_health-4a7919"),
        ("19_history_and_geography", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_history_and_geography-a181cb"),
        ("19_home_and_hobbies", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_home_and_hobbies-52899e"),
        ("19_industrial", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_industrial-365cbc"),
        ("19_literature", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_literature-8f2b65"),
        ("19_politics", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_politics-f0c570"),
        ("19_religion", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_religion-f71c2c"),
        (
            "19_science_math_and_technology",
            "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_science_math_and_technology-9676a0",
        ),
        ("19_social_life", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_social_life-e01f54"),
        ("19_software", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_software-e41483"),
        ("19_software_development", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_software_development-12bde0"),
        ("19_sports_and_fitness", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_sports_and_fitness-55671d"),
        ("19_transportation", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_transportation-8ef6d4"),
        ("19_travel_and_tourism", "tokenized/dolma3_dolmino_pool/common_crawl_hq_19_travel_and_tourism-0b507c"),
        ("20_adult_content", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_adult_content-60c0fc"),
        ("20_art_and_design", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_art_and_design-f6c229"),
        ("20_crime_and_law", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_crime_and_law-08910c"),
        ("20_education_and_jobs", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_education_and_jobs-6b6514"),
        (
            "20_electronics_and_hardware",
            "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_electronics_and_hardware-78eb1c",
        ),
        ("20_entertainment", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_entertainment-1c2777"),
        ("20_fashion_and_beauty", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_fashion_and_beauty-ec1192"),
        ("20_finance_and_business", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_finance_and_business-edf4dd"),
        ("20_food_and_dining", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_food_and_dining-495c5e"),
        ("20_games", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_games-a5427f"),
        ("20_health", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_health-50e664"),
        ("20_history_and_geography", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_history_and_geography-40625e"),
        ("20_home_and_hobbies", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_home_and_hobbies-ea57b3"),
        ("20_industrial", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_industrial-89e820"),
        ("20_literature", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_literature-1d4132"),
        ("20_politics", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_politics-6b2fc1"),
        ("20_religion", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_religion-553dce"),
        (
            "20_science_math_and_technology",
            "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_science_math_and_technology-8be542",
        ),
        ("20_social_life", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_social_life-f4a77b"),
        ("20_software", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_software-2eaab8"),
        ("20_software_development", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_software_development-bf8388"),
        ("20_sports_and_fitness", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_sports_and_fitness-32a5cd"),
        ("20_transportation", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_transportation-77fd33"),
        ("20_travel_and_tourism", "tokenized/dolma3_dolmino_pool/common_crawl_hq_20_travel_and_tourism-9b2bb7"),
    ],
    "dolmino_olmocr_pdfs_hq": [
        ("adult_content", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_adult_content-782fdf"),
        ("art_and_design", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_art_and_design-f66e82"),
        ("crime_and_law", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_crime_and_law-6bcf3f"),
        ("education_and_jobs", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_education_and_jobs-8c857e"),
        ("electronics_and_hardware", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_electronics_and_hardware-9efa57"),
        ("entertainment", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_entertainment-935e9d"),
        ("fashion_and_beauty", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_fashion_and_beauty-9e33a9"),
        ("finance_and_business", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_finance_and_business-1db9a9"),
        ("food_and_dining", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_food_and_dining-9c9ccd"),
        ("games", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_games-2247ec"),
        ("health", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_health-6592c9"),
        ("history_and_geography", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_history_and_geography-0aa517"),
        ("home_and_hobbies", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_home_and_hobbies-f2f3a6"),
        ("industrial", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_industrial-fdf3cc"),
        ("literature", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_literature-b8288a"),
        ("politics", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_politics-12e4e1"),
        ("religion", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_religion-79f07d"),
        (
            "science_math_and_technology",
            "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_science_math_and_technology-6b1eb9",
        ),
        ("software", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_software-07945d"),
        ("software_development", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_software_development-6b0967"),
        ("sports_and_fitness", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_sports_and_fitness-7e7c4c"),
        ("transportation", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_transportation-8bf699"),
        ("travel_and_tourism", "tokenized/dolma3_dolmino_pool/olmocr_pdfs_hq_travel_and_tourism-35e5a9"),
    ],
    "dolmino_stack_edu_fim": [
        ("C", "tokenized/dolma3_dolmino_pool/stack_edu_fim_C-421921"),
        ("CSharp", "tokenized/dolma3_dolmino_pool/stack_edu_fim_CSharp-826de5"),
        ("Cpp", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Cpp-acccdd"),
        ("Go", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Go-d945e3"),
        ("Java", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Java-f0c478"),
        ("JavaScript", "tokenized/dolma3_dolmino_pool/stack_edu_fim_JavaScript-f31610"),
        ("Markdown", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Markdown-b900ef"),
        ("PHP", "tokenized/dolma3_dolmino_pool/stack_edu_fim_PHP-6a476c"),
        ("Python", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Python-b0c1be"),
        ("Ruby", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Ruby-12aa83"),
        ("Rust", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Rust-d4ee4a"),
        ("SQL", "tokenized/dolma3_dolmino_pool/stack_edu_fim_SQL-0724c4"),
        ("Shell", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Shell-500b3c"),
        ("Swift", "tokenized/dolma3_dolmino_pool/stack_edu_fim_Swift-20bbbb"),
        ("TypeScript", "tokenized/dolma3_dolmino_pool/stack_edu_fim_TypeScript-917f94"),
    ],
    "dolmino_synth_code": [
        ("cranecode", "tokenized/dolma3_dolmino_pool/synth_code_cranecode-3d2447"),
    ],
    "dolmino_synth_instruction": [
        ("dolmino_flan", "tokenized/dolma3_dolmino_pool/synth_instruction_dolmino_flan-183f12"),
        ("tulu_3_sft", "tokenized/dolma3_dolmino_pool/synth_instruction_tulu_3_sft-00fa09"),
    ],
    "dolmino_synth_math": [
        ("cranemath", "tokenized/dolma3_dolmino_pool/synth_math_cranemath-2896f8"),
        ("dolmino_math", "tokenized/dolma3_dolmino_pool/synth_math_dolmino_math-6a90af"),
        ("megamatt", "tokenized/dolma3_dolmino_pool/synth_math_megamatt-862c18"),
        ("tinymath_mind", "tokenized/dolma3_dolmino_pool/synth_math_tinymath_mind-f01a63"),
        ("tinymath_pot", "tokenized/dolma3_dolmino_pool/synth_math_tinymath_pot-c60e19"),
        ("verifiable_gpt41", "tokenized/dolma3_dolmino_pool/synth_math_verifiable_gpt41-6e5533"),
        ("verifiable_o4mini", "tokenized/dolma3_dolmino_pool/synth_math_verifiable_o4mini-2cbec0"),
    ],
    "dolmino_synth_qa": [
        ("nemotron_synth_qa", "tokenized/dolma3_dolmino_pool/synth_qa_nemotron_synth_qa-4c6ea5"),
        ("reddit_to_flashcards", "tokenized/dolma3_dolmino_pool/synth_qa_reddit_to_flashcards-9acbf6"),
        ("wiki_to_rcqa", "tokenized/dolma3_dolmino_pool/synth_qa_wiki_to_rcqa-bd4afa"),
    ],
    "dolmino_synth_thinking": [
        ("code_meta_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_code_meta_reasoning-89ea11"),
        ("gemini_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_gemini_reasoning-fa77a6"),
        ("general_reasoning_mix", "tokenized/dolma3_dolmino_pool/synth_thinking_general_reasoning_mix-cb5cb6"),
        ("llama_nemotron_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_llama_nemotron_reasoning-1e9de1"),
        ("math_meta_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_math_meta_reasoning-c0fdb1"),
        ("omr_rewrite_fullthoughts", "tokenized/dolma3_dolmino_pool/synth_thinking_omr_rewrite_fullthoughts-e0eb6c"),
        ("openthoughts2_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_openthoughts2_reasoning-fd22f4"),
        ("program_verifiable", "tokenized/dolma3_dolmino_pool/synth_thinking_program_verifiable-bc5995"),
        ("qwq_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_qwq_reasoning-467558"),
        ("r1_reasoning", "tokenized/dolma3_dolmino_pool/synth_thinking_r1_reasoning-b23788"),
    ],
}

# Per-domain token counts (sum of underlying partition counts). Used to
# derive `PROPORTIONAL_WEIGHTS` and `TARGET_BUDGET` below.
DOMAIN_TOKEN_COUNTS: dict[str, int] = {
    "dolma3_cc/art_and_design_high": 114170169532,
    "dolma3_cc/art_and_design_low": 46340373707,
    "dolma3_cc/crime_and_law_high": 153674689073,
    "dolma3_cc/crime_and_law_low": 69031589100,
    "dolma3_cc/education_and_jobs_high": 251260875880,
    "dolma3_cc/education_and_jobs_low": 123073433190,
    "dolma3_cc/electronics_and_hardware_high": 132572355890,
    "dolma3_cc/electronics_and_hardware_low": 62313779937,
    "dolma3_cc/entertainment_high": 423412626925,
    "dolma3_cc/entertainment_low": 154085132409,
    "dolma3_cc/finance_and_business_high": 543963886806,
    "dolma3_cc/finance_and_business_low": 259194248668,
    "dolma3_cc/food_and_dining_high": 170839401766,
    "dolma3_cc/food_and_dining_low": 84090162862,
    "dolma3_cc/games_high": 225942508715,
    "dolma3_cc/games_low": 84710277508,
    "dolma3_cc/health_high": 435468411203,
    "dolma3_cc/health_low": 185597052864,
    "dolma3_cc/history_and_geography_high": 135111719214,
    "dolma3_cc/history_and_geography_low": 37182168838,
    "dolma3_cc/industrial_high": 81672506307,
    "dolma3_cc/industrial_low": 43243155264,
    "dolma3_cc/literature_high": 258340689382,
    "dolma3_cc/literature_low": 68881517798,
    "dolma3_cc/science_math_and_technology_high": 243037340439,
    "dolma3_cc/science_math_and_technology_low": 110943053451,
    "dolma3_arxiv": 28237567983,
    "dolma3_finemath_3plus": 34001855255,
    "dolma3_wikipedia": 3669138258,
    "dolma3_stack_edu": 134071054270,
    "dolmino_common_crawl_hq": 1317586261685,
    "dolmino_olmocr_pdfs_hq": 205902385099,
    "dolmino_stack_edu_fim": 133883067238,
    "dolmino_stem_heavy_crawl": 5213753236,
    "dolmino_synth_code": 18860808823,
    "dolmino_synth_instruction": 17980436519,
    "dolmino_synth_math": 21820894376,
    "dolmino_synth_qa": 527153343948,
    "dolmino_synth_thinking": 39897911717,
}

assert set(DOMAIN_CACHE_PATHS.keys()) | set(HIERARCHICAL_PARTITION_PATHS.keys()) == set(
    DOMAIN_TOKEN_COUNTS.keys()
), "Every domain must appear in either DOMAIN_CACHE_PATHS or HIERARCHICAL_PARTITION_PATHS"
assert (
    set(DOMAIN_CACHE_PATHS.keys()) & set(HIERARCHICAL_PARTITION_PATHS.keys()) == set()
), "DOMAIN_CACHE_PATHS and HIERARCHICAL_PARTITION_PATHS must be disjoint"
assert len(DOMAIN_TOKEN_COUNTS) == 39
assert len(HIERARCHICAL_PARTITION_PATHS) == 8
assert len(DOMAIN_CACHE_PATHS) == 31

TOTAL_TOKENS: int = sum(DOMAIN_TOKEN_COUNTS.values())

# Target budget for simulated epoching: the sum of Dolma 3 `common_crawl/*`
# partition tokens, anchoring the simulation to the Dolma 3 CC share of the
# pool so domains can be revisited proportionally when an experiment's
# token budget exceeds the natural mixture size.
TARGET_BUDGET: int = 6_325_183_647_689

PROPORTIONAL_WEIGHTS: dict[str, float] = {name: count / TOTAL_TOKENS for name, count in DOMAIN_TOKEN_COUNTS.items()}


def _build_component(domain: str, cache_path: str) -> DatasetComponent:
    cache_dir = InputName.hardcoded(cache_path)
    source = UrlDatasetSourceConfig(
        tags=[domain],
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_dir,
        format=TextLmDatasetFormat(),
    )
    return DatasetComponent(
        source=source,
        cache_dir=cache_dir,
        format=TextLmDatasetFormat(),
        tags=[domain],
    )


def _build_hierarchical_components() -> dict[str, ConcatDatasetComponent]:
    """Build one `ConcatDatasetComponent` per hierarchical Dolmino group.

    Each group's children are cache-backed `DatasetComponent`s pointing at
    their per-partition tokenized caches via region-relative
    `InputName.hardcoded` paths. Levanter loads the children's caches on the
    worker and concatenates them via `ConcatDataset`; the parent
    `LmDataConfig.shuffle` setting permutes the concatenated result globally.
    """
    components: dict[str, ConcatDatasetComponent] = {}
    for domain, partitions in HIERARCHICAL_PARTITION_PATHS.items():
        children = {
            partition_name: _build_component(f"{domain}/{partition_name}", rel_path)
            for partition_name, rel_path in partitions
        }
        components[domain] = ConcatDatasetComponent(children=children, tags=[domain])
    return components


# ---------------------------------------------------------------------------
# Launch wiring.
# ---------------------------------------------------------------------------


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker, run_id: str):
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_mix(config: GrugMoeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


_TARGET_STEPS: int = 2**14

# Compute-optimal (budget, hidden_dim) points for the scaling sweep. Each
# scale produces one v5p-8 training run; `build_from_heuristic` sizes the
# model, optimizer, batch, and step count from the pair.
_SCALES: tuple[tuple[float, int], ...] = (
    (2.19e17, 512),
    (1.70e18, 768),
    (9.00e18, 1024),
    (2.83e19, 1280),
    (9.00e19, 1536),
)

_CACHE_BACKED_COMPONENTS = {domain: _build_component(domain, path) for domain, path in DOMAIN_CACHE_PATHS.items()}
_HIERARCHICAL_COMPONENTS = _build_hierarchical_components()


def _build_scale_step(budget: float, hidden_dim: int) -> ExecutorStep:
    model, optimizer, batch, steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_TARGET_STEPS,
    )
    experiment_budget = batch * steps * model.max_seq_len
    assert experiment_budget <= TARGET_BUDGET, (
        f"experiment_budget ({experiment_budget}) exceeds TARGET_BUDGET ({TARGET_BUDGET}) for "
        f"d{hidden_dim} / {budget:.2e}"
    )

    components = {**_CACHE_BACKED_COMPONENTS, **_HIERARCHICAL_COMPONENTS}
    base_mixture = LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=components,
        train_weights=PROPORTIONAL_WEIGHTS,
        auto_build_caches=False,
        target_budget=TARGET_BUDGET,
        experiment_budget=experiment_budget,
    )
    data = add_validation_sets_to_mixture(
        base_mixture,
        default_validation_sets(tokenizer=marin_tokenizer),
    )
    slug = f"d{hidden_dim}-{budget:.2e}"
    run_id = _resolve_run_id(f"grug_moe_mix_{slug}")

    return ExecutorStep(
        name=f"grug/grug_moe_mix_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=data,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8", zone="us-east5-a")),
            steps=versioned(steps),
            batch_size=versioned(batch),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "dolma3_dolmino_mix", slug],
                group="moe-mix",
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


grug_moe_mix_steps: list[ExecutorStep] = [_build_scale_step(budget, dim) for budget, dim in _SCALES]


if __name__ == "__main__":
    executor_main(
        steps=grug_moe_mix_steps,
        description="Grug MoE on Dolma 3 + Dolmino top-level mixture, scales d512/d768/d1024/d1280/d1536.",
    )
