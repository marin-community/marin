# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen Europe runtime-cache identities for the Delphi TPP40 replay."""

EUROPE_RUNTIME_CACHE_REGION = "europe-west4"
EUROPE_RUNTIME_CACHE_PREFIX = "gs://marin-eu-west4"
EUROPE_HISTORICAL_STACK_INPUT_PREFIX = (
    f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/dolma3_pool_historical_full_document_v1/"
)
EUROPE_HISTORICAL_STACK_MERGED_PREFIX = (
    f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/merged/" "dolma3_dolmino_top_level_historical_full_document_v1/"
)
EUROPE_HISTORICAL_STACK_MERGED_PATH = f"{EUROPE_HISTORICAL_STACK_MERGED_PREFIX}dolma3_stack_edu-5eb331"
EUROPE_HISTORICAL_DOLMINO_PREFIX = (
    f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/dolma3_dolmino_pool_historical_full_document_v1/"
)
EUROPE_HISTORICAL_FLAN_EAST5_SUBSET_PATH = (
    f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/"
    "dolma3_dolmino_pool_historical_full_document_east5_subset_v1/"
    "synth_instruction_dolmino_flan-985ec1"
)
EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS = {
    "finemath_3plus": f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/finemath_3_plus_historical_full_document_v1-244ece",
    "dolmino_stem_heavy_crawl": (
        f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/merged/"
        "dolma3_dolmino_top_level_historical_full_document_v1/dolmino_stem_heavy_crawl-4f736e"
    ),
    "synth_instruction/dolmino_flan": EUROPE_HISTORICAL_FLAN_EAST5_SUBSET_PATH,
    "synth_math/dolmino_math": f"{EUROPE_HISTORICAL_DOLMINO_PREFIX}synth_math_dolmino_math-b671fa",
    "synth_qa/wiki_to_rcqa": f"{EUROPE_HISTORICAL_DOLMINO_PREFIX}synth_qa_wiki_to_rcqa-841841",
    "synth_thinking/code_meta_reasoning": f"{EUROPE_HISTORICAL_DOLMINO_PREFIX}synth_thinking_code_meta_reasoning-ba957d",
    "synth_thinking/math_meta_reasoning": f"{EUROPE_HISTORICAL_DOLMINO_PREFIX}synth_thinking_math_meta_reasoning-134f48",
    "synth_thinking/program_verifiable": f"{EUROPE_HISTORICAL_DOLMINO_PREFIX}synth_thinking_program_verifiable-38e32c",
}
EXPECTED_STACK_TOKENS = 134_071_054_270
EXPECTED_STACK_ELEMENTS = 167_063_162

# These hashes identify the successful Europe-local tokenization outputs found
# after the 38-domain preparation completed. Keeping them explicit prevents a
# training launcher from selecting an incomplete duplicate or rebuilding data.
EUROPE_SOURCE_CACHE_HASHES = {
    "common_crawl_hq/19_adult_content": "986941",
    "common_crawl_hq/19_art_and_design": "140701",
    "common_crawl_hq/19_crime_and_law": "7d7fdc",
    "common_crawl_hq/19_education_and_jobs": "90ee22",
    "common_crawl_hq/19_electronics_and_hardware": "39f802",
    "common_crawl_hq/19_entertainment": "ae89d3",
    "common_crawl_hq/19_fashion_and_beauty": "b5d2e1",
    "common_crawl_hq/19_finance_and_business": "20e87b",
    "common_crawl_hq/19_food_and_dining": "d57db7",
    "common_crawl_hq/19_games": "9e426a",
    "common_crawl_hq/19_health": "4a7919",
    "common_crawl_hq/19_history_and_geography": "a181cb",
    "common_crawl_hq/19_home_and_hobbies": "52899e",
    "common_crawl_hq/19_industrial": "365cbc",
    "common_crawl_hq/19_literature": "8f2b65",
    "common_crawl_hq/19_politics": "f0c570",
    "common_crawl_hq/19_religion": "f71c2c",
    "common_crawl_hq/19_science_math_and_technology": "9676a0",
    "common_crawl_hq/19_social_life": "e01f54",
    "common_crawl_hq/19_software": "e41483",
    "common_crawl_hq/19_software_development": "12bde0",
    "common_crawl_hq/19_sports_and_fitness": "55671d",
    "common_crawl_hq/19_transportation": "8ef6d4",
    "common_crawl_hq/19_travel_and_tourism": "0b507c",
    "common_crawl_hq/20_adult_content": "60c0fc",
    "common_crawl_hq/20_art_and_design": "f6c229",
    "common_crawl_hq/20_crime_and_law": "08910c",
    "common_crawl_hq/20_education_and_jobs": "6b6514",
    "common_crawl_hq/20_electronics_and_hardware": "78eb1c",
    "common_crawl_hq/20_entertainment": "1c2777",
    "common_crawl_hq/20_fashion_and_beauty": "ec1192",
    "common_crawl_hq/20_finance_and_business": "edf4dd",
    "common_crawl_hq/20_food_and_dining": "495c5e",
    "common_crawl_hq/20_games": "a5427f",
    "common_crawl_hq/20_health": "50e664",
    "common_crawl_hq/20_history_and_geography": "40625e",
    "common_crawl_hq/20_home_and_hobbies": "ea57b3",
    "common_crawl_hq/20_industrial": "89e820",
    "common_crawl_hq/20_literature": "1d4132",
    "common_crawl_hq/20_politics": "6b2fc1",
    "common_crawl_hq/20_religion": "553dce",
    "common_crawl_hq/20_science_math_and_technology": "8be542",
    "common_crawl_hq/20_social_life": "f4a77b",
    "common_crawl_hq/20_software": "2eaab8",
    "common_crawl_hq/20_software_development": "bf8388",
    "common_crawl_hq/20_sports_and_fitness": "32a5cd",
    "common_crawl_hq/20_transportation": "77fd33",
    "common_crawl_hq/20_travel_and_tourism": "9b2bb7",
    "olmocr_pdfs_hq/adult_content": "782fdf",
    "olmocr_pdfs_hq/art_and_design": "f66e82",
    "olmocr_pdfs_hq/crime_and_law": "6bcf3f",
    "olmocr_pdfs_hq/education_and_jobs": "8c857e",
    "olmocr_pdfs_hq/electronics_and_hardware": "9efa57",
    "olmocr_pdfs_hq/entertainment": "935e9d",
    "olmocr_pdfs_hq/fashion_and_beauty": "9e33a9",
    "olmocr_pdfs_hq/finance_and_business": "1db9a9",
    "olmocr_pdfs_hq/food_and_dining": "9c9ccd",
    "olmocr_pdfs_hq/games": "2247ec",
    "olmocr_pdfs_hq/health": "6592c9",
    "olmocr_pdfs_hq/history_and_geography": "0aa517",
    "olmocr_pdfs_hq/home_and_hobbies": "f2f3a6",
    "olmocr_pdfs_hq/industrial": "fdf3cc",
    "olmocr_pdfs_hq/literature": "b8288a",
    "olmocr_pdfs_hq/politics": "12e4e1",
    "olmocr_pdfs_hq/religion": "79f07d",
    "olmocr_pdfs_hq/science_math_and_technology": "6b1eb9",
    "olmocr_pdfs_hq/software": "07945d",
    "olmocr_pdfs_hq/software_development": "6b0967",
    "olmocr_pdfs_hq/sports_and_fitness": "7e7c4c",
    "olmocr_pdfs_hq/transportation": "8bf699",
    "olmocr_pdfs_hq/travel_and_tourism": "35e5a9",
    "stack_edu_fim/C": "421921",
    "stack_edu_fim/CSharp": "826de5",
    "stack_edu_fim/Cpp": "acccdd",
    "stack_edu_fim/Go": "d945e3",
    "stack_edu_fim/Java": "f0c478",
    "stack_edu_fim/JavaScript": "f31610",
    "stack_edu_fim/Markdown": "b900ef",
    "stack_edu_fim/PHP": "6a476c",
    "stack_edu_fim/Python": "b0c1be",
    "stack_edu_fim/Ruby": "12aa83",
    "stack_edu_fim/Rust": "d4ee4a",
    "stack_edu_fim/SQL": "0724c4",
    "stack_edu_fim/Shell": "500b3c",
    "stack_edu_fim/Swift": "20bbbb",
    "stack_edu_fim/TypeScript": "917f94",
    "synth_code/cranecode": "3d2447",
    "synth_instruction/dolmino_flan": "183f12",
    "synth_instruction/tulu_3_sft": "00fa09",
    "synth_math/cranemath": "2896f8",
    "synth_math/dolmino_math": "6a90af",
    "synth_math/megamatt": "862c18",
    "synth_math/tinymath_mind": "f01a63",
    "synth_math/tinymath_pot": "c60e19",
    "synth_math/verifiable_gpt41": "6e5533",
    "synth_math/verifiable_o4mini": "2cbec0",
    "synth_qa/nemotron_synth_qa": "4c6ea5",
    "synth_qa/reddit_to_flashcards": "9acbf6",
    "synth_qa/wiki_to_rcqa": "bd4afa",
    "synth_thinking/code_meta_reasoning": "89ea11",
    "synth_thinking/gemini_reasoning": "fa77a6",
    "synth_thinking/general_reasoning_mix": "cb5cb6",
    "synth_thinking/llama_nemotron_reasoning": "1e9de1",
    "synth_thinking/math_meta_reasoning": "c0fdb1",
    "synth_thinking/omr_rewrite_fullthoughts": "e0eb6c",
    "synth_thinking/openthoughts2_reasoning": "fd22f4",
    "synth_thinking/program_verifiable": "bc5995",
    "synth_thinking/qwq_reasoning": "467558",
    "synth_thinking/r1_reasoning": "b23788",
}

EUROPE_SOURCE_RUNTIME_CACHE_PATHS = {
    "arxiv": f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/dolma/arxiv-07a51f",
    "finemath_3plus": f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/finemath_3_plus-a26b0f",
    "wikipedia": f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/dolma/wiki-212315",
    **{
        partition_name: (
            f"{EUROPE_RUNTIME_CACHE_PREFIX}/tokenized/dolma3_dolmino_pool/"
            f"{partition_name.replace('/', '_')}-{cache_hash}"
        )
        for partition_name, cache_hash in EUROPE_SOURCE_CACHE_HASHES.items()
    },
    **{
        component: path
        for component, path in EUROPE_HISTORICAL_NONSTACK_REPAIR_PATHS.items()
        if component != "dolmino_stem_heavy_crawl"
    },
}
