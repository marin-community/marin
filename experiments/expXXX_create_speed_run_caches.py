"""
Script to create tokenized subcaches for speedrun
"""

from experiments.exp524_tokenizers import fineweb_edu_llama3_tokenized
from marin.execution import executor_main
from marin.export import upload_dir_to_hf
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.speedrun.slice_cache import slice_cache

base_cache = fineweb_edu_llama3_tokenized

# (Use this naming because we're going to use fineweb_edu_subcache_10B to mean the downloaded one)
fineweb_edu_subcache_10B_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10B",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000_000,
)

fineweb_edu_10B_repo_id = "marin-community/fineweb-edu-pretokenized-10B"

uploaded_cert = upload_dir_to_hf(fineweb_edu_subcache_10B_created, repo_id=fineweb_edu_10B_repo_id)

# 10M is mostly for debugging purposes
fineweb_edu_subcache_10M_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10M",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000,
)

fineweb_edu_10M_repo_id = "marin-community/fineweb-edu-pretokenized-10M"

uploaded_cert2 = upload_dir_to_hf(fineweb_edu_subcache_10M_created, repo_id=fineweb_edu_10M_repo_id)

if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_subcache_10B_created, uploaded_cert, fineweb_edu_subcache_10M_created, uploaded_cert2],
        description="Create subcaches of the fineweb-edu dataset.",
    )
