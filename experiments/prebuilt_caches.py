"""
Script to create tokenized subcaches for speedrun.

This script is used to create subcaches of the fineweb-edu dataset for use in Marin Speedrun.

Running this experiment will create two subcaches:
1. A 10B token subcache, which is a subset of the original fineweb-edu dataset consisting of approximately 10B tokens.
2. A 10M token subcache, which is a smaller subset of the original fineweb-edu dataset. (Mostly for testing purposes)

You can use these subcaches to get started faster by using the prebuilt caches instead of running the full tokenization
process:

```
from experiments.prebuilt_caches import fineweb_edu_subcache_10B

my_model = default_train(..., tokenized=fineweb_edu_subcache_10B, ...)
```

"""

from experiments.exp524_tokenizers import fineweb_edu_llama3_tokenized
from experiments.marin_models import marin_tokenizer
from marin.execution import executor_main
from marin.export import upload_dir_to_hf
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.processing.tokenize.download_pretokenized import download_pretokenized_cache
from marin.speedrun.slice_cache import slice_cache

# If you want to use prebuilt caches, you should use these steps and not the ones at the end of the file
fineweb_edu_10B_repo_id = "marin-community/fineweb-edu-pretokenized-10B"
fineweb_edu_subcache_10B = download_pretokenized_cache("fineweb-edu-10B", fineweb_edu_10B_repo_id, marin_tokenizer)

fineweb_edu_10M_repo_id = "marin-community/fineweb-edu-pretokenized-10M"
fineweb_edu_subcache_10M = download_pretokenized_cache("fineweb-edu-10M", fineweb_edu_10M_repo_id, marin_tokenizer)


### The following code is for creating the subcaches from the original dataset. You generally don't need to run this.

base_cache = fineweb_edu_llama3_tokenized

# (Use this naming because we're going to use fineweb_edu_subcache_10B to mean the downloaded one)
fineweb_edu_subcache_10B_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10B",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000_000,
)


uploaded_cert_10B = upload_dir_to_hf(fineweb_edu_subcache_10B_created, repo_id=fineweb_edu_10B_repo_id)

# 10M is mostly for debugging purposes
fineweb_edu_subcache_10M_created = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-10M",
    input_config=step_to_lm_mixture_component(fineweb_edu_llama3_tokenized, include_raw_paths=True),
    num_tokens=10_000_000,
)


uploaded_cert_10M = upload_dir_to_hf(fineweb_edu_subcache_10M_created, repo_id=fineweb_edu_10M_repo_id)

if __name__ == "__main__":
    executor_main(
        steps=[fineweb_edu_subcache_10B_created, uploaded_cert_10B, fineweb_edu_subcache_10M_created, uploaded_cert_10M],
        description="Create subcaches of the fineweb-edu dataset.",
    )
