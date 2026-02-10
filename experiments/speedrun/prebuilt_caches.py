# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""

Provides prebuilt "subcaches" of the fineweb-edu dataset for use in Marin Speedrun. There are currently two subcaches:

1. A 10B token subcache, which is a subset of the original fineweb-edu dataset consisting of approximately 10B tokens.
2. A 10M token subcache, which is a smaller subset of the original fineweb-edu dataset. (Mostly for testing purposes)


You can use these subcaches to get started faster by using the prebuilt caches instead of running the full tokenization
process:

```
from experiments.prebuilt_caches import fineweb_edu_subcache_10B

my_model = default_train(..., tokenized=fineweb_edu_subcache_10B, ...)
```

They are built with experiments.speedrun.build_prebuilt_caches.py.

"""

from experiments.marin_models import marin_tokenizer
from marin.processing.tokenize.download_pretokenized import download_pretokenized_cache

fineweb_edu_10B_repo_id = "marin-community/fineweb-edu-pretokenized-10B"
fineweb_edu_subcache_10B = download_pretokenized_cache("fineweb-edu-10B", fineweb_edu_10B_repo_id, marin_tokenizer)

fineweb_edu_10M_repo_id = "marin-community/fineweb-edu-pretokenized-10M"
fineweb_edu_subcache_10M = download_pretokenized_cache("fineweb-edu-10M", fineweb_edu_10M_repo_id, marin_tokenizer)
