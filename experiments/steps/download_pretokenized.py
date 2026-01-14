# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Step wrappers for downloading pretokenized caches from Hugging Face.

This module provides step definitions that wrap the library functions in
marin.processing.tokenize.download_pretokenized.
"""

import os

from levanter.data.text import LmDatasetFormatBase, TextLmDatasetFormat

from marin.execution import ExecutorStep, StepContext, ensure_versioned, step
from marin.processing.tokenize.download_pretokenized import (
    PretokenizedCacheDownloadConfig,
    download_pretokenized_cache as download_pretokenized_cache_lib,
)


@step(name="tokenized/subcache/{output_cache_path_name}", fn=download_pretokenized_cache_lib)
def download_pretokenized_cache(
    ctx: StepContext,
    output_cache_path_name: str,
    hf_repo_id: str,
    tokenizer: str,
    hf_revision: str | None = None,
    hf_token: str | None = None,
    tags: list[str] | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa: A002
) -> ExecutorStep[PretokenizedCacheDownloadConfig]:
    """
    Create a step to download a pre-tokenized Levanter cache from Hugging Face.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        output_cache_path_name: The logical name for this download step. The Executor will use this
                                to construct the actual output directory for the cache.
        hf_repo_id: The Hugging Face repository ID (e.g., "username/my_cache_repo").
        tokenizer: The name or path of the tokenizer associated with this cache.
        hf_revision: The specific revision, branch, or tag of the repository to download.
        hf_token: An optional Hugging Face API token for accessing private repositories.
        tags: Optional list of tags for the Levanter LMDatasetSourceConfig.
        format: The format of the dataset (default is TextLmDatasetFormat).

    Returns:
        An ExecutorStep that, when run, will download the cache and output a
        PretokenizedCacheDownloadConfig pointing to the downloaded data.
    """
    return PretokenizedCacheDownloadConfig(
        cache_path=ctx.output,
        tokenizer=ensure_versioned(tokenizer),
        hf_repo_id=ensure_versioned(hf_repo_id),  # type: ignore[call-arg]
        hf_revision=ensure_versioned(hf_revision),  # type: ignore[call-arg]
        hf_repo_type_prefix="datasets",  # Default for Hugging Face datasets
        hf_token=hf_token,
        tags=tags or [],
        format=format,
    )
