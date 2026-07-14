# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The code subsets of allenai/dolma3.5_pool.

* ``dolma_code`` -- Stack v2, reprocessed for code quality.
* ``dolma_code_prose`` -- web documents mixing prose and code.

The pool is one HF repo with a directory per component and is nearly 10T tokens in
full, so each subset downloads under a glob scoped to its own directory. The pool's
PDF subset is ``dolma4pdfs.py``.

``dolma_code_prose`` contains documents long enough to OOM a tokenizer worker at the
default ``ram="10g"``; tokenize it with more.
"""

from marin.datakit.download.hf_simple_util import hf_normalize_steps
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "allenai/dolma3.5_pool"
HF_REVISION = "d2bf6ae"

SUBSETS = ("dolma_code", "dolma_code_prose")


def dolma3_5_code_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for the dolma3.5_pool code subsets.

    Keyed by the registry name, which matches the upstream directory name.
    """
    return {
        subset: hf_normalize_steps(
            marin_name=subset,
            hf_dataset_id=HF_DATASET_ID,
            revision=HF_REVISION,
            staged_path=f"raw/dolma3_5_pool-{HF_REVISION}/{subset}",
            hf_urls_glob=(f"{subset}/**/*.jsonl.zst",),
            file_extensions=(".jsonl.zst",),
        )
        for subset in SUBSETS
    }
