# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Uncheatable Eval subsets as lazy validation ``Dataset`` handles.

The Uncheatable Eval dumps live in a GitHub repo with no pinned download, so the
raw data is a :func:`raw_download` handle (re-fetched on demand — a small repo) that
each subset tokenizer depends on.
"""

from marin.datakit.download.uncheatable_eval import (
    UncheatableEvalDownloadConfig,
    download_latest_uncheatable_eval,
)
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import raw_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# The English + code subsets included in the eval, mapped to their file globs.
UNCHEATABLE_SUBSETS = {
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}


def uncheatable_raw() -> ArtifactStep[TokenizedCache]:
    """The Uncheatable Eval GitHub dump as a raw-download handle."""
    return raw_download(
        "raw/uncheatable_eval",
        fn=download_latest_uncheatable_eval,
        build_config=lambda ctx: UncheatableEvalDownloadConfig(
            output_path=ctx.output_path,
            repo_owner="ziqing-huang",
            repo_name="uncheatable_eval",
            data_path="data",
            branch="master",
        ),
        version="2026.06.28",
    )


def uncheatable_validation(*, tokenizer: str = llama3_tokenizer) -> list[ArtifactStep[TokenizedCache]]:
    """One validation ``TokenizedCache`` handle per Uncheatable Eval subset, keyed by
    ``uncheatable_eval/<subset>-llama3``, each tokenizing from the shared raw download."""
    raw = uncheatable_raw()
    return [
        tokenized(
            f"uncheatable_eval/{subset}-llama3",
            tokenizer=tokenizer,
            version="2026.06.28",
            raw=raw,
            glob=pattern,
            validation=True,
        )
        for subset, pattern in UNCHEATABLE_SUBSETS.items()
    ]
