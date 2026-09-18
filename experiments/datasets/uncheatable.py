# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""UncheatableEval subsets from the pinned July 2026 release."""

import hashlib
from types import MappingProxyType

from marin.datakit.download.uncheatable_eval import UncheatableEvalTransformConfig, transform_uncheatable_eval
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer
from experiments.marin_tokenizer import marin_tokenizer

UNCHEATABLE_EVAL_DATASET_ID = "Jellyfish042/UncheatableEval-2026-07"
UNCHEATABLE_EVAL_REVISION = "65889535d56aa38d448ce7e07b08e6e36c031545"
UNCHEATABLE_EVAL_VERSION = "2026.08.24"

# The arXiv computer-science key retains its metric name although upstream
# shortened the category to ``arxiv_cs``.
UNCHEATABLE_SUBSETS = MappingProxyType(
    {
        "wikipedia_english": "wikipedia_english",
        "wikipedia_nonenglish": "wikipedia_nonenglish",
        "github_python": "github_python",
        "github_cpp": "github_cpp",
        "github_javascript": "github_javascript",
        "github_markdown": "github_markdown",
        "github_other": "github_other",
        "bbc_news": "bbc_news",
        "arxiv_physics": "arxiv_physics",
        "arxiv_computer_science": "arxiv_cs",
        "arxiv_math": "arxiv_math",
        "arxiv_other": "arxiv_other",
        "biorxiv_all": "biorxiv_all",
        "ao3_english": "ao3_english",
        "ao3_nonenglish": "ao3_nonenglish",
    }
)


def uncheatable_raw() -> ArtifactStep[Artifact]:
    """The pinned UncheatableEval Hugging Face release."""
    return hf_download(
        "raw/uncheatable_eval",
        hf_id=UNCHEATABLE_EVAL_DATASET_ID,
        revision=UNCHEATABLE_EVAL_REVISION,
        urls_glob=["data/*.parquet"],
        version=UNCHEATABLE_EVAL_VERSION,
    )


def uncheatable_processed(raw: ArtifactStep[Artifact] | None = None) -> ArtifactStep[Artifact]:
    """The release split into one normalized file per selected category."""
    raw = raw if raw is not None else uncheatable_raw()
    return ArtifactStep(
        name="processed/uncheatable_eval",
        version=UNCHEATABLE_EVAL_VERSION,
        artifact_type=Artifact,
        run=transform_uncheatable_eval,
        build_config=lambda ctx: UncheatableEvalTransformConfig(
            input_path=ctx.artifact_path(raw),
            output_path=ctx.output_path,
            categories=tuple(UNCHEATABLE_SUBSETS.values()),
        ),
        deps=(raw,),
    )


def uncheatable_dataset(
    subset: str, *, tokenizer: str = llama3_tokenizer, processed: ArtifactStep[Artifact] | None = None
) -> ArtifactStep[TokenizedCache]:
    """One Uncheatable Eval subset as a validation handle."""
    processed = processed if processed is not None else uncheatable_processed()
    category = UNCHEATABLE_SUBSETS[subset]
    if tokenizer == llama3_tokenizer:
        tokenizer_suffix = "llama3"
    elif tokenizer == marin_tokenizer:
        tokenizer_suffix = "marin"
    else:
        tokenizer_suffix = f"tokenizer-{hashlib.sha256(tokenizer.encode()).hexdigest()[:8]}"
    return tokenized(
        f"uncheatable_eval/{subset}-{tokenizer_suffix}",
        tokenizer=tokenizer,
        version=UNCHEATABLE_EVAL_VERSION,
        raw=processed,
        glob=f"{category}.jsonl.gz",
        validation=True,
    )


def uncheatable_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """All Uncheatable Eval subsets, keyed by subset name; one shared raw download."""
    processed = uncheatable_processed()
    return {
        subset: uncheatable_dataset(subset, tokenizer=tokenizer, processed=processed) for subset in UNCHEATABLE_SUBSETS
    }


if __name__ == "__main__":
    dataset_main(uncheatable_datasets())
