# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Pile download definitions for HuggingFace-hosted subsets."""

from dataclasses import dataclass

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class CommonPileDataset:
    hf_dataset_id: str
    revision: str
    hf_urls_glob: str
    output_name: str


COMMON_PILE_DATASETS: dict[str, CommonPileDataset] = {
    "peS2o": CommonPileDataset("common-pile/peS2o", "2caeba1", "v0/documents/*.json.gz", "peS2o"),
    "pubmed": CommonPileDataset("common-pile/pubmed", "648b8cf", "data/*.jsonl.gz", "pubmed"),
    "arxiv_papers": CommonPileDataset("common-pile/arxiv_papers", "963fe98", "*.jsonl.gz", "arxiv_papers"),
    "arxiv_abstracts": CommonPileDataset("common-pile/arxiv_abstracts", "828e35d", "*.jsonl.gz", "arxiv_abstracts"),
    "caselaw": CommonPileDataset(
        "common-pile/caselaw_access_project",
        "3c2cb50",
        "*.jsonl.gz",
        "caselaw_access_project",
    ),
    "doab": CommonPileDataset("common-pile/doab", "89e7a35", "v0/*.json.gz", "doab"),
    "uk_hansard": CommonPileDataset("common-pile/uk_hansard", "05eeb43", "uk_hansard/*.jsonl.gz", "uk_hansard"),
    "peps": CommonPileDataset(
        "common-pile/python_enhancement_proposals",
        "f932757",
        "raw/documents/*.jsonl.gz",
        "python_enhancement_proposals",
    ),
    "public_domain_review": CommonPileDataset(
        "common-pile/public_domain_review",
        "e9c7669",
        "v0/*.jsonl.gz",
        "public_domain_review",
    ),
}


def download_common_pile_step(name: str) -> StepSpec:
    dataset = COMMON_PILE_DATASETS[name]
    return download_hf_step(
        f"raw/common_pile/{dataset.output_name}",
        hf_dataset_id=dataset.hf_dataset_id,
        revision=dataset.revision,
        hf_urls_glob=[dataset.hf_urls_glob],
    )
