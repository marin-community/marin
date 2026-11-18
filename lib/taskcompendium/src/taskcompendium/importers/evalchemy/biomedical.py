# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PubMedQA likelihood prompt from lm-eval v0.4.12 (`pubmedqa` task, MIT license)."""

from taskcompendium.importers.fewshot import (
    ChoiceDocument,
    ChoiceTemplate,
    Document,
    FewshotPolicy,
    FewshotSampling,
)

PUBMEDQA_CHOICES = ("yes", "no", "maybe")


def pubmedqa_document(doc: Document) -> ChoiceDocument:
    """Render a `pubmed_qa_labeled_fold0_source` row; abstract sections are newline-joined."""
    abstract = "\n".join(doc["CONTEXTS"])
    prompt = f"Abstract: {abstract}\nQuestion: {doc['QUESTION']}\nAnswer:"
    return ChoiceDocument(prompt, PUBMEDQA_CHOICES, PUBMEDQA_CHOICES.index(doc["final_decision"]))


def pubmedqa_template() -> ChoiceTemplate:
    return ChoiceTemplate("pubmedqa", pubmedqa_document, FewshotPolicy(0, FewshotSampling.FIRST))
