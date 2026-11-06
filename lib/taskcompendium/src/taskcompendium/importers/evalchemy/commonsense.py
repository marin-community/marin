# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CommonsenseQA, COPA and WSC likelihood prompts from lm-eval v0.4.12."""

from taskcompendium.importers.evaluation_harness import ChoiceDocument
from taskcompendium.importers.fewshot import ChoiceLayout, ChoiceTemplate, Document, FewshotPolicy, FewshotSampling


def commonsense_qa_document(doc: Document) -> ChoiceDocument:
    letters = tuple("ABCDE")
    prompt = "Question: " + doc["question"].strip() + "\n"
    prompt += "\n".join(f"{letter}. {text}" for letter, text in zip(letters, doc["choices"]["text"], strict=True))
    return ChoiceDocument(prompt + "\nAnswer:", letters, letters.index(doc["answerKey"]))


def copa_document(doc: Document) -> ChoiceDocument:
    connector = {"cause": "because", "effect": "therefore"}[doc["question"]]
    choices = tuple(" " + doc[key][0].lower() + doc[key][1:] for key in ("choice1", "choice2"))
    return ChoiceDocument(doc["premise"].strip()[:-1] + " " + connector, choices, int(doc["label"]))


def wsc_document(doc: Document) -> ChoiceDocument:
    text = doc["text"].replace("  ", " ")
    options = []
    for option in doc["options"]:
        if doc["pronoun"].lower() in {"my", "his", "her", "our", "their"}:
            option += "'s"
        pronoun = option.split()[0]
        if text[doc["pronoun_loc"] - 2] != "." and pronoun in {
            "A",
            "An",
            "The",
            "She",
            "He",
            "It",
            "They",
            "My",
            "His",
            "Her",
            "Their",
        }:
            option = option.replace(pronoun, pronoun.lower())
        options.append(text[: doc["pronoun_loc"]] + option)
    ending = text[doc["pronoun_loc"] + len(doc["pronoun"]) :]
    return ChoiceDocument(ending, tuple(options), int(doc["label"]))


def commonsense_qa_template(shots: int, *, pool_file: str = "") -> ChoiceTemplate:
    # Both configured split names are unset in the pinned harness: draw n+1,
    # even when demonstrations happen to come from a separate training file.
    return ChoiceTemplate(
        "commonsense_qa",
        commonsense_qa_document,
        FewshotPolicy(shots, FewshotSampling.EXCLUDE_CURRENT, pool_file=pool_file),
    )


def copa_template() -> ChoiceTemplate:
    return ChoiceTemplate("copa", copa_document, FewshotPolicy(0, FewshotSampling.EXCLUDE_CURRENT))


def wsc_template() -> ChoiceTemplate:
    return ChoiceTemplate(
        "wsc273",
        wsc_document,
        FewshotPolicy(0, FewshotSampling.RANDOM),
        layout=ChoiceLayout.CONTEXT_CHOICES,
    )
