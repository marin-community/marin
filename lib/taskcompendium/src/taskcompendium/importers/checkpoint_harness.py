# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile checkpoint-menu protocols from pinned lm-eval v0.4.12 inputs.

Few-shot examples and BBH choice lists are retained in checkpoint_prompts.json,
from EleutherAI/lm-evaluation-harness 6d642546f4688648fced259eb3302efd36ece5af
(MIT). No harness objects or dependencies enter the published corpus.
"""

import csv
import json
import random
import re
import string
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.spec import ChoiceLikelihood, LikelihoodRequest
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evaluation_harness import FEWSHOT_SEED, ChoiceDocument, parquet_documents
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow


@cache
def checkpoint_prompts() -> dict:
    return json.loads((Path(__file__).parent / "evalchemy/assets/checkpoint_prompts.json").read_text())


@dataclass(frozen=True)
class CheckpointTemplate:
    family: str
    subject: str = ""
    shots: int = 0
    fewshot_file: str = ""


def clean_gpqa_answer(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def checkpoint_choice(doc: Mapping, template: CheckpointTemplate) -> ChoiceDocument:
    family = template.family
    if family == "commonsense_qa":
        letters = tuple("ABCDE")
        prompt = "Question: " + doc["question"].strip() + "\n"
        prompt += "\n".join(f"{letter}. {text}" for letter, text in zip(letters, doc["choices"]["text"], strict=True))
        return ChoiceDocument(prompt + "\nAnswer:", letters, letters.index(doc["answerKey"]))
    if family == "copa":
        connector = {"cause": "because", "effect": "therefore"}[doc["question"]]
        choices = tuple(" " + doc[key][0].lower() + doc[key][1:] for key in ("choice1", "choice2"))
        return ChoiceDocument(doc["premise"].strip()[:-1] + " " + connector, choices, int(doc["label"]))
    if family == "agieval_lsat_ar":
        return ChoiceDocument(doc["query"], tuple(doc["choices"]), int(doc["gold"][0]))
    if family == "leaderboard_mmlu_pro":
        choices = tuple(string.ascii_uppercase[: len(doc["options"])])
        prompt = (
            doc["question"]
            + "\n"
            + "\n".join(f"{letter}. {text}" for letter, text in zip(choices, doc["options"], strict=True))
            + "\nAnswer:"
        )
        return ChoiceDocument(prompt, choices, int(doc["answer_index"]))
    if family == "leaderboard_bbh":
        assets = checkpoint_prompts()["bbh/" + template.subject]
        choices = tuple(assets["doc_to_choice"])
        return ChoiceDocument("Q: " + doc["input"] + "\nA:", choices, choices.index(doc["target"]))
    if family == "leaderboard_gpqa":
        choices = tuple(f"({letter})" for letter in "ABCD")
        answers = doc["prepared_choices"]
        prompt = "What is the correct answer to this question:" + doc["Question"] + "\nChoices:\n"
        prompt += "\n".join(f"{letter} {text}" for letter, text in zip(choices, answers, strict=True)) + "\nAnswer: "
        return ChoiceDocument(prompt, choices, answers.index(clean_gpqa_answer(doc["Correct Answer"])))
    if family == "wsc273":
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
    raise ValueError(f"Unsupported checkpoint likelihood family: {family}")


def _generation_prompt(doc: Mapping, template: CheckpointTemplate) -> str:
    if template.family == "gsm8k_cot":
        return "Q: " + doc["question"] + "\nA:"
    if template.family == "bbh_cot_fewshot":
        return "Q: " + doc["input"] + "\nA: Let's think step by step.\n"
    if template.family == "minerva_math":
        return "Problem:\n" + doc["problem"] + "\n\nSolution:"
    raise ValueError(f"Unsupported checkpoint generation family: {template.family}")


def read_checkpoint_file(
    path: StoragePath, context: ConversionContext, *, template: CheckpointTemplate
) -> Iterator[dict]:
    if path.name.endswith(".csv"):
        with path.open("rt") as stream:
            documents = list(csv.DictReader(stream))
    else:
        documents = list(parquet_documents(path))
    family = template.family
    assets = checkpoint_prompts()
    description = ""
    if family in {"leaderboard_bbh", "bbh_cot_fewshot"}:
        source = assets[("bbh/" if family == "leaderboard_bbh" else "bbh_cot/") + template.subject]
        pool = source["fewshot_config"]["samples"]
        description = source["description"]
        # ContextSampler appends its few-shot delimiter to a nonempty description.
        description += "\n\n"
    elif family in {"gsm8k_cot", "minerva_math"}:
        pool = assets[family]
    elif template.fewshot_file:
        pool = list(parquet_documents(context.inputs["fewshot"] / template.fewshot_file))
    else:
        pool = documents
    sampler = random.Random(FEWSHOT_SEED)
    shuffle = random.Random(0)
    generation = family in {"gsm8k_cot", "bbh_cot_fewshot", "minerva_math"}
    first_n = generation or family in {"leaderboard_bbh", "leaderboard_mmlu_pro", "leaderboard_gpqa"}
    for original in documents:
        doc = dict(original)
        if family == "leaderboard_gpqa":
            answers = [
                clean_gpqa_answer(doc[key])
                for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")
            ]
            shuffle.shuffle(answers)
            doc["prepared_choices"] = answers
        indices = (
            list(range(template.shots))
            if first_n
            else sampler.sample(range(len(pool)), template.shots + int(pool is documents and bool(template.shots)))
        )
        if pool is documents:
            indices = [index for index in indices if pool[index] != original][: template.shots]
        prefix = description
        for index in indices:
            example = pool[index]
            if generation:
                target = example["solution"] if family == "minerva_math" else example["target"]
                prefix += (
                    _generation_prompt(example, template)
                    + ("" if family == "bbh_cot_fewshot" else " ")
                    + target
                    + "\n\n"
                )
            else:
                rendered = checkpoint_choice(example, template)
                prefix += rendered.prompt + " " + rendered.choices[rendered.gold] + "\n\n"
        if generation:
            yield {
                "document": doc,
                "prompt": prefix + _generation_prompt(doc, template),
                "profile": family,
                "subject": template.subject,
                "num_fewshot": template.shots,
            }
            continue
        rendered = checkpoint_choice(doc, template)
        requests = tuple(
            (
                LikelihoodRequest(context=choice, continuation=rendered.prompt)
                if family == "wsc273"
                else LikelihoodRequest(context=prefix + rendered.prompt, continuation=" " + choice)
            )
            for choice in rendered.choices
        )
        yield {
            "document": doc,
            "task_name": family,
            "operation": (
                ChoiceLikelihood(
                    choices=rendered.choices,
                    requests=requests,
                    gold=tuple(doc["gold"]) if family == "agieval_lsat_ar" else (rendered.gold,),
                ).model_dump(mode="json")
            ),
            "fewshot_indices": indices,
            "num_fewshot": template.shots,
        }


def read_humaneval_fewshot(path: StoragePath, _context: ConversionContext) -> Iterator[dict]:
    """Preserve the legacy ten-shot completion menu, including its test-text demonstrations."""
    documents = list(parquet_documents(path))
    sampler = random.Random(FEWSHOT_SEED)
    for document in documents:
        indices = sampler.sample(range(len(documents)), 11)
        selected = [documents[index] for index in indices if documents[index] != document][:10]
        prefix = "".join(
            example["prompt"] + " " + example["test"] + "\ncheck(" + example["entry_point"] + ")\n\n"
            for example in selected
        )
        yield {**document, "evaluation_prompt": prefix + document["prompt"]}


def convert_checkpoint_generation(row: RawRow, context: ConversionContext) -> TaskSpec:
    return conversation_task(
        row,
        events=(TextMessage(role="user", content=row.data["prompt"]),),
        package=source_package(
            "checkpoint_generation",
            {"profile": row.data["profile"], "document": row.data["document"]},
            environment=required_grader_environment(context),
        ),
        evidence=row.data,
    ).model_copy(update={"tags": ("eval", "source:lm-eval", "checkpoint")})
