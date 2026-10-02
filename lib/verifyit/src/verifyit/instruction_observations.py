# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare source instruction observations and existing primitive inputs."""

import json
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from verifyit.grade import InvalidTask
from verifyit.json_objects import unique_object
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec

MAPPED_IDS = {
    "keywords:frequency",
    "keywords:letter_frequency",
    "detectable_content:number_placeholders",
    "detectable_format:number_highlighted_sections",
    "change_case:capital_word_frequency",
    "punctuation:no_comma",
    "language:response_language",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "keywords:existence",
    "keywords:forbidden_words",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists",
    "detectable_format:multiple_sections",
    "detectable_format:constrained_response",
    "detectable_format:title",
    "detectable_format:json_format",
    "detectable_content:postscript",
    "startend:quotation",
    "startend:end_checker",
    "combination:repeat_prompt",
    "combination:two_responses",
}


def count_schema(expected: int, relation: str = "exactly") -> dict[str, Any]:
    if type(expected) is not int or expected < 0:
        raise InvalidTask("Instruction count must be a nonnegative integer")
    bounds = {"exactly": "const", "less than": "exclusiveMaximum", "at least": "minimum"}
    if relation not in bounds:
        raise InvalidTask("Unsupported instruction count relation")
    return {"type": "integer", bounds[relation]: expected}


def prepare_instruction_observations(
    family: str, identifier: str, instruction: Any, original: Any, text: str
) -> Sequence[tuple[dict[str, Any] | IfevalSpec, Any]] | None:
    """Return primitive inputs using trusted source builders and observation tools."""
    args = instruction.get_instruction_args() or {}
    if family == "IFEvalG" and identifier == "letters:letter_counting2":
        identifier = "keywords:letter_frequency"
    if family == "IFEvalG" and identifier not in MAPPED_IDS:
        tools = original.check_following.__globals__["instructions_util"]
        return prepare_extended_instruction_observations(identifier, args, tools, text)
    if family not in {"LiveBench", "IFEval", "IFEvalG"} or identifier not in MAPPED_IDS:
        return None
    if identifier == "combination:two_responses":
        return [(IfevalSpec((Constraint(identifier, {}),), empty_output=EmptyOutputPolicy.GRADE), text)]
    if identifier in {"language:response_language", "change_case:english_capital", "change_case:english_lowercase"}:
        langdetect = original.check_following.__globals__["langdetect"]

        expected = args["language"] if identifier == "language:response_language" else "en"
        if not isinstance(expected, str) or not expected:
            raise InvalidTask("Instruction language must be nonempty text")
        components: list[tuple[dict[str, Any] | IfevalSpec, Any]] = []
        if identifier != "language:response_language":
            normalized = text.upper() if identifier.endswith("capital") else text.lower()
            components = [
                (IfevalSpec((Constraint(identifier, {}),), empty_output=EmptyOutputPolicy.GRADE), text),
                ({"const": normalized}, text),
            ]
        try:
            detected = langdetect.detect(text)
        except langdetect.LangDetectException:
            detected = None
        components.append(({"type": "string", "const": expected}, detected))
        return components
    schema: dict[str, Any] = {"type": "string"}
    instance: Any = text
    if identifier in {"keywords:existence", "keywords:forbidden_words"}:
        forbidden = identifier.endswith("forbidden_words")
        patterns = args["forbidden_words" if forbidden else "keywords"]
        if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
            raise InvalidTask("Instruction patterns must be strings")
        constraints = [{"pattern": "(?i)" + (r"\b" + pattern + r"\b" if forbidden else pattern)} for pattern in patterns]
        schema["allOf"] = [{"not": item} for item in constraints] if forbidden else constraints
    elif identifier in {"length_constraints:number_words", "length_constraints:number_sentences"}:
        kind = "words" if identifier.endswith("words") else "sentences"
        schema = count_schema(args["num_" + kind], args["relation"])
        namespace = original.check_following.__globals__
        tokenizer = (
            namespace["count_" + kind]
            if family == "IFEval"
            else getattr(namespace["instructions_util"], "count_" + kind)
        )
        instance = tokenizer(text)
    elif identifier == "keywords:frequency":
        schema = count_schema(args["frequency"], args["relation"])
        try:
            parser = re.compile(args["keyword"], re.IGNORECASE)
        except re.error as error:
            raise InvalidTask("Invalid trusted keyword pattern") from error
        instance = len(parser.findall(text))
    elif identifier == "keywords:letter_frequency":
        schema = count_schema(args["let_frequency"], args["let_relation"])
        instance = text.lower().count(args["letter"])
    elif identifier == "detectable_content:number_placeholders":
        schema = count_schema(args["num_placeholders"], "at least")
        instance = len(re.findall(r"\[.*?\]", text))
    elif identifier == "detectable_format:number_highlighted_sections":
        schema = count_schema(args["num_highlights"], "at least")
        single = [value.strip("*").strip() for value in re.findall(r"\*[^\n\*]*\*", text)]
        double = [value.removeprefix("**").removesuffix("**").strip() for value in re.findall(r"\*\*[^\n\*]*\*\*", text)]
        instance = len([value for value in single + double if value])
    elif identifier == "change_case:capital_word_frequency":
        schema = count_schema(args["capital_frequency"], args["capital_relation"])
        namespace = original.check_following.__globals__
        nltk = namespace["nltk"] if family == "IFEval" else namespace["instructions_util"].nltk
        instance = len([word for word in nltk.word_tokenize(text) if word.isupper()])
    elif identifier == "punctuation:no_comma":
        schema["not"] = {"pattern": ","}
    elif identifier == "length_constraints:number_paragraphs":
        parts = [part.strip() for part in re.split(r"\s?\*\*\*\s?", text)]
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        count = args["num_paragraphs"]
        count_schema(count)
        schema = {"type": "array", "minItems": count, "maxItems": count, "items": {"type": "string", "minLength": 1}}
        instance = parts
    elif identifier == "length_constraints:nth_paragraph_first_word":
        parts = re.split(r"\n\n", text)
        nth = args["nth_paragraph"]
        if type(nth) is not int or nth < 1:
            raise InvalidTask("Paragraph index must be positive")
        words = parts[nth - 1].strip().split() if nth <= len(parts) else []
        first = re.split(r"""[.,?!'"]""", words[0].lstrip("'").lstrip('"'))[0].lower() if words else ""
        schema = {
            "type": "object",
            "properties": {
                "count": {**count_schema(args["num_paragraphs"]), "minimum": nth},
                "first": {"const": args["first_word"]},
            },
        }
        instance = {"count": len([part for part in parts if part.strip()]), "first": first}
    elif identifier == "detectable_format:number_bullet_lists":
        instance = len(re.findall(r"^\s*\*[^\*].*$", text, re.MULTILINE)) + len(
            re.findall(r"^\s*-.*$", text, re.MULTILINE)
        )
        schema = count_schema(args["num_bullets"])
    elif identifier == "detectable_format:multiple_sections":
        pattern = r"\s?" + args["section_spliter"] + r"\s?\d+\s?"
        schema = count_schema(args["num_sections"], "at least")
        try:
            parser = re.compile(pattern)
        except re.error as error:
            raise InvalidTask("Invalid trusted section delimiter") from error
        instance = len(parser.split(text)) - 1
    elif identifier == "detectable_format:constrained_response":
        schema["anyOf"] = [{"pattern": re.escape(value)} for value in instruction._constrained_responses]
        instance = text.strip()
    elif identifier == "detectable_format:title":
        instance = [title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", text)]
        schema = {"type": "array", "contains": {"type": "string", "minLength": 1}}
    elif identifier == "detectable_format:json_format":
        value = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        schema = {"type": "object", "required": ["parsed"]}
        try:
            instance = {"parsed": json.loads(value, object_pairs_hook=unique_object)}
        except (ValueError, RecursionError):
            instance = {}
    elif identifier == "detectable_content:postscript":
        marker = args["postscript_marker"]
        pattern = {"P.P.S": r"\s*p\.\s?p\.\s?s.*$", "P.S.": r"\s*p\.\s?s\..*$"}.get(
            marker, r"\s*" + marker.lower() + r".*$"
        )
        schema["pattern"] = "(?m)" + pattern
        instance = text.lower()
    elif identifier == "startend:quotation":
        schema.update(minLength=2, pattern='(?s)^".*"$')
        instance = text.strip()
    elif identifier == "startend:end_checker":
        schema["pattern"] = re.escape(args["end_phrase"].strip().lower()) + r"\Z"
        instance = text.strip().strip('"').lower()
    elif identifier == "combination:repeat_prompt":
        prompt = args["prompt_to_repeat"]
        if not isinstance(prompt, str):
            raise InvalidTask("Prompt prefix must be text")
        schema["pattern"] = "^" + re.escape(prompt.strip().lower())
        instance = text.strip().lower()
    return [(schema, instance)]


def prepare_extended_instruction_observations(
    identifier: str, args: dict[str, Any], tools: Any, text: str
) -> list[tuple[dict[str, Any], Any]] | None:
    """Prepare extended instruction tokens, counts, and positional schemas."""
    schema: dict[str, Any] = {"type": "string"}
    instance: Any = text
    if identifier in {"copy:copy", "copy:copying_simple", "new:copy_span_idx", "copy:copying_multiple"}:
        expected = args["prompt_to_repeat"]
        if identifier == "new:copy_span_idx":
            expected = expected[args["n_start"] : args["n_end"]]
        expected = expected.strip().lower()
        schema["const"] = expected
        instance = text.strip().lower()
        if identifier == "copy:copying_multiple":
            count_schema(args["N"])
            schema = {"type": "array", "minItems": args["N"], "maxItems": args["N"], "items": schema}
            instance = [part.strip().lower() for part in text.split("******")]
    elif identifier == "copy:repeat_phrase":
        words = args["phrase"].split()
        count = args["small_n"]
        count_schema(count)
        if not words:
            raise InvalidTask("Repeated phrase must contain words")
        variants = []
        for changed in range(len(words)):
            items = [
                {"not": {"const": word}} if index == changed else {"const": word} for index, word in enumerate(words)
            ]
            variants.append({"type": "array", "prefixItems": items, "minItems": len(words), "maxItems": len(words)})
        schema = {"type": "array", "minItems": count, "maxItems": count, "items": {"oneOf": variants}}
        instance = [part.split() for part in re.findall(rf"{words[0]} .*? {words[-1]}", text)]
    elif identifier in {"keywords:word_once", "keywords:word_count_different_numbers", "count:count_increment_word"}:
        frequencies = (
            [(args["keyword"], 1, "exactly")]
            if identifier == "keywords:word_once"
            else (
                [(args["keyword"], args["frequency"], args["relation"])]
                if identifier == "keywords:word_count_different_numbers"
                else [(args["keyword1"], 1, "exactly"), (args["keyword2"], 2, "exactly")]
            )
        )
        return [
            (count_schema(count, relation), len(re.findall(pattern, text, flags=re.IGNORECASE)))
            for pattern, count, relation in frequencies
        ]
    elif identifier == "keywords:exclude_word_harder":
        schema["not"] = {"pattern": re.escape(" " + args["keyword"] + " ")}
    elif identifier in {"punctuation:punctuation_dot", "punctuation:punctuation_exclamation"}:
        schema["not"] = {"pattern": r"\." if identifier.endswith("dot") else "!"}
    elif identifier in {"count:lowercase_counting", "letters:letter_counting"}:
        instance = len(re.findall(r"\b[a-z]+\b" if identifier.startswith("count:") else r"[a-zA-Z]", text))
        schema = count_schema(args["N"], args.get("relation", "exactly"))
        if identifier.startswith("count:"):
            schema = {"type": "integer", "maximum": args["N"]}
    elif identifier in {"paragraphs:paragraphs", "paragraphs:paragraphs2", "count:counting_composition"}:
        parts = [
            part.strip() for part in re.split(r"\n\n" if identifier.endswith("paragraphs2") else r"\s?\*\*\*\s?", text)
        ]
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        if identifier == "count:counting_composition":
            parts = [part.strip() for part in re.split(r"\s?\*\*\*\s?", text)]
        count = 3 if identifier == "count:counting_composition" else 2
        schema = {"type": "array", "minItems": count, "maxItems": count, "items": {"type": "string", "minLength": 1}}
        instance = parts
        if identifier == "count:counting_composition":
            count_schema(args["n_sent"])
            count_schema(args["n_words"])
            counts = [
                [len(tools.nltk.word_tokenize(sentence)) for sentence in tools.split_into_sentences(part)]
                for part in parts
            ]
            return [
                (schema, parts),
                (
                    {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": args["n_sent"],
                            "maxItems": args["n_sent"],
                            "items": count_schema(args["n_words"]),
                        },
                    },
                    counts,
                ),
            ]
    elif identifier.startswith(("first_word:", "last_word:")):
        first = identifier.startswith("first_word:")
        parts = tools.split_into_sentences(text) if identifier.endswith("sent") else [text]
        words = [part.split() for part in parts]
        instance = [(part[0] if first else re.sub(r"[^\w\s]", "", part[-1])).lower() if part else None for part in words]
        schema = {
            "type": "array",
            "minItems": 1,
            "items": {"const": args["first_word" if first else "last_word"].lower()},
        }
    elif identifier == "detectable_format:square_brackets":
        schema = {"type": "array", "minItems": 1, "items": {"type": "string", "pattern": r"^\[.*\]$"}}
        instance = text.split()
    elif identifier == "detectable_format:bigram_wrapping":
        words = text.split()
        instance = [words[index : index + 2] for index in range(0, len(words) - 1, 2)]
        schema = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "array", "prefixItems": [{"pattern": "^<<"}, {"pattern": ">>$"}]},
        }
    elif identifier == "detectable_format:sentence_hyphens":
        pairs = list(zip(text.split("-"), tools.split_into_sentences(text.replace("-", " ")), strict=False))
        schema = {
            "type": "array",
            "minItems": 1,
            "prefixItems": [{"const": [gold, sentence.strip()]} for sentence, gold in pairs] or [False],
        }
        instance = [[sentence, sentence] for sentence, _ in pairs]
    elif identifier == "keywords:no_adjacent_consecutive":
        letters = [word[0].lower() for word in text.split()]
        schema = {
            "type": "object",
            "properties": {
                "lengths": {"type": "array", "minItems": 1, "items": {"const": 1}},
                "differences": {"type": "array", "items": {"not": {"const": 1}}},
            },
        }
        instance = {
            "lengths": [len(letter) for letter in letters],
            "differences": [
                ord(right) - ord(left) if len(left) == len(right) == 1 else None for left, right in pairwise(letters)
            ],
        }
    elif identifier == "count:count_unique":
        schema = {"type": "array", "minItems": 1, "uniqueItems": True}
        instance = tools.nltk.word_tokenize(text)
    elif identifier == "keywords:palindrome":
        words = text.split()
        schema = {
            "type": "object",
            "minProperties": 1,
            "anyOf": (
                [{"properties": {str(index): {"const": word[::-1]}}} for index, word in enumerate(words)] or [False]
            ),
        }
        instance = {str(index): word for index, word in enumerate(words)}
    elif identifier == "keywords:keyword_specific_position":
        n, m = args["n"], args["m"]
        if type(n) is not int or type(m) is not int or min(n, m) < 1:
            raise InvalidTask("Instruction word and sentence positions must be positive integers")
        sentences = tools.split_into_sentences(text)
        words = tools.nltk.word_tokenize(sentences[n - 1]) if len(sentences) >= n else []
        schema["const"] = args["keyword"]
        instance = words[m - 1] if len(words) >= m else None
    elif identifier == "keywords:start_end":
        words = tools.nltk.word_tokenize(text)
        schema = {"type": "array", "minItems": 2, "prefixItems": [{"const": words[-1].lower()}] if words else [False]}
        instance = [word.lower() for word in words]
    else:
        return None
    return [(schema, instance)]
