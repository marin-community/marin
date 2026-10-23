# IFEval constraint checkers for RLVR-style instruction-following rewards.
#
# The constraint functions below are vendored verbatim from allenai/open-instruct
# (`open_instruct/if_functions.py`, IF_FUNCTIONS_MAP), which is the canonical verifier
# for the `allenai/RLVR-IFeval` dataset. They cover the 25 IFEval-taxonomy constraints;
# the dataset's `ground_truth` is a JSON blob carrying `func_name` plus the kwargs each
# function expects. Source: https://github.com/allenai/open-instruct (Apache-2.0).
#
# `validate_response_language` needs the optional `langdetect` dependency; its import is
# deferred so this module imports cleanly without it (that func_name does not appear in
# the RLVR-IFeval `ground_truth` set).

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# include keywords: Include keywords {keyword1}, {keyword2} in your response
def verify_keywords(text, keyword_list):
    """Verify if the response contains all the specified keywords (case-insensitive)."""
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text, word, N):
    """Verify a keyword appears exactly N times (case-insensitive, word-boundary)."""
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


def verify_keyword_frequency_relation(text: str, keyword_list: list[str], N: int, quantifier: str) -> bool:
    """Verify that every keyword obeys the requested frequency relation."""
    words = re.findall(r"\b\w+\b", text.lower())
    counts = [sum(word == keyword.lower() for word in words) for keyword in keyword_list]
    if quantifier == "exactly":
        return all(count == N for count in counts)
    if quantifier == "at least":
        return all(count >= N for count in counts)
    if quantifier == "less than":
        return all(count < N for count in counts)
    return False


# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text, forbidden_words):
    """Validate that none of the forbidden words appear (case-insensitive)."""
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


# Letter Frequency: In your response, the letter {letter} should appear {N} times.
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """Verify a given letter appears exactly N times (case-sensitive)."""
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    actual_count = text.count(letter)
    return actual_count == N


# Response Language: Your ENTIRE response should be in {language}.
def validate_response_language(text, language):
    """Validate the entire response is in the specified language code (e.g. 'en')."""
    import langdetect

    detected_language = langdetect.detect(text)
    return detected_language == language


# Number Paragraphs: separated by the markdown divider '* * *'.
def verify_paragraph_count(text: str, N: int) -> bool:
    """Verify the text contains exactly N paragraphs separated by '* * *'."""

    def clean_text(t: str) -> str:
        return "\n".join(line.strip() for line in t.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


# Number Words: Answer with at least / around / at most {N} words.
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """Validate the word count against an 'at least'/'around'/'at most' quantifier."""
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "more than":
        return actual_count > N
    elif quantifier == "less than":
        return actual_count < N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """Verify the sentence count against an 'at least'/'around'/'at most' quantifier."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    actual_count = len(sentences)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "more than":
        return actual_count > N
    elif quantifier == "less than":
        return actual_count < N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


# Number Paragraphs + First Word in i-th Paragraph (paragraphs separated by two line breaks).
def validate_paragraphs(text, N, first_word, i):
    """Validate N paragraphs and that the i-th (1-indexed) starts with first_word."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


# Postscript: add a postscript starting with {postscript marker}.
def verify_postscript(text, postscript_marker):
    """Verify the text contains a non-empty postscript starting with the marker."""
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


# Number Placeholder: at least {N} placeholders in square brackets, e.g. [address].
def validate_placeholders(text: str, N: int) -> bool:
    """Validate the text contains at least N [bracketed] placeholders."""
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


# Number Bullets: exactly {N} markdown bullet points (* or -).
def verify_bullet_points(text: str, N: int) -> bool:
    """Verify the text contains exactly N markdown bullet points."""
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


# Title: must contain a title wrapped in double angular brackets, e.g. <<poem of joy>>.
def validate_title(text: str) -> bool:
    """Validate the text contains a <<...>> title."""
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


# Choose: Answer with one of the following options: {options}.
def validate_choice(text: str, options: list) -> bool:
    """Validate the text contains one of the allowed options (substring match)."""
    return any(option in text for option in options)


# Minimum Highlighted Sections: at least {N} *highlighted* sections.
def validate_highlighted_sections(text: str, N: int) -> bool:
    """Validate the text contains at least N *highlighted* sections."""
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


# Multiple Sections: {N} sections, each beginning with {section splitter} X.
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """Validate the text splits into exactly N sections by the splitter."""
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


# JSON Format: entire output wrapped in JSON.
def validate_json_format(text: str) -> bool:
    """Validate the entire output parses as JSON."""
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


# Repeat Prompt: repeat the request verbatim, then answer.
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """Validate the response begins with the original prompt verbatim."""
    return bool(text.startswith(original_prompt))


# Two Responses: two responses separated by exactly '******'.
def validate_two_responses(text: str) -> bool:
    """Validate exactly two distinct responses separated by '******'."""
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


# All Uppercase.
def validate_uppercase(text: str) -> bool:
    """Validate the entire response is uppercase."""
    return text == text.upper()


# All Lowercase.
def validate_lowercase(text: str) -> bool:
    """Validate the entire response is lowercase."""
    return text == text.lower()


# Frequency of All-capital Words: at least / around / at most {N}.
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """Validate the count of ALL-CAPS words against a quantifier."""
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return abs(len(words) - N) <= max(round(N * 0.1), 1)
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


# End Checker: finish with this exact phrase {end phrase}.
def validate_end(text: str, end_phrase: str) -> bool:
    """Validate the response ends with the exact end phrase."""
    return bool(text.endswith(end_phrase))


# Quotation: wrap the entire response in double quotation marks.
def validate_quotation(text: str) -> bool:
    """Validate the response is wrapped in double quotes."""
    return bool(text.startswith('"') and text.endswith('"'))


# No Commas.
def validate_no_commas(text: str) -> bool:
    """Validate the response contains no commas."""
    return "," not in text


# Canonical func_name -> checker mapping (open-instruct IF_FUNCTIONS_MAP).
IF_FUNCTIONS_MAP: Dict[str, Callable[..., bool]] = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "verify_keyword_frequency_relation": verify_keyword_frequency_relation,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

# Argument names (besides the response `text`) each checker consumes, drawn from the
# RLVR-IFeval `ground_truth` JSON. Used to filter the spec down to the kwargs a given
# func actually accepts (the spec carries all possible keys, most set to null).
_FUNC_ARG_NAMES: Dict[str, tuple] = {
    "verify_keywords": ("keyword_list",),
    "verify_keyword_frequency": ("word", "N"),
    "verify_keyword_frequency_relation": ("keyword_list", "N", "quantifier"),
    "validate_forbidden_words": ("forbidden_words",),
    "verify_letter_frequency": ("letter", "N"),
    "validate_response_language": ("language",),
    "verify_paragraph_count": ("N",),
    "validate_word_constraint": ("N", "quantifier"),
    "verify_sentence_constraint": ("N", "quantifier"),
    "validate_paragraphs": ("N", "first_word", "i"),
    "verify_postscript": ("postscript_marker",),
    "validate_placeholders": ("N",),
    "verify_bullet_points": ("N",),
    "validate_title": (),
    "validate_choice": ("options",),
    "validate_highlighted_sections": ("N",),
    "validate_sections": ("N", "section_splitter"),
    "validate_json_format": (),
    "validate_repeat_prompt": ("original_prompt",),
    "validate_two_responses": (),
    "validate_uppercase": (),
    "validate_lowercase": (),
    "validate_frequency_capital_words": ("N", "quantifier"),
    "validate_end": ("end_phrase",),
    "validate_quotation": (),
    "validate_no_commas": (),
}


def _normalize_constraint(ground_truth: Mapping[str, Any]) -> dict[str, Any]:
    raw_spec = dict(ground_truth)
    func_name = raw_spec.get("func_name")
    if func_name not in IF_FUNCTIONS_MAP:
        raise ValueError(f"Unknown IFEval func_name: {func_name!r}.")

    normalized = {"func_name": func_name}
    for argument in _FUNC_ARG_NAMES[func_name]:
        if argument not in raw_spec or raw_spec[argument] is None:
            raise ValueError(f"IFeval {func_name!r} requires non-null {argument!r}.")
        normalized[argument] = raw_spec[argument]
    return normalized


def normalize_ground_truth(ground_truth: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    """Validate and canonicalize an IFEval constraint for dataset preparation.

    Runtime scoring deliberately treats malformed examples as incorrect instead of
    crashing a rollout. Dataset builders should use this stricter boundary before
    writing an artifact so that malformed or incomplete constraints cannot silently
    become permanently unlearnable rows.
    """
    if isinstance(ground_truth, Mapping):
        normalized: dict[str, Any] | list[dict[str, Any]] = _normalize_constraint(ground_truth)
    else:
        if not ground_truth:
            raise ValueError("IFeval ground_truth constraint list must be non-empty.")
        normalized = [_normalize_constraint(constraint) for constraint in ground_truth]
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def check_constraint(response: str, ground_truth: str) -> bool:
    """Score a model response against a single IFEval constraint spec.

    All 25 RLVR-IFeval ``func_name`` values are implemented (``IF_FUNCTIONS_MAP``), so the
    unknown-func_name branch should never fire on the real dataset. It is treated as an
    unsatisfied constraint (score 0) + a logged warning rather than a raise, so a stray /
    out-of-distribution spec penalizes the one sample instead of crashing the RL rollout
    (``env.step`` is NOT wrapped in try/except in the SkyRL generator). A malformed spec
    (missing ``func_name``) is treated the same way. The RLVR-IFeval reward is binary
    per-example (one constraint per row), matching open-instruct ``IF_FUNCTIONS_MAP``.

    Args:
        response: The model's generated text.
        ground_truth: A JSON string with a ``func_name`` key plus the kwargs the
            corresponding checker consumes (the RLVR-IFeval `reward_model.ground_truth`).

    Returns:
        True if the response satisfies the constraint, False otherwise (including the
        unknown/malformed-spec and validator-error cases).
    """
    spec: Dict[str, Any] = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    func_name = spec.get("func_name")
    if func_name is None:
        logger.warning("ifeval: ground_truth missing 'func_name' (%r); scoring 0.", spec)
        return False
    if func_name not in IF_FUNCTIONS_MAP:
        logger.warning(
            "ifeval: unknown/unimplemented func_name %r (covered=%d); scoring 0.",
            func_name,
            len(IF_FUNCTIONS_MAP),
        )
        return False

    func = IF_FUNCTIONS_MAP[func_name]
    arg_names = _FUNC_ARG_NAMES[func_name]
    kwargs = {name: spec[name] for name in arg_names if spec.get(name) is not None}
    try:
        return bool(func(response, **kwargs))
    except Exception as e:  # a bad completion must never crash the trainer — penalize it
        logger.warning("ifeval: checker %r raised %s; scoring 0.", func_name, e)
        return False


def compute_score(response: str, ground_truth: str, verifyit_enabled: bool = False) -> Dict[str, Any]:
    """Compute the IFEval constraint-satisfaction reward for a response.

    ``score`` is the fraction of constraints satisfied and ``acc`` reports whether all
    constraints passed. Single-constraint artifacts retain binary scoring.
    """
    if verifyit_enabled:
        from skyrl_gym.envs.instruction_verifyit import grade_standalone_instructions

        return grade_standalone_instructions(response, ground_truth)
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    constraints = spec if isinstance(spec, list) else [spec]
    satisfied = [check_constraint(response, constraint) for constraint in constraints]
    satisfied_count = sum(satisfied)
    total = len(constraints)
    return {
        "score": satisfied_count / total if total else 0.0,
        "acc": all(satisfied),
        "func_name": spec.get("func_name") if isinstance(spec, dict) else None,
        "constraints_satisfied": satisfied_count,
        "constraints_total": total,
    }
