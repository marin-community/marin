# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The IFEval constraint checks, keyed by instruction id such as ``keywords:letter_frequency``.

Each check takes the candidate text and the constraint's params and returns ``(passed, detail)``.
The detail is a short human-readable string that lands in the reward's ``detail`` field. Checks are
pure stdlib: this module has to run inside every task image without an extra.

Several IFEval instruction ids describe an instruction the model was given rather than a property
that can be measured exactly (``count:count_unique``, ``language:response_language``,
``detectable_format:bigram_wrapping``). Those are approximations, and the docstring on each says so.
"""

import json
import re
from collections.abc import Callable

Check = Callable[[str, dict], tuple[bool, str]]

CONSTRAINTS: dict[str, Check] = {}

PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z0-9'\-]+")
BULLET = re.compile(r"^\s*[\*\-]\s+")
SQUARE_BRACKETS = re.compile(r"\[[^\[\]\n]+\]")
TITLE_MARKER = re.compile(r"<<[^<>\n]+>>")
JSON_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)
MARKDOWN_HIGHLIGHT = re.compile(r"\*{1,2}[^*\n]+\*{1,2}")
BOLD_BIGRAM = re.compile(r"\*\*\s*\w+\s+\w+\s*\*\*")

CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
TWO_RESPONSE_SEPARATOR = "******"
PARAGRAPH_DIVIDER = "***"

# Closed-class words that are cheap evidence a response is in the requested language. Scripts with
# no entry (zh, ja, ko, ar, th) are checked by their share of non-ASCII characters instead.
LANGUAGE_HINTS: dict[str, set[str]] = {
    "en": {"the", "and", "of", "to", "is", "in", "a"},
    "es": {"el", "la", "y", "de", "que", "en", "un", "es"},
    "fr": {"le", "la", "et", "de", "que", "un", "une", "est"},
    "de": {"der", "die", "und", "ist", "ein", "eine", "zu", "von"},
    "it": {"il", "la", "e", "di", "che", "un", "una", "è"},
    "pt": {"o", "a", "e", "de", "que", "um", "uma", "é"},
    "ru": {"и", "в", "не", "на", "с", "но"},  # noqa: RUF001 - Cyrillic es, not a latin c.
    "zh": set(),
    "ja": set(),
    "ko": set(),
    "ar": set(),
    "hi": {"और", "है", "का", "की"},
    "th": set(),
    "vi": {"và", "là", "có", "không", "trong"},
    "tr": {"ve", "bir", "bu", "için", "ile"},
    "nl": {"de", "het", "en", "een", "van", "is"},
}
NON_ASCII_MIN = 10
NON_ASCII_SHARE = 0.2
UNIQUE_WORDS_MIN = 5
UNIQUE_WORDS_SHARE = 0.5
PALINDROME_MIN_LETTERS = 3


def constraint(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CONSTRAINTS[name] = check
        return check

    return register


def paragraphs(text: str) -> list[str]:
    return [p for p in PARAGRAPH_SEPARATOR.split(text.strip()) if p.strip()]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def words(text: str) -> list[str]:
    return WORD.findall(text)


def count_keyword(text: str, keyword: str) -> int:
    return len(re.findall(r"\b" + re.escape(keyword) + r"\b", text, flags=re.IGNORECASE))


def contains_keyword(text: str, keyword: str) -> bool:
    return count_keyword(text, keyword) > 0


def compare(actual: int, relation: str, expected: int) -> tuple[bool, str] | None:
    """Apply an IFEval ``relation``, or ``None`` when the relation is not one IFEval defines."""
    detail = f"actual={actual} rel={relation} n={expected}"
    if relation == "at least":
        return actual >= expected, detail
    if relation == "less than":
        return actual < expected, detail
    if relation == "at most":
        return actual <= expected, detail
    if relation in ("equal", "exactly"):
        return actual == expected, detail
    return None


def _with_relation(actual: int, relation: str, expected: int, prefix: str) -> tuple[bool, str]:
    result = compare(actual, relation, expected)
    if result is None:
        return False, f"unsupported relation: {relation}"
    passed, detail = result
    return passed, f"{prefix} {detail}"


# === length_constraints =====================================================


@constraint("length_constraints:number_paragraphs")
def number_paragraphs(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_paragraphs")
    if not isinstance(expected, int):
        return False, "missing num_paragraphs"
    actual = len(paragraphs(text))
    return actual == expected, f"paragraphs={actual} expected={expected}"


@constraint("length_constraints:number_words")
def number_words(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_words")
    if not isinstance(expected, int):
        return False, "missing num_words"
    return _with_relation(len(words(text)), params.get("relation", "at least"), expected, "words:")


@constraint("length_constraints:number_sentences")
def number_sentences(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_sentences")
    if not isinstance(expected, int):
        return False, "missing num_sentences"
    return _with_relation(len(sentences(text)), params.get("relation", "at least"), expected, "sentences:")


@constraint("length_constraints:nth_paragraph_first_word")
def nth_paragraph_first_word(text: str, params: dict) -> tuple[bool, str]:
    expected_paragraphs = params.get("num_paragraphs")
    nth = params.get("nth_paragraph")
    first_word = params.get("first_word")
    if not isinstance(expected_paragraphs, int) or not isinstance(nth, int) or not isinstance(first_word, str):
        return False, "missing kwargs"
    found = paragraphs(text)
    if len(found) != expected_paragraphs or nth < 1 or nth > len(found):
        return False, f"paragraphs={len(found)} need={expected_paragraphs} nth={nth}"
    actual = words(found[nth - 1])
    if not actual or actual[0].lower() != first_word.lower():
        return False, f"got first_word={actual[:1]} expected={first_word!r}"
    return True, "ok"


# === keywords ===============================================================


@constraint("keywords:forbidden_words")
def forbidden_words(text: str, params: dict) -> tuple[bool, str]:
    forbidden = params.get("forbidden_words", [])
    if not isinstance(forbidden, list):
        return False, "forbidden_words must be list"
    hits = [w for w in forbidden if isinstance(w, str) and contains_keyword(text, w)]
    return not hits, f"hits={hits}"


@constraint("keywords:word_once")
def word_once(text: str, params: dict) -> tuple[bool, str]:
    keyword = params.get("keyword")
    if not isinstance(keyword, str):
        return False, "missing keyword"
    count = count_keyword(text, keyword)
    return count == 1, f"count={count} keyword={keyword!r}"


@constraint("keywords:letter_frequency")
def letter_frequency(text: str, params: dict) -> tuple[bool, str]:
    letter = params.get("letter")
    expected = params.get("let_frequency")
    if not (isinstance(letter, str) and len(letter) == 1 and isinstance(expected, int)):
        return False, "bad kwargs"
    actual = sum(1 for c in text.lower() if c == letter.lower())
    return _with_relation(actual, params.get("let_relation", "at least"), expected, f"letter={letter}")


@constraint("keywords:existence")
def keyword_existence(text: str, params: dict) -> tuple[bool, str]:
    keywords = params.get("keywords", [])
    if not isinstance(keywords, list):
        return False, "keywords must be list"
    missing = [w for w in keywords if isinstance(w, str) and not contains_keyword(text, w)]
    return not missing, f"missing={missing}"


@constraint("keywords:frequency")
def keyword_frequency(text: str, params: dict) -> tuple[bool, str]:
    keyword = params.get("keyword")
    expected = params.get("frequency")
    if not isinstance(keyword, str) or not isinstance(expected, int):
        return False, "bad kwargs"
    actual = count_keyword(text, keyword)
    return _with_relation(actual, params.get("relation", "at least"), expected, f"keyword={keyword!r}")


@constraint("keywords:word_count_different_numbers")
def word_count_different_numbers(text: str, params: dict) -> tuple[bool, str]:
    """Same params as ``keywords:frequency`` in the Nemotron dataset."""
    return keyword_frequency(text, params)


@constraint("keywords:keyword_specific_position")
def keyword_specific_position(text: str, params: dict) -> tuple[bool, str]:
    """The keyword must be the ``m``-th word of the ``n``-th sentence."""
    keyword = params.get("keyword")
    sentence_index = params.get("n")
    word_index = params.get("m")
    if not isinstance(keyword, str) or not isinstance(sentence_index, int) or not isinstance(word_index, int):
        return False, "bad kwargs"
    found = sentences(text)
    if sentence_index < 1 or sentence_index > len(found):
        return False, f"only {len(found)} sentences, need sentence {sentence_index}"
    in_sentence = words(found[sentence_index - 1])
    if word_index < 1 or word_index > len(in_sentence):
        return False, f"sentence has {len(in_sentence)} words, need pos {word_index}"
    actual = in_sentence[word_index - 1]
    return (
        actual.lower() == keyword.lower(),
        f"pos[{sentence_index},{word_index}]={actual!r} expected={keyword!r}",
    )


@constraint("keywords:palindrome")
def palindrome(text: str, params: dict) -> tuple[bool, str]:
    """The response must contain at least one palindromic word of three letters or more."""
    for word in words(text):
        lowered = word.lower()
        if len(lowered) >= PALINDROME_MIN_LETTERS and lowered == lowered[::-1]:
            return True, f"palindrome={lowered!r}"
    return False, "no palindrome found"


@constraint("keywords:no_adjacent_consecutive")
def no_adjacent_consecutive(text: str, params: dict) -> tuple[bool, str]:
    """No two adjacent words may be identical, ignoring case."""
    lowered = [w.lower() for w in words(text)]
    for index in range(1, len(lowered)):
        if lowered[index] == lowered[index - 1]:
            return False, f"adjacent dup at pos {index}: {lowered[index]!r}"
    return True, "no adjacent duplicates"


@constraint("keywords:start_end")
def start_end(text: str, params: dict) -> tuple[bool, str]:
    """The response must start and end with the same word."""
    found = words(text)
    if not found:
        return False, "empty"
    return found[0].lower() == found[-1].lower(), f"start={found[0]!r} end={found[-1]!r}"


# === change_case ============================================================


@constraint("change_case:english_capital")
def english_capital(text: str, params: dict) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no letters"
    passed = all(c.isupper() for c in letters)
    return passed, f"all_upper={passed}"


@constraint("change_case:english_lowercase")
def english_lowercase(text: str, params: dict) -> tuple[bool, str]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no letters"
    passed = all(c.islower() for c in letters)
    return passed, f"all_lower={passed}"


@constraint("change_case:capital_word_frequency")
def capital_word_frequency(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("capital_frequency")
    if not isinstance(expected, int):
        return False, "missing capital_frequency"
    actual = sum(1 for w in words(text) if len(w) >= 2 and w.isupper())
    return _with_relation(actual, params.get("capital_relation", "at least"), expected, "all_caps_words:")


# === punctuation ============================================================


@constraint("punctuation:no_comma")
def no_comma(text: str, params: dict) -> tuple[bool, str]:
    return "," not in text, f"has_comma={',' in text}"


@constraint("punctuation:punctuation_dot")
def no_dot(text: str, params: dict) -> tuple[bool, str]:
    """IFEval semantics: the response must contain no periods at all."""
    return "." not in text, f"has_dot={'.' in text}"


@constraint("punctuation:punctuation_exclamation")
def no_exclamation(text: str, params: dict) -> tuple[bool, str]:
    return "!" not in text, f"has_exclam={'!' in text}"


# === startend ===============================================================


@constraint("startend:end_checker")
def end_checker(text: str, params: dict) -> tuple[bool, str]:
    end_phrase = params.get("end_phrase")
    if not isinstance(end_phrase, str):
        return False, "missing end_phrase"
    return text.rstrip().endswith(end_phrase), f"ends_with={end_phrase!r}"


@constraint("startend:quotation")
def quotation(text: str, params: dict) -> tuple[bool, str]:
    """The whole response must be wrapped in quotes, straight or typographic."""
    stripped = text.strip()
    passed = (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("“") and stripped.endswith("”"))
        or (stripped.startswith("'") and stripped.endswith("'"))
    )
    return passed, f"quoted={passed}"


# === detectable_format ======================================================


@constraint("detectable_format:number_bullet_lists")
def number_bullet_lists(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_bullets")
    if not isinstance(expected, int):
        return False, "missing num_bullets"
    actual = sum(1 for line in text.splitlines() if BULLET.match(line))
    return actual == expected, f"bullets={actual} expected={expected}"


@constraint("detectable_format:title")
def title(text: str, params: dict) -> tuple[bool, str]:
    found = bool(TITLE_MARKER.search(text))
    return found, f"has_title_marker={found}"


@constraint("detectable_format:json_format")
def json_format(text: str, params: dict) -> tuple[bool, str]:
    stripped = text.strip()
    fence = JSON_FENCE.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    try:
        json.loads(stripped)
    except ValueError as error:
        return False, f"json_parse_error: {error}"
    return True, "json_parses"


@constraint("detectable_format:multiple_sections")
def multiple_sections(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_sections")
    splitter = params.get("section_spliter") or params.get("section_splitter") or "Section"
    if not isinstance(expected, int) or not isinstance(splitter, str):
        return False, "bad kwargs"
    actual = len(re.findall(re.escape(splitter) + r"\s*\d+", text, flags=re.IGNORECASE))
    return actual == expected, f"sections={actual} expected={expected} splitter={splitter!r}"


@constraint("detectable_format:number_highlighted_sections")
def number_highlighted_sections(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_highlights")
    if not isinstance(expected, int):
        return False, "missing num_highlights"
    actual = len(MARKDOWN_HIGHLIGHT.findall(text))
    return actual >= expected, f"highlights={actual} need>={expected}"


@constraint("detectable_format:square_brackets")
def square_brackets(text: str, params: dict) -> tuple[bool, str]:
    found = bool(SQUARE_BRACKETS.search(text))
    return found, f"has_square_brackets={found}"


@constraint("detectable_format:sentence_hyphens")
def sentence_hyphens(text: str, params: dict) -> tuple[bool, str]:
    """Every sentence must join its words with hyphens instead of spaces."""
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        body = sentence.rstrip(".!?")
        if " " in body or "\t" in body:
            return False, f"sentence has whitespace: {body[:40]!r}"
        if "-" not in body:
            return False, f"sentence missing hyphen: {body[:40]!r}"
    return True, f"hyphenated sentences={len(found)}"


@constraint("detectable_format:bigram_wrapping")
def bigram_wrapping(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: at least one word pair is wrapped in markdown bold."""
    found = bool(BOLD_BIGRAM.search(text))
    return found, f"bigram_bold={found}"


@constraint("detectable_format:constrained_response")
def constrained_response(text: str, params: dict) -> tuple[bool, str]:
    stripped = text.strip()
    passed = stripped in CONSTRAINED_RESPONSES
    return passed, f"matched_constrained={passed}"


# === detectable_content =====================================================


@constraint("detectable_content:postscript")
def postscript(text: str, params: dict) -> tuple[bool, str]:
    marker = params.get("postscript_marker")
    if not isinstance(marker, str):
        return False, "missing postscript_marker"
    found = bool(re.search(r"^\s*" + re.escape(marker), text, flags=re.MULTILINE | re.IGNORECASE))
    return found, f"has_postscript={marker!r}: {found}"


@constraint("detectable_content:number_placeholders")
def number_placeholders(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_placeholders")
    if not isinstance(expected, int):
        return False, "missing num_placeholders"
    actual = len(SQUARE_BRACKETS.findall(text))
    return actual >= expected, f"placeholders={actual} need>={expected}"


# === paragraphs =============================================================


@constraint("paragraphs:paragraphs")
def divided_paragraphs(text: str, params: dict) -> tuple[bool, str]:
    """IFEval writes ``***`` between paragraphs; blank-line separation is accepted too."""
    by_divider = [p for p in text.split(PARAGRAPH_DIVIDER) if p.strip()]
    by_blank_line = paragraphs(text)
    if len(by_divider) >= 2:
        return True, f"*** divided paragraphs={len(by_divider)}"
    if len(by_blank_line) >= 2:
        return True, f"blank-line paragraphs={len(by_blank_line)}"
    return False, f"need >=2 paragraphs (divider={len(by_divider)} blank={len(by_blank_line)})"


@constraint("paragraphs:paragraphs2")
def divided_paragraphs2(text: str, params: dict) -> tuple[bool, str]:
    return divided_paragraphs(text, params)


# === first_word / last_word =================================================


@constraint("first_word:first_word_answer")
def first_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = words(text)
    actual = found[0] if found else ""
    return actual.lower() == expected.lower(), f"first={actual!r} expected={expected!r}"


@constraint("first_word:first_word_sent")
def first_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = words(sentence)
        if not in_sentence or in_sentence[0].lower() != expected.lower():
            return False, f"sent starts with {in_sentence[:1]} need={expected!r}"
    return True, f"all {len(found)} sentences start with {expected!r}"


@constraint("last_word:last_word_answer")
def last_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = words(text)
    actual = found[-1] if found else ""
    return actual.lower() == expected.lower(), f"last={actual!r} expected={expected!r}"


@constraint("last_word:last_word_sent")
def last_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = words(sentence.rstrip(".!?\"' "))
        if not in_sentence or in_sentence[-1].lower() != expected.lower():
            return False, f"sent ends with {in_sentence[-1:]} need={expected!r}"
    return True, f"all {len(found)} sentences end with {expected!r}"


# === count ==================================================================


@constraint("count:count_unique")
def count_unique(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: the response must use mostly distinct words."""
    lowered = [w.lower() for w in words(text)]
    if not lowered:
        return False, "empty"
    unique = len(set(lowered))
    passed = unique >= max(UNIQUE_WORDS_MIN, int(UNIQUE_WORDS_SHARE * len(lowered)))
    return passed, f"unique={unique} total={len(lowered)}"


@constraint("count:count_increment_word")
def count_increment_word(text: str, params: dict) -> tuple[bool, str]:
    """The first keyword group must appear exactly one more time than the second."""
    first_group = params.get("keyword1")
    second_group = params.get("keyword2")
    if not isinstance(first_group, list) or not isinstance(second_group, list):
        return False, "bad kwargs"
    first = sum(count_keyword(text, w) for w in first_group if isinstance(w, str))
    second = sum(count_keyword(text, w) for w in second_group if isinstance(w, str))
    passed = first > 0 and first == second + 1
    return passed, f"kw1_count={first} kw2_count={second} (need kw1 == kw2+1)"


@constraint("count:lowercase_counting")
def lowercase_counting(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    actual = sum(1 for w in words(text) if w.isalpha() and w.islower())
    return actual >= expected, f"lower_words={actual} need>={expected}"


@constraint("count:counting_composition")
def counting_composition(text: str, params: dict) -> tuple[bool, str]:
    expected_sentences = params.get("n_sent")
    expected_words = params.get("n_words")
    if not isinstance(expected_sentences, int) or not isinstance(expected_words, int):
        return False, "bad kwargs"
    found = sentences(text)
    if len(found) < expected_sentences:
        return False, f"sentences={len(found)} need>={expected_sentences}"
    for index, sentence in enumerate(found[:expected_sentences]):
        count = len(words(sentence))
        if count < expected_words:
            return False, f"sent[{index}] has {count} words, need>={expected_words}"
    return True, f"first {expected_sentences} sentences each have >={expected_words} words"


# === letters ================================================================


@constraint("letters:letter_counting")
def letter_counting(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: the Nemotron rows carry no letter threshold, so this counts words."""
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    return _with_relation(len(words(text)), params.get("relation", "at least"), expected, "words:")


@constraint("letters:letter_counting2")
def letter_counting2(text: str, params: dict) -> tuple[bool, str]:
    """Same params as ``keywords:letter_frequency``."""
    return letter_frequency(text, params)


# === language ===============================================================


@constraint("language:response_language")
def response_language(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: closed-class hint words, or a share of non-ASCII for non-Latin scripts."""
    language = params.get("language")
    if not isinstance(language, str):
        return False, "missing language"
    language = language.lower()
    hints = LANGUAGE_HINTS.get(language)
    if hints:
        matched = any(contains_keyword(text, hint) for hint in hints)
        return matched, f"lang={language} hint_match={matched}"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    passed = non_ascii >= max(NON_ASCII_MIN, int(NON_ASCII_SHARE * max(len(text), 1)))
    return passed, f"lang={language} non_ascii={non_ascii}"


# === copy ===================================================================


@constraint("copy:repeat_phrase")
def repeat_phrase(text: str, params: dict) -> tuple[bool, str]:
    phrase = params.get("phrase")
    expected = params.get("small_n")
    if not isinstance(phrase, str) or not isinstance(expected, int):
        return False, "bad kwargs"
    actual = len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))
    return actual >= expected, f"phrase_count={actual} need>={expected}"


# === combination ============================================================


@constraint("combination:two_responses")
def two_responses(text: str, params: dict) -> tuple[bool, str]:
    parts = [p for p in text.split(TWO_RESPONSE_SEPARATOR) if p.strip()]
    return len(parts) >= 2, f"sections_by_******={len(parts)}"
