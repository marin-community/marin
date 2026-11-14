# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The IFEval constraint checks, keyed by instruction id such as ``keywords:letter_frequency``.

Each check takes the candidate text and the constraint's params and returns ``(passed, detail)``.
The detail is a short human-readable string that lands in the reward's ``detail`` field. Checks are
pure stdlib: this module has to run inside every task image without an extra.

``language:response_language`` describes an instruction the model was given rather than a property
that can be measured exactly, so its check is an approximation, and its docstring says so. The
prompts of ``change_case:english_capital`` and ``change_case:english_lowercase`` also ask for
English; those checks look only at letter case.

A word is a letter or digit followed by any letters, digits and combining marks in any script,
with apostrophes and zero-width joiners or non-joiners inside it, so ``don't`` is one word,
``high-risk`` is two and a Devanagari vowel sign stays in its word. A run of Han or kana characters
is one word, which undercounts Chinese and Japanese, except in a check that compares words, where
each such letter is one word, and where a number with separators, such as ``1,000`` or ``10:30``,
and single letters joined by dots, such as ``U.S.``, are one word each. A sentence ends at a line
break, at ``.``, ``!``, ``?`` or an ellipsis before whitespace, after any closing quotes or
brackets, and at a Chinese, Japanese, Devanagari, Arabic, Armenian or Ethiopic sentence-final mark
wherever it stands, after any closing quotes or brackets that follow it. Only
``detectable_format:sentence_hyphens`` reads the dot of an abbreviation, such as ``Dr.`` or
``e.g.``, as no sentence end.
"""

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from itertools import chain, pairwise

Check = Callable[[str, dict], tuple[bool, str]]

CONSTRAINTS: dict[str, Check] = {}

PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
SENTENCE_FINAL = ".!?\u2026"
# Closing quotes and brackets, including the >> of <<title>>, that may stand between a sentence's
# final mark and the space after it.
CLOSERS = "\"'\u201d\u2019)]}>\u00bb\u300d\u300f\uff09"
# Sentence-final marks of Chinese, Japanese, Devanagari, Arabic, Urdu, Armenian and Ethiopic text,
# which end a sentence wherever they stand.
SCRIPT_SENTENCE_MARKS = "\u3002\uff0e\uff01\uff1f\uff61\u0964\u0965\u06d4\u061f\u0589\u1362"
# A run of final marks, captured, and any closers after it; and a run of a script's sentence-final
# marks and any closers after it. A match starts only at the first mark of a run and never gives
# marks back, so a long run of marks takes linear time.
FINAL_MARK_RUN = rf"(?<![{re.escape(SENTENCE_FINAL)}])([{re.escape(SENTENCE_FINAL)}]++)[{re.escape(CLOSERS)}]*+"
SCRIPT_MARK_RUN = rf"(?<![{SCRIPT_SENTENCE_MARKS}])[{SCRIPT_SENTENCE_MARKS}]++[{re.escape(CLOSERS)}]*+"
SENTENCE_END = re.compile(rf"{FINAL_MARK_RUN}(?=\s)|{SCRIPT_MARK_RUN}")
SENTENCE_ENDING_MARKS = tuple(SENTENCE_FINAL + SCRIPT_SENTENCE_MARKS)
APOSTROPHES = "'\u2019"
# Inside a word these join word characters: "don't", and Persian or Indic words written with a
# zero-width non-joiner or joiner. A hyphen separates words, so "high-risk" is two.
WORD_JOINERS = APOSTROPHES + "\u200c\u200d"


# Combining marks occur only in planes 0 and 1 and among the variation selectors of plane 14, so the
# scan below skips the other code points.
MARK_CODE_POINTS = (range(0x20000), range(0xE0100, 0xE01F0))


def _combining_mark_ranges() -> str:
    """Every Unicode combining mark (category M), as ranges for a regex character class."""
    ranges: list[list[int]] = []
    for code in chain(*MARK_CODE_POINTS):
        if unicodedata.category(chr(code))[0] != "M":
            continue
        if ranges and ranges[-1][1] == code - 1:
            ranges[-1][1] = code
        else:
            ranges.append([code, code])
    return "".join(f"{chr(low)}-{chr(high)}" for low, high in ranges)


# A word character is a letter, a digit or a combining mark, such as a Devanagari vowel sign.
WORD_CHARACTER = rf"(?:[^\W_]|[{_combining_mark_ranges()}])"
WORD = re.compile(rf"[^\W_](?:{WORD_CHARACTER}|[{re.escape(WORD_JOINERS)}](?={WORD_CHARACTER}))*")
# Han ideographs, hiragana and katakana. Chinese and Japanese leave no space between words, so a check that compares
# words takes each of these letters as a word.
UNSPACED_RANGES = (
    "\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003ffff"
)
UNSPACED_LETTER = re.compile(f"[{UNSPACED_RANGES}]")
SPACED_WORD_CHARACTER = rf"(?![{UNSPACED_RANGES}]){WORD_CHARACTER}"
SPACED_LETTER = rf"(?![{UNSPACED_RANGES}])[^\W\d_]"
# In a check that compares words, a number written with separators, such as "1,000", "3.5" or "10:30", and single
# letters joined by dots, such as "U.S." or "e.g.", are one word each, so "1,000" and "2,000" share no word "000".
SEPARATED_NUMBER = r"\d+(?:[.,:]\d+)+"
DOTTED_LETTERS = rf"(?<![\w.]){SPACED_LETTER}(?:\.{SPACED_LETTER})+(?![^\W_])"
COMPARED_WORD = re.compile(
    rf"{SEPARATED_NUMBER}|{DOTTED_LETTERS}|[{UNSPACED_RANGES}]|(?![{UNSPACED_RANGES}])[^\W_]"
    rf"(?:{SPACED_WORD_CHARACTER}|[{re.escape(WORD_JOINERS)}](?={SPACED_WORD_CHARACTER}))*"
)
BULLET = re.compile(r"^\s*[\*\-]\s+\S")
BULLET_ONLY_LINE = re.compile(r"^[\s*\-+]*$")
# A span that holds a non-space character; the lookahead keeps matching linear in the span's length.
SQUARE_BRACKETS = re.compile(r"\[(?=[^\[\]\n]*[^\[\]\s])[^\[\]\n]*\]")
ONE_WORD_BRACKETS = re.compile(r"\[[^\[\]\s]*\]")
TITLE_MARKER = re.compile(r"<<(?=[^<>\n]*[^<>\s])[^<>\n]*>>")
DOUBLE_ANGLE_SPAN = re.compile(r"<<((?:(?!<<|>>)[^\n])*)>>")
JSON_FENCE = re.compile(r"^```(?:json)?[ \t]*\r?\n(.*)\n```\s*$", re.DOTALL)
# One to three asterisks on each side; as in CommonMark, the span does not start or end with whitespace.
MARKDOWN_HIGHLIGHT = re.compile(r"(?<!\*)(\*{1,3})(?![\s*])([^*\n]*?[^\s*])\1(?!\*)")
# How far sentence_hyphens reads around final marks with no space after them for an initial, an abbreviation or an
# address; a longer run without spaces is none of these, and the bound keeps a reply without spaces linear in length.
TOKEN_WINDOW = 32
# Dashes that are not hyphens, such as the en and em dashes; one does not join two sentences as a hyphen does.
NON_HYPHEN_DASHES = "\u2012\u2013\u2014\u2015\u2e3a\u2e3b\ufe58"
# A sentence end that no hyphen joins to the next sentence: final marks and any closers before
# whitespace, before a hyphen and whitespace, before a dash that is not a hyphen, or right before a
# letter or digit, which sentence_hyphens reads as the next sentence's start or not; or a script's
# sentence-final marks and any closers before anything but a hyphen and the next sentence.
UNJOINED_SENTENCE_END = re.compile(
    rf"{FINAL_MARK_RUN}(?:\s|-\s|(?=[{NON_HYPHEN_DASHES}]|[^\W_]))|{SCRIPT_MARK_RUN}(?!-\S|\s*$)"
)
# Exactly three asterisks; a longer run is not the divider.
DIVIDER = r"(?<!\*)\*\*\*(?!\*)"
DIVIDER_SPLIT = re.compile(rf"\s?{DIVIDER}\s?")
# The counting_composition prompt writes the divider as "* * *", the same Markdown divider as "***".
COMPOSITION_DIVIDER_SPLIT = re.compile(rf"\s?(?:{DIVIDER}|\* \* \*)\s?")
# Bold italics, as in "a ***big*** one", divide nothing. They follow whitespace, an opening bracket or quote or a Han or
# kana letter, or start the text; after a full stop, three asterisks are a divider, as in "One.***Two.".
BOLD_ITALIC = re.compile(
    rf"(?<![^\s(\[{{\"'\u201c\u2018\u00ab\u300c\u300e\uff08{UNSPACED_RANGES}])"
    r"\*\*\*([^\s*](?:[^*\n]*[^\s*])?)\*\*\*(?!\*)"
)
COMPOSITION_PARAGRAPHS = 3
# Whitespace and Markdown markup that may open a marker line, as in "**P.S.**".
MARKDOWN_LEAD = " \t#*_>+-"
MARKDOWN_LINE_START = rf"^[{MARKDOWN_LEAD}]*"
# Markdown markup and opening quotes or brackets that may stand before a section marker.
SECTION_MARKER_LEAD = MARKDOWN_LEAD + "\"'\u201c\u2018([{\u00ab\u300c\u300e\uff08"

CONSTRAINED_RESPONSES = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
# Exactly six asterisks; a longer run is not the separator.
TWO_RESPONSE_SEPARATOR = re.compile(r"(?<!\*)\*{6}(?!\*)")
# The prompt shows "." and calls it a dot: that mark, its fullwidth and small forms, the one- and two-dot leaders and
# the ellipsis "…". The full stops of other scripts, such as the hollow U+3002, are different marks.
DOTS = ".\uff0e\ufe52\u2024\u2025\u2026"
# Markdown emphasis or straight double quotes around the whole answer, as in "**My answer is yes.**".
ANSWER_WRAPPING = re.compile(r'^(\*{1,2}|")(.*)\1$', re.DOTALL)
# Abbreviations whose dot ends no sentence in detectable_format:sentence_hyphens. "No.", "Fig." and
# "Vol." are ones only before a number, as in "No. 5", since "no", "fig" and "vol" are also words.
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "cf", "approx", "ph"})
NUMBER_ABBREVIATIONS = frozenset({"no", "fig", "vol"})
# Letters joined by dots, such as "e.g", "U.S" or "9a.m"; in detectable_format:sentence_hyphens the dot
# after such a token ends no sentence.
DOTTED_TOKEN = re.compile(r"\d*[^\W\d_]+(?:\.[^\W\d_]+)+")

# Closed-class words that are cheap evidence a response is in the requested language. Languages with
# no hint words (zh, ja, ko, ar, th and the languages not listed) are checked by their share of
# non-ASCII characters instead, so a language written in Latin script needs hint words.
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
    "fi": {"ja", "ei", "että", "oli", "mutta", "kuin", "myös", "ovat"},
    "pl": {"się", "nie", "jest", "że", "na", "jak", "dla", "oraz"},
    "sw": {"na", "ya", "wa", "kwa", "ni", "za", "katika", "kuwa"},
}
NON_ASCII_MIN = 10
NON_ASCII_SHARE = 0.2
PALINDROME_MIN_LETTERS = 3
# Different letters and digits a word palindrome needs, which "www" and the "000" of "2,000" lack.
PALINDROME_MIN_DIFFERENT = 2
# A phrase palindrome, such as "Never odd or even", spans at most this many words and has at least this many letters
# and digits; shorter runs, such as "at a" or "on no", read the same backward by accident in most English texts. It
# also needs this many different letters and digits, which "I I I I I I I" and "no on no on" lack.
PHRASE_PALINDROME_MAX_WORDS = 12
PHRASE_PALINDROME_MIN_LETTERS = 7
PHRASE_PALINDROME_MIN_DIFFERENT = 3


def constraint(name: str) -> Callable[[Check], Check]:
    def register(check: Check) -> Check:
        CONSTRAINTS[name] = check
        return check

    return register


def paragraphs(text: str, separator: re.Pattern[str]) -> list[str]:
    """The parts between separators that hold a word; a divider line or a run of emoji is no paragraph."""
    return [part for part in separator.split(text) if WORD.search(part)]


def between_dividers(text: str, divider: re.Pattern[str]) -> list[str]:
    """The paragraphs between Markdown dividers; bold italics divide nothing."""
    return paragraphs(BOLD_ITALIC.sub(r"\1", text), divider)


def sentences(text: str) -> list[str]:
    """The sentences of the text: a line break ends one, and a part with no word, such as a divider, does not count."""
    found = []
    for line in text.splitlines():
        start = 0
        for end in SENTENCE_END.finditer(line):
            found.append(line[start : end.end()])
            start = end.end()
        found.append(line[start:])
    return [sentence.strip() for sentence in found if WORD.search(sentence)]


def words(text: str) -> list[str]:
    return WORD.findall(text)


def compared_words(text: str) -> list[str]:
    """The words of the text for a check that compares them: each Han or kana letter is a word."""
    return COMPARED_WORD.findall(text)


def starts_with_word(found: list[str], word: str) -> bool:
    """Whether the words start with the required word, which may split into several, as "well-known" and "世界" do."""
    keys = [word_key(part) for part in compared_words(word)]
    return bool(keys) and [word_key(part) for part in found[: len(keys)]] == keys


def ends_with_word(found: list[str], word: str) -> bool:
    """Whether the words end with the required word, which may split into several."""
    keys = [word_key(part) for part in compared_words(word)]
    return bool(keys) and [word_key(part) for part in found[-len(keys) :]] == keys


def token_before(text: str, end: int) -> str:
    """The text before ``end`` back to the last whitespace or hyphen."""
    start = end
    while start > 0 and not text[start - 1].isspace() and text[start - 1] != "-":
        start -= 1
    return text[start:end]


def abbreviation_dot(token: str, following: str) -> bool:
    """Whether the dot after ``token`` ends an abbreviation, such as "e.g." or "Dr.", rather than a sentence."""
    word = re.sub(r"^\W+", "", token).lower()
    if word in NUMBER_ABBREVIATIONS:
        return following.isdigit()
    return bool(DOTTED_TOKEN.fullmatch(word)) or word in ABBREVIATIONS


def starts_line_or_sentence(text: str, start: int) -> bool:
    """Whether only markup and opening quotes stand between ``start`` and a line start, or between ``start``
    and a sentence end with its closing quotes or brackets."""
    index = start
    while index > 0 and text[index - 1] in SECTION_MARKER_LEAD:
        index -= 1
    if index == 0 or text[index - 1] == "\n":
        return True
    while index > 0 and text[index - 1] in CLOSERS:
        index -= 1
    return index > 0 and text[index - 1] in SENTENCE_ENDING_MARKS


def word_key(word: str) -> str:
    """The word lowercased and without a trailing 's, so "Space's" and "space" share a key, as do "it's" and "it"."""
    lowered = word.lower()
    return lowered[:-2] if len(lowered) > 2 and lowered[-2] in APOSTROPHES and lowered[-1] == "s" else lowered


def same_word(word: str, target: str) -> bool:
    return word_key(word) == word_key(target)


def count_keyword(text: str, keyword: str) -> int:
    """Occurrences of the keyword that no letter or digit precedes, directly or through combining marks, and that
    no letter, digit or combining mark follows.

    An underscore is none of these, so ``_keyword_`` and ``user_keyword`` hold the keyword, and here neither is a Han
    or kana letter, since Chinese and Japanese leave no space between words. Combining marks after a symbol do not join
    the keyword to it, so an emoji with a variation selector may stand right before it.
    """
    pattern = re.compile(rf"(?i:{re.escape(keyword)})(?!{SPACED_WORD_CHARACTER})")
    count, position = 0, 0
    while position <= len(text) and (found := pattern.search(text, position)):
        start = found.start()
        while start > 0 and unicodedata.category(text[start - 1])[0] == "M":
            start -= 1
        if start > 0 and text[start - 1].isalnum() and not UNSPACED_LETTER.match(text[start - 1]):
            position = found.start() + 1
        else:
            count += 1
            position = max(found.end(), found.start() + 1)
    return count


def contains_keyword(text: str, keyword: str, word_boundary: bool = True) -> bool:
    return count_keyword(text, keyword) > 0 if word_boundary else keyword.lower() in text.lower()


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
    """The prompt says that the markdown divider ``***`` separates paragraphs."""
    expected = params.get("num_paragraphs")
    if not isinstance(expected, int):
        return False, "missing num_paragraphs"
    parts = between_dividers(text, DIVIDER_SPLIT)
    return len(parts) == expected, f"paragraphs={len(parts)} expected={expected}"


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
    found = paragraphs(text, PARAGRAPH_SEPARATOR)
    if len(found) != expected_paragraphs or nth < 1 or nth > len(found):
        return False, f"paragraphs={len(found)} need={expected_paragraphs} nth={nth}"
    actual = compared_words(found[nth - 1])
    if not starts_with_word(actual, first_word):
        return False, f"got first_word={actual[:1]} expected={first_word!r}"
    return True, "ok"


# === keywords ===============================================================


@constraint("keywords:forbidden_words")
def forbidden_words(text: str, params: dict) -> tuple[bool, str]:
    forbidden = params.get("forbidden_words", [])
    if not isinstance(forbidden, list) or not all(isinstance(word, str) for word in forbidden):
        return False, "forbidden_words must be a list of strings"
    if type(params.get("word_boundary", True)) is not bool:
        raise ValueError("word_boundary must be boolean")
    hits = [w for w in forbidden if contains_keyword(text, w, params.get("word_boundary", True))]
    return not hits, f"hits={hits}"


@constraint("keywords:word_once")
def word_once(text: str, params: dict) -> tuple[bool, str]:
    """The prompt asks to include the keyword; "once" is only in the instruction id."""
    keyword = params.get("keyword")
    if not isinstance(keyword, str):
        return False, "missing keyword"
    count = count_keyword(text, keyword)
    return count >= 1, f"count={count} keyword={keyword!r}"


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
    if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
        return False, "keywords must be a list of strings"
    if type(params.get("word_boundary", True)) is not bool:
        raise ValueError("word_boundary must be boolean")
    missing = [w for w in keywords if not contains_keyword(text, w, params.get("word_boundary", True))]
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
    """The prompt asks for the keyword N times and does not show the stored relation, so the count must be N."""
    return keyword_frequency(text, {**params, "relation": "exactly"})


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
    in_sentence = compared_words(found[sentence_index - 1])
    if word_index < 1 or word_index > len(in_sentence):
        return False, f"sentence has {len(in_sentence)} words, need pos {word_index}"
    actual = in_sentence[word_index - 1]
    return (
        starts_with_word(in_sentence[word_index - 1 :], keyword),
        f"pos[{sentence_index},{word_index}]={actual!r} expected={keyword!r}",
    )


@constraint("keywords:palindrome")
def palindrome(text: str, params: dict) -> tuple[bool, str]:
    """The response must contain a word whose letters and digits, at least three and two different ones, read the same
    backward, or a phrase of consecutive words whose letters and digits, at least seven and three different ones, do,
    as in "Never odd or even"."""
    found = words(text)
    # Apostrophes and combining marks are left out, as in a Devanagari "Malayalam".
    keys = ["".join(c for c in word.lower() if c.isalnum()) for word in found]
    for start, key in enumerate(keys):
        if len(key) >= PALINDROME_MIN_LETTERS and len(set(key)) >= PALINDROME_MIN_DIFFERENT and key == key[::-1]:
            return True, f"palindrome={found[start]!r}"
        for end in range(start + 1, min(len(keys), start + PHRASE_PALINDROME_MAX_WORDS)):
            key += keys[end]
            enough = len(key) >= PHRASE_PALINDROME_MIN_LETTERS and len(set(key)) >= PHRASE_PALINDROME_MIN_DIFFERENT
            if enough and key == key[::-1]:
                return True, f"palindrome={' '.join(found[start : end + 1])!r}"
    return False, "no palindrome found"


def has_alphabet(letter: str) -> bool:
    return letter.isalpha() and not UNSPACED_LETTER.match(letter)


@constraint("keywords:no_adjacent_consecutive")
def no_adjacent_consecutive(text: str, params: dict) -> tuple[bool, str]:
    """No two adjacent words may start with consecutive letters, in either order, as "apple banana" and "big apple" do.

    Letters of any script are compared by code point, so "в городе" fails too, except Han and kana, which have no
    alphabet.
    """
    # U+0130 lowercases to an "i" and a combining dot.
    initials = [w[0].lower()[0] for w in words(text)]
    for index in range(1, len(initials)):
        left, right = initials[index - 1], initials[index]
        if has_alphabet(left) and has_alphabet(right) and abs(ord(right) - ord(left)) == 1:
            return False, f"adjacent words at pos {index} start with {left!r} and {right!r}"
    return True, "no adjacent consecutive initials"


@constraint("keywords:start_end")
def start_end(text: str, params: dict) -> tuple[bool, str]:
    """The response must start and end with the same word, with nothing after the last word."""
    stripped = text.strip()
    found = list(COMPARED_WORD.finditer(stripped))
    if not found:
        return False, "empty"
    first, last = found[0].group(), found[-1]
    if last.end() != len(stripped):
        return False, f"text ends with {stripped[last.end() :][:20]!r} after the last word"
    return same_word(first, last.group()), f"start={first!r} end={last.group()!r}"


# === change_case ============================================================


def case_letters(text: str) -> list[str]:
    """The letters of the text except modifier letters, such as the Hawaiian okina, which have no case."""
    return [c for c in text if c.isalpha() and unicodedata.category(c) != "Lm"]


@constraint("change_case:english_capital")
def english_capital(text: str, params: dict) -> tuple[bool, str]:
    """Only the letter case; the prompt also asks for English, which this check does not look at."""
    letters = case_letters(text)
    if not letters:
        return False, "no letters"
    passed = all(c.isupper() for c in letters)
    return passed, f"all_upper={passed}"


@constraint("change_case:english_lowercase")
def english_lowercase(text: str, params: dict) -> tuple[bool, str]:
    """Only the letter case; the prompt also asks for English, which this check does not look at."""
    letters = case_letters(text)
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


def named_punctuation(text: str, name: str) -> list[str]:
    """The different punctuation characters of the text whose Unicode name holds ``name``."""
    return sorted({c for c in text if unicodedata.category(c)[0] == "P" and name in unicodedata.name(c, "")})


@constraint("punctuation:no_comma")
def no_comma(text: str, params: dict) -> tuple[bool, str]:
    """The prompt forbids any commas, so the fullwidth, ideographic and Arabic ones count too."""
    found = named_punctuation(text, "COMMA")
    return not found, f"commas={found}"


@constraint("punctuation:punctuation_dot")
def no_dot(text: str, params: dict) -> tuple[bool, str]:
    """IFEval semantics: the response must contain no dots at all, an ellipsis "…" included."""
    found = sorted({c for c in text if c in DOTS})
    return not found, f"dots={found}"


@constraint("punctuation:punctuation_exclamation")
def no_exclamation(text: str, params: dict) -> tuple[bool, str]:
    """The prompt forbids exclamation marks, so the fullwidth and inverted ones count too."""
    found = named_punctuation(text, "EXCLAMATION")
    return not found, f"exclamation_marks={found}"


# === startend ===============================================================


@constraint("startend:end_checker")
def end_checker(text: str, params: dict) -> tuple[bool, str]:
    """The response must end with the phrase, case ignored as in the postscript and section markers. The prompt
    forbids other words after it, so a closing quote, Markdown markup or an emoji may follow, as in
    "**Any other questions?**"."""
    end_phrase = params.get("end_phrase")
    if not isinstance(end_phrase, str):
        return False, "missing end_phrase"
    # The look-ahead finds overlapping occurrences; each match is as long as the phrase, so the last ends latest.
    occurrences = re.finditer(f"(?=({re.escape(end_phrase)}))", text, flags=re.IGNORECASE)
    end = max((found.end(1) for found in occurrences), default=-1)
    passed = end >= 0 and WORD.search(text, end) is None
    return passed, f"ends_with={end_phrase!r}"


@constraint("startend:quotation")
def quotation(text: str, params: dict) -> tuple[bool, str]:
    """The whole response must be wrapped in double quotation marks, straight or typographic."""
    stripped = text.strip()
    passed = len(stripped) > 1 and (
        (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("“") and stripped.endswith("”"))
    )
    return passed, f"quoted={passed}"


# === detectable_format ======================================================


@constraint("detectable_format:number_bullet_lists")
def number_bullet_lists(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_bullets")
    if not isinstance(expected, int):
        return False, "missing num_bullets"
    # A line of only markers, such as the divider "* * *" or "---", is not a bullet point.
    actual = sum(1 for line in text.splitlines() if BULLET.match(line) and not BULLET_ONLY_LINE.match(line))
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
    """A section begins with the splitter and its number at the start of a line or a sentence, after
    optional Markdown markup or an opening quote; a mention inside a sentence is no marker."""
    expected = params.get("num_sections")
    splitter = params.get("section_spliter") or params.get("section_splitter") or "Section"
    if not isinstance(expected, int) or not isinstance(splitter, str):
        return False, "bad kwargs"
    marker = re.compile(r"(?<![^\W_])" + re.escape(splitter) + r"\s*\d+", flags=re.IGNORECASE)
    markers = [found for found in marker.finditer(text) if starts_line_or_sentence(text, found.start())]
    bodies = [
        text[found.end() : following.start() if following else len(text)]
        for found, following in pairwise([*markers, None])
    ]
    if any(not WORD.search(body) for body in bodies):
        return False, "a section has no text after its marker"
    return len(markers) == expected, f"sections={len(markers)} expected={expected} splitter={splitter!r}"


@constraint("detectable_format:number_highlighted_sections")
def number_highlighted_sections(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("num_highlights")
    if not isinstance(expected, int):
        return False, "missing num_highlights"
    actual = sum(1 for match in MARKDOWN_HIGHLIGHT.finditer(text) if WORD.search(match.group(2)))
    return actual >= expected, f"highlights={actual} need>={expected}"


def outside_spans(text: str, spans: list[re.Match]) -> str | None:
    """The first word of ``text`` that no span covers; ``spans`` are in text order and do not overlap."""
    index = 0
    for word in WORD.finditer(text):
        while index < len(spans) and spans[index].end() <= word.start():
            index += 1
        if index == len(spans) or not spans[index].start() < word.start() < word.end() < spans[index].end():
            return word.group()[:20]
    return None


@constraint("detectable_format:square_brackets")
def square_brackets(text: str, params: dict) -> tuple[bool, str]:
    """Every word must sit in a bracket span with no whitespace; punctuation may stay outside: "[Hello], [world]"."""
    if not WORD.search(text):
        return False, "no words"
    uncovered = outside_spans(text, list(ONE_WORD_BRACKETS.finditer(text)))
    return uncovered is None, f"word outside one-word brackets: {uncovered!r}" if uncovered else "every word bracketed"


def starts_unspaced_sentence(text: str, match: re.Match) -> bool:
    """Whether what follows final marks with no space, as in "One.Two.", is the next sentence: a dash other than a
    hyphen is, and so is a letter or digit after "!" or "?". After a dot or an ellipsis only a capital or caseless
    letter is, and not after an initial, an acronym or an abbreviation or inside an address, as in "U.S.Army",
    "ASP.NET", "Dr.Smith" or "john.Doe@example.com"; a digit or a lowercase letter there, as in "3.5" or
    "example.com", is not. It reads at most TOKEN_WINDOW characters on each side of the marks."""
    following = text[match.end()]
    if not following.isalnum() or "!" in match.group(1) or "?" in match.group(1):
        return True
    if following.isdigit() or following.islower():
        return False
    start = match.start()
    window_start = max(0, start - TOKEN_WINDOW)
    letters_start = start
    while letters_start > window_start and text[letters_start - 1].isalpha():
        letters_start -= 1
    letters = text[letters_start:start]
    if len(letters) == 1 or letters.isupper():
        return False
    token = token_before(text[window_start:start], start - window_start)
    if abbreviation_dot(token, following):
        return False
    address = token + re.match(r"\S*", text[match.end() : match.end() + TOKEN_WINDOW]).group(0)
    return not ("@" in address or "://" in address or address.lower().startswith("www."))


@constraint("detectable_format:sentence_hyphens")
def sentence_hyphens(text: str, params: dict) -> tuple[bool, str]:
    """Sentences must be joined by hyphens with no whitespace between them, as in "One.-Two.". A line break is such
    whitespace, so a reply must stay on one line. Sentences that touch with nothing between them, as in "One.Two.",
    or that a dash other than a hyphen joins are not joined by a hyphen either. A reply with no sentence end to join
    is one sentence and passes."""
    stripped = text.strip()
    if not WORD.search(stripped):
        return False, "no sentences"
    if len(stripped.splitlines()) > 1:
        return False, "a line break stands between two lines"
    for match in UNJOINED_SENTENCE_END.finditer(stripped):
        if match.group(1) is not None and not match.group(0)[-1].isspace():
            if not starts_unspaced_sentence(stripped, match):
                continue
        else:
            following = stripped[match.end() : match.end() + 1]
            if match.group(1) == "." and abbreviation_dot(token_before(stripped, match.start()), following):
                continue
        before = stripped[max(0, match.start() - 20) : match.start() + 1].strip()
        return False, f"sentence end without a hyphen at {before!r}"
    return True, "sentences joined by hyphens"


@constraint("detectable_format:bigram_wrapping")
def bigram_wrapping(text: str, params: dict) -> tuple[bool, str]:
    """Every word must sit in a <<...>> span of two words. The last span of a line or a sentence may hold one, but
    spans of one word may not outnumber spans of two, so a reply with every word in its own span fails. Each Han or
    kana letter counts as a word here."""
    if not WORD.search(text):
        return False, "no words"
    sizes = Counter()
    for line in text.splitlines():
        spans = list(DOUBLE_ANGLE_SPAN.finditer(line))
        uncovered = outside_spans(line, spans)
        if uncovered is not None:
            return False, f"word outside <<...>>: {uncovered!r}"
        for index, span in enumerate(spans):
            count = len(compared_words(span.group(1)))
            # The sentence's final mark sits inside the span, before any closing quote, or right after it.
            closes = (
                index == len(spans) - 1
                or span.group(1).rstrip().rstrip(CLOSERS).endswith(SENTENCE_ENDING_MARKS)
                or line.startswith(SENTENCE_ENDING_MARKS, span.end())
            )
            if count != 2 and not (count == 1 and closes):
                return False, f"span {span.group(0)[:20]!r} holds {count} words"
            sizes[count] += 1
    if sizes[1] > sizes[2]:
        return False, f"{sizes[1]} spans of one word against {sizes[2]} of two"
    return True, "every bigram wrapped"


@constraint("detectable_format:constrained_response")
def constrained_response(text: str, params: dict) -> tuple[bool, str]:
    stripped = text.strip()
    wrapped = ANSWER_WRAPPING.match(stripped)
    if wrapped:
        stripped = wrapped.group(2).strip()
    passed = stripped in CONSTRAINED_RESPONSES
    return passed, f"matched_constrained={passed}"


# === detectable_content =====================================================


@constraint("detectable_content:postscript")
def postscript(text: str, params: dict) -> tuple[bool, str]:
    marker = params.get("postscript_marker")
    if not isinstance(marker, str):
        return False, "missing postscript_marker"
    match = re.search(MARKDOWN_LINE_START + re.escape(marker), text, flags=re.MULTILINE | re.IGNORECASE)
    found = match is not None and WORD.search(text, match.end()) is not None
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
    """Exactly two paragraphs, separated by the markdown divider ``***``."""
    return number_paragraphs(text, {"num_paragraphs": 2})


@constraint("paragraphs:paragraphs2")
def divided_paragraphs2(text: str, params: dict) -> tuple[bool, str]:
    """Exactly two paragraphs, separated by blank lines."""
    found = paragraphs(text, PARAGRAPH_SEPARATOR)
    return len(found) == 2, f"blank-line paragraphs={len(found)}"


# === first_word / last_word =================================================


@constraint("first_word:first_word_answer")
def first_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = compared_words(text)
    actual = found[0] if found else ""
    return starts_with_word(found, expected), f"first={actual!r} expected={expected!r}"


@constraint("first_word:first_word_sent")
def first_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("first_word")
    if not isinstance(expected, str):
        return False, "missing first_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = compared_words(sentence)
        if not starts_with_word(in_sentence, expected):
            return False, f"sent starts with {in_sentence[:1]} need={expected!r}"
    return True, f"all {len(found)} sentences start with {expected!r}"


@constraint("last_word:last_word_answer")
def last_word_answer(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = compared_words(text)
    actual = found[-1] if found else ""
    return ends_with_word(found, expected), f"last={actual!r} expected={expected!r}"


@constraint("last_word:last_word_sent")
def last_word_sent(text: str, params: dict) -> tuple[bool, str]:
    expected = params.get("last_word")
    if not isinstance(expected, str):
        return False, "missing last_word"
    found = sentences(text)
    if not found:
        return False, "no sentences"
    for sentence in found:
        in_sentence = compared_words(sentence)
        if not ends_with_word(in_sentence, expected):
            return False, f"sent ends with {in_sentence[-1:]} need={expected!r}"
    return True, f"all {len(found)} sentences end with {expected!r}"


# === count ==================================================================


@constraint("count:count_unique")
def count_unique(text: str, params: dict) -> tuple[bool, str]:
    """No word may appear twice, ignoring case and a trailing 's; in Chinese and Japanese, no letter may."""
    counts = Counter(word_key(w) for w in compared_words(text))
    if not counts:
        return False, "empty"
    repeated = sorted(word[:20] for word, count in counts.items() if count > 1)
    return not repeated, f"unique={len(counts)} total={counts.total()} repeated={repeated[:5]}"


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
    """No lowercase word may appear more than N times; a word with a capital letter, such as "The", is not counted."""
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    counts = Counter(w for w in compared_words(text) if w.islower())
    repeated = sorted(word[:20] for word, count in counts.items() if count > expected)
    return not repeated, f"most={max(counts.values(), default=0)} need<={expected} repeated={repeated[:5]}"


@constraint("count:counting_composition")
def counting_composition(text: str, params: dict) -> tuple[bool, str]:
    """Three paragraphs between ``* * *`` dividers, each of n_sent sentences of n_words words."""
    expected_sentences = params.get("n_sent")
    expected_words = params.get("n_words")
    if not isinstance(expected_sentences, int) or not isinstance(expected_words, int):
        return False, "bad kwargs"
    parts = between_dividers(text, COMPOSITION_DIVIDER_SPLIT)
    if len(parts) != COMPOSITION_PARAGRAPHS:
        return False, f"paragraphs={len(parts)} need={COMPOSITION_PARAGRAPHS}"
    for number, part in enumerate(parts):
        found = sentences(part)
        if len(found) != expected_sentences:
            return False, f"paragraph {number} has {len(found)} sentences, need={expected_sentences}"
        counts = [len(words(sentence)) for sentence in found]
        if any(count != expected_words for count in counts):
            return False, f"paragraph {number} sentence word counts {counts}, need={expected_words}"
    return True, f"{COMPOSITION_PARAGRAPHS} paragraphs of {expected_sentences} sentences of {expected_words} words"


# === letters ================================================================


@constraint("letters:letter_counting")
def letter_counting(text: str, params: dict) -> tuple[bool, str]:
    """The prompt counts letters ("Answer with less than N letters"); letters of every script count."""
    expected = params.get("N")
    if not isinstance(expected, int):
        return False, "missing N"
    actual = sum(1 for c in text if c.isalpha())
    return _with_relation(actual, params.get("relation", "at least"), expected, "letters:")


@constraint("letters:letter_counting2")
def letter_counting2(text: str, params: dict) -> tuple[bool, str]:
    """Same params as ``keywords:letter_frequency``."""
    return letter_frequency(text, params)


# === language ===============================================================


@constraint("language:response_language")
def response_language(text: str, params: dict) -> tuple[bool, str]:
    """Approximate: one closed-class hint word is enough, or a share of non-ASCII for non-Latin scripts."""
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
    """The phrase must appear exactly small_n times, each time with one word other than the first and last replaced."""
    phrase = params.get("phrase")
    expected = params.get("small_n")
    if not isinstance(phrase, str) or not isinstance(expected, int):
        return False, "bad kwargs"
    target = [word_key(w) for w in compared_words(phrase)]
    found = [word_key(w) for w in compared_words(text)]
    changes = []
    index = 0
    while target and index + len(target) <= len(found):
        differing = [j for j, word in enumerate(target) if found[index + j] != word]
        if len(differing) <= 1:
            changes.append(differing)
            index += len(target)
        else:
            index += 1
    inner = range(1, len(target) - 1)
    passed = len(changes) == expected and all(len(change) == 1 and change[0] in inner for change in changes)
    return passed, f"copies={len(changes)} need={expected} changed_positions={changes}"


# === combination ============================================================


@constraint("combination:two_responses")
def two_responses(text: str, params: dict) -> tuple[bool, str]:
    sections = TWO_RESPONSE_SEPARATOR.split(text)
    if any(not WORD.search(section) for section in sections[1:-1]):
        return False, "empty section between responses"
    # A part with no words, such as the "." after a closing separator, is not a response.
    parts = [section.strip() for section in sections if WORD.search(section)]
    passed = len(parts) == 2 and parts[0] != parts[1]
    return passed, f"sections_by_******={len(parts)}"
