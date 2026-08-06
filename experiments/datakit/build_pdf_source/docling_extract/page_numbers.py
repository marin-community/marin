# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recognising running page numbers, in the fifty-odd languages the crawl is written in.

A page number is a short numeric string in the first or last text block on a page, optionally
introduced by a word meaning "page" and optionally followed by a total. The FinePDFs patterns are
ported here unchanged, because the set of forms they cover is the contribution -- ``1``, ``1/10``,
``Page 1 of 2``, ``第1页共10页``, ``1. oldal``, ``עמ' 1`` and so on.

The number itself is ``0`` or ``1``-``9999`` with no leading zeros, which is what keeps a year, a
part number, or a lone figure caption from being read as pagination.
"""

import re
from functools import cache

# Words meaning "page" and the connector each language puts between the number and the total.
_PAGE_WORDS_BY_LANGUAGE = {
    "en": (["Page", "p", "pp"], ["of"]),
    "zh": (["页"], ["的"]),
    "hi": (["पृष्ठ", "पेज"], ["का", "से"]),
    "es": (["Página", "Pág"], ["de"]),
    "fr": (["Page", "p"], ["sur", "de"]),
    "ar": (["صفحة"], ["من"]),
    "bn": (["পৃষ্ঠা", "পাতা"], ["এর"]),
    "ru": (["Страница", "Стр"], ["из"]),
    "pt": (["Página", "Pág"], ["de"]),
    "id": (["Halaman", "Hal"], ["dari"]),
    "ur": (["صفحہ"], ["کا"]),
    "de": (["Seite", "S"], ["von"]),
    "ja": (["ページ", "P"], ["の"]),
    "sw": (["Ukurasa", "Uk"], ["wa"]),
    "mr": (["पान", "पृष्ठ"], ["चे"]),
    "te": (["పేజీ", "పుట"], ["లో"]),
    "tr": (["Sayfa", "S"], ["den"]),
    "ta": (["பக்கம்", "பக்"], ["இல்"]),
    "vi": (["Trang", "Tr"], ["của"]),
    "ko": (["페이지", "쪽"], ["의"]),
    "it": (["Pagina", "Pag"], ["di"]),
    "th": (["หน้า"], ["ของ"]),
    "gu": (["પાનું", "પેજ"], ["નું"]),
    "pl": (["Strona", "Str"], ["z"]),
    "uk": (["Сторінка", "Стор"], ["з"]),
    "kn": (["ಪುಟ", "ಪೇಜ್"], ["ರ"]),
    "ml": (["പേജ്", "താൾ"], ["ന്റെ"]),
    "or": (["ପୃଷ୍ଠା", "ପେଜ୍"], ["ର"]),
    "pa": (["ਪੰਨਾ", "ਪੇਜ"], ["ਦਾ"]),
    "ro": (["Pagina", "Pag"], ["din"]),
    "nl": (["Pagina", "Pag", "p"], ["van"]),
    "hu": (["oldal", "o"], ["ból"]),
    "el": (["Σελίδα", "Σελ"], ["από"]),
    "cs": (["Strana", "Str"], ["z"]),
    "be": (["Старонка", "Стар"], ["з"]),
    "he": (["עמוד", "עמ"], ["מתוך"]),
    "sv": (["Sida", "S"], ["av"]),
    "az": (["Səhifə", "Səh"], ["dan"]),
    "bg": (["Страница", "Стр"], ["от"]),
    "ms": (["Muka surat", "Ms"], ["daripada"]),
    "uz": (["Sahifa", "Sah"], ["dan"]),
    "ne": (["पृष्ठ", "पेज"], ["को"]),
    "si": (["පිටුව", "පි"], ["හි"]),
    "kk": (["Бет", "Б"], ["дан"]),
    "am": (["ገጽ"], ["ከ"]),
    "ka": (["გვერდი", "გვ"], ["დან"]),
    "no": (["Side", "S"], ["av"]),
    "da": (["Side", "S"], ["af"]),
    "fi": (["Sivu", "S"], ["ja"]),
    "sk": (["Stránka", "Str"], ["z"]),
    "hr": (["Stranica", "Str"], ["od"]),
}

# 0 or 1-9999, no leading zeros: narrow enough that a year or a part number is not pagination.
_NUMBER = r"(?:0|[1-9]\d{0,3})"


def _language_specific_patterns() -> list[str]:
    """Forms that do not fit the ``<word> <number> <connector> <number>`` shape."""
    return [
        rf"^\s*第\s*{_NUMBER}\s*页\s*$",
        rf"^\s*第\s*{_NUMBER}\s*页\s*/\s*{_NUMBER}\s*$",
        rf"^\s*第\s*{_NUMBER}\s*页\s*共\s*{_NUMBER}\s*页\s*$",
        rf"^\s*第\s*{_NUMBER}\s*頁\s*$",
        rf"^\s*第\s*{_NUMBER}\s*頁\s*/\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*ページ\s*$",
        rf"^\s*{_NUMBER}\s*ページ\s*/\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*/\s*{_NUMBER}\s*ページ\s*$",
        rf"^\s*P\.\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*페이지\s*$",
        rf"^\s*{_NUMBER}\s*페이지\s*/\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*/\s*{_NUMBER}\s*페이지\s*$",
        rf"^\s*{_NUMBER}\s*쪽\s*$",
        rf"^\s*{_NUMBER}\.\s*oldal\s*$",
        rf"^\s*{_NUMBER}\.\s*oldal\s*/\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*صفحة\s*$",
        rf"^\s*{_NUMBER}\s*صفحہ\s*$",
        rf"^\s*หน้า\s*{_NUMBER}\s*$",
        rf"^\s*หน้า\s*{_NUMBER}\s*/\s*{_NUMBER}\s*$",
        rf"^\s*עמוד\s*{_NUMBER}\s*$",
        rf"^\s*עמ'\s*{_NUMBER}\s*$",
    ]


def _patterns_for_word(word: str, connectors: list[str]) -> list[str]:
    """Build the pattern set for one language's word for "page".

    An abbreviation (``p``, ``Pág.``, ``S``) may sit hard against its number; a full word needs
    whitespace, so that ``Pages`` or a sentence beginning ``Side by side`` is not matched.
    """
    escaped = re.escape(word)
    separator = r"\s*" if ("." in word or len(word) <= 2 or word.islower()) else r"\s+"
    patterns = [
        rf"^\s*{escaped}{separator}{_NUMBER}\s*$",
        rf"^\s*{escaped}{separator}{_NUMBER}\s*/\s*{_NUMBER}\s*$",
        rf"^\s*{escaped}{separator}{_NUMBER}\s*{_NUMBER}[a-z\/]{_NUMBER}\s*$",
        rf"^\s*{escaped}:\s*{_NUMBER}\s*$",
        rf"^\s*{escaped}:\s*{_NUMBER}\s*/\s*{_NUMBER}\s*$",
    ]
    # A document in any language may still write its totals with the English "of".
    patterns += [
        rf"^\s*{escaped}{separator}{_NUMBER}\s+{re.escape(connector)}\s+{_NUMBER}\s*$"
        for connector in [*connectors, "of"]
    ]
    return patterns


@cache
def page_number_patterns() -> tuple[re.Pattern[str], ...]:
    """Every page-number pattern, compiled once per process."""
    patterns = [
        rf"^\s*{_NUMBER}\s*$",
        rf"^\s*{_NUMBER}\s*/\s*{_NUMBER}\s*$",
        *_language_specific_patterns(),
    ]
    for words, connectors in _PAGE_WORDS_BY_LANGUAGE.values():
        for word in words:
            patterns.extend(_patterns_for_word(word, connectors))
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def is_page_number(text: str) -> bool:
    """Whether a block of text is a running page number rather than content."""
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in page_number_patterns())
