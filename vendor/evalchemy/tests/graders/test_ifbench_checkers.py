"""Regression cases for IFBench checker defects.

Each case id starts with its finding id in the marin-community/marin#10060 audit,
https://github.com/XiaoyuanWang111/ifbench-audit/blob/a31c6841dae2632dafda740bd51496843d9e5f38/reports/AUDIT.md,
and each expected verdict follows the instruction text.
"""

import pytest

from eval.chat_benchmarks.IFBench import grader

REQUEST = "explain the context, code and theory behind the paper Identifying Candidate Spaces for Advert Implantation."
ALPHABET = (
    '"Ants move." Bees move. Cats move. Dogs move. Eels move. Foxes move. Goats move. Hens move. Ibises move. '
    "Jays move. Kites move. Lions move. Moles move. Newts move. Owls move. Pigs move. Quails move. Rats move. "
    "Seals move. Toads move. Urchins move. Voles move. Wolves move. Xerus move. Yaks move. Zebras move."
)
EMPTY_FIELD_TABLE = (
    '"StudentID"\t"Subject"\t"Grade"\t"Semester"\t"Score"\n"1"\t"Math"\t"A"\t"Fall"\t"95"\n'
    '"2"\t\t"B"\t"Fall"\t"85"\n"3"\t"Art"\t"C"\t"Fall"\t"75"'
)

CASES = {
    "F1-word-indices": (
        "repeat:repeat_span",
        {"prompt_to_repeat": REQUEST, "n_start": 3, "n_end": 6},
        "code and theory behind",
        True,
    ),
    "F1-character-slice": (
        "repeat:repeat_span",
        {"prompt_to_repeat": REQUEST, "n_start": 3, "n_end": 6},
        "lain",
        False,
    ),
    "F2-word-trigrams": (
        "ratio:overlap",
        {"reference_text": "the cat sat on the mat", "percentage": 0},
        "mat the on sat cat the",
        True,
    ),
    "F6-repeated-words": ("ratio:sentence_words", {}, "I am here. I am here. I am here.", False),
    "F7-whole-words": (
        "count:keywords_multiple",
        {"keyword1": "door", "keyword2": "bread", "keyword3": "blue", "keyword4": "lamp", "keyword5": "river"},
        "The door opened onto a doorway. bread bread. blue blue blue. lamp lamp lamp lamp lamp. "
        "river river river river river river river.",
        True,
    ),
    "F8-punctuation-tokens": (
        "words:words_position",
        {"keyword": "vibrant"},
        "Hello, vibrant world, vibrant day.",
        True,
    ),
    "F9-opening-quote": ("custom:sentence_alphabet", {}, ALPHABET, True),
    "F15-empty-field": ("custom:csv_quotes", {}, EMPTY_FIELD_TABLE, False),
}


@pytest.mark.parametrize(("instruction_id", "kwargs", "response", "expected"), CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("mode", ["strict", "loose"])
def test_ifbench_checker_follows_instruction_text(instruction_id, kwargs, response, expected, mode):
    example = grader.InputExample(key=0, instruction_id_list=[instruction_id], prompt="", kwargs=[kwargs])
    check = getattr(grader, f"test_instruction_following_{mode}")

    assert check(example, {"": response}).follow_instruction_list == [expected]
