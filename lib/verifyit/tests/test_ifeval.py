# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
import unicodedata
from types import SimpleNamespace

import pytest
from verifyit.execution.worker import call_bounded
from verifyit.grade import InvalidTask, Status
from verifyit.instruction_observations import prepare_extended_instruction_observations
from verifyit.modes import grade_ifeval
from verifyit.modes.ifeval import CONSTRAINTS
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec

# A passing and a failing response per constraint, written the way a model would answer.
CASES = [
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "The first thing to know.\n***\nThe second thing to know.",
        "The first thing to know.\n\nThe second thing to know.",
    ),
    (
        "length_constraints:number_words",
        {"relation": "at least", "num_words": 6},
        "One two three four five six seven",
        "Far too short",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at most", "num_sentences": 2},
        "First sentence. Second sentence.",
        "First sentence. Second sentence. Third sentence.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "crash"},
        "Opening thoughts.\n\nCrash reports arrived late.\n\nClosing thoughts.",
        "Opening thoughts.\n\nReports arrived late.\n\nClosing thoughts.",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["banana", "mango"]},
        "I would rather have an apple with breakfast.",
        "I would rather have a banana with breakfast.",
    ),
    (
        "keywords:existence",
        {"keywords": ["quantum", "entangled"]},
        "The quantum pair stayed entangled across the lab.",
        "The quantum pair stayed together across the lab.",
    ),
    (
        "keywords:frequency",
        {"keyword": "cat", "relation": "at least", "frequency": 2},
        "The cat watched another cat from the window.",
        "The cat watched a dog from the window.",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_relation": "at least", "let_frequency": 3},
        "A zebra dozed in the zoo.",
        "A zebra slept in the yard.",
    ),
    (
        "change_case:english_lowercase",
        {},
        "everything here stays lowercase, as asked.",
        "Everything here stays lowercase, as asked.",
    ),
    (
        "change_case:english_capital",
        {},
        "EVERYTHING HERE SHOUTS.",
        "Everything here shouts.",
    ),
    (
        "change_case:english_lowercase",
        {},
        "we spent a week in hawai\u02bbi.",
        "We spent a week in Hawai\u02bbi.",
    ),
    (
        "change_case:english_capital",
        {},
        "DON\u02bcT WORRY ABOUT IT.",
        "Don\u02bct worry about it.",
    ),
    (
        "punctuation:no_comma",
        {},
        "No commas appear anywhere in this reply.",
        "Commas, sadly, appear here.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "That covers the deployment. Any other questions?",
        "That covers the deployment. Let me know.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "Rimas con «supermán»: talismán, capitán, volcán, huracán y alacrán.\n\n**Any other questions?**",
        "Finish your response with this exact phrase Any other questions?. No other words should follow this phrase.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        '"That covers the deployment. Any other questions?"',
        '"Any other questions?" That covers the deployment.',
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "that covers the deployment. any other questions?",
        "that covers the deployment. any other questions? thanks",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Is there anything else I can help with?"},
        "MIX WARM WATER WITH A LITTLE VINEGAR, WIPE THE GLASS WITH A MICROFIBER CLOTH AND DRY IT WITH NEWSPAPER. "
        "IS THERE ANYTHING ELSE I CAN HELP WITH?",
        "Pencils contain graphite, not lead, so they are safe to use. Is there anything else I can help you with?",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 3},
        "* first\n* second\n* third",
        "* first\n* second",
    ),
    (
        "detectable_format:title",
        {},
        "<<The Lantern Keeper>>\n\nShe lit the wick at dusk.",
        "The Lantern Keeper\n\nShe lit the wick at dusk.",
    ),
    (
        "detectable_format:json_format",
        {},
        '{"status": "ok", "items": [1, 2]}',
        "status: ok, items: 1 and 2",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Send it to [name] at [address] before Friday.",
        "Send it to [name] before Friday.",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "crash"},
        "Crash barriers were installed overnight.",
        "The crash barriers were installed overnight.",
    ),
    (
        "last_word:last_word_answer",
        {"last_word": "contest"},
        "Everyone practised hard before the contest",
        "Everyone practised hard before the match",
    ),
    (
        "count:lowercase_counting",
        {"N": 2},
        "The cat sat on a mat, then the dog sat.",
        "the cat and the dog and the bird",
    ),
    (
        "count:lowercase_counting",
        {"N": 2},
        "The sun rose. The sky cleared. We saw the sea.",
        "The sun rose. We saw the sky, the sea and the hills.",
    ),
    (
        "combination:two_responses",
        {},
        "First take on the question.\n******\nSecond take on the question.",
        "Only one take on the question.",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "Zebras eat grass quietly.",
        "An apple boat.",
    ),
    (
        "keywords:start_end",
        {},
        "Hello there, hello",
        "Hello there, hello.",
    ),
    (
        "startend:quotation",
        {},
        '"A quoted answer."',
        "'A quoted answer.'",
    ),
    (
        "detectable_format:square_brackets",
        {},
        "[Hello], [world].",
        "[Hello] world",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "The sky is blue.-The grass is green.",
        "The sky is blue. The grass is green.",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        '<<She said>> <<"Stop.">> <<I am>> <<at home>>',
        "<<I am>> at home",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        "Section 1\nSetup steps.\nSection 2\nTest steps.",
        "Section 1 covers setup, and Section 2 covers tests.",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        "Here is the plan. Section 1 Setup steps. Section 2 Test steps.",
        "Here is the plan for Section 1 setup. Section 2 Test steps.",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        '"Section 1\nSetup steps.\nSection 2\nTest steps."',
        "Section 1\nSetup steps, which Section 2 repeats.",
    ),
    (
        "paragraphs:paragraphs",
        {},
        "First paragraph.\n***\nSecond paragraph.",
        "First.\n***\nSecond.\n***\nThird.",
    ),
    (
        "paragraphs:paragraphs2",
        {},
        "First paragraph.\n\nSecond paragraph.",
        "First.\n\nSecond.\n\nThird.",
    ),
    (
        "count:count_unique",
        {},
        "Every word here differs.",
        "Each word differs, except this word.",
    ),
    (
        "count:counting_composition",
        {"n_sent": 3, "n_words": 2},
        "Go now. Be fast. Run far.\n* * *\nEat well. Sleep early. Stay calm.\n* * *\n"
        "Read books. Write notes. Think deeply.",
        "Go now. Be fast. Run far.",
    ),
    (
        "letters:letter_counting",
        {"N": 3, "relation": "less than"},
        "42",
        "Yes",
    ),
    (
        "copy:repeat_phrase",
        {"phrase": "The pen is mightier than sword", "small_n": 2},
        "The pen is stronger than sword. The pen is sharper than sword.",
        "The pen is mightier than sword. The pen is mightier than sword.",
    ),
    (
        "punctuation:punctuation_dot",
        {},
        "No dots appear here",
        "Wait for it…",
    ),
    (
        "detectable_format:constrained_response",
        {},
        "**My answer is yes.**",
        "My answer is yes. It works.",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "Main text.\n**P.S.** Bring water.",
        "Main text.\nP.S.",
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "at least", "capital_frequency": 2},
        "СССР и США",  # noqa: RUF001 - Cyrillic capitals.
        "СССР и мир",  # noqa: RUF001 - Cyrillic capitals.
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "第一句。第二句。",
        "只有一句。",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 2},
        "ಪ್ರಪಂಚ",
        "ಮತ್ತು ಪ್ರಪಂಚ",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "第一句。-第二句。",
        "第一句。第二句。",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "He said “Stop!”-Then he left.",
        "He said “Stop!” Then he left.",
    ),
    (
        "keywords:palindrome",
        {},
        "मलयालम एक भाषा है।",
        "हिंदी एक भाषा है।",
    ),
    (
        "keywords:palindrome",
        {},
        "A man, a plan, a canal: Panama.",
        "I am here.",
    ),
    (
        "keywords:palindrome",
        {},
        "Taco cat is a classic.",
        "We met at a cafe.",
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 1},
        "Use *this* one.",
        "int *a, *b;",
    ),
    (
        "paragraphs:paragraphs2",
        {},
        "First paragraph.\n\n***\n\nSecond paragraph.",
        "First.\n\n---\n\nSecond.\n\nThird.",
    ),
    (
        "count:counting_composition",
        {"n_sent": 3, "n_words": 2},
        "Go now. Be fast. Run far.\n* * *\nEat well. Sleep early. Stay calm.\n* * *\n"
        "Read books. Write notes. Think deeply.",
        "Go now. Be fast. Run far.\n* * *\nEat well. Sleep early.\n* * *\nRead books. Write notes. Think deeply.",
    ),
    (
        "count:counting_composition",
        {"n_sent": 3, "n_words": 2},
        "Go now. Be fast. Run far.\n* * *\nEat well. Sleep early. Stay calm.\n* * *\n"
        "Read books. Write notes. Think deeply.",
        "Go now. Be fast. Run very far.\n* * *\nEat well. Sleep early. Stay calm.\n* * *\n"
        "Read books. Write notes. Think deeply.",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>> <<at home>>",
        "<<I am at>> <<home>>",
    ),
    (
        "copy:repeat_phrase",
        {"phrase": "The pen is mightier than sword", "small_n": 2},
        "The pen is stronger than sword. The pen is sharper than sword.",
        "A pen is mightier than sword. The pen is sharper than sword.",
    ),
    (
        "copy:repeat_phrase",
        {"phrase": "The pen is mightier than sword", "small_n": 2},
        "The pen is stronger than sword. The pen is sharper than sword.",
        "The pen is stronger than sword. The pen is sharper than sword. The pen is greater than sword.",
    ),
    (
        "detectable_format:square_brackets",
        {},
        "[Hello], [world].",
        "[Hello world]",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "We met Dr. Smith.-He waved.",
        "The price is $3.50. Buy it now.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "太好了！！-我们走吧。",  # noqa: RUF001 - Chinese punctuation.
        "太好了！！我们走吧。",  # noqa: RUF001 - Chinese punctuation.
    ),
    (
        "keywords:start_end",
        {},
        "Love is all you need, love",
        "Love is all you need, love ❤️",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "他说：“走吧。”-我们就走了。",  # noqa: RUF001 - Chinese punctuation.
        "他说：“走吧。”我们就走了。",  # noqa: RUF001 - Chinese punctuation.
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "We use e.g. this one.-Then stop.",
        "We use e.g. this one. Then stop.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "See Fig. 3 for the plot.-It helps.",
        "I ate a fig. It was sweet.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "See No. 5 for details.-It helps.",
        "My answer is no. Sorry about that.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "First sentence.-Second sentence.",
        "First sentence.Second sentence.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "Is it done?-Yes!-Good.",
        "Is it done?Yes!Good.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "Stop here.-Wait.",
        "Stop here.\u2013Wait.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "The U.S.Army paid 3.5 dollars at example.com.-We met Dr.Smith.",
        "The army paid.Then it left.",
    ),
    (
        "count:count_unique",
        {},
        "We sold 1,000 apples, 2,000 pears at 10:30 in the U.S. and U.K.",
        "We sold 1,000 apples and 1,000 pears.",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<It costs>> <<3.50 dollars>>",
        "<<It costs 3.50>> <<dollars>>",
    ),
    (
        "last_word:last_word_sent",
        {"last_word": "baseball"},
        "<<We love baseball!>> Everyone plays baseball.",
        "<<Go to Taiwan!>> Everyone plays baseball.",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at most", "num_sentences": 1},
        "“只有一句。”",
        "“第一句。”“第二句。”",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "space"},
        "Space's edge is far.",
        "Spaces are vast.",
    ),
    (
        "letters:letter_counting",
        {"N": 3, "relation": "less than"},
        "42",
        "答案是四十二",
    ),
    (
        "letters:letter_counting",
        {"N": 3, "relation": "at least"},
        "答案是四十二",
        "Да",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "Big dogs run.",
        "Big apples run.",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        "Section 1\nA.\nSection 2\nB.",
        "Section 1\nSection 2",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        "He said “Go.” Section 1 Setup. Section 2 Tests.",
        "He said “Go” Section 1 Setup. Section 2 Tests.",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* One\n* * *\n* Two",
        "* One\n* * *",
    ),
    (
        "detectable_format:title",
        {},
        "<<A Title>>\nText.",
        "Here: << >>",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 1},
        "Fill in [name] later.",
        "Fill in [ ] later.",
    ),
    (
        "count:count_unique",
        {},
        "One cat saw a dog.",
        "The cat saw the dog.",
    ),
    (
        "punctuation:punctuation_dot",
        {},
        "No dots here",
        "Done．",  # noqa: RUF001 - fullwidth full stop.
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>> <<home>>. <<You are>> <<out>>.",
        "<<I am>> <<home>>, <<you are>> <<out>>.",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 2},
        "می\u200cخواهم",  # noqa: RUF001 - Persian letters.
        "می خواهم",
    ),
    (
        "keywords:word_once",
        {"keyword": "fix"},
        "Fix the bug, then fix the test.",
        "Repair the bug, then test it.",
    ),
    (
        "keywords:word_count_different_numbers",
        {"keyword": "chicken", "frequency": 2, "relation": "less than"},
        "The chicken ate. The chicken slept.",
        "The chicken ate.",
    ),
    (
        "keywords:word_count_different_numbers",
        {"keyword": "chicken", "frequency": 2, "relation": "at least"},
        "The chicken ate. The chicken slept.",
        "The chicken ate. The chicken slept. The chicken woke.",
    ),
    (
        "punctuation:punctuation_dot",
        {},
        "No dots here",
        "Wait\u2025",
    ),
    (
        "punctuation:punctuation_dot",
        {},
        "\u4f60\u597d\u3002",
        "\u4f60\u597d.",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 2},
        "क्\u200dष",
        "क् ष",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "Wait… Then go.",
        "Wait, then go.",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "पहला वाक्य। दूसरा वाक्य",
        "पहला वाक्य दूसरा वाक्य",
    ),
    (
        "keywords:existence",
        {"keywords": ["front"]},
        "Use the _front_ door.",
        "Use the frontal door.",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["name"]},
        "Set the user field.",
        "Set the user_name field.",
    ),
    (
        "keywords:existence",
        {"keywords": ["cafe"]},
        "Meet at the cafe.",
        "Meet at the cafe\u0301.",
    ),
    (
        "keywords:word_once",
        {"keyword": "bar"},
        "Meet me at the \u2764\ufe0fbar tonight.",
        "Meet me at the barn tonight.",
    ),
    (
        "keywords:existence",
        {"keywords": ["\u0915\u0940"]},
        "\u0930\u093e\u092e \u0915\u0940 \u0915\u093f\u0924\u093e\u092c",
        "\u092c\u093e\u0915\u0940 \u0938\u092c \u0920\u0940\u0915",
    ),
    (
        "keywords:existence",
        {"keywords": ["\u0926\u0940"]},
        "\u0909\u0938\u0928\u0947 \u0915\u093f\u0924\u093e\u092c \u0926\u0940",
        "\u092e\u0941\u091d\u0947 \u0939\u093f\u0902\u0926\u0940 \u092a\u0938\u0902\u0926 \u0939\u0948",
    ),
    (
        "language:response_language",
        {"language": "hi"},
        "यह अच्छा है।",
        "This is good.",
    ),
    (
        "language:response_language",
        {"language": "fi"},
        "Tämä on hyvä kysymys, ja vastaus on yksinkertainen.",
        "This is a good question, and the answer is simple.",
    ),
    (
        "language:response_language",
        {"language": "pl"},
        "To jest dobre pytanie, a odpowiedź nie jest trudna.",
        "This is a good question, and the answer is not hard.",
    ),
    (
        "language:response_language",
        {"language": "sw"},
        "Hili ni swali zuri, na jibu lake ni rahisi.",
        "This is a good question, and its answer is easy.",
    ),
    (
        "keywords:keyword_specific_position",
        {"keyword": "cat", "n": 1, "m": 3},
        "我爱cat。",
        "我cat。",
    ),
    (
        "keywords:start_end",
        {},
        "我爱你\uff0c你也爱我",
        "我爱北京",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "senior"},
        "Senior工程师很忙。",
        "工程师很忙。",
    ),
    (
        "copy:repeat_phrase",
        {"phrase": "The pen is mightier than sword", "small_n": 2},
        "The pen is stronger than sword. The pen is sharper than sword.",
        "The pen is mightier than sword. The pen is sharper than sword.",
    ),
    (
        "count:count_unique",
        {},
        "我爱北京",
        "我爱北京我爱上海",
    ),
    (
        "count:count_unique",
        {},
        "こんにちは世界",
        "ありがとうありがとう",
    ),
    (
        "count:lowercase_counting",
        {"N": 1},
        "the世界",
        "the世界the",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "Section"},
        "- Section 1\nSetup steps.\n- Section 2\nTest steps.",
        "- Section 1\nSetup steps.",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "Main text.\n- P.S. Bring water.",
        "Main text.\n- Bring water.",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>> <<at home>>",
        "<<I>> <<am>> <<at>> <<home>>",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>>\n<<at home>>",
        "<<I>>\n<<am>>\n<<at>>\n<<home>>",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>> <<home>>. <<You are>> <<out>>.",
        "<<I>>. <<am>>. <<at>>. <<home>>.",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<I am>> <<home.>> <<You are>> <<out.>>",
        "<<I.>> <<am.>> <<at.>> <<home.>>",
    ),
    (
        "punctuation:no_comma",
        {},
        "First then second.",
        "First\uff0c then second.",
    ),
    (
        "punctuation:no_comma",
        {},
        "First then second.",
        "First\u3001 then second.",
    ),
    (
        "punctuation:no_comma",
        {},
        "First then second.",
        "First\u060c then second.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "First point.-Second point.-Third point.",
        "First point\nSecond point\nThird point",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "One.-Two.",
        "One.-\nTwo.",
    ),
    (
        "detectable_format:sentence_hyphens",
        {},
        "One long sentence with no final mark",
        "One long line with no final mark\nAnd another line",
    ),
    (
        "keywords:palindrome",
        {},
        "Step on no pets!",
        "I I I I I I I",
    ),
    (
        "keywords:palindrome",
        {},
        "Never odd or even.",
        "no on no on",
    ),
    (
        "length_constraints:number_words",
        {"relation": "at least", "num_words": 2},
        "high-risk",
        "risk",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 4},
        "the-cat-sat",
        "the-cat-sat-on-the-mat",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "well"},
        "Well-known facts.",
        "Known facts.",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "Overview\nThe plan works.",
        "The plan works.",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "less than", "num_sentences": 2},
        "One. . . .",
        "One. Two.",
    ),
    (
        "first_word:first_word_sent",
        {"first_word": "peace"},
        "Peace now.\n***\nPeace then.",
        "Peace now\nWe rest\nPeace then",
    ),
    (
        "last_word:last_word_sent",
        {"last_word": "rest"},
        "We rest.\nThey rest",
        "We rest\nThey run\nWe rest",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "well-known"},
        "Well-known facts are here.",
        "Well facts are here.",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "世界"},
        "世界很大。",
        "世人很大。",
    ),
    (
        "first_word:first_word_sent",
        {"first_word": "long-term"},
        "Long-term plans win. Long-term goals stay.",
        "Long-term plans win. Term goals stay.",
    ),
    (
        "last_word:last_word_answer",
        {"last_word": "follow-up"},
        "This is the follow-up",
        "This is the up",
    ),
    (
        "last_word:last_word_sent",
        {"last_word": "self-care"},
        "I love self-care. You need self-care.",
        "I love self-care. You need care.",
    ),
    (
        "keywords:keyword_specific_position",
        {"keyword": "long-term", "n": 1, "m": 2},
        "A long-term plan wins.",
        "A term plan wins.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "well-being"},
        "Intro here.\n\nWell-being matters.",
        "Intro here.\n\nBeing matters.",
    ),
    (
        "keywords:existence",
        {"keywords": ["Python"]},
        "我喜欢Python语言",
        "我喜欢Java语言",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["banana"]},
        "我喜欢苹果的味道",
        "我喜欢banana的味道",
    ),
    (
        "keywords:word_once",
        {"keyword": "senior"},
        "Senior工程师很忙。",
        "工程师很忙。",
    ),
    (
        "count:count_unique",
        {},
        "high-risk zone",
        "high-risk risk",
    ),
    (
        "count:count_unique",
        {},
        "the-cat-and-a-dog",
        "the-cat-and-the-dog",
    ),
    (
        "count:lowercase_counting",
        {"N": 1},
        "the-cat a-dog",
        "the-cat the-dog",
    ),
    (
        "count:lowercase_counting",
        {"N": 2},
        "The-cat-sat-on-the-mat",
        "The-cat-sat-on-the-mat-and-the-dog-sat-on-the-rug",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "apple-cherry",
        "apple-banana",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "\u4e00 \u4e01",
        "apple banana",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "\u0432 \u0434\u043e\u043c\u0435",
        "\u0432 \u0433\u043e\u0440\u043e\u0434\u0435",
    ),
    (
        "keywords:no_adjacent_consecutive",
        {},
        "\u0130stanbul Kars",
        "\u0130stanbul Japonya",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 1},
        "This is ***very*** important.",
        "This is.\n***\nImportant.",
    ),
    (
        "paragraphs:paragraphs",
        {},
        "A ***big*** one.\n***\nThe other one.",
        "The first one.\n****\nThe other one.",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "The first one.***The other one.",
        "The first one. The other one.",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 1},
        "这是***很***重要的。",
        "这很重要。\n***\n那也重要。",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 1},
        "This is (***very***) important.",
        "This is.\n***\n(Important.)",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "第一段。***第二段。",
        "第一段。第二段。",
    ),
    (
        "punctuation:punctuation_exclamation",
        {},
        "Great news.",
        "Great news\uff01",
    ),
    (
        "punctuation:punctuation_exclamation",
        {},
        "Great news.",
        "\u00a1Great news",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<well-known>> <<facts are>>",
        "<<well-known fact>>",
    ),
    (
        "detectable_format:bigram_wrapping",
        {},
        "<<你好>> <<世界>>",
        "<<你好>> 世界",
    ),
    (
        "keywords:palindrome",
        {},
        "A race-car passed.",
        "A race-card passed.",
    ),
    (
        "keywords:palindrome",
        {},
        "Mom called.",
        "It costs 2,000.",
    ),
    (
        "keywords:palindrome",
        {},
        "Mom called.",
        "Visit www.site.org",
    ),
]


# Degenerate output, such as a long run of marks or a bracket that never closes, must stay linear in the
# text's length.
@pytest.mark.parametrize(
    "name, params, text, passed",
    [
        (
            "length_constraints:number_sentences",
            {"relation": "at least", "num_sentences": 2},
            "Wait" + "." * 40_000,
            False,
        ),
        ("detectable_format:sentence_hyphens", {}, "Wait" + "!" * 40_000, True),
        ("detectable_format:sentence_hyphens", {}, "U.S." * 25_000, True),
        ("detectable_format:sentence_hyphens", {}, "Dr." * 33_000, True),
        ("count:count_unique", {}, "1." * 50_000, True),
        ("detectable_content:number_placeholders", {"num_placeholders": 1}, "See [" + "a" * 100_000, False),
        ("detectable_format:title", {}, "<<" + "a" * 100_000, False),
        ("detectable_format:square_brackets", {}, " ".join(["[word]"] * 20_000), True),
        ("detectable_format:json_format", {}, "```" + "\n" * 100_000 + "x", False),
        ("keywords:existence", {"keywords": ["b"]}, "a" + "\u0301" * 100_000 + "b", False),
        ("keywords:palindrome", {}, " ".join(["ab"] * 20_000), False),
        ("startend:end_checker", {"end_phrase": "a a"}, "a " * 50_000, True),
    ],
    ids=[
        "sentence_marks",
        "hyphen_marks",
        "initials_without_spaces",
        "abbreviations_without_spaces",
        "separated_number_without_spaces",
        "open_bracket",
        "open_title",
        "many_brackets",
        "fence_newlines",
        "keyword_after_marks",
        "many_words",
        "overlapping_phrase",
    ],
)
def test_check_of_degenerate_text_finishes_within_its_deadline(name, params, text, passed):
    assert call_bounded(CONSTRAINTS[name], text, params, timeout=5)[0] is passed


def test_empty_keyword_finishes_within_its_deadline():
    # An empty keyword matches at every position, so the search must still advance.
    assert call_bounded(CONSTRAINTS["keywords:existence"], "Hello world.", {"keywords": [""]}, timeout=5)[0] is True


def test_combining_mark_stays_inside_its_word():
    check = CONSTRAINTS["length_constraints:number_words"]
    marks = (chr(code) for code in range(sys.maxunicode + 1) if unicodedata.category(chr(code))[0] == "M")
    assert all(check(f"a{mark}b", {"relation": "less than", "num_words": 2})[0] for mark in marks)


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def reward_for(workspace, tests_dir, constraints, text=None):
    if text is not None:
        answer(workspace, text)
    return grade_ifeval.grade(IfevalSpec(constraints=constraints), tests_dir, workspace)


@pytest.mark.parametrize("name, params, passing, failing", CASES, ids=[case[0] for case in CASES])
def test_constraint_separates_a_satisfying_response_from_a_violating_one(
    tmp_path, workspace, name, params, passing, failing
):
    constraints = (Constraint(name, params),)
    assert reward_for(workspace, tmp_path, constraints, passing).reward == 1.0
    violated = reward_for(workspace, tmp_path, constraints, failing)
    assert violated.reward == 0.0
    assert violated.detail["failed"] == [name]


def test_every_constraint_must_hold_for_a_full_reward(tmp_path, workspace):
    # The exemplar task: four paragraphs whose third starts with "crash", ending on "contest".
    constraints = (
        Constraint(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 4, "nth_paragraph": 3, "first_word": "crash"},
        ),
        Constraint("last_word:last_word_answer", {"last_word": "contest"}),
    )
    body = "One.\n\nTwo.\n\nCrash tests came third.\n\nWe entered the contest"
    assert reward_for(workspace, tmp_path, constraints, body).reward == 1.0

    wrong_ending = reward_for(workspace, tmp_path, constraints, body.replace("contest", "raffle"))
    assert wrong_ending.reward == 0.0
    assert wrong_ending.detail["failed"] == ["last_word:last_word_answer"]

    out_of_order = "Crash tests came first.\n\nTwo.\n\nThree.\n\nWe entered the contest"
    reordered = reward_for(workspace, tmp_path, constraints, out_of_order)
    assert reordered.detail["failed"] == ["length_constraints:nth_paragraph_first_word"]


def test_detail_records_a_verdict_for_every_constraint(tmp_path, workspace):
    constraints = (Constraint("punctuation:no_comma", {}), Constraint("keywords:word_once", {"keyword": "otter"}))
    reward = reward_for(workspace, tmp_path, constraints, "An otter swam past, twice.")
    assert [(entry["name"], entry["passed"]) for entry in reward.detail["constraints"]] == [
        ("punctuation:no_comma", False),
        ("keywords:word_once", True),
    ]


@pytest.mark.parametrize("text", [None, "", "  \n"])
def test_absent_or_blank_output_scores_zero_with_no_output(tmp_path, workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = reward_for(workspace, tmp_path, (Constraint("punctuation:no_comma", {}),))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail == {"reason": "no_output"}


def test_unknown_constraint_is_an_invalid_task(tmp_path, workspace):
    answer(workspace, "anything at all")
    with pytest.raises(InvalidTask, match="unknown ifeval constraints"):
        reward_for(workspace, tmp_path, (Constraint("keywords:no_such_check", {}),))


def test_spec_without_constraints_is_an_invalid_task(tmp_path, workspace):
    answer(workspace, "anything at all")
    with pytest.raises(InvalidTask, match="no constraints"):
        reward_for(workspace, tmp_path, ())


def test_constraint_missing_its_parameters_fails_the_candidate(tmp_path, workspace):
    constraints = (Constraint("length_constraints:number_paragraphs", {}),)
    reward = reward_for(workspace, tmp_path, constraints, "Two.\n\nParagraphs.")
    assert reward.reward == 0.0
    assert reward.detail["constraints"][0]["detail"] == "missing num_paragraphs"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First answer.******Second answer.", 1.0),
        ("******First answer.******Second answer.******", 1.0),
        ("Same answer.****** Same answer. ", 0.0),
        ("First.******Second.******Third.", 0.0),
        ("First.************Second.", 0.0),
        ("First.*******Second.", 0.0),
        ("Responses should be separated by 6 asterisk symbols: ******.", 0.0),
    ],
)
def test_two_responses_requires_exactly_two_distinct_answers(tmp_path, workspace, text, expected):
    result = reward_for(workspace, tmp_path, (Constraint("combination:two_responses", {}),), text)
    assert (result.reward, result.status) == (expected, Status.SCORED)


@pytest.mark.parametrize("policy", [EmptyOutputPolicy.ZERO, EmptyOutputPolicy.GRADE])
def test_direct_candidate_preserves_file_empty_and_constraint_results(tmp_path, policy):
    spec = IfevalSpec((Constraint("punctuation:no_comma"),), empty_output=policy)
    for candidate in ["plain", "one,two", ""]:
        (tmp_path / "answer.txt").write_text(candidate)
        direct = grade_ifeval.grade_ifeval_candidate(spec, candidate)
        from_file = grade_ifeval.grade(spec, tmp_path, tmp_path)
        assert direct == from_file
        assert direct.reward == float("," not in candidate and (bool(candidate) or policy is EmptyOutputPolicy.GRADE))


def test_custom_constraint_cannot_replace_builtin_scoring():
    spec = IfevalSpec((Constraint("punctuation:no_comma"),))
    with pytest.raises(InvalidTask):
        grade_ifeval.grade_ifeval_candidate(
            spec, "one,two", registry={"punctuation:no_comma": lambda text, params: (True, "")}
        )
    assert grade_ifeval.grade_ifeval_candidate(spec, "one,two").reward == 0


@pytest.mark.parametrize("value,detail", [(1, ""), ("passed", ""), (None, ""), (True, {})])
def test_custom_constraint_truthy_malformed_result_cannot_award_credit(value, detail):
    spec = IfevalSpec((Constraint("custom:decision"),))
    with pytest.raises(RuntimeError):
        grade_ifeval.grade_ifeval_candidate(
            spec, "candidate", registry={"custom:decision": lambda text, params: (value, detail)}
        )


@pytest.mark.parametrize(
    "name,key", [("keywords:existence", "keywords"), ("keywords:forbidden_words", "forbidden_words")]
)
def test_keyword_substring_policy_preserves_lowercase_and_word_default(name, key):
    expected = name == "keywords:existence"
    default = IfevalSpec((Constraint(name, {key: ["cat"]}),))
    substring = IfevalSpec((Constraint(name, {key: ["cat"], "word_boundary": False}),))
    assert grade_ifeval.grade_ifeval_candidate(default, "SCATTER").reward == float(not expected)
    assert grade_ifeval.grade_ifeval_candidate(substring, "SCATTER").reward == float(expected)
    malformed = IfevalSpec((Constraint(name, {key: [], "word_boundary": "false"}),))
    assert grade_ifeval.grade_ifeval_candidate(malformed, "candidate").reward == 0


@pytest.mark.parametrize(
    "name,key,valid",
    [("keywords:existence", "keywords", "candidate"), ("keywords:forbidden_words", "forbidden_words", "absent")],
)
def test_malformed_keyword_members_cannot_silently_disappear(name, key, valid):
    for words in ([None], [valid, None]):
        spec = IfevalSpec((Constraint(name, {key: words}),))
        assert grade_ifeval.grade_ifeval_candidate(spec, "candidate").reward == 0
    for words in ([], [valid]):
        spec = IfevalSpec((Constraint(name, {key: words}),))
        assert grade_ifeval.grade_ifeval_candidate(spec, "candidate").reward == 1


@pytest.mark.parametrize(
    "text,language,expected",
    [("YES THIS IS ENGLISH", "en", 1.0), ("Yes this is English", "en", 0.0), ("OUI", "fr", 0.0)],
)
def test_prepared_observations_require_both_constraint_and_tool_result(text, language, expected):
    verdict = grade_ifeval.grade_instruction_observations(
        [
            (IfevalSpec((Constraint("change_case:english_capital"),)), text),
            ({"type": "string", "const": "en"}, language),
        ]
    )
    assert (verdict.status, verdict.reward) == (Status.SCORED, expected)


def test_invalid_prepared_reference_cannot_preserve_a_passing_observation():
    with pytest.raises(InvalidTask):
        grade_ifeval.grade_instruction_observations(
            [({"const": "valid"}, "valid"), ({"type": "array", "minItems": "many"}, [])]
        )
    empty = grade_ifeval.grade_instruction_observations([])
    assert (empty.status, empty.reward) == (Status.INVALID_TASK, 0.0)


@pytest.mark.parametrize("identifier", ["detectable_format:sentence_hyphens", "keywords:start_end"])
def test_empty_instruction_tokens_score_zero_without_invalidating_task(identifier):
    tools = SimpleNamespace(split_into_sentences=str.split, nltk=SimpleNamespace(word_tokenize=str.split))
    observations = prepare_extended_instruction_observations(identifier, {}, tools, "")
    verdict = grade_ifeval.grade_instruction_observations(observations)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 0.0)


def test_repeated_phrase_requires_exactly_one_changed_word_per_match():
    args = {"phrase": "red warm soft fox", "small_n": 2}
    for candidate, expected in [
        ("red cold soft fox red warm furry fox", 1.0),
        ("red cold furry fox red warm furry fox", 0.0),
        ("red warm soft fox red warm furry fox", 0.0),
        ("red cold soft fox", 0.0),
    ]:
        observations = prepare_extended_instruction_observations("copy:repeat_phrase", args, None, candidate)
        assert grade_ifeval.grade_instruction_observations(observations).reward == expected


def test_source_copy_span_uses_exclusive_character_end():
    args = {"prompt_to_repeat": "A small bird", "n_start": 2, "n_end": 7}
    for candidate, expected in [("SMALL", 1.0), ("small b", 0.0), ("small bird", 0.0)]:
        observations = prepare_extended_instruction_observations("new:copy_span_idx", args, None, candidate)
        assert grade_ifeval.grade_instruction_observations(observations).reward == expected
