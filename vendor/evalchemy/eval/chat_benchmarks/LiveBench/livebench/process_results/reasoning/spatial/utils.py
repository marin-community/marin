import re
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def spatial_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    word_to_number = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    bold_words = re.findall(r"\*\*([^\*]+)\*\*", llm_answer)
    score = 0

    # allow the answer to be within the last 3 bolded words
    words_to_check = []
    for i in range(3):
        if bold_words and len(bold_words) > i:
            words_to_check.append(bold_words[-i - 1].strip().lower())

    for i, word in enumerate(words_to_check):
        if word == ground_truth.strip().lower():
            score = 1

        # allow the answer to be the number spelled out
        if word in word_to_number and word_to_number[word] == ground_truth.strip().lower():
            score = 1

        # allow certain cases like "two tetrahedra" == "tetrahedra" and "equilateral triangle" == "triangle"
        # while still disallowing cases like "circle square triangle" == "triangle"
        for answer in ["tetrahedra", "tetrahedron", "triangle", "square"]:
            if ground_truth.strip().lower() == answer and answer in word and len(word) < (2 * len(answer) + 5):
                score = 1

    allow_boxed = True
    if score == 0 and allow_boxed:
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)
            parsed_answer = parsed_answer.replace("\\textbf{", "")
            parsed_answer = parsed_answer.replace("\\mathbf{", "")
            parsed_answer = parsed_answer.replace("}", "")
            if parsed_answer == ground_truth:
                score = 1

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth.strip().lower())
        if bold_words:
            print("BOLD WORDS:", bold_words[-1].strip().lower())
        if last_boxed:
            print("LAST BOXED", last_boxed)
            print("PARSED ANSWER", parsed_answer)
        print("END OF OUTPUT", llm_answer[-50:])

    return score
