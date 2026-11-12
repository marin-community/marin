import re
import itertools
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def web_of_lies_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    score = 0
    parsed_answer = None
    # pull out words in bold
    bold_words = re.findall(r"\*\*(.*?)\*\*", llm_answer)

    if bold_words:
        bold_words = [
            word.lower().strip().replace(",", "").replace(".", "")[0 : max(len(word), 3)]
            for match in bold_words
            for word in match.split()
        ]
        parsed_answer = []
        i = len(bold_words) - 1
        while i >= 0 and len(parsed_answer) < 3:
            if bold_words[i] in ["yes", "no"]:
                parsed_answer = [bold_words[i]] + parsed_answer
            i -= 1
        if len(parsed_answer) > 0:
            parsed_answer = ", ".join(parsed_answer)
        else:
            parsed_answer = None

    allow_latex = True
    if parsed_answer is None or parsed_answer.strip() == "" and allow_latex:
        llm_answer = llm_answer.replace("\\\\boxed{\\\\textbf{", "\\\\boxed{")
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        llm_answer = llm_answer.replace("\\textbf{", "\\boxed{")

        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)

    allow_plain = True
    if allow_plain and parsed_answer is None:
        combs = itertools.product(["yes", "no"], repeat=3)
        # find all instances of these combinations in the answer, then treat the last one as the actual answer
        # to compare to the ground truth
        final_comb = None
        final_comb_index = -1
        for comb in combs:
            index = llm_answer.lower().find(", ".join(comb))
            if index != -1 and index > final_comb_index:
                final_comb = comb
                final_comb_index = index
        if final_comb is not None:
            parsed_answer = ", ".join(final_comb)

    # Check if parsed_answer is an exact match of ground_truth
    if parsed_answer and parsed_answer == ground_truth.lower():
        score = 1

    # Check if parsed_answer contains the ground_truth
    if (
        parsed_answer
        and parsed_answer.count("yes") + parsed_answer.count("no") == 3
        and ground_truth.lower() in parsed_answer
    ):
        score = 1

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth)
        if parsed_answer:
            print("PARSED ANSWER", parsed_answer)
        else:
            print("NO PARSED ANSWER")
        print("END OF OUTPUT", llm_answer[-50:])

    return score
