from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed
import re


def mathcontest_process_results(ground_truth: str, llm_answer: str, question_text: str, debug=False) -> int:
    score = 0
    # the reference answer must be a single capital letter from A to E (I.e., the multiple choice answer)
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and "A" <= ground_truth <= "E"):
        raise ValueError("amc_answer must be a single capital letter between A and E.")

    # The LLM was prompted to repeat letter answer 5 times, to make it easy to pull out the answer
    if ground_truth * 4 in llm_answer:
        score = 1

    allow_boxed = True
    if score == 0 and allow_boxed:
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed).replace("\\text{", "").replace("}", "").replace("\\", "").lower()
            if parsed_answer == ground_truth.lower():
                score = 1

    allow_answer_values = True
    if score == 0 and allow_answer_values:
        value = extract_answer(question_text, ground_truth)
        length_to_check = 20 + len(value)
        if value in llm_answer[-length_to_check:]:
            score = 1

    if debug and score == 0:
        # check if the LLM guessed a letter, even if it was wrong
        letter_answer = False
        for letters in ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]:
            if letters in llm_answer:
                letter_answer = True

        if not letter_answer and score == 0:
            print("INCORRECT")
            print("GROUND TRUTH", ground_truth.strip().lower())
            if last_boxed:
                print("PARSED ANSWER:", parsed_answer)
            print("END OF OUTPUT", llm_answer[-200:])

    return score


def extract_answer(statement, letter):
    pattern = r"\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)"
    matches = re.findall(pattern, statement)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None)

    if not answer or answer == "":
        # this only happens for one question, which is too long for the LLMs to repeat
        answer = "FAILURE"

    answer = answer.strip()
    answer = answer.strip("$")
    answer = answer.strip("~")

    return answer


def aime_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    score = 0
    if ground_truth in llm_answer[-50:]:
        score = 1

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth)
        print("SOLUTION", llm_answer[-200:])
    return score
