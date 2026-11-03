import re


def house_traversal_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    # pull out words in bold
    bold_words = re.findall(r"(\*{2,})(.*?)\1", llm_answer.lower())
    if not len(bold_words):
        if debug:
            print("INCORRECT")
            print("GROUND TRUTH", ground_truth)
            print("SOLUTION", llm_answer)
        return 0

    last_bold = bold_words[-1][1]
    ground_truth_names = ground_truth.lower().split(" ")

    # check if all the ground truth names are in the last few bolded words, in order
    if len(bold_words) >= len(ground_truth_names):
        if all([name in bold_words[-1 - i] for i, name in enumerate(ground_truth_names[::-1])]):
            return 1

    score = 1
    # check if all the ground truth names are in the last bolded part, in order
    last_index = -1
    for name in ground_truth_names:
        index = last_bold.find(name)
        if index == -1:
            score = 0
            break
        elif index <= last_index:
            score = 0
            break
        else:
            last_index = index

    if debug and score == 0:
        print("INCORRECT", score)
        print("GROUND TRUTH", ground_truth)
        print("SOLUTION", llm_answer)
    return score
