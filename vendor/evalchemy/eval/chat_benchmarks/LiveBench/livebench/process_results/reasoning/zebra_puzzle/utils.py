import re
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def zebra_puzzle_process_results_old(ground_truth: str, llm_answer: str, debug=False) -> int:
    # Mapping of numbers to words for 1 to 9
    number_to_word = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    # Pull out words in bold
    bold_words = re.findall(r"\*\*\*(\w+)\*\*\*", llm_answer)

    score = 0

    # Remove any trailing punctuation from the last bold word if exists
    if bold_words:
        if (
            bold_words[-1].lower() == ground_truth.lower()
            or (bold_words[-1] in number_to_word and number_to_word[bold_words[-1]].lower() == ground_truth.lower())
            or bold_words[-1].lower() + " movies" == ground_truth.lower()
        ):
            score = 1
    else:
        # Split the text into words and remove punctuation.
        words = re.findall(r"\b\w+\b", llm_answer)
        last_word = words[-1] if words else ""
        # Check if the last bold word is a number and matches the word representation of the ground_truth
        if (
            last_word.lower() == ground_truth.lower()
            or (last_word in number_to_word and number_to_word[last_word].lower() == ground_truth.lower())
            or last_word.lower() + " movies" == ground_truth.lower()
        ):
            score = 1

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth.lower())
        if bold_words:
            print("LAST BOLD WORD", bold_words[-1].lower())
        else:
            print("LAST WORD", last_word.lower())
        print("END OF OUTPUT", llm_answer[-50:])
    return score


def zebra_puzzle_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    # Mapping of numbers to words for 1 to 9
    word_to_num = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    ground_truth = ground_truth.split(",")

    # extract text from <solution></solution> tags
    solution_matches = re.findall(r"<solution>(.*?)</solution>", llm_answer)

    if len(solution_matches) == 0:
        solution_matches = re.findall(r"</solution>(.*?)</solution>", llm_answer)

    allow_boxed = True
    if len(solution_matches) == 0 and allow_boxed:
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            boxed_removed = remove_boxed(last_boxed)
            boxed_removed = boxed_removed.replace("\\text{", "").replace("}", "").replace("\\", "")
            solution_matches.append(boxed_removed)

    if len(solution_matches) == 0:
        last_line = llm_answer.strip().split("\n")[-1]
        if last_line.count(",") == len(ground_truth) - 1:
            solution_matches.append(last_line)

    if len(solution_matches) == 0:
        if debug:
            print("No solution text found for zebra puzzle")
            print("GROUND TRUTH", ground_truth)
            print("END OF OUTPUT", llm_answer[-100:])
        return 0

    if len(solution_matches) > 1:
        if debug:
            print("WARNING: Multiple solution texts found for zebra puzzle, combining starting from last")
        all_solution_text = []
        for match in solution_matches:
            all_solution_text += match.split(",")
        # get last len(ground_truth) elements
        solution_text = all_solution_text[-len(ground_truth) :]
    else:
        solution_text = solution_matches[-1].split(",")

    if debug and len(solution_text) < len(ground_truth):
        print(
            f"WARNING: LLM does not have an answer for all zebra puzzle questions (expected {len(ground_truth)}, got {len(solution_text)})"
        )

    num_correct = 0
    total = len(ground_truth)
    for i in range(total):
        gt_word = ground_truth[i].strip().lower().replace("-", " ")
        if i >= len(solution_text):
            continue
        llm_word = solution_text[i].strip().lower().replace("-", " ").replace("position", "")
        if llm_word in word_to_num:
            # llm_word = word_to_num[llm_word]
            if debug:
                print("WARNING: LLM answer contains numbers in word form")

        if i < len(solution_text) and (gt_word == llm_word or gt_word in llm_word):
            num_correct += 1
    if debug and num_correct < total:
        print("INCORRECT", num_correct / total)
        print("GROUND TRUTH", ground_truth)
        print("SOLUTION", solution_text)
        print("END OF OUTPUT", llm_answer[-50:])
    return ((num_correct == total) + num_correct / total) / 2


def get_zebra_puzzle_evaluator(release_date: str):
    if release_date < "2024-11-25":
        return zebra_puzzle_process_results_old
    return zebra_puzzle_process_results
