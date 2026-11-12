import re
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def group_words(words):
    groups = [set()]
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append(set())
        groups[-1].add(word)
    return groups


def connections_process_results_old(ground_truth: str, llm_answer: str, debug=False) -> int:
    # pull out words in bold
    bold_words = re.findall(r"\*\*(.*?)\*\*", llm_answer.replace("\n", ""))

    if not bold_words:
        if debug:
            print("No bold words found for connections")
            print("END OF OUTPUT", llm_answer[-50:])
        return 0

    bold_words = [words.split(",") for words in bold_words]

    ground_truth_groups = group_words(ground_truth.split(","))
    max_score = 0
    for output_groups in list(map(group_words, bold_words)):
        correct_groups = 0
        for ground_truth_group in ground_truth_groups:
            for output_group in output_groups:
                if all([word in output_group for word in ground_truth_group]):
                    correct_groups += 1
                    break

        max_score = max(max_score, correct_groups / len(ground_truth_groups))
    if debug and max_score < 1:
        print("INCORRECT", max_score)
        print("GROUND TRUTH", ground_truth_groups)
        print("SOLUTION", output_groups)
        print("END OF OUTPUT", llm_answer[-50:])
    return max_score


def connections_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    # extract text from <solution></solution> tags
    solution_matches = re.findall(r"<solution>(.*?)<\/solution>", llm_answer)
    if len(solution_matches) == 0:
        solution_matches = re.findall(r"<solution>(.*?)<\/solution>", llm_answer.replace("\n", ""))
    if len(solution_matches) == 0:
        solution_matches = re.findall(r"</solution>(.*?)<\/solution>", llm_answer)
    if len(solution_matches) == 0:
        solution_matches = re.findall(r"</solution>(.*?)<\/solution>", llm_answer.replace("\n", ""))

    ground_truth_words = ground_truth.split(",")

    if len(solution_matches) == 0 and "\\boxed" in llm_answer:
        boxed = last_boxed_only_string(llm_answer)
        no_box = remove_boxed(boxed)
        solution_matches = [no_box.replace("\\text{", "").replace("}", "").replace("\\", "")]

    solution_matches = [match.replace("\n", "") for match in solution_matches]

    if len(solution_matches) == 0:
        if debug:
            print("No solution text found for connections")
            print(llm_answer[-500:])
        return 0
    elif len(solution_matches) > 1:
        if debug:
            print("Multiple solution texts found for connections. Combining starting from last")
        all_words = []
        num_words = len(ground_truth_words)
        for match in solution_matches:
            all_words.extend(match.split(","))
        solution_words = all_words[-num_words:]
    else:
        solution_words = solution_matches[-1].split(",")

    if len(solution_words) != len(ground_truth_words):
        if debug:
            print(
                f"Number of words in solution does not match number of words in ground truth ({len(solution_words)} vs expected {len(ground_truth_words)})"
            )

    llm_groups = group_words(solution_words)
    ground_truth_groups = group_words(ground_truth_words)

    correct_groups = 0
    for llm_group in llm_groups:
        if llm_group in ground_truth_groups:
            correct_groups += 1

    if debug and correct_groups / len(ground_truth_groups) < 1:
        print("INCORRECT", correct_groups / len(ground_truth_groups))
        print("GROUND TRUTH", sorted([sorted(list(group)) for group in ground_truth_groups]))
        print("SOLUTION", sorted([sorted(list(group)) for group in llm_groups]))
        print("END OF OUTPUT", llm_answer[-500:])

    return correct_groups / len(ground_truth_groups)


def get_connections_puzzle_evaluator(release_date: str):
    if release_date < "2024-11-25":
        return connections_process_results_old
    return connections_process_results
