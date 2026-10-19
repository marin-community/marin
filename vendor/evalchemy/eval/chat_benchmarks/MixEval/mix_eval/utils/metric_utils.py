import re
import random

random.seed(42)
import numpy as np
import ast

from mix_eval.utils.judge_freeform_parser import (
    ChatGPTJudgeCloseendFreeform,
    ClaudeJudgeCloseendFreeform,
    GeminiJudgeCloseendFreeform,
)
from mix_eval.utils.judge_multichoice_parser import (
    ChatGPTJudgeCloseendMultichoice,
    ClaudeJudgeCloseendMultichoice,
    GeminiJudgeCloseendMultichoice,
)
from mix_eval.utils.common_utils import (
    extract_basemodel_response_3e,
    extract_basemodel_response_2e,
)


def find_all_sub(s, sub):
    """Find all occurrences of a substring in a string using regular expressions."""
    pattern = re.escape(sub)  # Escape the substring to handle special regex characters
    matches = [match.start() for match in re.finditer(pattern, s)]
    return matches


def parse_multi_choice_response_rule(args, response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    GET_FIRST = True  # get the first match or last

    if hasattr(args, "model_type"):
        if args.model_type == "BaseModel":
            response = extract_basemodel_response_2e(response)
        elif args.model_type == "ChatModel":
            pass
        elif args.model_type == "APIModelBase":
            pass
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []
    patterns = [
        lambda choice: f" {choice} ",  # e.g., A B C D
        lambda choice: f" {choice}\n",  # e.g., A\n B\n C\n D\n
        lambda choice: f"\n{choice} ",  # e.g., \nA B C D
        lambda choice: f"\n{choice}\n",  # e.g., \nA\n B\n C\n D\n
        lambda choice: f" {choice}. ",  # e.g., A. B. C. D.
        lambda choice: f" {choice}.\n",  # e.g., A.\n B.\n C.\n D.\n
        lambda choice: f"\n{choice}. ",  # e.g., \nA. \nB. \nC. \nD.
        lambda choice: f"\n{choice}.\n",  # e.g., \nA.\n \nB.\n \nC.\n \nD.\n
        lambda choice: f"({choice})",  # e.g., (A) (B) (C) (D)
        lambda choice: f"**{choice} ",  # e.g., **A **B **C **D
        lambda choice: f" {choice}**",  # e.g., A** B** C** D**
        lambda choice: f"**{choice}. ",  # e.g., **A. **B. **C. **D.
        lambda choice: f" {choice}.**",  # e.g., A.** B.** C.** D.**
    ]

    for choice in all_choices:
        for pattern in patterns:
            ids = find_all_sub(response, pattern(choice))
            for id in ids:
                candidates.append((choice, id))

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            ids = find_all_sub(response.lower(), ans.lower())
            for id in ids:
                candidates.append((index, id))

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = [candidate[1] for candidate in candidates]
        if GET_FIRST:
            pred_index = candidates[np.argmin(start_indexes)][0]
        else:
            pred_index = candidates[np.argmax(start_indexes)][0]
    else:  # if only one candidate, use it.
        pred_index = candidates[0][0]
    return pred_index


def get_option_from_judge(judge_response):
    """
    Get the option from the judge response.
    """
    double_brackets_pattern = re.compile("\[\[([A-Z])\]\]")
    single_brackets_pattern = re.compile("\[([A-Z])\]")

    match = re.search(double_brackets_pattern, judge_response)
    if not match:
        match = re.search(single_brackets_pattern, judge_response)

    if match:
        option = match.groups()[0]
    else:
        option = -1

    return str(option)


def is_option(variable):
    return isinstance(variable, str) and len(variable) == 1 and variable.isupper()


def parse_multi_choice_response_model(args, tasks):
    tasks_remained = tasks
    tasks_judged = []
    model_judge = ChatGPTJudgeCloseendMultichoice(args)

    MAX_RETRY_NUM = 10
    for _ in range(MAX_RETRY_NUM):
        tasks_judged_p = model_judge.annotate_parallel(tasks_remained)
        # retry those failed cases whose "judge_response" is None or no valid score found inside
        tasks_remained = []
        for task in tasks_judged_p:
            if task["judge_response"] is not None and is_option(get_option_from_judge(task["judge_response"])):
                task["judge_option"] = get_option_from_judge(task["judge_response"])
                tasks_judged.append(task)
            else:
                tasks_remained.append(task)

        if len(tasks_remained) == 0:
            break
        else:
            print(f"Still {len(tasks_remained)} tasks remained to be judged. Retry...")

    if len(tasks_remained) > 0:
        print(
            f"Max retry number {MAX_RETRY_NUM} reached, while some tasks are still not judged. "
            "Randomly assign the options for them.\n"
            "This is expected during parsing. "
            "The main cause may be that the evaluated model's response does not contain a valid answer."
        )
        # randomly assign the option for each entry
        for task in tasks_remained:
            options = task["options"]
            option_letters = [chr(ord("A") + i) for i in range(len(options))]
            task["judge_option"] = random.choice(option_letters)
            tasks_judged.append(task)

    assert len(tasks_judged) == len(tasks), "The number of tasks judged is not equal to the number of input tasks."

    return tasks_judged


def check_is_number(string):
    """
    Check if the given string a number.
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbers
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def parse_freeform_response_rule(args, response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    if hasattr(args, "model_type"):
        if args.model_type == "BaseModel":
            response = extract_basemodel_response_3e(response)
        elif args.model_type == "ChatModel":
            pass
        elif args.model_type == "APIModelBase":
            pass
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = (
                None  # the shortest response that may contain the answer (tail part of the response)
            )
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_score_from_judge(judge_response):
    """
    Get the score from the judge response.
    """
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

    match = re.search(one_score_pattern, judge_response)
    if not match:
        match = re.search(one_score_pattern_backup, judge_response)

    if match:
        rating = ast.literal_eval(match.groups()[0])
    else:
        rating = -1

    return float(rating)


def parse_freeform_response_model(args, tasks):
    tasks_remained = tasks
    tasks_judged = []
    model_judge = ChatGPTJudgeCloseendFreeform(args)

    MAX_RETRY_NUM = 10
    for _ in range(MAX_RETRY_NUM):
        tasks_judged_p = model_judge.annotate_parallel(tasks_remained)
        # retry those failed cases whose "judge_response" is None or no valid score found inside
        tasks_remained = []
        for task in tasks_judged_p:
            if task["judge_response"] is not None and 0 <= get_score_from_judge(task["judge_response"]) <= 1:
                task["judge_score"] = get_score_from_judge(task["judge_response"])
                tasks_judged.append(task)
            else:
                tasks_remained.append(task)

        if len(tasks_remained) == 0:
            break
        else:
            print(f"Still {len(tasks_remained)} tasks remained to be judged. Retry...")

    if len(tasks_remained) > 0:
        print(
            f"Max retry number {MAX_RETRY_NUM} reached, "
            "while some tasks are still not judged. "
            "Randomly assign the scores for them.\n"
            "This is expected during parsing. "
            "The main cause may be that the evaluated model's response does not contain a valid answer."
        )
        # randomly assign the score for each entry
        for task in tasks_remained:
            task["judge_score"] = round(random.random(), 1)
            tasks_judged.append(task)

    assert len(tasks_judged) == len(tasks), "The number of tasks judged is not equal to the number of input tasks."

    return tasks_judged


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_freeform_rule(gold_i, pred_i):
    """
    Evaluate an open question instance
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def eval_freeform_model(args, tasks):
    return parse_freeform_response_model(args, tasks)


if __name__ == "__main__":
    response = "Sandy will have more tokens than any sibling by 1/8 million."
    preds = parse_freeform_response_rule(response)
    print(preds)
