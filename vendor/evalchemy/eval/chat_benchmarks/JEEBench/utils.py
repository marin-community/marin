def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


QUES_TYPES = ["MCQ", "MCQ(multiple)", "Integer", "Numeric"]


def compute_score(gold, resp, question_type):
    """
    from https://github.com/dair-iitd/jeebench/blob/main/compute_metrics.py
    """
    if resp is None:
        return 0
    assert question_type in QUES_TYPES
    if question_type == "MCQ(multiple)":
        gold = set([c for c in ["A", "B", "C", "D"] if c in gold])
        resp = set([c for c in ["A", "B", "C", "D"] if c in resp])
        if resp == gold:
            return 1.0
        else:
            if len(resp - gold) == 0:
                return 0.25 * len(resp)
            return 0.0  # If response contains something not in the gold set, give 0
    elif question_type == "MCQ":
        gold = set([c for c in ["A", "B", "C", "D"] if c in gold])
        resp = set([c for c in ["A", "B", "C", "D"] if c in resp])
        return int(gold == resp)
    else:
        if resp == "None":
            return 0.0
        g = float(gold)
        try:
            r = float(resp)
            return int(abs(g - r) <= 0.01)
        except:  # failed to parse resp
            return 0
