import json


from .templates.ZEBRA_GRID import ZEBRA_GRID
from .templates.MCQA import MCQA
from .templates.OEQA import OEQA, OEQA_DIRECT


def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string


def apply_mc_template(item):
    question, choices = item["question"], item["choices"]
    prompt_str = MCQA[:]
    prompt_str = prompt_str.replace("{question}", question)
    prompt_str = prompt_str.replace("{choices}", generate_choice_string(choices))
    return prompt_str


def apply_oeqa_template(item, question_key="question", cot=True):
    question = item[question_key]
    if cot:
        prompt_str = OEQA[:]
    else:
        prompt_str = OEQA_DIRECT[:]
    prompt_str = prompt_str.replace("{question}", question)
    return prompt_str


def apply_lgp_grid_template(item):
    prompt_str = ZEBRA_GRID[:]
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i+1}"] = {columns[j]: "___" for j in range(1, len(columns))}
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str
