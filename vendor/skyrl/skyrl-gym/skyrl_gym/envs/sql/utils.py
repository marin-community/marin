"""The final-query format for multi-turn SQL tasks."""

import re

THINK_START = "<think>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"


def final_sql(output: str) -> str | None:
    """Return the final SQL query when the conversation obeys the source format."""
    if output.count(SOLUTION_START) != 1:
        return None
    pre_solution, tail = output.split(SOLUTION_START, 1)
    if tail.count(SOLUTION_END) != 1:
        return None
    solution_text, _ = tail.split(SOLUTION_END, 1)
    if re.search(r"</?(think|sql|observation)\b", solution_text, re.I):
        return None
    if not re.search(r"<think>.*?</think>", output, re.S):
        return None
    for match in re.finditer(r"</observation>", pre_solution, re.I):
        if not pre_solution[match.end() :].lstrip().lower().startswith(THINK_START):
            return None
    return solution_text.strip()
