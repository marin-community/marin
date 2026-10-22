from eval.contracts.eval_spec import ReducerSpec, SourceRef
"""CodeForces dataset and published prompt."""
import logging

from datasets import load_dataset

from eval.sources.common import text_specs
from eval.sources.programs import difficulty, exact_tests
from eval.task import BaseBenchmark

def make_html_problem(problem):
    test_cases = problem['examples']
    title = problem['title']
    html_output = '<html><body>'
    html_output += f'<h1>{title}</h1>'
    html_output += f"<div>Time limit per test: {problem['time_limit']} s</div>"
    html_output += f'<h2>Description</h2>'
    html_output += f"<div>{problem['description']}</div>"
    html_output += f'<h2>Input Format</h2>'
    html_output += f"<div>{problem['input_format']}</div>"
    html_output += f'<h2>Output Format</h2>'
    html_output += f"<div>{problem['output_format']}</div>"
    if len(test_cases) > 0:
        for tc in test_cases:
            html_output += f'<h2>Example</h2>'
            html_output += f'<h3>Input</h3>'
            html_output += f"<div>{tc['input']}</div>"
            html_output += f'<h3>Output</h3>'
            html_output += f"<div>{tc['output']}</div>"
    if problem['interaction_format']:
        html_output += f'<h2>Interaction</h2>'
        html_output += f"<div>{problem['interaction_format']}</div>"
    if problem['note']:
        html_output += f'<h2>Note</h2>'
        html_output += f"<div>{problem['note']}</div>"
    if problem['editorial']:
        html_output += f'<h2>Editorial</h2>'
        html_output += f"<div>{problem['editorial']}</div>"
    html_output += '</body></html>'
    return html_output

INSTRUCTION = 'You are a coding expert. Given a competition-level coding problem, you need to write a Python program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```. The code should take stdin as input and print the output. Your program should be a Python function generated from the given prompt. Simply call the function after the definition.'


class CodeForcesBenchmark(BaseBenchmark):
    SOURCE = SourceRef("open-r1/codeforces")
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err", repeat_ddof=1, group_suffix="_avg")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"

    def __init__(self, debug=False, seed=(0, 1234, 1234, 1234), max_tokens=32768,
                 logger: logging.Logger | None = None, system_instruction=None):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 3
        self.filter_interaction_questions = True

    def load_questions(self):
        rows = load_dataset("open-r1/codeforces")["test"]
        if self.debug:
            rows = list(rows)[:10]
        return ({**row, "difficulty": difficulty(row["rating"])} for row in rows
                if not self.filter_interaction_questions or not row["interaction_format"])

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda row: INSTRUCTION + "\n\n" + make_html_problem(row),
            target=lambda row: row["official_tests"], grader="program",
            reference=lambda row: {"tests": exact_tests(row["official_tests"]), "stdin": True,
                                   "timeout": row["time_limit"], "comparison": "stdout_normalized",
                                   "deadline_grace": 60},
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens}, repeats=self.n_repeat,
            tags=lambda row: {"difficulty": row["difficulty"]},
        )
