# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL declarations turn source-shaped rows into tasks with the right graders."""

import json
from copy import deepcopy
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import EVIDENCE_PATH
from taskcompendium.grader import grader_config
from taskcompendium.models import (
    ConversationTrace,
    GradingAttempt,
    NoGrader,
    ScriptGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from taskcompendium.pipeline.sources import SourceShard, staged_raw_file_rows
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.task_grading import grade_task

from experiments.post_training.task_curation.datasets.skyrl import code, ifeval, math, mcq, preference
from experiments.post_training.task_curation.pipeline import CurationRecipe, source_files
from experiments.post_training.task_curation.tests.conversion import (
    convert_row,
    converted_task,
    fixture_context,
)

RECIPES = {
    source.name: cast(CurationRecipe, source.config)
    for module in (math, code, ifeval, mcq, preference)
    for source in module.sources()
}

SUM_TESTS = {"inputs": ["1 2\n"], "outputs": ["3\n"]}
SUM_SOLUTION = "a, b = map(int, input().split())\nprint(a + b)"
HH_ROW = {
    "chosen": "\n\nHuman: How do I boil an egg?\n\nAssistant: Simmer it for nine minutes.",
    "rejected": "\n\nHuman: How do I boil an egg?\n\nAssistant: I don't know.",
}


def kto_row(component: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": "Name a primary color."}],
        "completion": [{"role": "assistant", "content": "Blue."}],
        "label": True,
        "kto_component_provenance": {
            "component": component,
            "parent_dataset": preference.PARENT_REPO,
            "parent_revision": preference.PARENT_REVISION,
            "parent_file": preference.TRAIN_FILE,
            "parent_rows": [0],
        },
    }


ROWS: dict[str, dict] = {
    "aime24": {"id": 60, "problem": "Find the remainder when $2^{10}$ is divided by $1000$.", "answer": "24"},
    "aime_1983_2024": {
        "ID": "1983-1",
        "Year": "1983",
        "Problem Number": "1",
        "Question": "What is $6 \\cdot 7$?",
        "Answer": "42",
        "Part": "",
    },
    "asdiv": {
        "ID": "nluds-0001",
        "Grade": "1",
        "Source": "http://www.k5learning.com",
        "Body": "Seven red apples and two green apples are in the basket.",
        "Question": "How many apples are in the basket?",
        "Solution-Type": "Addition",
        "Answer": "9 (apples)",
        "Formula": "7+2=9",
    },
    "dapo_math": {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": "What is $3 + 4$? Put the final answer in a box."}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "7", "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": "a1"},
    },
    "deepscaler": {"problem": "Compute $2 + 2$.", "answer": "4", "solution": "$2 + 2 = 4$."},
    "gsm8k": {
        "question": "Natalia sold clips to 48 friends in April and half as many in May. How many clips did she sell?",
        "answer": "In May she sold 48/2 = 24 clips.\nAltogether she sold 48+24 = 72 clips.\n#### 72",
    },
    "hardmath": {
        "question": "Find the limit of $x/(2x+1)$ as $x \\to \\infty$.",
        "solution": "Divide numerator and denominator by $x$.",
        "ground_truths": "\\frac{1}{2}",
    },
    "hendrycks_math": {
        "problem": "What is $2+3$?",
        "level": "Level 1",
        "type": "Algebra",
        "solution": "We have $2+3=\\boxed{5}$.",
    },
    "math500": {
        "problem": "What is $10 - 3$?",
        "solution": "$10 - 3 = \\boxed{7}$.",
        "answer": "7",
        "subject": "Prealgebra",
        "level": 1,
        "unique_id": "test/prealgebra/1.json",
    },
    "numina_math": {
        "source": "olympiads",
        "problem": "Find $x$ if $2x = 10$.",
        "solution": "Dividing by two gives $x = \\boxed{5}$.",
    },
    "rlvr_math": {
        "messages": [{"role": "user", "content": "Question: What is 1+1?\nAnswer: 2\n\nQuestion: What is 2+3?"}],
        "ground_truth": "5",
        "dataset": "MATH",
        "constraint_type": None,
        "constraint": None,
    },
    "svamp": {
        "ID": "chal-777",
        "Body": "There are 290 bananas organized into 2 equal groups.",
        "Question": "How big is each group of bananas?",
        "Equation": "( 290.0 / 2.0 )",
        "Answer": "145",
        "Type": "Common-Division",
    },
    "gpqa": {
        "Question": "Which particle carries no electric charge?",
        "Correct Answer": "Neutron",
        "Incorrect Answer 1": "Proton",
        "Incorrect Answer 2": "Electron",
        "Incorrect Answer 3": "Positron",
    },
    "openscience": {
        "input": (
            "Which gas is most abundant in Earth's atmosphere?\nA: Oxygen\nB: Nitrogen\nC: Argon\nD: Carbon dioxide"
        ),
        "output": "Nitrogen makes up about 78% of the atmosphere. The answer is \\boxed{B}.",
    },
    "apps": {
        "problem_id": 0,
        "question": "Read two integers and print their sum.",
        "solutions": json.dumps(["", SUM_SOLUTION]),
        "input_output": json.dumps(SUM_TESTS),
        "difficulty": "introductory",
        "url": "https://codeforces.com/problemset/problem/1/A",
        "starter_code": "",
    },
    "eurus2_code": {
        "data_source": "taco",
        "prompt": [
            {"role": "system", "content": "Solve the programming task."},
            {"role": "user", "content": "Read two integers and print their sum."},
        ],
        "ability": "code",
        "reward_model": {"ground_truth": json.dumps(SUM_TESTS), "style": "rule"},
        "extra_info": {"split": "train", "index": 0},
    },
    "verifiable_code": {
        "source": "codeforces",
        "task_type": "verifiable_code",
        "in_source_id": "1A",
        "problem_statement": "Read two integers and print their sum.",
        "gold_standard_solution": f"```python\n{SUM_SOLUTION}\n```",
        "verification_info": {
            "language": "python",
            "test_cases": [{"input": "1 2\n", "output": "3\n", "type": "stdin_stdout", "fn_name": None}],
        },
        "metadata": {"difficulty": "easy"},
        "problem_id": "vfc_1",
    },
    "gretel_text_to_sql": {
        "id": 1,
        "sql_complexity": "basic SQL",
        "sql_task_type": "analytics and reporting",
        "sql_prompt": "List every value in t, including duplicates.",
        "sql_context": "CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (1), (1), (2);",
        "sql": "SELECT v FROM t",
        "sql_explanation": "Return each stored value.",
    },
    "nemotron_if": {
        "input": [{"role": "user", "content": "Write a haiku about rain. Your entire response should be in lowercase."}],
        "args": {"instruction_id_list": ["change_case:english_lowercase"], "instruction_kwargs": [{}]},
        "category": "instruction following",
        "license": "odc-by",
    },
    "rlvr_ifeval": {
        "messages": [{"role": "user", "content": "Write a haiku about rain. Do not use any commas."}],
        "ground_truth": json.dumps({"func_name": "validate_no_commas"}),
        "dataset": "ifeval",
        "constraint_type": "No Commas",
    },
    "hh_harmless_base": HH_ROW,
    "hh_helpful_base": HH_ROW,
    "hh_helpful_online": HH_ROW,
    "hh_helpful_rejection_sampled": HH_ROW,
    **{name: kto_row(component) for name, (component, _metadata) in preference.KTO_COMPONENTS.items()},
}

# A correct and an incorrect final reply for every source graded in process.
ANSWERS = {
    "aime24": ("24", "25"),
    "aime_1983_2024": ("\\boxed{42}", "\\boxed{41}"),
    "asdiv": ("\\boxed{9}", "\\boxed{8}"),
    "dapo_math": ("\\boxed{7}", "\\boxed{6}"),
    "deepscaler": ("\\boxed{4}", "\\boxed{5}"),
    "gsm8k": ("\\boxed{72}", "\\boxed{24}"),
    "hardmath": ("\\boxed{\\frac{1}{2}}", "\\boxed{2}"),
    "hendrycks_math": ("\\boxed{5}", "\\boxed{6}"),
    "math500": ("\\boxed{7}", "\\boxed{13}"),
    "numina_math": ("\\boxed{5}", "\\boxed{10}"),
    "rlvr_math": ("\\boxed{5}", "\\boxed{2}"),
    "svamp": ("145", "580"),
    "openscience": ("B", "A"),
}
CODE_SOURCES = {"apps", "eurus2_code", "verifiable_code", "gretel_text_to_sql"}
PREFERENCE_SOURCES = {name for name in ROWS if name.startswith(("hh_", "kto_"))}


def grade_reply(task: TaskSpec, reply: str):
    return grade_task(
        task,
        GradingAttempt(ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=reply)))),
    )


def verifier_file(task: TaskSpec, path: str) -> bytes:
    return resource_bytes(next(resource for resource in task.resources.verifier if resource.path == path))


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_in_process_grader_scores_the_known_answer(name):
    task = converted_task(RECIPES[name], ROWS[name])
    correct, wrong = ANSWERS[name]
    assert isinstance(task.grader, VerifyitGrader) and task.grader.environment is None
    assert grade_reply(task, correct).reward == 1.0
    assert grade_reply(task, wrong).reward == 0.0


def test_gpqa_shuffles_choices_and_keys_the_correct_option():
    task = converted_task(RECIPES["gpqa"], ROWS["gpqa"])
    prompt = task.context.events[-1].content
    letter = next(line[0] for line in prompt.splitlines() if line.endswith(". Neutron"))
    wrong = next(line[0] for line in prompt.splitlines() if line.endswith(". Proton"))
    assert grade_reply(task, letter).reward == 1.0
    assert grade_reply(task, wrong).reward == 0.0


@pytest.mark.parametrize("name", sorted(CODE_SOURCES))
def test_code_task_golden_control_replays_the_known_solution(name):
    task = converted_task(RECIPES[name], ROWS[name])
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert grader.environment == fixture_context(RECIPES[name]).grader_environment
    final_prompt = task.context.events[-1]
    assert isinstance(final_prompt, TextMessage) and final_prompt.role == "user"
    golden = code.reference_solution(task)
    if name == "eurus2_code":
        assert golden is None
        return
    assert golden is not None
    expected = {
        "apps": f"```python\n{SUM_SOLUTION}\n```",
        "verifiable_code": f"```python\n{SUM_SOLUTION}\n```",
        "gretel_text_to_sql": "<solution>SELECT v FROM t</solution>",
    }
    assert golden.event == TextMessage(role="assistant", content=expected[name])


def test_eurus2_code_selects_code_rows_and_keeps_every_prompt_message():
    pipeline = RECIPES["eurus2_code"]
    assert pipeline.source.select is not None
    assert not pipeline.source.select({**ROWS["eurus2_code"], "ability": "math"}, fixture_context(pipeline))
    task = converted_task(pipeline, ROWS["eurus2_code"])
    assert [event.role for event in task.context.events] == ["system", "user"]
    assert grader_config(task)["test_cases"] == [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]


@pytest.mark.parametrize(
    ("name", "constraints"),
    [
        ("nemotron_if", [{"func_name": "validate_lowercase"}]),
        ("rlvr_ifeval", [{"func_name": "validate_no_commas"}]),
    ],
)
def test_ifeval_task_gives_the_scorer_skyrl_constraints(name, constraints):
    task = converted_task(RECIPES[name], ROWS[name])
    assert isinstance(task.grader, ScriptGrader)
    assert task.grader.environment == fixture_context(RECIPES[name]).grader_environment
    assert grader_config(task)["constraints"] == constraints


def test_nemotron_repeat_prompt_constraint_receives_the_instruction():
    row = {
        **ROWS["nemotron_if"],
        "args": {
            "instruction_id_list": ["combination:repeat_prompt"],
            "instruction_kwargs": [{"prompt_to_repeat": "Write a haiku about rain."}],
        },
    }
    task = converted_task(RECIPES["nemotron_if"], row)
    (constraint,) = grader_config(task)["constraints"]
    assert constraint == {"func_name": "validate_repeat_prompt", "original_prompt": row["input"][0]["content"]}


@pytest.mark.parametrize("name", sorted(PREFERENCE_SOURCES))
def test_preference_task_hides_candidates_and_has_no_grader(name):
    task = converted_task(RECIPES[name], ROWS[name])
    assert isinstance(task.grader, NoGrader)
    final = task.context.events[-1]
    assert isinstance(final, TextMessage) and final.role == "user"
    contract = task.grader.contract
    if name.startswith("hh_"):
        assert contract["kind"] == "pairwise"
        assert [message["content"] for message in contract["chosen"] + contract["rejected"]] == [
            " Simmer it for nine minutes.",
            " I don't know.",
        ]
        return
    assert (contract["kind"], contract["preferred"]) == ("binary", True)
    evidence = json.loads(verifier_file(task, EVIDENCE_PATH))
    assert evidence["kto_component_provenance"]["component"] == preference.KTO_COMPONENTS[name][0]


REJECTIONS = [
    *(
        ("apps", {"input_output": encoded}, ImportFailureKind.SOURCE_DEFECT, "invalid_test_contract")
        for encoded in ("", "{", "null", "[]", '{"inputs": "1 2", "outputs": ["3"]}')
    ),
    *(
        ("apps", {"solutions": solutions}, ImportFailureKind.SOURCE_DEFECT, "invalid_solutions")
        for solutions in ("[", '{"solution": "print(1)"}')
    ),
    (
        "verifiable_code",
        {
            "verification_info": {
                "language": "cpp",
                "test_cases": [{"input": "1", "output": "1", "type": "stdin_stdout"}],
            }
        },
        ImportFailureKind.UNSUPPORTED,
        "unsupported_test_cases",
    ),
    (
        "verifiable_code",
        {"metadata": {"problem_url": "https://practice.geeksforgeeks.org/problems/destructive-year/1"}},
        ImportFailureKind.SOURCE_DEFECT,
        "function_template_with_example_tests",
    ),
    (
        "eurus2_code",
        {"reward_model": {"ground_truth": json.dumps({"inputs": ["1 2"], "outputs": ["3"], "fn_name": "add"})}},
        ImportFailureKind.SOURCE_DEFECT,
        "invalid_test_cases",
    ),
    *(
        ("gretel_text_to_sql", changes, ImportFailureKind.UNSUPPORTED, "unsupported_sql_context")
        for changes in (
            {"sql_context": "CREATE TABLE t (v INTEGER);"},
            {"sql_context": f"{ROWS['gretel_text_to_sql']['sql_context']} CREATE VIEW w AS SELECT v FROM t;"},
            {"sql": "SELECT missing FROM t"},
        )
    ),
    (
        "numina_math",
        {"problem": "Prove that $x^2 + 1 > x$ for all real $x$.", "solution": "Hence $\\boxed{x^2 + 1 > x}$."},
        ImportFailureKind.UNSUPPORTED,
        "unsupported_proof_contract",
    ),
    ("hendrycks_math", {"solution": "The answer is 5."}, ImportFailureKind.UNSUPPORTED, "missing_final_answer"),
    ("gpqa", {"Incorrect Answer 3": " Neutron "}, ImportFailureKind.SOURCE_DEFECT, "duplicate_options"),
    (
        "nemotron_if",
        {"args": {"instruction_id_list": ["unknown:instruction"], "instruction_kwargs": [{}]}},
        ImportFailureKind.UNSUPPORTED,
        "unsupported_ifeval_constraint",
    ),
    (
        "hh_helpful_base",
        {"rejected": "\n\nHuman: How do I fry an egg?\n\nAssistant: Use butter."},
        ImportFailureKind.SOURCE_DEFECT,
        "preference_prompt_conflict",
    ),
    ("kto_component_capybara", {"label": "true"}, ImportFailureKind.SOURCE_DEFECT, "invalid_binary_preference"),
]


@pytest.mark.parametrize(("name", "changes", "kind", "reason"), REJECTIONS)
def test_skyrl_converter_rejects_unusable_rows(name, changes, kind, reason):
    result = convert_row(RECIPES[name], {**ROWS[name], **changes})
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (kind, reason)


def test_apps_golden_skips_python2_solutions_for_the_first_python3_one():
    python2 = "a, b = map(int, raw_input().split())\nprint a + b"
    row = {**ROWS["apps"], "solutions": json.dumps([python2, SUM_SOLUTION])}
    golden = code.reference_solution(converted_task(RECIPES["apps"], row))
    assert golden is not None
    assert golden.event == TextMessage(role="assistant", content=f"```python\n{SUM_SOLUTION}\n```")


def test_apps_accepts_alternative_expected_outputs():
    row = {**ROWS["apps"], "input_output": json.dumps({"inputs": [""], "outputs": [["a", "b"]]})}
    assert isinstance(converted_task(RECIPES["apps"], row).grader, ScriptGrader)


def test_asdiv_reader_yields_problem_fields(tmp_path):
    path = tmp_path / "ASDiv.xml"
    path.write_text(
        '<Machine-Reading-Corpus-File><ProblemSet><Problem ID="nluds-0001" Grade="1">'
        "<Body>Seven apples.</Body><Question>How many?</Question><Answer>7 (apples)</Answer>"
        "</Problem></ProblemSet></Machine-Reading-Corpus-File>"
    )
    (row,) = math.asdiv_rows(StoragePath(str(path)), fixture_context(RECIPES["asdiv"]))
    assert row == {
        "ID": "nluds-0001",
        "Grade": "1",
        "Body": "Seven apples.",
        "Question": "How many?",
        "Answer": "7 (apples)",
    }


def pair(component: str, request: str, system: str = "Keep full context") -> dict:
    history = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier response"},
        {"role": "user", "content": request},
    ]
    return {
        "dataset": component,
        "chosen": [*history, {"role": "assistant", "content": "Chosen"}],
        "rejected": [*history, {"role": "assistant", "content": "Rejected"}],
    }


def observations(parent: list[dict]) -> list[dict]:
    return [
        {"prompt": deepcopy(row[field][:-1]), "completion": deepcopy(row[field][-1:]), "label": label}
        for row in parent
        for field, label in (("chosen", True), ("rejected", False))
    ]


def staged_component_rows(tmp_path, parent: list[dict], kto: list[dict], name: str):
    (tmp_path / "kto/data").mkdir(parents=True)
    (tmp_path / "parent/data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(parent), tmp_path / "parent" / preference.TRAIN_FILE)
    pq.write_table(pa.Table.from_pylist(kto), tmp_path / "kto" / preference.TRAIN_FILE)
    return staged_raw_file_rows(
        str(tmp_path / "kto"),
        SourceShard(preference.TRAIN_FILE, 0, 1, None),
        source_files(RECIPES[name].source),
        fixture_context(RECIPES[name], {preference.PARENT_INPUT: StoragePath(str(tmp_path / "parent"))}),
    )


COMPONENTS = [component for component, _metadata in preference.KTO_COMPONENTS.values()]


def test_kto_component_rows_keep_history_labels_order_and_duplicates(tmp_path):
    parent = [
        pair(COMPONENTS[0], "Shared final request", "First system"),
        pair(COMPONENTS[1], "Shared final request", "Other system"),
    ]
    parent.append(deepcopy(parent[0]))
    kto = observations(parent)[::-1]
    selected = list(staged_component_rows(tmp_path, parent, kto, "kto_component_capybara"))
    assert [row["index"] for row in selected] == [0, 1, 4, 5]
    assert [row["data"]["label"] for row in selected] == [False, True, False, True]
    for row in selected:
        assert row["data"]["kto_component_provenance"]["parent_rows"] == [0, 2]
        task = converted_task(RECIPES["kto_component_capybara"], row["data"])
        first, *_ = task.context.events
        assert isinstance(first, TextMessage) and first.content == "First system"
        assert len(task.context.events) == 4


@pytest.mark.parametrize(
    "failure",
    [
        "unmatched",
        "wrong_label",
        "missing_observation",
        "cross_component_ambiguity",
        "malformed_label",
        "moved_boundary",
    ],
)
def test_kto_join_failure_releases_no_component_row(tmp_path, failure):
    parent = [pair(COMPONENTS[0], "Valid request"), pair(COMPONENTS[1], "Other request")]
    kto = observations(parent)
    if failure == "unmatched":
        kto[-1]["prompt"][-1]["content"] = "Not in parent"
    elif failure == "wrong_label":
        kto[-1]["label"] = True
    elif failure == "missing_observation":
        kto.pop()
    elif failure == "cross_component_ambiguity":
        parent[1] = {**deepcopy(parent[0]), "dataset": COMPONENTS[1]}
        kto = observations(parent)
    elif failure == "moved_boundary":
        kto[-1]["completion"] = [kto[-1]["prompt"].pop(), *kto[-1]["completion"]]
    else:
        for row in kto:
            row["label"] = str(row["label"])
    rows = staged_component_rows(tmp_path, parent, kto, "kto_component_capybara")
    with pytest.raises(preference.KtoComponentError) as error:
        next(rows)
    expected = ImportFailureKind.SOURCE_DEFECT if failure == "malformed_label" else ImportFailureKind.UNSUPPORTED
    assert error.value.kind == expected
    assert "Valid request" not in str(error.value)
