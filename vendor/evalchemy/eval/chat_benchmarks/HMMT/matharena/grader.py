import os
import json
import json5
from loguru import logger
from datetime import datetime
from difflib import SequenceMatcher
import yaml
import re
from datasets import load_dataset

from matharena.api import APIQuery
from matharena.cot_solver import CoTSolver
from matharena.parser import parse_grading, WarningType


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.8  # Allow minor formatting differences


def clean_string_to_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\n(.*?)\n```", r"\1", text, flags=re.DOTALL)
    text = text.replace("`", "")
    return text


def format_grading_scheme(scheme, problem_id):
    formatted_str = ""
    if scheme["problem_idx"] != problem_id:
        raise ValueError(f"Incorrect schema given for problem {problem_id}")
    total_points = 0
    for category in scheme["grading_scheme"]:
        total_points += category["points"]
        formatted_str += f"Category: {category['title']}\n"
        formatted_str += f"Available points: {category['points']}\n"
        formatted_str += f"Description: {category['desc']}\n\n"

    if total_points != scheme["points"]:
        raise ValueError(
            f"Total points in schema for problem {problem_id} totals {total_points}, but should be {scheme['points']}"
        )

    return formatted_str


def run_grader(
    grader_config,
    solver_config_path,
    competition,
    skip_existing=False,
    output_folder="outputs",
    grading_folder="autogrades",
    competition_config_folder="competition_configs",
    autograding_config_path="configs/autograding/config.yaml",
):
    model = grader_config["model"]
    n = grader_config["n"]
    api = grader_config["api"]

    with open(autograding_config_path, "r") as f:
        autograding_config = yaml.safe_load(f)

    with open(f"{competition_config_folder}/{competition}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)

    n_evals = autograding_config["n_evals"]
    date_comp = datetime.strptime(competition_config["date"], "%Y-%m-%d")

    max_tokens = grader_config.get("max_tokens", competition_config["default_max_tokens"])
    temperature = grader_config.get("temperature", competition_config["default_temperature"])
    kwargs = grader_config.copy()
    del kwargs["model"]
    del kwargs["n"]
    del kwargs["api"]
    del kwargs["human_readable_id"]
    if "date" in kwargs:
        date_model = datetime.strptime(kwargs["date"], "%Y-%m-%d")
        if date_model > date_comp:
            logger.warning(f"Model date is after competition date. Model: {model}, Competition: {competition}")
        del kwargs["date"]
    kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature

    logger.info(f"New run, model: {model}, competition: {competition}")

    prompt_template = f"{autograding_config['grading_instruction']}"

    problems = load_dataset(competition_config["dataset_path"], split="train").to_list()

    output_dir = os.path.join(f"{output_folder}/{competition}/", solver_config_path.replace(".yaml", ""))
    autograder_dir = f"{grading_folder}/{competition}/"

    batch_prompts = []
    batch_idx_to_problem_idx = {}
    marking_schemas = {}

    all_messages_per_problem = {i: [] for i in range(len(problems))}
    all_evals_per_problem_per_solution = {i: {} for i in range(len(problems))}

    for i, problem in enumerate(problems):
        problem_id = problem["problem_idx"]
        output_file = os.path.join(output_dir, f"{problem_id}.json")

        if not os.path.exists(output_file):
            raise ValueError(f"Could not find the solutions for {problem_id} in {output_dir}")
        else:
            data_file = json.load(open(output_file))
            problem["anon_id"] = data_file["anonymous_id"]
            messages = data_file["messages"]
            all_evals_per_problem_per_solution[i] = {i: [] for i in range(n_evals)}
            messages = [messages_one for messages_one in messages if len(messages_one[-1]["content"]) > 0]
            all_messages_per_problem[i] = messages

        marking_schema = format_grading_scheme(problem, problem_id)
        marking_schemas[i] = problem["grading_scheme"]

        for j in range(n_evals):
            auto_grading_file = os.path.join(
                autograder_dir, f"{problem_id}/{problem['anon_id']}_{grader_config['model'].split('/')[-1]}-{j}.json"
            )

            if skip_existing and os.path.exists(auto_grading_file):
                data_file = json.load(open(auto_grading_file))
                messages = [messages_one["raw"] for messages_one in data_file]
                all_evals_per_problem_per_solution[i][j] = messages
                if len(all_evals_per_problem_per_solution[i][j]) == n:
                    calculate_grading_results(
                        problem,
                        autograder_dir,
                        all_evals_per_problem_per_solution[i][j],
                        marking_schemas[i],
                        i,
                        j,
                        grader_model_name=grader_config["model"].split("/")[-1],
                    )
                continue
            for _, message in enumerate(messages):
                problem_statement = problem["problem"]
                grading_prompt = prompt_template.format(
                    problem_statement=problem_statement,
                    marking_schema=marking_schema,
                    correct_solution=problem["sample_solution"],
                    example_grading=problem["sample_grading"],
                    solution=message if skip_existing and os.path.exists(auto_grading_file) else message[-1]["content"],
                )
                batch_idx_to_problem_idx[len(batch_prompts)] = (i, j)
                batch_prompts.append((grading_prompt, None))

    logger.info("Collected all queries, now running")

    if len(batch_prompts) == 0:
        return
    api = APIQuery(model=model, api=api, **kwargs)

    cot_solver = CoTSolver(
        querier=api,
    )

    for idx, messages, _ in cot_solver.solve(batch_prompts):
        problem_idx, grader_idx = batch_idx_to_problem_idx[idx]
        problem = problems[problem_idx]
        all_evals_per_problem_per_solution[problem_idx][grader_idx].append(messages[-1]["content"])
        # check if the whole problem is finished
        if len(all_evals_per_problem_per_solution[problem_idx][grader_idx]) == n:
            calculate_grading_results(
                problem,
                autograder_dir,
                all_evals_per_problem_per_solution[problem_idx][grader_idx],
                marking_schemas[problem_idx],
                problem_idx,
                grader_idx,
                grader_model_name=grader_config["model"].split("/")[-1],
            )


def calculate_grading_results(
    problem, output_dir, gradings_per_solution, marking_schema, problem_idx, grader_idx, grader_model_name
):
    problem_id = problem["problem_idx"]
    anon_id = problem["anon_id"]

    output_file = os.path.join(output_dir, f"{problem_id}/{anon_id}_{grader_model_name}-{grader_idx}.json")
    os.makedirs(f"{output_dir}/{problem_id}", exist_ok=True)

    outputs = [{} for _ in gradings_per_solution]

    for i, message in enumerate(gradings_per_solution):
        outputs[i]["raw"] = message
        warning = WarningType.NONE
        parsed_grading = {}
        try:
            try:
                parsed_grading = json.loads(clean_string_to_json(message), strict=False)
            except json.JSONDecodeError:
                parsed_grading = json5.loads(clean_string_to_json(message), strict=False)
            except Exception:
                parsed_grading = parse_grading(message)
            if not "points" in parsed_grading:
                logger.error(f"Final points were not generated for grader {grader_idx} of {problem_idx}:\n {message}")
                warning = max(warning, WarningType.MAJOR)
            if not "details" in parsed_grading:
                if not "scheme" in parsed_grading:
                    logger.error(f"Not scoring details found for grader {grader_idx} of {problem_idx}:\n {message}")
                    warning = max(warning, WarningType.MAJOR)
                else:
                    parsed_grading["details"] = parsed_grading["scheme"]
            elif len(parsed_grading["details"]) != len(marking_schema):
                logger.error(f"Mismatch between marking schema lengths")
                warning = max(warning, WarningType.MAJOR)
            else:
                if anon_id == "ecddbb":
                    breakpoint()
                final_points = 0
                for given, expected in zip(parsed_grading["details"], marking_schema):
                    if not similar(given["title"], expected["title"]):
                        logger.error(f"Title mismatch: '{given['title']}' vs '{expected['title']}'")
                        warning = max(warning, WarningType.MAJOR)
                    elif given["points"] > expected["points"]:
                        logger.warning(
                            f"Warning: Given points ({given['points']}) exceed max allowed ({expected['points']}) for category '{given['title']}'"
                        )
                        warning = max(warning, WarningType.MINOR)
                        given["points"] = expected["points"]
                    elif given["points"] < 0:
                        logger.warning(
                            f"Warning: Given points ({given['points']}) are negative for category '{given['title']}'"
                        )
                        warning = max(warning, WarningType.MINOR)
                        given["points"] = 0

                    given["title"] = expected["title"]
                    final_points += given["points"]
                parsed_grading["points"] = final_points

        except Exception as e:
            logger.error(e)
            warning = max(warning, WarningType.MAJOR)
            parsed_grading = {
                "points": 0,
                "details": [
                    {"title": item["title"], "points": 0, "desc": "The grading could not be parsed."}
                    for item in marking_schema
                ],
            }

        outputs[i]["warning"] = warning.value
        for k in parsed_grading:
            outputs[i][k] = parsed_grading[k]

    with open(output_file, "w") as f:
        json.dump(outputs, f)
