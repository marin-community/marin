import json
from collections import defaultdict
import os
from tabulate import tabulate
from datasets import load_dataset

from .eval_utils import load_model_results, extract_last_complete_json, model_name_replacement

private_solutions = {}


def load_private_solutions():
    global private_solutions
    private_zebra_data = load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")
    for item in private_zebra_data:
        private_solutions[item["id"]] = item["solution"]
    return


def grade_model(model, data):
    global private_solutions

    parsed_results = []  # New list to store parsed results
    for item in data:
        # solution = item["solution"]
        solution = private_solutions[item["id"]]

        # Process the solution
        solution_table = {}
        num_houses = len(solution["rows"])
        columns = solution["header"]
        assert columns[0] == "House"
        solution_table = {}
        this_total_cells = 0
        for i in range(num_houses):
            solution_table[f"House {i+1}"] = {columns[j]: solution["rows"][i][j] for j in range(1, len(columns))}
            this_total_cells += len(columns) - 1

        # Read and Parse the prediction from model output
        prediction_str = item["output"][0]
        prediction_json = extract_last_complete_json(prediction_str)
        if prediction_json is None or "solution" not in prediction_json or prediction_json["solution"] is None:
            # print("-"*100)
            # prediction_str = prediction_str.replace("\n", "")
            # print([prediction_str])
            # json.loads(prediction_str)
            parsed_results.append({**item, "correct_cells": 0, "total_cells": this_total_cells, "solved": False, "answered": False})
            # print(item["id"])
            continue
        reason = prediction_json.get("reasoning", "")
        prediction_table = prediction_json["solution"]

        this_correct_cells = 0  # number in the solution_table
        for house in solution_table:
            for column in solution_table[house]:
                # if prediction_table[house][column] not exist then pass
                if house in prediction_table and column in prediction_table[house]:
                    truth_cell = solution_table[house][column].lower().strip()
                    if prediction_table[house][column] is None or len(prediction_table[house][column]) == 0:
                        continue
                    if type(prediction_table[house][column]) == list:
                        predicted_cell = prediction_table[house][column][0].lower().strip()
                    elif type(prediction_table[house][column]) == str:
                        predicted_cell = prediction_table[house][column].lower().strip()
                    else:
                        raise ValueError(f"Unknown type: {type(prediction_table[house][column])}")
                    if truth_cell.lower().strip() == predicted_cell.lower().strip():
                        this_correct_cells += 1
        parsed_item = item.copy()
        parsed_item["reasoning"] = reason
        parsed_item["correct_cells"] = this_correct_cells
        parsed_item["total_cells"] = this_total_cells
        parsed_item["solved"] = this_correct_cells == this_total_cells
        parsed_item["answered"] = True
        parsed_results.append(parsed_item)
    return parsed_results


def aggregate_model(model, graded):
    answered = [row for row in graded if row["answered"]]
    easy_sizes = {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"}
    easy = [row for row in graded if row["size"] in easy_sizes]
    hard = [row for row in graded if row["size"] not in easy_sizes]
    return {
        "Model": model_name_replacement(model.split("%")[0]),
        "Mode": model.split("%")[1],
        "Puzzle Acc": f"{sum(row['solved'] for row in graded) / len(graded) * 100:.2f}",
        "Cell Acc": f"{sum(row['correct_cells'] for row in graded) / sum(row['total_cells'] for row in graded) * 100:.2f}",
        "No answer": f"{(len(graded) - len(answered)) / len(graded) * 100:.2f}",
        "Easy Puzzle Acc": f"{sum(row['solved'] for row in easy) / len(easy) * 100:.2f}" if easy else "0.00",
        "Hard Puzzle Acc": f"{sum(row['solved'] for row in hard) / len(hard) * 100:.2f}" if hard else "0.00",
        "Total Puzzles": len(graded),
        "Reason Lens": f"{sum(len(row['reasoning']) for row in answered) / len(answered):.2f}",
    }


def eval_model(model, filepath):
    with open(filepath) as stream:
        data = json.load(stream)
    graded = grade_model(model, data)
    return aggregate_model(model, graded), graded


def gen_results(run_name_folders):
    model_results = load_model_results(run_name_folders)

    columns = [
        "Model",
        "Mode",
        "Puzzle Acc",
        "Easy Puzzle Acc",
        "Hard Puzzle Acc",
        "Cell Acc",
        "No answer",
        "Total Puzzles",
        "Reason Lens",
    ]
    rows = []
    for model_name, filepath in model_results.items():
        result, parsed_results = eval_model(model_name, filepath)
        # Save the parsed_results to the same filepath with a new prefix
        parsed_results_filepath = filepath.replace("result_dirs", "result_dirs_parsed")
        # Create folders if they don't exist
        os.makedirs(os.path.dirname(parsed_results_filepath), exist_ok=True)
        # Save parsed results
        with open(parsed_results_filepath, "w") as f:
            json.dump(parsed_results, f, indent=2)
        rows.append(result)

    # sort the rows by puzzle accuracy
    rows = sorted(rows, key=lambda x: -float(x["Puzzle Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to json file
    with open("result_dirs/zebra-grid.summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # write to markdown file
    with open(f"result_dirs/zebra-grid.summary.md", "w") as f:
        f.write(tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center"))


if __name__ == "__main__":
    run_name_folders = {
        "greedy": "result_dirs/zebra-grid",
        "sampling": "result_dirs/zebra-grid/sampling",
    }
    load_private_solutions()
    gen_results(run_name_folders)
