import json
from collections import defaultdict
import os
from tabulate import tabulate
import re
import sys
from .eval_utils import (
    load_model_results,
    extract_values_from_json,
    extract_first_complete_json,
    model_specific_extraction,
    model_name_replacement,
)


def grade_model(model, data):
    global private_solutions

    parsed_results = []
    for item in data:
        # Read and Parse the prediction from model output
        prediction_str = item["output"][0]
        prediction_json = extract_first_complete_json(prediction_str)
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
            # print("-")
        if prediction_json is None or "answer" not in prediction_json:
            try_extracted_answer = model_specific_extraction(model, prediction_str)
            if try_extracted_answer:
                # print(f"Extracted answer from model: {try_extracted_answer}")
                prediction_json["answer"] = try_extracted_answer
            else:
                parsed_results.append({**item, "matched": False, "answered": False})
                if False and "3.1" in model:  # used for debugging the format of the output
                    print("--------------------------")
                    print(f"No answer for {item['id']}")
                    print(prediction_str)
                    print(prediction_json)
                    print(correct_answer)
                continue
        reason = prediction_json.get("reasoning", "")

        # We use string to compare the answers, so we need to strip the quotes
        model_answer = str(prediction_json["answer"]).strip("'\"").replace("\n", "\\n")
        correct_answer = str(item["answer"]).strip("'\"")

        correct = False
        if model_answer and correct_answer:
            if model_answer == correct_answer:
                correct = True
                # To debug the correct examples
                if False and "SimPO" in model:
                    if raw_model_answer != correct_answer:
                        print(f"Raw Model Answer: {raw_model_answer}")
                        print(f"Model Answer: {model_answer}, Truth: {correct_answer}")
                        print(
                            f"Extracted from model: {first_number_in_model_answer.group()}, Extracted from truth: {first_number_in_correct_answer.group()}"
                        )
                        print("--- correct")
            else:
                # To debug the wrong examples
                if False and "3.1" in model:
                    print(f"--- incorrect {item['id']} ---")
                    print(f"Model: {model_answer}, Truth: {correct_answer}")
                    print(f"Problem: {item['question']}")
                    print(f"Json: {prediction_json}")
                    print(f"Extracted from model: {model_answer}, Extracted from truth: {correct_answer}")
        # consider both model_answer and correct_answer are empty string
        elif model_answer == "" and correct_answer == "":
            correct = True
        # For Debugging:
        if False and "SimPO" in model:
            if not correct:
                print(item["id"], "incorrect")
        parsed_item = item.copy()
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = model_answer
        parsed_item["correct_answer"] = correct_answer
        parsed_item["matched"] = correct
        parsed_item["answered"] = True
        parsed_results.append(parsed_item)
    return parsed_results


def aggregate_model(model, graded):
    answered = [row for row in graded if row["answered"]]
    return {
        "Model": model_name_replacement(model.split("%")[0]),
        "Mode": model.split("%")[1],
        "Acc": f"{sum(row['matched'] is True for row in graded) / len(graded) * 100:.2f}",
        "No answer": f"{(len(graded) - len(answered)) / len(graded) * 100:.2f}",
        "Total": len(graded),
        "Reason Lens": f"{sum(len(row['reasoning']) for row in answered) / len(answered):.2f}",
    }


def eval_model(model, filepath):
    with open(filepath) as stream:
        data = json.load(stream)
    graded = grade_model(model, data)
    return aggregate_model(model, graded), graded


def gen_results(run_name_folders):
    model_results = load_model_results(run_name_folders)

    columns = ["Model", "Mode", "Acc", "No answer", "Total", "Reason Lens"]
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
    rows = sorted(rows, key=lambda x: -float(x["Acc"]))
    # Convert rows to the expected format for tabulate
    table_data = [[row[col] for col in columns] for row in rows]

    print(tabulate(table_data, headers=columns, tablefmt="fancy_outline", stralign="center", numalign="center"))
    # print(tabulate(rows, headers=columns, tablefmt="github"))

    # write to markdown file
    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/4666e72d-4202-4283-8e78-e5ce2b030dcf" alt="zebra_banner" style="width: 69%;" />
</div>


"""
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(
            banner_header
            + tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center")
        )

    # write to json file
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    data_name = "crux"  # by default if there is no sys.argv[1]
    if len(sys.argv) > 1:
        data_name = sys.argv[1]
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}",
        "sampling": f"result_dirs/{data_name}/sampling",
        "greedy@no_cot": f"result_dirs/{data_name}/greedy@no_cot",
    }
    gen_results(run_name_folders)
