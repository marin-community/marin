import json
from collections import defaultdict
import os
from tabulate import tabulate
import re
import sys
from eval_utils import (
    load_model_results,
    extract_values_from_json,
    extract_first_complete_json,
    model_specific_extraction,
    model_name_replacement,
)


def eval_model(model, filepath):
    global private_solutions
    with open(filepath, "r") as f:
        print(f"Processing {filepath}")
        data = json.load(f)

    solved_examples = 0
    num_total_examples = len(data)
    no_answer = 0

    reason_lens = []
    parsed_results = []
    for item in data:
        # Read and Parse the prediction from model output

        prediction_str = item["output"][0]
        prediction_json = extract_first_complete_json(prediction_str)
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(prediction_str, allow_no_quotes=True)
        if (
            prediction_json is None
            or "answer" not in prediction_json
            or prediction_json["answer"] is None
            or prediction_json["answer"] == ""
        ):
            try_extracted_answer = model_specific_extraction(model, prediction_str)
            if try_extracted_answer:
                # print(f"Extracted answer from model: {try_extracted_answer}")
                prediction_json["answer"] = try_extracted_answer
            else:
                no_answer += 1
                # print the no answer examples for debugging
                if False and "Llama-3.1" in model:
                    print(f"No answer for {item['id']}")
                    print(prediction_str)
                    print(prediction_json)
                    print(correct_answer)
                continue
        reason = prediction_json.get("reasoning", "")
        model_answer = prediction_json["answer"]
        correct_answer = item["correct_answer"]
        index_of_correct_answer = item["choices"].index(correct_answer)
        label_of_correct_answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index_of_correct_answer]
        if model_answer == label_of_correct_answer or f"{label_of_correct_answer})" in model_answer:
            solved_examples += 1
            correct = True
        else:
            correct = False
            if False and "Llama-3.1" in model:  # for debugging
                print(f"## Example ID {item['id']}")
                # print(f"Input: {item['chat_history'][0]}")
                print(f"\n### Question:\n\n {item['question']}")
                print(f"\n### Choices:\n\n")
                for choice_index, choice in enumerate(item["choices"]):
                    print(f"- {chr(65+choice_index)}) {choice}")
                print(f"\n### Correct Answer:\n\n {label_of_correct_answer}")
                print(f"\n### Model's reasoning:\n\n {reason}")
                print(f"\n### Model's prediction:\n\n {model_answer}")
                print("\n\n--------------------------------\n\n")
        reason_lens.append(len(reason))
        parsed_item = item.copy()
        parsed_item["reasoning"] = reason
        parsed_item["model_answer"] = model_answer
        parsed_item["correct_answer"] = label_of_correct_answer
        parsed_item["matched"] = correct
        parsed_results.append(parsed_item)

    result = {}
    result["Model"] = model.split("%")[0]
    result["Mode"] = model.split("%")[1]
    result["Acc"] = f"{solved_examples/num_total_examples*100:.2f}"
    result["No answer"] = f"{no_answer/num_total_examples*100:.2f}"
    result["Total"] = num_total_examples
    result["Reason Lens"] = f"{sum(reason_lens)/len(reason_lens):.2f}"

    result["Model"] = model_name_replacement(result["Model"])
    return result, parsed_results


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

    # write to json file
    with open(f"result_dirs/{data_name}.summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # write to markdown file
    banner_header = """
<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/aba4df22-61dd-4a8e-b2fb-217ec18865b0" alt="zebra_banner" style="width: 69%;" />
</div>


"""
    with open(f"result_dirs/{data_name}.summary.md", "w") as f:
        f.write(
            banner_header
            + tabulate(table_data, headers=columns, tablefmt="github", stralign="center", numalign="center")
        )


if __name__ == "__main__":
    data_name = sys.argv[1]
    if data_name not in ["mmlu-redux"]:
        print(f"Invalid data name: {data_name}")
        sys.exit(1)
    run_name_folders = {
        "greedy": f"result_dirs/{data_name}",
    }
    gen_results(run_name_folders)
