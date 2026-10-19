import re
import os
import json

total_num_examples = {"gsm": 1319, "mmlu-redux": 2778, "zebra-grid": 1000, "crux": 800, "math-l5": 721}


def model_name_replacement(model_name):
    model_name = model_name.replace("gemma-2-9b-it@nvidia", "gemma-2-9b-it")
    model_name = model_name.replace("gemma-2-9b-it@together", "gemma-2-9b-it")
    model_name = model_name.replace("gemma-2-27b-it@together", "gemma-2-27b-it")
    model_name = model_name.replace("gemma-2-27b-it@nvidia", "gemma-2-27b-it")
    model_name = model_name.replace("deepseek-chat", "deepseek-v2-chat-0628")
    model_name = model_name.replace("deepseek-coder", "deepseek-v2-coder-0614")
    model_name = model_name.replace("DeepSeek-Coder-V2-0724", "deepseek-v2-coder-0724")
    model_name = model_name.replace("Llama-3.1-405B-Inst-fp8", "Llama-3.1-405B-Inst-fp8@together")
    model_name = model_name.replace("Llama-3.1-405B-Instruct-Turbo", "Llama-3.1-405B-Inst-fp8@together")
    model_name = model_name.replace("Meta-Llama-3.1-405B-Instruct@hyperbolic", "Llama-3.1-405B-Inst@hyperbolic")
    return model_name


def model_specific_extraction(model_name, prediction_str):
    if "Llama-3.1" in model_name:
        if "boxed" in prediction_str[-30:]:
            # print(prediction_str)
            # extract "$\boxed{36}$" --> 36
            # print(prediction_str)
            match = re.search(r"\\boxed{([\w\d]+)}", prediction_str)
            if match:
                return match.group(1)
    return None


def load_model_results(run_name_folders):
    model_results = {}

    for run_name, folder in run_name_folders.items():
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue
        # iterate all json files under the folder
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not filename.endswith(".json"):
                continue
            model_name = filename.replace(".json", "")
            model_name = f"{model_name}%{run_name}"
            model_results[model_name] = filepath
    return model_results


def extract_values_from_json(json_string, keys=["reasoning", "answer"], allow_no_quotes=False):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f"{key}\\s*:\\s*([^,\\s]*)"
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def extract_first_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None

    return None


def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start : i + 1]
                    last_json_start = None

    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass

    return None


if __name__ == "__main__":
    json_test = """
    {
        "reasoning": "Calculate shipping cost ($1.40 per pound x 4 pounds) and mileage cost ($0.08 per mile x 20 miles), then add them together ($3.00). Determine refund amount (75% of $32 = $24) and subtract it from the total shipping cost to find Milly’s loss (-$21).",
        "answer": -21
    }
    """
    print(json.dumps(extract_last_complete_json(json_test), indent=2))
