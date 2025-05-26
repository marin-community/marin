import ast
import glob
import random
import re

from marin.optimizer_sweep.utils_simp import (
    calculate_data_tag,
    check_baseline_run,
    create_configs,
    grab_best_run,
)


def extract_job_id(log_name):
    # Open the log file for reading
    with open(log_name, "r") as file:
        # Iterate over each line in the log file
        for line in file:
            # Use regex to search for the job id pattern:
            # "Job submitted with ID:" followed by whitespace and then the id starting with "raysubmit"
            match = re.search(r"Job submitted with ID:\s*(raysubmit\S+)", line)
            if match:
                # Return the captured job id
                return match.group(1)
    # Return None if no job id is found
    return None


def parse_command_file(filename):
    """
    Parses the given Python file and extracts the following variables:
      - sweep_grids
      - baseline_config
      - model_size
      - target_chinchilla
      - optimizer_name (extracted from the third argument in the call to template)

    Returns:
      A dictionary containing these parsed values.
    """
    with open(filename, "r") as f:
        content = f.read()

    tree = ast.parse(content)
    result = {}

    # Set of variable names we want to capture from assignments
    target_vars = {"sweep_grids", "model_size", "target_chinchilla", "baseline_config"}

    # Walk the AST to look for assignment nodes and the template() call
    for node in ast.walk(tree):
        # Look for assignments like: var_name = <value>
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in target_vars:
                    try:
                        # Use literal_eval to safely evaluate constant expressions
                        result[target.id] = ast.literal_eval(node.value)
                    except Exception as e:
                        print(f"Could not evaluate {target.id}: {e}")
        # Look for the call to template(...)
        elif isinstance(node, ast.Call):
            # Ensure the function being called is named "template"
            if isinstance(node.func, ast.Name) and node.func.id == "template":
                # We expect the third argument to be the optimizer name.
                if len(node.args) >= 3:
                    try:
                        optimizer_name = ast.literal_eval(node.args[2])
                        result["optimizer_name"] = optimizer_name
                    except Exception as e:
                        print(f"Could not evaluate optimizer name: {e}")

    return result


def extract_baseline_from_line(line):
    """
    Extracts the baseline dictionary from a line if it matches the expected pattern.

    Expected pattern:
    "Choose: { ... }"
    """
    match = re.search(r"Choose:\s*(\{.*\})", line)
    if match:
        dict_str = match.group(1)
        try:
            baseline = ast.literal_eval(dict_str)
            return baseline
        except Exception as e:
            print("Error parsing baseline dictionary:", e)
    return None


def extract_baseline_from_file(log_name):
    """
    Opens the log file and returns the first baseline dictionary found.
    """
    with open(log_name, "r") as file:
        for line in file:
            baseline = extract_baseline_from_line(line)
            if baseline is not None:
                return baseline
    return None


# Define the pattern to match the desired files
pattern = "logs/*.txt"

# Use glob to get a list of all files matching the pattern
file_list = glob.glob(pattern)


def replace_random_suffix(file_path):
    # Read the entire file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Generate a random letter between 'a' and 'z'
    random_letter = random.choice("abcdefghijklmnopqrstuvwxyz")

    # Replace "random_suffix = None" with "random_suffix = 'random_letter'"
    new_lines = []
    for line in lines:
        if "my_suffix = " in line:
            new_lines.append(line.split("my_suffix = ")[0] + f"my_suffix = '{random_letter}'\n")
        else:
            new_lines.append(line)

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        for line in new_lines:
            file.write(line)

    print(f"Updated random_suffix to '{random_letter}' in {file_path}")


def num_left(baseline_config):
    target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
    # use wandb to avoid rerunning
    num_left = 0
    for config in config_in_dict:
        if not check_baseline_run(config, tags):
            num_left += 1
    return num_left


def judge_success(first_baseline, tags):
    current_best_config, approximate_best_config_list = grab_best_run(first_baseline.keys(), tags)
    min_num = 10000
    min_config = None
    for approximate_best_config in approximate_best_config_list:
        num_left_config = num_left(approximate_best_config)
        if num_left_config == 0:
            print(f"Found: {approximate_best_config}")
        if num_left_config < min_num:
            min_num = num_left_config
            min_config = approximate_best_config
    return min_num, min_config


if __name__ == "__main__":
    for file_path in file_list:
        try:
            print(f"Going over: {file_path}")
            name = file_path.split("/")[1].split(".txt")[0]
            log_name = file_path
            filename = f"experiments/optimizer_sweep/PhaseI_Bound/{name}.py"
            parsed_data = parse_command_file(filename)
            first_baseline = parsed_data["baseline_config"]
            sweep_grids = parsed_data["sweep_grids"]
            model_size = parsed_data["model_size"]
            target_chinchilla = parsed_data["target_chinchilla"]
            optimizer = parsed_data["optimizer_name"]
            target_data, data_size = calculate_data_tag(model_size, target_chinchilla)
            tags = (model_size, data_size, optimizer)
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    for line in lines:
                        if "Stupid Ray" in line:
                            print("Stupid Ray Found")
                            replace_random_suffix(filename)
        except Exception as e:
            print(f"Error in {file_path}: {e}")
