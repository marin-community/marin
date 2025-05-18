import ast
import json
import os
import re


def parse_sweep_file(file_path):
    """Parse a single sweep file to extract sweep configuration."""
    with open(file_path, "r") as f:
        content = f.read()

    # Parse the file content into an AST
    tree = ast.parse(content)

    result = {
        "sweep_grids": None,
        "baseline_config": None,
        "model_size": None,
        "target_chinchilla": None,
        "optimizer": None,
    }

    # Extract optimizer from filename
    # Example: exp725_adamwsweep_130M_1.py -> adamw
    filename = os.path.basename(file_path)
    optimizer_match = re.search(r"_(\w+)sweep_", filename)
    if optimizer_match:
        result["optimizer"] = optimizer_match.group(1).lower()

    # Walk through the AST to find assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None

            if target_name == "sweep_grids":
                result["sweep_grids"] = ast.literal_eval(node.value)
            elif target_name == "baseline_config":
                result["baseline_config"] = ast.literal_eval(node.value)
            elif target_name == "model_size":
                result["model_size"] = ast.literal_eval(node.value)
            elif target_name == "target_chinchilla":
                result["target_chinchilla"] = ast.literal_eval(node.value)

    return result


def collect_all_sweeps(directory):
    """Collect sweep configurations from all relevant files in the directory."""
    all_sweeps = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and "sweep" in file.lower():
                file_path = os.path.join(root, file)
                try:
                    sweep_config = parse_sweep_file(file_path)
                    if any(sweep_config.values()):  # Only add if we found some configuration
                        sweep_config["source_file"] = file
                        all_sweeps.append(sweep_config)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing {file}: {e}")

    return all_sweeps


def main():
    # Assuming the script is run from the project root directory
    optimizer_sweep_dir = "experiments/optimizer_sweep/PhaseIII_16x"
    output_file = "experiments/optimizer_sweep/Analysis/PhaseIII_16x/sweep_configurations.json"

    # Collect all sweep configurations
    all_sweeps = collect_all_sweeps(optimizer_sweep_dir)

    # Organize by optimizer, model_size, and chinchilla ratio
    organized_sweeps = {}
    for sweep in all_sweeps:
        optimizer = sweep["optimizer"]
        model_size = sweep["model_size"]
        chinchilla = sweep["target_chinchilla"]

        if optimizer not in organized_sweeps:
            organized_sweeps[optimizer] = {}
        if model_size not in organized_sweeps[optimizer]:
            organized_sweeps[optimizer][model_size] = {}
        if chinchilla not in organized_sweeps[optimizer][model_size]:
            organized_sweeps[optimizer][model_size][chinchilla] = []

        organized_sweeps[optimizer][model_size][chinchilla].append(sweep)

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(organized_sweeps, f, indent=2)

    print(f"Sweep configurations have been saved to {output_file}")


if __name__ == "__main__":
    main()
