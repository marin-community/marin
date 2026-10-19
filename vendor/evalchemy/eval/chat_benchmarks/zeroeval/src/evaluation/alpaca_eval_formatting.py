import json
import sys


def remove_configs_from_json(input_file, output_file):
    # Read the JSON data from the input file
    with open(input_file, "r") as file:
        data = json.load(file)

    # Remove the "configs" key-value pairs from each dictionary in the list
    for item in data:
        del item["configs"]
        del item["chat_history"]

    # Write the modified data back to the output file
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)


# Example usage
input_file = sys.argv[1]
output_file = input_file.replace(".json", ".ae.json")
remove_configs_from_json(input_file, output_file)
