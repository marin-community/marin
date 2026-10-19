import json
import csv
import sys


def jsonl_to_csv(input_filename, output_filename, task):
    """
    Converts a JSONL file to a CSV file with specific fields.

    Args:
        input_filename (str): The path to the input JSONL file.
        output_filename (str): The path to the output CSV file.
    """
    with (
        open(input_filename, "r", encoding="utf-8") as jsonl_file,
        open(output_filename, "w", encoding="utf-8", newline="") as csv_file,
    ):
        # Define the CSV writer and write the header
        csv_writer = csv.writer(csv_file)
        header = ["question_id", "citation", "prompt"]
        if "coding_completion" in task:
            header.append("partial_solution")
            header.append("remainder")
        csv_writer.writerow(header)

        # Process each line in the JSONL file
        for line in jsonl_file:
            # Parse the JSON object
            data = json.loads(line)

            # Extract the required fields
            question_id = data.get("question_id", "")
            citation = data.get("citation", "").split(" ")[0]
            turns = data.get("turns", [])
            if "coding_completion" in task:
                partial_solution = data.get("partial_solution", "")
                remainder = data.get("remainder", "")

            # Get the first turn as the prompt, if available
            prompt = turns[0] if turns else ""

            # Write the row to the CSV file
            row = [question_id, citation, prompt]
            if "coding_completion" in task:
                row.append(partial_solution)
                row.append(remainder)
            csv_writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python code_question_to_csv.py <input_filename> <output_filename> <task>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    task = sys.argv[3]

    jsonl_to_csv(input_filename, output_filename, task)
