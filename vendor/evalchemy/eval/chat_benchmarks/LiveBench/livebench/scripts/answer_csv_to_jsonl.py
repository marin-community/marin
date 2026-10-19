import csv
import json
import shortuuid
import time
import sys


def csv_to_jsonl(input_csv, output_jsonl):
    try:
        # Open the input CSV file
        with open(input_csv, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            # Open the output JSONL file
            with open(output_jsonl, mode="w", encoding="utf-8") as jsonl_file:
                for row in reader:
                    # Extract the question_id and output from the CSV row
                    question_id = row["question_id"]
                    output = row["output"]

                    # Generate the JSON object
                    json_object = {
                        "question_id": question_id,
                        "answer_id": shortuuid.uuid(),
                        "model_id": "o1",
                        "choices": [
                            {
                                "index": 0,
                                "turns": ["```python\n" + output + "\n```"],
                            }
                        ],
                        "tstamp": time.time(),
                    }

                    # Write the JSON object to the JSONL file
                    jsonl_file.write(json.dumps(json_object) + "\n")

        print(f"Successfully converted {input_csv} to {output_jsonl}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Ensure the script is called with the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <output_jsonl>")
        sys.exit(1)

    # Get the input and output file names from command line arguments
    input_csv = sys.argv[1]
    output_jsonl = sys.argv[2]

    # Call the conversion function
    csv_to_jsonl(input_csv, output_jsonl)
