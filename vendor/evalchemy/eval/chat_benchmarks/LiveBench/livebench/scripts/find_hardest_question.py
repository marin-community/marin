import json
from collections import defaultdict
import sys


def find_max_zero_scores(filename):
    # Dictionary to store count of zero scores per question_id
    zero_scores = defaultdict(set)

    # Read the JSONL file and process each line
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["score"] == 0:
                # Add the model to the set of models with zero score for this question
                zero_scores[data["question_id"]].add(data["model"])

    # Find the question_id(s) with the maximum number of models scoring zero
    max_zero_count = max(len(models) for models in zero_scores.values())
    max_zero_questions = [qid for qid, models in zero_scores.items() if len(models) == max_zero_count]

    # Print results
    print(f"Question(s) with the most models scoring zero ({max_zero_count} models):")
    for qid in max_zero_questions:
        print(f"Question ID: {qid}")
        print(f"Models with zero score: {sorted(list(zero_scores[qid]))}")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_jsonl_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    find_max_zero_scores(input_file)

# Created/Modified files during execution:
# (No files created or modified - script only reads from input file)
