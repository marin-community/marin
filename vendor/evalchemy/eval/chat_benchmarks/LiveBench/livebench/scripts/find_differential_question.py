import json
from collections import defaultdict
import sys
import argparse


def find_differential_problems(filename, model1, model2):
    # Dictionary to store scores for each question_id and model
    scores = defaultdict(dict)

    # Read the JSONL file and process each line
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            model = data["model"]
            score = data["score"]

            if model in [model1, model2]:
                scores[qid][model] = score

    # Find questions where model1 succeeds (score > 0) and model2 fails (score = 0)
    differential_cases = []
    for qid, model_scores in scores.items():
        if len(model_scores) == 2:  # ensure we have scores for both models
            if model_scores[model1] > 0 and model_scores[model2] == 0:
                differential_cases.append((qid, model_scores[model1], model_scores[model2]))

    # Print results
    print(f"\nFound {len(differential_cases)} questions where {model1} succeeds but {model2} fails:")
    for qid, score1, score2 in differential_cases:
        print(f"\nQuestion ID: {qid}")
        print(f"{model1} score: {score1}")
        print(f"{model2} score: {score2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find problems where one model succeeds and another fails")
    parser.add_argument("filename", help="Input JSONL file")
    parser.add_argument("model1", help="First model name")
    parser.add_argument("model2", help="Second model name")

    args = parser.parse_args()

    find_differential_problems(args.filename, args.model1, args.model2)
