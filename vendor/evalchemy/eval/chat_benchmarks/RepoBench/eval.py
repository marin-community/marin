import os
import json
from evaluation.metrics import exact_match_score, edit_similarity_score, codebleu_score
import fire


def eval(
    path="results/deepseek-coder-1.3b-base-python",
    language="python",  # to calculate codebleu, we need to specify the language
):
    total_data_points = 0
    total_em_model, total_es_model, total_cb_model = 0, 0, 0

    for level in ["cross_file_first", "cross_file_random", "in_file"]:
        filepath = os.path.join(path, f"{level}.jsonl")
        seen_indices = set()  # Track seen indices for the current level

        # check if the file exists
        if not os.path.exists(filepath):
            print(f"Level: {level} not found for the model")
            continue

        with open(filepath, "r") as f:
            data = []
            for line in f:
                entry = json.loads(line.strip())
                idx = entry["idx"]

                # Skip duplicate indices based on the chosen policy (here, keeping the former)
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    data.append(entry)

            data_points = len(data)

            if data_points == 0:
                continue

            ground_truth = [d["gt"] for d in data]
            generated = [d["pred"] for d in data]

            em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
            es_model = round(edit_similarity_score(ground_truth, generated), 2)
            cb_model = round(codebleu_score(generated, ground_truth, language) * 100, 2)

            # accumulate the data points and the metrics
            total_data_points += data_points
            total_em_model += em_model * data_points
            total_es_model += es_model * data_points
            total_cb_model += cb_model * data_points

            print(f"Level: {level} with {data_points} data points")
            print(f"EM: {em_model}, ES: {es_model}, CB: {cb_model}")
            print("-" * 30)

    # calculate the weighted averages
    if total_data_points > 0:
        avg_em_model = round(total_em_model / total_data_points, 2)
        avg_es_model = round(total_es_model / total_data_points, 2)
        avg_cb_model = round(total_cb_model / total_data_points, 2)

        print("Weighted Averages:")
        print(f"EM: {avg_em_model}, ES: {avg_es_model}, CB: {avg_cb_model}\n")

    else:
        print("No data points were found for evaluation.")


if __name__ == "__main__":
    fire.Fire(eval)
