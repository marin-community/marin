"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.eval.compare_classifiers \
    --ground_truth_path <path> \
    --prediction_path <path> \
    --ground_truth_attribute_name <attribute_name> \
    --prediction_attribute_name <attribute_name> \
    --label_name <label_name> \
    --threshold <threshold>
"""

import argparse
import json
import os
from typing import List, Tuple

import fsspec
import ray

from marin.utils import fsspec_glob, fsspec_isdir


@ray.remote
def process_file(
    ground_truth_file: str,
    prediction_file: str,
    ground_truth_attribute_name: str,
    prediction_attribute_name: str,
    label_name: str,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    ground_truth_labels: List[int] = []
    prediction_labels: List[int] = []

    with (
        fsspec.open(ground_truth_file, "rt", compression="gzip") as gt_f,
        fsspec.open(prediction_file, "rt", compression="gzip") as pred_f,
    ):
        for gt_line, pred_line in zip(gt_f, pred_f, strict=False):
            gt_data = json.loads(gt_line)
            pred_data = json.loads(pred_line)

            gt_label = 1 if gt_data["attributes"][ground_truth_attribute_name][label_name] >= threshold else 0
            pred_label = 1 if pred_data["attributes"][prediction_attribute_name][label_name] >= threshold else 0

            ground_truth_labels.append(gt_label)
            prediction_labels.append(pred_label)

    return ground_truth_labels, prediction_labels


@ray.remote(runtime_env={"pip": ["evaluate", "scikit-learn"]})
def generate_report(ground_truth_list: List[int], predictions_list: List[int]):
    from evaluate import load
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(ground_truth_list, predictions_list)
    confusion_matrix = confusion_matrix(ground_truth_list, predictions_list)

    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(confusion_matrix))

    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")
    accuracy_metric = load("accuracy")

    precision = precision_metric.compute(predictions=predictions_list, references=ground_truth_list, average="macro")[
        "precision"
    ]
    recall = recall_metric.compute(predictions=predictions_list, references=ground_truth_list, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predictions_list, references=ground_truth_list, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions_list, references=ground_truth_list)["accuracy"]

    results = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    return results


def _get_filepaths(path: str) -> List[str]:
    if fsspec_isdir(path):
        return fsspec_glob(os.path.join(path, "**/*.jsonl.gz"))
    else:
        return [path]


def main(args):
    ground_truth_files = _get_filepaths(args.ground_truth_path)
    prediction_files = _get_filepaths(args.prediction_path)

    process_file_tasks = []
    for gt_file, pred_file in zip(ground_truth_files, prediction_files, strict=False):
        process_file_tasks.append(
            process_file.remote(
                gt_file,
                pred_file,
                args.ground_truth_attribute_name,
                args.prediction_attribute_name,
                args.label_name,
                args.threshold,
            )
        )

    all_predictions = []
    all_ground_truth = []
    for task in process_file_tasks:
        ground_truth, predictions = ray.get(task)
        all_ground_truth.extend(ground_truth)
        all_predictions.extend(predictions)

    results = generate_report.remote(all_ground_truth, all_predictions)

    print("\nEvaluation Metrics:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classification report for predictions")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path containing ground truth files")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path containing prediction files")
    parser.add_argument(
        "--ground_truth_attribute_name", type=str, required=True, help="Name of the attribute to evaluate"
    )
    parser.add_argument("--prediction_attribute_name", type=str, required=True, help="Name of the attribute to evaluate")
    parser.add_argument("--label_name", type=str, required=True, help="Name of the label to evaluate")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold to use for classification")
    args = parser.parse_args()

    ray.init()
    main(args)
