import argparse
import json
import os
from typing import Dict, List, Tuple

import fsspec
import ray

from marin.utils import fsspec_glob


@ray.remote
def process_file(
    ground_truth_file: str, prediction_file: str, ground_truth_attribute_name: str, prediction_attribute_name: str
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

            # NOTE(chris): A bit of a hack assuming that both of them take in __label__hq as the key
            # Reformat this to be more generic.
            gt_label = 1 if gt_data["attributes"][ground_truth_attribute_name]["__label__hq"] >= 0.5 else 0
            pred_label = 1 if pred_data["attributes"][prediction_attribute_name]["__label__hq"] >= 0.5 else 0

            ground_truth_labels.append(gt_label)
            prediction_labels.append(pred_label)

    return ground_truth_labels, prediction_labels


@ray.remote(runtime_env={"pip": ["evaluate", "scikit-learn"]})
class ClassificationReportActor:
    def __init__(self):
        self.all_ground_truth: List[int] = []
        self.all_predictions: List[int] = []

    def add_results(self, ground_truth: List[int], predictions: List[int]) -> Dict[str, float]:
        self.all_ground_truth.extend(ground_truth)
        self.all_predictions.extend(predictions)

    def generate_report(self):
        from evaluate import load
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(self.all_ground_truth, self.all_predictions)
        confusion_matrix = confusion_matrix(self.all_ground_truth, self.all_predictions)

        print("Validation Report:\n" + report)
        print("Confusion Matrix:\n" + str(confusion_matrix))

        precision_metric = load("precision")
        recall_metric = load("recall")
        f1_metric = load("f1")
        accuracy_metric = load("accuracy")

        precision = precision_metric.compute(
            predictions=self.all_predictions, references=self.all_ground_truth, average="macro"
        )["precision"]
        recall = recall_metric.compute(
            predictions=self.all_predictions, references=self.all_ground_truth, average="macro"
        )["recall"]
        f1 = f1_metric.compute(predictions=self.all_predictions, references=self.all_ground_truth, average="macro")["f1"]
        accuracy = accuracy_metric.compute(predictions=self.all_predictions, references=self.all_ground_truth)[
            "accuracy"
        ]

        results = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

        return results


def main(args):
    ray.init()

    # NOTE(Chris): Make this generic to take in any filepath whether it be directory or file.
    ground_truth_files = fsspec_glob(os.path.join(args.ground_truth_dir, "**/*.jsonl.gz"))
    prediction_files = fsspec_glob(os.path.join(args.prediction_dir, "**/*.jsonl.gz"))

    report_actor = ClassificationReportActor.remote()

    process_file_tasks = []
    for gt_file, pred_file in zip(ground_truth_files, prediction_files, strict=False):
        process_file_tasks.append(
            process_file.remote(gt_file, pred_file, args.ground_truth_attribute_name, args.prediction_attribute_name)
        )

    report_actor_tasks = []
    for task in process_file_tasks:
        ground_truth, predictions = ray.get(task)
        report_actor_tasks.append(report_actor.add_results.remote(ground_truth, predictions))

    ray.get(report_actor_tasks)

    results = ray.get(report_actor.generate_report.remote())

    print("\nEvaluation Metrics:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classification report for predictions")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Directory containing ground truth files")
    parser.add_argument("--prediction_dir", type=str, required=True, help="Directory containing prediction files")
    parser.add_argument(
        "--ground_truth_attribute_name", type=str, required=True, help="Name of the attribute to evaluate"
    )
    parser.add_argument("--prediction_attribute_name", type=str, required=True, help="Name of the attribute to evaluate")
    args = parser.parse_args()

    main(args)
