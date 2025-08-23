"""Evaluate classifier on scoring text.

The input is a ground truth dataset of the form
{
    "text": str,
    "label": int,
}

We then take in a classifier that is a GCS path to a model checkpoint.
We then use the classifier to score the text and compare the labels.

We report the accuracy, confusion matrix, and F1 score. If desired,
we also write down where the predictions differ or agree with the ground truth.
"""

import glob
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import draccus
import fsspec
import numpy as np
import pandas as pd
import ray
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from experiments.evals.resource_configs import ResourceConfig
from marin.processing.classification.classifier import AutoClassifier

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluateClassifierConfig:
    """Configuration for evaluating a classifier.

    Attributes:
        validation_dataset_path: Path to the ground truth dataset (JSONL format).
        model_path: Path to the classifier model (local or GCS).
        output_results_path: Path to write the evaluation results JSON file.
        model_type: The type of model, e.g., 'fasttext', 'bert'. If None, it's inferred from model_path.
        attribute_name: The name of the attribute the classifier predicts.
        batch_size: The batch size for prediction.
        text_column: The name of the column containing the text in the dataset.
        label_column: The name of the column containing the ground truth label.
        use_wandb: Enable logging to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """

    validation_dataset_path: str
    model_path: str
    output_results_path: str
    model_type: str
    resource_config: ResourceConfig
    attribute_name: str = "quality"
    batch_size: int = 32
    text_column: str = "text"
    label_column: str = "label"
    run_name: str | None = field(default=None, metadata={"help": "W&B run name."})
    use_wandb: bool = field(default=False, metadata={"help": "Enable logging to Weights & Biases."})
    model_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional keyword arguments for the classifier."}
    )


def get_predicted_label(quality_dict: dict[str, float], model_type: str) -> int:
    """Extracts a single integer label from a classifier's output dictionary."""
    if not quality_dict:
        return -1  # Default for empty predictions

    # Find the label with the highest score/probability
    predicted_label_str = max(quality_dict, key=quality_dict.get)

    if model_type == "fasttext":
        return int(predicted_label_str.replace("__label__", ""))
    elif model_type == "gte":
        return quality_dict["int_score"]

    logger.warning(f"Could not parse integer from predicted label: {predicted_label_str}. Defaulting to -1.")
    return -1


@ray.remote(
    memory=64 * 1024 * 1024 * 1024,
)
def _evaluate_classifier(config: EvaluateClassifierConfig):
    """
    Evaluates a classifier based on the provided configuration.
    """
    logger.info(f"Starting classifier evaluation with config: {config}")

    if config.use_wandb:
        try:
            import wandb
        except ImportError:
            logger.error("wandb not installed. Please install it with 'pip install wandb'")
            raise

        # Initialize wandb
        assert os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_ENTITY") is not None
        assert config.run_name is not None
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=config.run_name,
        )

    # 1. Load the classifier
    logger.info(f"Loading model '{config.model_path}' of type '{config.model_type}'")
    classifier = AutoClassifier(
        model_name=config.model_path,
        attribute_name=config.attribute_name,
        model_type=config.model_type,
        **config.model_kwargs,
    )

    # 2. Load the validation dataset
    logger.info(f"Loading validation dataset from {config.validation_dataset_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        fs, fs_path = fsspec.core.url_to_fs(config.validation_dataset_path)
        fs.get(fs_path + "/*", tmp_dir)
        # Get all JSON/JSONL files from the temporary directory

        json_files = glob.glob(os.path.join(tmp_dir, "**/*.json*"), recursive=True)
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {tmp_dir} after downloading from {config.validation_dataset_path}"
            )
        dataset = load_dataset("json", data_files=json_files, split="train", keep_in_memory=True)

    # 3. Perform predictions
    all_predictions = []
    all_ground_truth = []
    detailed_results = []

    logger.info(f"Running predictions in batches of {config.batch_size}...")

    for i in tqdm(range(0, len(dataset), config.batch_size), desc="Evaluating classifier"):
        batch = dataset[i : i + config.batch_size]
        texts = batch[config.text_column]
        ground_truths = batch[config.label_column]

        # Log CPU and memory usage before classifier call
        # cpu_percent = psutil.cpu_percent(interval=1)
        # memory = psutil.virtual_memory()
        # print(
        #     f"Batch {i//config.batch_size + 1}: CPU {cpu_percent:.1f}%, Mem {memory.percent:.1f}%"
        # )

        # The classifier updates the batch dictionary with an 'attributes' key
        classifier({"text": texts, "attributes": []})
        results = classifier({"text": texts})["attributes"]

        for j, result in enumerate(results):
            quality_dict = result.get(config.attribute_name, {})
            predicted_label = get_predicted_label(quality_dict, config.model_type)
            ground_truth_label = ground_truths[j]

            all_predictions.append(predicted_label)
            all_ground_truth.append(ground_truth_label)

            detailed_results.append(
                {
                    "text": texts[j],
                    "ground_truth": ground_truth_label,
                    "prediction": predicted_label,
                }
            )

    # 4. Calculate and report metrics
    logger.info("Calculating metrics...")
    all_predictions_np = np.array(all_predictions)
    all_ground_truth_np = np.array(all_ground_truth)

    accuracy = accuracy_score(all_ground_truth_np, all_predictions_np)
    report = classification_report(all_ground_truth_np, all_predictions_np, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_ground_truth_np, all_predictions_np)
    labels = sorted(list(set(all_ground_truth_np) | set(all_predictions_np)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Metrics for classifying labels >= 3
    labels_3_plus = (all_ground_truth_np >= 3).astype(int)
    preds_3_plus = (all_predictions_np >= 3).astype(int)
    accuracy_3_plus = accuracy_score(labels_3_plus, preds_3_plus)
    precision_3_plus = precision_score(labels_3_plus, preds_3_plus, average="binary", zero_division=0)
    recall_3_plus = recall_score(labels_3_plus, preds_3_plus, average="binary", zero_division=0)
    f1_3_plus = f1_score(labels_3_plus, preds_3_plus, average="binary", zero_division=0)

    print("\\n" + "=" * 30)
    print(" " * 8 + "Evaluation Results")
    print("=" * 30)
    print(f"Accuracy: {accuracy:.4f}\\n")
    print("Classification Report:")
    print(pd.DataFrame(report).transpose().to_string())
    print("\\nConfusion Matrix:")
    print(cm_df.to_string())
    print("=" * 30 + "\\n")

    # 5. Save results to JSON file
    results_payload = {
        "metrics": {
            "accuracy": accuracy,
            "classification_report": report,
            "accuracy_3_plus": accuracy_3_plus,
            "precision_3_plus": precision_3_plus,
            "recall_3_plus": recall_3_plus,
            "f1_3_plus": f1_3_plus,
        },
        "confusion_matrix": cm_df.to_dict(),
    }

    if config.use_wandb:
        metrics_to_log = {
            "accuracy": accuracy,
            "precision_macro": report.get("macro avg", {}).get("precision", 0),
            "recall_macro": report.get("macro avg", {}).get("recall", 0),
            "f1_macro": report.get("macro avg", {}).get("f1-score", 0),
            "accuracy_3_plus": accuracy_3_plus,
            "precision_3_plus": precision_3_plus,
            "recall_3_plus": recall_3_plus,
            "f1_3_plus": f1_3_plus,
        }
        wandb.log(metrics_to_log)

        # Log confusion matrix as a table
        cm_records = cm_df.reset_index().rename(columns={"index": "actual_label"}).to_records(index=False)
        cm_table = wandb.Table(data=list(cm_records), columns=list(cm_records.dtype.names))
        wandb.log({"confusion_matrix": cm_table})

        wandb.finish()

    logger.info(f"Saving evaluation results to {config.output_results_path}")
    fs, path = fsspec.core.url_to_fs(config.output_results_path)
    with fs.open(f"{path}/results.json", "w") as f:
        json.dump(results_payload, f, indent=4)

    with fs.open(f"{path}/predictions.jsonl", "w") as f:
        for result in detailed_results:
            f.write(json.dumps(result) + "\n")

    logger.info("Evaluation finished successfully.")


def run_evaluate_classifier(config: EvaluateClassifierConfig):
    ray.get(
        _evaluate_classifier.options(
            resources={"TPU": config.resource_config.num_tpu, f"{config.resource_config.tpu_type}-head": 1},
        ).remote(config)
    )


@draccus.wrap()
def main(config: EvaluateClassifierConfig):
    """Main entry point for evaluation script."""
    run_evaluate_classifier(config)


if __name__ == "__main__":
    main()
