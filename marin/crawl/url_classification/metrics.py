#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction


def url_classifier_compute_eval_metrics(labels2id: dict[str, int], eval_predictions: EvalPrediction):
    """
    Function for computing metrics when training BERT-based URL classifiers.

    Unfortunately, this needs to be defined in a separate module, rather than
    inline in the training script. Otherwise, Python's default pickling can't
    find the function because it's in __main__ (rather than in a proper
    importable module)
    """
    positive_label_id = labels2id["True"]
    logits, labels = eval_predictions
    predictions = np.argmax(logits, axis=-1)

    accuracy = float(accuracy_score(labels, predictions))

    binary_precision = float(precision_score(labels, predictions, pos_label=positive_label_id, average="binary"))
    binary_recall = float(recall_score(labels, predictions, pos_label=positive_label_id, average="binary"))
    binary_f1 = float(f1_score(labels, predictions, pos_label=positive_label_id, average="binary"))
    return {
        "accuracy": accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
    }
