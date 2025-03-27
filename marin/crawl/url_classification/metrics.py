#!/usr/bin/env python3
import numpy as np
from evaluate import load_metric
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

    accuracy = load_metric("accuracy").compute(predictions=predictions, references=labels)
    binary_precision = load_metric("precision").compute(
        predictions=predictions, references=labels, pos_label=positive_label_id, average="binary"
    )
    binary_recall = load_metric("recall").compute(
        predictions=predictions, references=labels, pos_label=positive_label_id, average="binary"
    )
    binary_f1 = load_metric("f1").compute(
        predictions=predictions, references=labels, pos_label=positive_label_id, average="binary"
    )
    return {
        "accuracy": accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
    }
