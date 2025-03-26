#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction


def url_classifier_compute_eval_metrics(labels2id: dict, eval_predictions: EvalPrediction):
    positive_label_id = labels2id[True]
    labels = eval_predictions.label_ids
    logits = eval_predictions.predictions.argmax(-1)
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(predictions=predictions, references=labels)
    binary_precision = precision_score(
        predictions=predictions, references=labels, pos_label=positive_label_id, average="binary"
    )
    binary_recall = recall_score(
        predictions=predictions, references=labels, pos_label=positive_label_id, average="binary"
    )
    binary_f1 = f1_score(predictions=predictions, references=labels, pos_label=positive_label_id, average="binary")
    return {
        "accuracy": accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
    }
