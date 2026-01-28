"""
MME (MultiModal Evaluation) Benchmark Utilities.

MME is a comprehensive evaluation benchmark for VLMs with 14 perception subtasks
and 6 cognition subtasks. Each subtask has Yes/No questions.

Score = accuracy_positive + accuracy_negative for each subtask.
"""

import re
from typing import Any, Dict, List


def doc_to_image(doc: Dict[str, Any]) -> List:
    """Extract image from document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Format the prompt for MME questions."""
    question = doc.get("question", "")
    # MME uses Yes/No format
    prompt = f"<image>\n{question}\nPlease answer yes or no."
    return prompt


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute metrics.

    Args:
        doc: Document containing question, answer, category, etc.
        results: List of model outputs (typically single string)

    Returns:
        Dictionary with accuracy and category info for aggregation
    """
    prediction = results[0].strip().lower() if results else ""
    reference = doc.get("answer", "").lower()

    # Extract yes/no from prediction
    if "yes" in prediction:
        pred_answer = "yes"
    elif "no" in prediction:
        pred_answer = "no"
    else:
        pred_answer = prediction.split()[0] if prediction else ""

    # Check correctness
    is_correct = pred_answer == reference

    # Get category and question type for MME score calculation
    category = doc.get("category", "unknown")
    question_id = doc.get("question_id", "")

    return {
        "acc": 1.0 if is_correct else 0.0,
        "category": category,
        "question_id": question_id,
        "prediction": pred_answer,
        "reference": reference,
        "is_correct": is_correct,
    }


def mme_score(results: List[Dict], docs: List[Dict]) -> float:
    """
    Compute MME score for a category.

    MME Score = accuracy_positive + accuracy_negative
    where positive/negative refers to whether the correct answer is Yes or No.

    This is computed per category and then averaged.
    """
    # Group by category
    category_results = {}
    for result, doc in zip(results, docs):
        category = result.get("category", doc.get("category", "unknown"))
        if category not in category_results:
            category_results[category] = {"positive": [], "negative": []}

        reference = result.get("reference", doc.get("answer", "").lower())
        is_correct = result.get("is_correct", False)

        if reference == "yes":
            category_results[category]["positive"].append(1.0 if is_correct else 0.0)
        else:
            category_results[category]["negative"].append(1.0 if is_correct else 0.0)

    # Compute score per category
    total_score = 0.0
    num_categories = 0

    for category, data in category_results.items():
        if data["positive"] and data["negative"]:
            acc_positive = sum(data["positive"]) / len(data["positive"])
            acc_negative = sum(data["negative"]) / len(data["negative"])
            category_score = (acc_positive + acc_negative) * 100  # Scale to 0-200
            total_score += category_score
            num_categories += 1

    return total_score / num_categories if num_categories > 0 else 0.0


def aggregate_mme_score(results: List[float]) -> float:
    """Aggregate MME scores across categories."""
    return sum(results) / len(results) if results else 0.0
