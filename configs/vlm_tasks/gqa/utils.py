"""
GQA (Visual Reasoning) Benchmark Utilities.

GQA is a large-scale visual reasoning dataset with compositional questions
that require multi-step reasoning over images.
"""

import re
import string
from typing import Any, Dict, List


def doc_to_image(doc: Dict[str, Any]) -> List:
    """Extract image from document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Format the prompt for GQA questions."""
    question = doc.get("question", "")
    prompt = f"<image>\n{question}\nAnswer the question using a single word or phrase."
    return prompt


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    - Convert to lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    """
    answer = answer.lower().strip()

    # Remove articles
    articles = ["a", "an", "the"]
    words = answer.split()
    words = [w for w in words if w not in articles]
    answer = " ".join(words)

    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    answer = " ".join(answer.split())

    return answer


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute accuracy.

    Args:
        doc: Document containing question and answer
        results: List of model outputs

    Returns:
        Dictionary with accuracy metric
    """
    prediction = results[0].strip() if results else ""
    reference = doc.get("answer", "")

    # Normalize both for comparison
    pred_normalized = normalize_answer(prediction)
    ref_normalized = normalize_answer(reference)

    # Exact match after normalization
    is_correct = pred_normalized == ref_normalized

    return {
        "acc": 1.0 if is_correct else 0.0,
        "prediction": prediction,
        "reference": reference,
    }
