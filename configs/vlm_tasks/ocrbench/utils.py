"""
OCRBench Benchmark Utilities.

OCRBench evaluates OCR capabilities of VLMs across 5 categories:
1. Text Recognition - Basic text recognition in images
2. Scene Text - Text in natural scenes
3. Document - Document understanding
4. Table - Table structure and content
5. KIE (Key Information Extraction) - Extract key information

Contains 29 subtasks in total.
"""

import re
from typing import Any, Dict, List


def doc_to_image(doc: Dict[str, Any]) -> List:
    """Extract image from document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Format the prompt for OCR tasks."""
    question = doc.get("question", "")

    # OCRBench questions are typically direct
    prompt = f"<image>\n{question}"
    return prompt


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower().strip()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def check_answer_match(prediction: str, reference: str) -> bool:
    """
    Check if prediction matches reference.

    Uses fuzzy matching for OCR tasks since exact character matching
    can be too strict for OCR outputs.
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    # Exact match after normalization
    if pred_norm == ref_norm:
        return True

    # Check if reference is contained in prediction (for verbose outputs)
    if ref_norm in pred_norm:
        return True

    # For short answers, check if prediction contains the reference
    if len(ref_norm) <= 20 and ref_norm in pred_norm:
        return True

    return False


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute accuracy.

    Args:
        doc: Document containing question and answer
        results: List of model outputs

    Returns:
        Dictionary with accuracy metric and category info
    """
    prediction = results[0].strip() if results else ""
    reference = doc.get("answer", "")

    # Handle list of acceptable answers
    if isinstance(reference, list):
        is_correct = any(check_answer_match(prediction, ref) for ref in reference)
    else:
        is_correct = check_answer_match(prediction, str(reference))

    # Get category for analysis
    category = doc.get("type", doc.get("category", "unknown"))
    subtask = doc.get("subtask", doc.get("dataset", "unknown"))

    return {
        "acc": 1.0 if is_correct else 0.0,
        "prediction": prediction[:100],  # Truncate for logging
        "reference": str(reference)[:100] if isinstance(reference, str) else str(reference[0])[:100] if reference else "",
        "category": category,
        "subtask": subtask,
    }
