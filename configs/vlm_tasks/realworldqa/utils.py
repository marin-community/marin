"""
RealWorldQA Benchmark Utilities.

RealWorldQA is a benchmark for real-world visual understanding,
including scenarios like driving, navigation, and everyday scenes.
Questions are in multiple-choice format (A/B/C/D).
"""

import re
from typing import Any, Dict, List


def doc_to_image(doc: Dict[str, Any]) -> List:
    """Extract image from document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Format the prompt with question and choices."""
    question = doc.get("question", "")
    choices = doc.get("choices", [])

    # Format choices as A, B, C, D
    choice_text = ""
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        choice_text += f"{letter}. {choice}\n"

    prompt = f"<image>\n{question}\n\n{choice_text}\nAnswer with the option letter (A, B, C, or D)."
    return prompt


def extract_answer_letter(text: str) -> str:
    """Extract answer letter (A, B, C, D) from model output."""
    text = text.strip().upper()

    # Try to find a letter at the start
    if text and text[0] in "ABCD":
        return text[0]

    # Look for patterns like "A.", "A)", "(A)", "Answer: A"
    patterns = [
        r"^([A-D])[.\):]",
        r"\(([A-D])\)",
        r"[Aa]nswer[:\s]+([A-D])",
        r"[Oo]ption[:\s]+([A-D])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # If no pattern matched, return the first letter if it's A-D
    for char in text:
        if char in "ABCD":
            return char

    return ""


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute accuracy.

    Args:
        doc: Document containing question, choices, and answer
        results: List of model outputs

    Returns:
        Dictionary with accuracy metric
    """
    prediction = results[0] if results else ""
    reference = doc.get("answer", "")

    # Extract letter from prediction
    pred_letter = extract_answer_letter(prediction)

    # Reference might be the letter or the index
    if isinstance(reference, int):
        ref_letter = chr(ord("A") + reference)
    elif isinstance(reference, str) and len(reference) == 1 and reference.upper() in "ABCD":
        ref_letter = reference.upper()
    else:
        ref_letter = extract_answer_letter(str(reference))

    is_correct = pred_letter == ref_letter

    return {
        "acc": 1.0 if is_correct else 0.0,
        "prediction": pred_letter,
        "reference": ref_letter,
    }
