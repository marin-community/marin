"""
MMStar Benchmark Utilities.

MMStar is a vision-indispensable benchmark where questions cannot be
answered without looking at the image. It covers 6 core capabilities:
- Coarse perception
- Fine-grained perception
- Instance reasoning
- Logical reasoning
- Science & technology
- Math
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

    # MMStar has options as separate fields or as a list
    options = []
    for letter in "ABCD":
        opt_key = f"option_{letter.lower()}"
        if opt_key in doc:
            options.append(doc[opt_key])

    if not options and "options" in doc:
        options = doc["options"]

    # Format choices
    choice_text = ""
    for i, choice in enumerate(options):
        letter = chr(ord("A") + i)
        choice_text += f"{letter}. {choice}\n"

    prompt = f"<image>\n{question}\n\n{choice_text}\nAnswer with the option letter (A, B, C, or D)."
    return prompt


def extract_answer_letter(text: str) -> str:
    """Extract answer letter from model output."""
    text = text.strip().upper()

    if text and text[0] in "ABCD":
        return text[0]

    patterns = [
        r"^([A-D])[.\):]",
        r"\(([A-D])\)",
        r"[Aa]nswer[:\s]+([A-D])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    for char in text:
        if char in "ABCD":
            return char

    return ""


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process model output and compute accuracy.

    Args:
        doc: Document containing question, options, and answer
        results: List of model outputs

    Returns:
        Dictionary with accuracy metric
    """
    prediction = results[0] if results else ""
    reference = doc.get("answer", "")

    pred_letter = extract_answer_letter(prediction)

    if isinstance(reference, int):
        ref_letter = chr(ord("A") + reference)
    elif isinstance(reference, str) and len(reference) == 1 and reference.upper() in "ABCD":
        ref_letter = reference.upper()
    else:
        ref_letter = extract_answer_letter(str(reference))

    is_correct = pred_letter == ref_letter

    # Get category for analysis
    category = doc.get("category", doc.get("l2_category", "unknown"))

    return {
        "acc": 1.0 if is_correct else 0.0,
        "prediction": pred_letter,
        "reference": ref_letter,
        "category": category,
    }
