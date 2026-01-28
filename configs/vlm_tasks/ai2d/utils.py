"""
AI2D (AI2 Diagrams) Benchmark Utilities.

AI2D is a dataset for diagram understanding with science-focused questions.
Questions are multiple choice and require understanding of scientific diagrams
including food webs, cycles, and other scientific illustrations.
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
    choices = doc.get("choices", doc.get("options", []))

    # Format choices
    choice_text = ""
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        choice_text += f"{letter}. {choice}\n"

    prompt = f"<image>\n{question}\n\n{choice_text}\nAnswer with the option letter."
    return prompt


def extract_answer_letter(text: str) -> str:
    """Extract answer letter from model output."""
    text = text.strip().upper()

    if text and text[0] in "ABCDEFGH":
        return text[0]

    patterns = [
        r"^([A-H])[.\):]",
        r"\(([A-H])\)",
        r"[Aa]nswer[:\s]+([A-H])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    for char in text:
        if char in "ABCDEFGH":
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

    pred_letter = extract_answer_letter(prediction)

    if isinstance(reference, int):
        ref_letter = chr(ord("A") + reference)
    elif isinstance(reference, str) and len(reference) == 1 and reference.upper() in "ABCDEFGH":
        ref_letter = reference.upper()
    else:
        ref_letter = extract_answer_letter(str(reference))

    is_correct = pred_letter == ref_letter

    return {
        "acc": 1.0 if is_correct else 0.0,
        "prediction": pred_letter,
        "reference": ref_letter,
    }
