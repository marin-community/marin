import re
from collections import defaultdict
from matharena.parser import extract_answer


def extract_numbers(text):
    """
    Extract numbers from the text.
    Returns a list of tuples: (number_string, start_index, end_index)
    """
    # This regex handles integers and decimals.
    pattern = r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)"
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def check_number_proximity_any_order(gold, model, threshold=20):
    """
    Check if the numbers from the gold answer appear close together
    in the model answer, regardless of their order.

    The function finds the smallest window in the model answer that contains
    at least one occurrence of each number from the gold answer. If the span
    of that window is less than or equal to the threshold, we flag it.
    """
    if len(gold) < 5:  # too easy, let's not do this
        return False
    gold_numbers = extract_numbers(str(gold))
    if not gold_numbers:
        return False  # Nothing to check if no numbers in the gold answer.

    threshold = max(2 * len(str(gold)), threshold)

    # We assume each number is considered only once.
    gold_set = set(num for num, _, _ in gold_numbers)

    model_numbers = extract_numbers(model)
    if not model_numbers:
        return False  # No numbers in model answer.

    # Gather occurrences of gold numbers in the model answer: (position, number)
    occurrences = [(start, num) for num, start, _ in model_numbers if num in gold_set]
    if not occurrences:
        return False  # None of the gold numbers appear in the model answer.

    # Sort occurrences by their position in the text.
    occurrences.sort(key=lambda x: x[0])

    # Use a sliding window to find the minimal span that covers all gold numbers.
    count = defaultdict(int)
    have = 0
    need = len(gold_set)
    min_window = float("inf")
    left = 0

    for right in range(len(occurrences)):
        pos_right, num_right = occurrences[right]
        count[num_right] += 1
        if count[num_right] == 1:  # first time this gold number appears in the window
            have += 1

        # Once the window covers all gold numbers, try to shrink it from the left.
        while have == need and left <= right:
            pos_left, num_left = occurrences[left]
            window_span = pos_right - pos_left
            if window_span < min_window:
                min_window = window_span
            # Move the left pointer; if removing this occurrence loses a needed number, update 'have'
            count[num_left] -= 1
            if count[num_left] == 0:
                have -= 1
            left += 1

    return min_window <= threshold


def check_all_numbers(text, gold_answer):
    if extract_answer(text, strict_parsing=True)[0] is not None:
        return False
    numbers = re.findall(r"\d+", text)
    return any(num == gold_answer for num in numbers)


def check_output_length(length):
    if length < 1000:
        return False
    if length % 1000 == 0:
        return True
    while length % 10 == 0 and length > 1:
        length /= 10
    while length % 2 == 0 and length > 1:
        length /= 2
    return length == 1
