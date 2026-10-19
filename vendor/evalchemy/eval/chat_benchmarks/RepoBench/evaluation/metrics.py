from fuzzywuzzy import fuzz
from codebleu import calc_codebleu


def exact_match_score(predictions, ground_truths):
    """
    This function computes the average exact match score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of exact match between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes exactly match their corresponding
    ground truth codes and a value of 0 means none of the predicted codes exactly match their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes

    Returns:
    Float, the average exact match score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    exact_match = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.split() == gt.split():
            exact_match += 1

    return round(exact_match / len(predictions), 5)


def edit_similarity_score(predictions, ground_truths):
    """
    This function computes the average edit similarity score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes

    Returns:
    Float, the average edit similarity score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    edit_sim = 0.0
    for pred, gt in zip(predictions, ground_truths):
        edit_sim += fuzz.ratio(pred, gt)

    return round(edit_sim / len(predictions), 5)


def accuracy_at_k(prediction_list, golden_index_list, k):
    """
    This function computes the accuracy at k. It returns a float value between 0 and 1 indicating the
    accuracy at k, where a value of 1 means the correct code is retrieved at the top k positions and
    a value of 0 means the correct code is not retrieved at the top k positions.

    Args:
    prediction_list: list, a list of lists, where each list contains the indices of the retrieved codes.
    golden_index_list: list, a list of integers, where each integer is the index of the correct code.
    k: int, the number of retrieved codes.

    Returns:
    Float, the accuracy at k.
    """

    if len(golden_index_list) == 0:
        raise ValueError("The list of golden indices should not be empty.")

    assert len(golden_index_list) == len(prediction_list), (
        "The length of the golden indices list should be equal to the length of the prediction list, however, "
        f"the length of the golden indices list is {len(golden_index_list)} and the length of the prediction list is {len(prediction_list)}."
    )

    acc = 0

    for i in range(len(prediction_list)):
        golden_index = golden_index_list[i]
        index_list = prediction_list[i]

        if len(index_list) < k:
            raise ValueError("The number of retrieved codes should be greater than k.")

        top_k_indices = index_list[:k]

        if golden_index not in top_k_indices:
            continue
        else:
            acc += 1

    return round(acc / len(golden_index_list), 5)


def codebleu_score(predictions, ground_truths, language, weight=[0.25, 0.25, 0.25, 0.25]):
    """
    This function computes the average codebleu score between the predicted codes and the ground truth codes.
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding
    ground truth codes.

    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes
    language: str, the programming language of the codes
    weight: list, the weights for each n-gram

    Returns:
    Float, the average codebleu score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    # remove \r for both pred and gt
    predictions = [pred.replace("\r", "") for pred in predictions]
    ground_truths = [gt.replace("\r", "") for gt in ground_truths]

    res_list = calc_codebleu(ground_truths, predictions, language, weight, tokenizer=None)

    return res_list["codebleu"]
