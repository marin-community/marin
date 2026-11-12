

def score_results(follow_all_instructions, follow_instruction_list, threshold=0.2):
    """
    This function will return 10 if ground_truth == output,
    0 if output is off by more than 20%,
    and a linearly scaled value between 0 and 10 if the difference
    is between 0% and 20%.
    """
    # `follow_all_instructions` is either True or False, denotes if all instructions were followed.
    # `follow_instruction_list` is a list of booleans, each element denotes if the corresponding instruction was followed in a prompt.
    score_1 = 1 if follow_all_instructions else 0
    score_2 = [1 if follow else 0 for follow in follow_instruction_list]
    score_2 = sum(score_2) / len(score_2)
    avg_score = (score_1 + score_2) / 2
    return avg_score
