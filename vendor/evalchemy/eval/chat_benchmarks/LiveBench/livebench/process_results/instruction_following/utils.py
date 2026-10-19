import re
from eval.chat_benchmarks.LiveBench.livebench.if_runner.instruction_following_eval import evaluation_main


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


def instruction_following_process_results(questions, model_answers, task: str, model_id: str, debug=False) -> int:
    result_log_dir = f"eval/chat_benchmarks/LiveBench/data/live_bench/instruction_following/{task}/model_judgment"
    results = evaluation_main.evaluator(questions, model_answers, result_log_dir, model_id)
    results = results["strict"]  # or "loose" depending on what we want to use.
    scores = []
    for result in results:
        question_id = result.question_id
        follow_all_instructions = result.follow_all_instructions
        follow_instruction_list = result.follow_instruction_list
        scores.append(
            {"question_id": question_id, "score": score_results(follow_all_instructions, follow_instruction_list)}
        )
        if debug and scores[-1]["score"] < 1:
            question = next((q for q in questions if q["question_id"] == question_id), None)
            print("INCORRECT", scores[-1]["score"], question_id)
            print("PROMPT", question["turns"][0].split("-------\n")[-1])
            print(
                "RESULT",
                {
                    question["instruction_id_list"][i]: result.follow_instruction_list[i]
                    for i in range(len(question["instruction_id_list"]))
                },
            )
            print("SOLUTION", model_answers[model_id][question_id]["choices"][0]["turns"][0])

    return scores
