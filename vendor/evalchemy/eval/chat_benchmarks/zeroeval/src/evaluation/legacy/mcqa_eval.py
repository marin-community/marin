import json
import sys

filepath = sys.argv[1]

# Load the prediction file
# filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Llama-3-8B-Tulu-330K.json"
# filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Magpie-Pro-SFT-v0.1.json"
# filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Llama-3-8B-OpenHermes-243K.json"
# filepath = "ZeroEval/result_dirs/mmlu-redux/cot=false/Llama-3-8B-Ultrachat-200K.json"

filepath = filepath.replace("ZeroEval/", "")
with open(filepath, "r") as f:
    predictions = json.load(f)

answers = {}
total = 0
correct = 0
eval_ed_ids = set()

total_examples = 0
correct = 0
for prediction_item in predictions:
    total_examples += 1
    answer = prediction_item["correct_answer"]
    # remove the endding "." from the answer
    choices = []
    for choice in prediction_item["choices"]:
        while choice.endswith(".") or choice.endswith(","):
            choice = choice[:-1]
        choices.append(choice)
    while answer.endswith(".") or answer.endswith(","):
        answer = answer[:-1]
    assert answer in choices
    labels = ["(A)", "(B)", "(C)", "(D)"]
    answer_label = labels[choices.index(answer)]
    output = prediction_item["output"][0]
    if f": {answer_label}" in output:
        correct += 1
    elif f": {answer}" in output:
        correct += 1
    elif f"is {answer_label}" in output:
        correct += 1
    elif f"is {answer}" in output:
        correct += 1
    elif f"be {answer_label}" in output:
        correct += 1
    elif f"be {answer}" in output:
        correct += 1

print("---------------------------------")
print(f"File: {filepath}")
# print(f"Total examples: {total_examples}")
# print(f"Correct: {correct}")
print(f"Accuracy: {correct/total_examples}")
print("---------------------------------")

exit()


"""
echo "Start"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-OpenHermes-243K.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-Ultrachat-200K.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-Tulu-330K.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-WildChat.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-ShareGPT-112K.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Magpie-Pro-SFT-v0.1.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-WizardLM-196K.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Meta-Llama-3-8B-Instruct.json" 
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Pro-SFT-200K-v0.1.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Pro-SFT-100K-v0.1.json"
python evaluation/eval.py "result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Air-SFT-v0.1.json"

echo "Done"


"""


"""

---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-Ultrachat-200K.json
Accuracy: 0.5075593952483801
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-Tulu-330K.json
Accuracy: 0.46976241900647947
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-WildChat.json
Accuracy: 0.5259179265658748
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-ShareGPT-112K.json
Accuracy: 0.5068394528437725
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Magpie-Pro-SFT-v0.1.json
Accuracy: 0.521598272138229
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-WizardLM-196K.json
Accuracy: 0.52735781137509
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Meta-Llama-3-8B-Instruct.json
Accuracy: 0.5860331173506119
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Pro-SFT-200K-v0.1.json
Accuracy: 0.5143988480921526
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Pro-SFT-100K-v0.1.json
Accuracy: 0.5255579553635709
---------------------------------
File: result_dirs/mmlu-redux/cot=false/Llama-3-8B-Magpie-Air-SFT-v0.1.json
Accuracy: 0.5233981281497481
---------------------------------
"""

example_prediction_item = """
{
    "session_id": "mmlu-redux-anatomy-#0",
    "chat_history": [
      "## Instruction \n\nYou will be presented with a question with multiple choices. \nYour task is to read the question carefully, consider the provided options, and select the one answer that you believe is the best choice. \n\n\n## Formatting \n\nPlease output both the letter and the answer text of the selected choice. \nFor example, if you believe the answer is \"(C) candles\", you should finally output \"### Answer: (C) candles\".\n\n## Question\n\nWhich of the following is a disorder characterized by uncontrollable episodes of falling asleep during the day?\n\n### Choices\n\n- (A) Dyslexia\n- (B) Epilepsy\n- (C) Hydrocephalus\n- (D) Narcolepsy\n\n"
    ],
    "model_input": "n/a",
    "output": [
      "### Answer: (D) Narcolepsy"
    ],
    "generator": "openai/gpt-3.5-turbo-0125",
    "configs": {
      "engine": "openai",
      "repetition_penalty": 1,
      "temperature": 0.0,
      "top_p": 1.0,
      "max_tokens": 4096
    },
    "dataset": "mmlu-redux",
    "id": "mmlu-redux-anatomy-#0",
    "question": "Which of the following is a disorder characterized by uncontrollable episodes of falling asleep during the day?",
    "choices": [
      "Dyslexia",
      "Epilepsy",
      "Hydrocephalus",
      "Narcolepsy"
    ],
    "correct_answer": "Narcolepsy",
    "source": "edinburgh-dawg/mmlu-redux",
    "config": "anatomy",
    "task_type": "multiple_choice"
  }
  
"""
