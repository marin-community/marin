from datasets import load_dataset
import json


validation = load_dataset("cais/mmlu", "all", split="validation")
auxiliary_train = load_dataset("cais/mmlu", "all", split="auxiliary_train")

for data in validation:
    question = data["question"]
    choices = data["choices"]
    input_format = f"Q: {question.strip()}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\nA:"
    answer = data["answer"]
    output_format = ['(A)', '(B)', '(C)', '(D)'][answer]
    subject = data["subject"]
    with open(f"mmlu_validation_{subject}.jsonl", "a") as f:
        f.write(json.dumps({"input": input_format, "output": output_format}) + "\n")
    

for data in auxiliary_train:
    question = data["question"]
    choices = data["choices"]
    input_format = f"Q: {question.strip()}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\nA:"
    answer = data["answer"]
    output_format = ['(A)', '(B)', '(C)', '(D)'][answer]
    subject = "auxiliary_train"
    with open(f"mmlu_validation_{subject}.jsonl", "a") as f:
        f.write(json.dumps({"input": input_format, "output": output_format}) + "\n")
        
    
    