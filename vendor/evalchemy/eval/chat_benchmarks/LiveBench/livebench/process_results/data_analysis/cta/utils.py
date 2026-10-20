import re
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w]", "", text)
    return text


def cta_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    parsed_answer = llm_answer

    if "\\boxed{" in parsed_answer:
        parsed_answer = last_boxed_only_string(parsed_answer)
        parsed_answer = remove_boxed(parsed_answer)
        parsed_answer = parsed_answer.replace("\\text{", "").replace("}", "").replace("\\", "")

    if clean_text(ground_truth) == clean_text(parsed_answer):
        return 1
    elif clean_text(ground_truth) == clean_text(parsed_answer)[-len(clean_text(ground_truth)) :]:
        return 1
    else:
        if debug:
            print("INCORRECT")
            print("GROUND TRUTH", ground_truth)
            print("SOLUTION", parsed_answer)
            if parsed_answer != llm_answer:
                print("END OF OUTPUT", llm_answer[-100:])

        return 0
