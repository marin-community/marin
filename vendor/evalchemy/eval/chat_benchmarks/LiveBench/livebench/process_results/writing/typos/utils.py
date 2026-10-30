import difflib
import re


def extract_answer(llm_answer):
    pattern = r".* --- (.*?) --- .*"
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer


def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    llm_answer = " ".join(list(filter(None, llm_answer.split("\n"))))

    llm_answer = extract_answer(llm_answer)

    if debug and ground_truth not in llm_answer:
        a = ground_truth
        b = llm_answer
        m = difflib.SequenceMatcher(a=a, b=b)
        pad = 10

        for tag, i1, i2, j1, j2 in m.get_opcodes():
            length = min(len(llm_answer), len(ground_truth))
            mi1, mi2, mj1, mj2 = max(0, i1 - pad), min(length, i2 + pad), max(0, j1 - pad), min(length, j2 + pad)
            if tag == "replace":
                print("<changed>", a[i1:i2], b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
            if tag == "delete":
                print("<deleted>", a[i1:i2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])
            if tag == "insert":
                print("<inserted>", b[j1:j2], "::::", a[mi1:mi2], "-->", b[mj1:mj2])

    if debug and not int(ground_truth in llm_answer):
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth)
        print("SOLUTION", llm_answer)

    return int(ground_truth in llm_answer)
