import re
import numpy as np
from ast import literal_eval
import json
import ast
from eval.chat_benchmarks.LiveBench.livebench.process_results.util import last_boxed_only_string, remove_boxed


def clean_llm_output(s):
    try:
        match_d = literal_eval(s)
    except:
        matches = re.findall("%s(.*?)%s" % ("```python", "```"), s.replace("\n", ""), re.MULTILINE)
        if len(matches) == 0:
            matches = re.findall("%s(.*?)%s" % ("```", "```"), s.replace("\n", ""), re.MULTILINE)
        if len(matches) == 0:
            if "\\boxed" in s:
                boxed = last_boxed_only_string(s.replace("\n", ""))
                if boxed:
                    no_boxed = remove_boxed(boxed)
                    matches = [re.sub(r"\\text{[\'|\"](.*?)[\'|\"]}", r"'\1'", no_boxed).replace("\\", "")]
        if len(matches) == 0:
            matches = [s]
        if len(matches) >= 1:
            matches = matches[-1]
        matches = matches.replace("null", "None")
        try:
            match_d = literal_eval(matches)
        except:
            return {}
    if not isinstance(match_d, dict):
        return {}
    else:
        keys = list(match_d.keys())
        for k in keys:
            if match_d[k] is None:
                del match_d[k]
        return match_d


def joinmap_process_results(_, ground_truth, llm, debug=False):
    if type(ground_truth) != dict:
        ground_truth = ast.literal_eval(ground_truth)
    llm_clean = clean_llm_output(llm)
    if len(llm_clean) == 0:
        if debug:
            print("could not parse output")
            print("GROUND TRUTH", ground_truth)
            print("END OF OUTPUT", llm[-min(500, len(llm)) :])
        return 0.0
    tp = 0
    fp = 0
    fn = 0
    for k, v in llm_clean.items():
        gt = ground_truth.get(k, None)
        if not gt:
            fp += 1
        elif gt == v:
            tp += 1
        else:
            fp += 1
            fn += 1
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    for k, v in ground_truth.items():
        llm_resp = llm_clean.get(k, None)
        if not llm_resp:
            fn += 1
    result = np.round(((2 * tp) / ((2 * tp) + fp + fn)), 2)
    if debug and result < 1:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth)
        print("SOLUTION", llm_clean)
        print("END OF OUTPUT", llm[-min(500, len(llm)) :])
    return result
