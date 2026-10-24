from eval.chat_benchmarks.LiveBench.livebench.lcb_runner.lm_styles import LMStyle


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        if model_output[0] == "`" and model_output[-1] == "`":
            return model_output[1:-1]
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
