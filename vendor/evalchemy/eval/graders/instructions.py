"""Per-prompt instruction checks with explicit instruction-count weights."""

from collections import defaultdict
from importlib import import_module

from eval.contracts.eval_spec import Grade


def instruction_following(spec, responses):
    protocol = spec.grader.reference["protocol"]
    package = {
        "ifbench": "eval.chat_benchmarks.IFBench.grader",
        "ifeval": "eval.chat_benchmarks.IFEval.evaluation_main",
    }[protocol]
    checker = import_module(package)
    doc = spec.doc
    example = checker.InputExample(
        key=doc["key"], instruction_id_list=doc["instruction_id_list"], prompt=doc["prompt"], kwargs=doc["kwargs"],
    )
    metrics, weights, audit = {}, {}, {}
    for mode in ("strict", "loose"):
        check = getattr(checker, f"test_instruction_following_{mode}")
        outcome = check(example, {doc["prompt"]: str(responses[-1])})
        flags = outcome.follow_instruction_list
        prompt_name = f"{mode}_prompt_accuracy" if protocol == "ifbench" else f"prompt_level_{mode}"
        instruction_name = f"{mode}_instruction_accuracy" if protocol == "ifbench" else f"instruction_level_{mode}"
        metrics[prompt_name] = float(all(flags))
        metrics[instruction_name] = sum(flags) / len(flags) if flags else 0.0
        weights[instruction_name] = len(flags)
        audit[f"{mode}_instruction_pass"] = flags
        if protocol == "ifeval" and mode == "loose":
            metrics["prompt-level"] = metrics[prompt_name]
            metrics["instruction-level"] = metrics[instruction_name]
            weights["instruction-level"] = len(flags)
        if protocol == "ifbench":
            grouped = defaultdict(list)
            for instruction, passed in zip(example.instruction_id_list, flags, strict=True):
                grouped[instruction.split(":")[0]].append(passed)
            for category, values in grouped.items():
                name = f"{mode}_per_type:{category}"
                metrics[name] = sum(values) / len(values)
                weights[name] = len(values)
    return Grade(metrics, audit, weights)
