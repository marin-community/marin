from datasets import load_dataset
from ._TEMPLATES import apply_mc_template, apply_lgp_grid_template, apply_oeqa_template


def mapping_task_names(data_name):
    """
    Mapping the task names to the dataset and id name.
    """
    id_name = "id"
    if data_name == "mmlu-redux":
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
    elif data_name == "gsm":
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
    elif data_name == "zebra-grid":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    elif data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    elif data_name == "numersense-v2":
        dataset = load_dataset("yuchenlin/zero-eval", "numersense-v2", split="test")
    elif data_name == "crux":
        dataset = load_dataset("flydust/zero-eval", "crux", split="test")
    elif data_name == "math-l5":
        dataset = load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return dataset, id_name


def prompt_generation(data_name, data_item, args):
    """
    Generate prompt for different tasks.
    """
    if data_name in ["mmlu-redux"]:  # and other multiple-choice QA dataset
        prompt = apply_mc_template(data_item)
    elif data_name in ["alpaca_eval"]:
        prompt = data_item["instruction"]
    elif data_name in ["zebra-grid"]:
        prompt = apply_lgp_grid_template(data_item)
    elif data_name in ["gsm", "math-l5"]:
        question_key = "question"
        if data_name == "math-l5":
            question_key = "problem"
        prompt = apply_oeqa_template(data_item, question_key=question_key)
    elif data_name in ["crux"]:
        prompt = apply_oeqa_template(data_item, cot=True)  # cot?
    elif data_name in ["numersense-v2"]:
        if "no_cot" in args.run_name:
            prompt = apply_oeqa_template(data_item, cot=False)
        prompt = apply_oeqa_template(data_item)
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return prompt


def result_format(output_item, args):
    """
    Modify the output format for different tasks if needed.
    """
    if args.data_name in ["alpaca_eval"]:
        output_item["output"] = output_item["output"][0]  # use str instead of list
    elif args.data_name in ["zebra-grid"]:
        del output_item["solution"]
    else:
        pass
    return output_item
