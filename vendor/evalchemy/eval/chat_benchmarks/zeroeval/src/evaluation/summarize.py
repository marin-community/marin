import pandas as pd
from eval_utils import model_name_replacement, total_num_examples

# Load the JSON files
gsm_file = "result_dirs/gsm.summary.json"
math_file = "result_dirs/math-l5.summary.json"
mmlu_file = "result_dirs/mmlu-redux.summary.json"
zebra_file = "result_dirs/zebra-grid.summary.json"
crux_file = "result_dirs/crux.summary.json"


gsm_data = pd.read_json(gsm_file)
math_data = pd.read_json(math_file)
mmlu_data = pd.read_json(mmlu_file)
zebra_data = pd.read_json(zebra_file)
crux_data = pd.read_json(crux_file)


# only keep the models that have the same number of examples as the total_num_examples
def filter_models_by_num_examples(cur_data, total_num_examples, total_key="Total"):
    cur_data = cur_data[cur_data[total_key] == total_num_examples]
    return cur_data


gsm_data = filter_models_by_num_examples(gsm_data, total_num_examples["gsm"])
mmlu_data = filter_models_by_num_examples(mmlu_data, total_num_examples["mmlu-redux"])
zebra_data = filter_models_by_num_examples(zebra_data, total_num_examples["zebra-grid"], "Total Puzzles")
crux_data = filter_models_by_num_examples(crux_data, total_num_examples["crux"])
math_data = filter_models_by_num_examples(math_data, total_num_examples["math-l5"])


# replace the value from "gemma-2-9b-it@nvidia" to "gemma-2-9b-it" for all data. only when the model name is "gemma-2-9b-it@nvidia"
def replace_model_names(cur_data):
    # remove the data for model Qwenya-2-9b-it@nvidia
    cur_data = cur_data[cur_data["Model"] != "Qwen2.5-32B-Instruct"]
    cur_data["Model"] = model_name_replacement(cur_data["Model"])
    return cur_data


gsm_data = replace_model_names(gsm_data)
mmlu_data = replace_model_names(mmlu_data)
zebra_data = replace_model_names(zebra_data)
crux_data = replace_model_names(crux_data)
math_data = replace_model_names(math_data)


# Add suffixes to the columns
gsm_data = gsm_data[["Model", "Acc"]]
gsm_data = gsm_data.add_suffix("_gsm")
gsm_data = gsm_data.rename(columns={"Model_gsm": "Model"}).rename(columns={"Acc_gsm": "GSM8K"})

mmlu_data = mmlu_data[["Model", "Acc"]]
mmlu_data = mmlu_data.add_suffix("_mmlu")
mmlu_data = mmlu_data.rename(columns={"Model_mmlu": "Model"}).rename(columns={"Acc_mmlu": "MMLU<br/>-Redux"})

# remove the rows when the mode = sampling
zebra_data = zebra_data[zebra_data["Mode"] != "sampling"]
# zebra_data = zebra_data[['Model', "Easy Puzzle Acc"]] #, 'Puzzle Acc']] #  'Puzzle Acc',  # Easy
zebra_data = zebra_data[["Model", "Puzzle Acc"]]  # , 'Puzzle Acc']] #  'Puzzle Acc',  # Easy
zebra_data = zebra_data.add_suffix("_zebra")
zebra_data = (
    zebra_data.rename(columns={"Model_zebra": "Model"})
    .rename(columns={"Puzzle Acc_zebra": "ZebraLogic"})
    .rename(columns={"Easy Puzzle Acc_zebra": "ZebraLogic<br/>-Easy"})
    .rename(columns={"Puzzle Acc_zebra": "ZebraLogic<br/>-Full"})
)


crux_data = crux_data[["Model", "Acc"]]
crux_data = crux_data.add_suffix("_crux")
crux_data = crux_data.rename(columns={"Model_crux": "Model"}).rename(columns={"Acc_crux": "CRUX"})

math_data = math_data[["Model", "Acc"]]
math_data = math_data.add_suffix("_math")
math_data = math_data.rename(columns={"Model_math": "Model"}).rename(columns={"Acc_math": "MATH<br/>-L5"})


# Generate the summary of the results (merging and only keep the models that have all results on MMLU, ZEBRA, CRUX, MATH)
def generate_summary():
    # Merge the dataframes on the "Model" column
    # merged_data = pd.merge(gsm_data, mmlu_data, on='Model')
    merged_data = pd.merge(mmlu_data, zebra_data, on="Model")
    merged_data = pd.merge(merged_data, crux_data, on="Model")
    merged_data = pd.merge(merged_data, math_data, on="Model")

    # add a final column to do average of the scores except for Model name
    merged_data["Average"] = merged_data.drop(columns=["Model"]).mean(axis=1)

    # rank the models based on the average score
    merged_data = merged_data.sort_values(by="Average", ascending=False)

    # Generate a Markdown table
    markdown_table = merged_data.to_markdown(index=False, floatfmt=".2f")

    # rename the <br> in column names with "\n"
    # merged_data.columns = merged_data.columns.str.replace('<br/>', '\n')
    markdown_table_lite = merged_data.to_markdown(index=False, floatfmt=".2f", tablefmt="grid")
    print(markdown_table_lite)

    # print(markdown_table)
    with open("result_dirs/summary.md", "w") as f:
        f.write(markdown_table)

    # save the json output
    # merged_data.columns = merged_data.columns.str.replace('<br/>', '\n')
    with open("result_dirs/summary.json", "w") as f:
        f.write(merged_data.to_json(orient="records", lines=False))


# Generate the full summary of the results (merging and leave NaN for the models that do not have all results on MMLU, ZEBRA, CRUX, MATH)
def generate_full():
    # Merge the dataframes on the "Model" column
    merged_data = pd.merge(gsm_data, mmlu_data, on="Model", how="outer")
    merged_data = pd.merge(merged_data, zebra_data, on="Model", how="outer")
    merged_data = pd.merge(merged_data, crux_data, on="Model", how="outer")
    merged_data = pd.merge(merged_data, math_data, on="Model", how="outer")

    # sort by the average,  just for sorting purposes
    merged_data["Average"] = merged_data.drop(columns=["Model"]).mean(axis=1)
    merged_data = merged_data.sort_values(by="Average", ascending=False)
    merged_data = merged_data.drop(columns=["Average"])

    # Generate a Markdown table
    markdown_table = merged_data.to_markdown(index=False, floatfmt=".2f")

    # rename the <br> in column names with "\n"
    # merged_data.columns = merged_data.columns.str.replace('<br/>', '\n')
    markdown_table_lite = merged_data.to_markdown(index=False, floatfmt=".2f", tablefmt="grid")
    print(markdown_table_lite)

    # print(markdown_table)
    with open("result_dirs/summary_full.md", "w") as f:
        f.write(markdown_table)

    # save the json output
    # merged_data.columns = merged_data.columns.str.replace('<br/>', '\n')
    with open("result_dirs/summary_full.json", "w") as f:
        f.write(merged_data.to_json(orient="records", lines=False))


generate_summary()
generate_full()
