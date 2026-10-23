import json
import traceback
import pandas as pd
from io import StringIO
import re
import math
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


def read_df_func(df_type, df_str):
    if df_type == "json":
        try:
            return pd.read_json(StringIO(df_str), orient="index", encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="records", lines=False, encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="records", lines=True, encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="table", encoding="utf-8")
        except:
            pass
        return pd.read_json(StringIO(df_str), orient="values", encoding="utf-8")
    elif df_type == "jsonl":
        return pd.read_json(StringIO(df_str), orient="records", lines=True, encoding="utf-8")
    elif df_type == "html":
        return pd.concat(pd.read_html(StringIO(df_str), encoding="utf-8"), axis=0)
    elif df_type == "csv":
        return pd.read_csv(StringIO(df_str), encoding="utf-8")
    elif df_type == "markdown":
        return pd.read_table(StringIO(df_str), sep="|", header=0, index_col=1, skipinitialspace=True)
    elif df_type == "tsv":
        return pd.read_csv(StringIO(df_str), sep="\t", encoding="utf-8")


def clean_llm_output(s):
    pattern_json = r"```json\n(.*?)```"
    matches = re.findall(pattern_json, s, re.DOTALL)
    if len(matches) > 0:
        return matches[-1].strip()
    pattern_html = r"```html\n(.*?)```"
    matches = re.findall(pattern_html, s, re.DOTALL)
    if len(matches) > 0:
        return matches[-1].strip()
    pattern_a = r"^```.*\n"
    s = re.sub(pattern_a, "", s)
    # re.findall(pattern, text, re.MULTILINE)
    s = s.replace("&amp;", "&")
    return s.replace("```", "").strip()


def read_sep_table_from_text(text, header, sep=","):
    text = text.strip()
    # look for the first line that contains the header
    header_line = 0
    while header_line < len(text.split("\n")) and text.split("\n")[header_line].strip() != header.strip():
        header_line += 1
    if header_line == len(text.split("\n")) or text.split("\n")[header_line].strip() != header.strip():
        return None
    # read the table from the header index
    table = text.split("\n")[header_line:]
    # read the table as a csv
    parsed_table = None
    while parsed_table is None:
        try:
            parsed_table = pd.read_csv(StringIO("\n".join(table)), sep=sep)
        except:
            # in case there's extra text after the table
            table = table[:-1]
    return parsed_table


def read_jsonl_table_from_text(text, header):
    text = text.strip().split("\n")
    table = []
    for line in text:
        if len(line) < 2 or line[0] != "{" or line[-1] != "}":
            continue
        if not all(key in line for key in header):
            continue
        try:
            table.append(json.loads(line))
        except:
            continue
    if len(table) == 0:
        return None
    return pd.DataFrame(table)


def remove_initial_phrase(text):
    # remove phrases like "Here is the updated table:" , "Here is the table in a new format:"
    pattern = r"^\s*(Here|Input)\b.*?\b(format|table)\s*[:)]\s*"
    modified_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return modified_text.strip()


def table_process_results(input_command: str, ground_truth: str, llm_answer: str, debug=False) -> int:
    input_format = input_command.split("Please convert the Input Table from ")[1].split(" format")[0].lower()
    output_format = (
        input_command.split("Please convert the Input Table from ")[1]
        .split("format to ")[1]
        .split(" format")[0]
        .lower()
    )
    gt_df = read_df_func(output_format, ground_truth)
    try:
        gt_df = read_df_func(output_format, ground_truth)
    except:
        print("Error when reading the ground-truth table")
        return "err"
    llm_clean = clean_llm_output(llm_answer)
    # if there's an initial phrase before the table, remove it and try to score again
    llm_clean = remove_initial_phrase(llm_clean)
    # first check the raw LLM output
    llm_df = None
    try:
        llm_df = read_df_func(output_format, llm_clean)
    except:
        if output_format == "csv" or output_format == "tsv":
            header = (",", "\t")[output_format == "tsv"].join(gt_df.columns)
            llm_df = read_sep_table_from_text(llm_clean, header, sep="," if output_format == "csv" else "\t")
        elif output_format == "jsonl":
            llm_df = read_jsonl_table_from_text(llm_clean, gt_df.columns)
        if llm_df is None:
            print("Could not read the LLM output")
            print("GROUND TRUTH\n", ground_truth)
            print("END OF OUTPUT\n", llm_answer[-min(3000, len(llm_answer)) :])
            return 0
    score = check_table_reformat(output_format, llm_df, gt_df, debug)

    if score == 0:
        # try to read table directly from the text, in case the LLM added some text before/after the table
        if output_format == "csv":
            header = ",".join(gt_df.columns)
            llm_df = read_sep_table_from_text(llm_clean, header, sep="," if output_format == "csv" else "\t")

        elif output_format == "jsonl":
            llm_df = read_jsonl_table_from_text(llm_clean, gt_df.columns)

        if llm_df is not None:
            score = check_table_reformat(output_format, llm_df, gt_df, debug)
    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH\n", gt_df)
        print("LLM DF\n", llm_df)
        print("LLM ANSWER\n", llm_clean)
    return score


def check_table_reformat(output_format, llm_df, gt_df, debug=False):
    try:
        gt_df.columns = [s.strip() for s in gt_df.columns]
        llm_df.columns = [s.strip() for s in llm_df.columns]
        assert len(llm_df) == len(
            gt_df
        ), f"DataFrame Length does not match, {len(llm_df)} (LLM) vs {len(gt_df)} (Ground Truth)"
        assert list(sorted(llm_df.columns)) == list(
            sorted(gt_df.columns)
        ), f"Columns do not match:\n{sorted(llm_df.columns)} (LLM)\n{sorted(gt_df.columns)} (Ground Truth)"
        # for test_col in llm_df.columns:
        #     assert sorted(llm_df[test_col].tolist()) == sorted(gt_df[test_col].tolist()), f"Column content {test_col} does not match"
        for i in range(len(llm_df)):
            for key in llm_df.columns:
                llm_val = llm_df.iloc[i][key]
                gt_val = gt_df.iloc[i][key]
                if isinstance(llm_val, str):
                    llm_val = llm_val.strip()
                if isinstance(gt_val, str):
                    gt_val = gt_val.strip()

                if (isinstance(llm_val, float) or is_numeric_dtype(llm_val)) and (
                    isinstance(gt_val, float) or is_numeric_dtype(gt_val)
                ):
                    try:
                        llm_val = float(llm_val)
                        gt_val = float(gt_val)
                    except:
                        assert (
                            str(llm_val).strip() == str(gt_val).strip()
                        ), f"Mismatched types of values {llm_val} (LLM) vs {gt_val} (Ground Truth) for key {key} in row {i}"
                        continue
                    if math.isnan(llm_val) and math.isnan(gt_val):
                        continue
                    assert (
                        abs(llm_val - gt_val) < 1e-6
                    ), f"Unequal float values {llm_val} (LLM) vs {gt_val} (Ground Truth) for key {key} in row {i}"
                else:
                    assert (
                        llm_val == gt_val
                    ), f"Value {llm_val} (LLM) vs {gt_val} (Ground Truth) for key {key} in row {i}"
    except AssertionError as e:
        if debug:
            print(e)
        return 0
    except Exception as e:
        if debug:
            print(e)
            traceback.print_exc()
        return 0
    return 1
