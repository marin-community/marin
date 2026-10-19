import pandas as pd

df = pd.read_json(
    "LiveBench/question_generation/instruction_following/instruction_following_eval/data/input_data.jsonl", lines=True
)

# changing the column name and format a little to match the expected of livebench model response generation script
df.rename(columns={"prompt": "turns"}, inplace=True)
df.rename(columns={"key": "question_id"}, inplace=True)
df["turns"] = df["turns"].apply(lambda x: [x])

# adding column "category" with value "instruction_following"
df["category"] = "instruction_following"

# from IPython import embed; embed()

df.to_json("LiveBench/livebench/data/live_bench/instruction_following/question.jsonl", orient="records", lines=True)
