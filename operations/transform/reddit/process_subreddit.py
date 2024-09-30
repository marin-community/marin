"""
Adapted from DCLM's code for filtering subreddit dumps for desired columns and removing low-quality posts.

This script takes the unfiltered subreddit shards processed from extract_subreddit.py and
filters out posts that don't meet the minimum comment and submission score thresholds. It then uploads these
processed jsonl files to the specified output directory.

We adapt the code to use Ray because it is very memory-intensive;
it must fit the entire uncompressed jsonl files in memory.
We set some reasonable defaults for the minimum comment and submission scores
using what was used in the original DCLM paper.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass

import draccus
import fsspec
import pandas as pd
import ray
from tqdm import tqdm

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


@dataclass
class ProcessSubredditConfig:
    """Configuration class for processing and filtering subreddit jsonl files after the extraction step.

    Attributes:
        input_parent_dir (str): The path to the directory containing the jsonl files from the previous extraction step.
                                In other words, it is the output dir of the extract_subreddit.py script.
        subreddit (str): The subreddit to process (e.g. "explainlikeimfive")
        output_dir (str): The directory to write the output files which could be a local or remote directory
        shard_size (int): The number of documents in each shard
        min_comments (int): The minimum number of comments a post must have to be included
        min_comment_score (int): The minimum score a comment must have to be included
        min_submission_score (int): The minimum score a submission must have to be included
        min_comment_length (int): The minimum length a comment must have to be included
    """

    input_parent_dir: str
    subreddit: str = "explainlikeimfive"
    output_dir: str
    shard_size: int | None = 5000
    min_comments: int = 3
    min_comment_score: int = 5
    min_submission_score: int = 0
    min_comment_length: int = 10


def remote_glob(input_dir: str) -> list[str]:
    file_path = os.path.join(input_dir, "*.jsonl")

    # Use fsspec to get a list of files
    fs = fsspec.core.url_to_fs(file_path)[0]
    protocol = fsspec.core.split_protocol(file_path)[0]

    def join_protocol(file):
        if protocol:
            return f"{protocol}://{file}"
        return file

    return [join_protocol(file) for file in fs.glob(file_path)]


def read_df_from_shards(shard_dir: str) -> pd.DataFrame:
    shards = []
    for f in tqdm(remote_glob(shard_dir)):
        df = pd.read_json(f, lines=True)
        shards.append(df)
    df = pd.concat(shards)
    return df


def write_jsonl(lines: list[dict], file: str):
    with fsspec.open(file, "w", compression="gzip") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def get_url_regex():
    # Run the subprocess and wait for it to finish
    import tempfile

    from retrie.retrie import Blacklist

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_file:
        subprocess.run(
            [
                "wget",
                "-O",
                tmp_file.name,
                "https://raw.githubusercontent.com/mlfoundations/dclm/main/baselines/mappers/iana_tlds.txt",
            ],
            check=True,
        )

        # Get rid of standalone urls with a regex based on top-level domains (taken from IANA)
        tmp_file.seek(0)
        tlds_list = [re.escape(tld) for tld in tmp_file.read().splitlines()]
    tlds_regex = Blacklist(tlds_list, match_substrings=True).compiled
    url_regex = re.compile(
        rf"\s{{0,10}}(?:((https?|ftp)://))?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}\.({tlds_regex.pattern})\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    )
    return url_regex


def get_regex_patterns_to_remove():
    # Removes the eli5 pattern that leads many titles
    eli5_pattern = re.compile(r"eli5:?\s*", re.IGNORECASE)

    # Get rid of [removed] or [deleted] for many selftexts
    removed_pattern = re.compile(r"(\[removed\]|\[deleted\])")

    # Get rid of common formatting for quotes in these reddit dumps
    gt_pattern = re.compile(r"=?&gt;?")

    # Get rid of links and replace with the text when they appear
    hyperlink_pattern = re.compile(r"\[(.*?)\]\s*(\(http.*?\))")

    return eli5_pattern, removed_pattern, gt_pattern, hyperlink_pattern


def get_columns_to_keep():
    SUB_COLUMNS_TO_KEEP = [
        "title",
        "selftext",
        "score",
        "ups",
        "downs",
        "url",
        "subreddit_id",
        "subreddit",
        "media",
        "over_18",
        "is_self",
        "Locked",
        "created_utc",
        "gilded",
        "distinguished",
        "num_comments",
        "name",
        "id",
        "is_original_content",
        "is_video",
        "selftext_html",
    ]
    COM_COLUMNS_TO_KEEP = ["parent_id", "id", "body", "score"]
    return SUB_COLUMNS_TO_KEEP, COM_COLUMNS_TO_KEEP


def get_dolma_formatted_row(row: dict):
    return {
        "id": row["id"],
        "text": row["text"],
        "created": row["created_utc"],
        "source": "reddit",
        "metadata": {
            "subreddit": row["subreddit"],
            "score": row["score"],
            "ups": row["ups"],
            "downs": row["downs"],
        },
    }


@ray.remote(memory=200 * 1024 * 1024 * 1024, runtime_env={"pip": ["retrie"]})
def process_subreddit(cfg: ProcessSubredditConfig):
    eli5_pattern, removed_pattern, gt_pattern, hyperlink_pattern = get_regex_patterns_to_remove()
    sub_columns_to_keep, com_columns_to_keep = get_columns_to_keep()
    url_regex = get_url_regex()

    # Read single dfs for comments and submissions and keep only the necessary columns
    log.info("Reading in data")
    com_df = read_df_from_shards(os.path.join(cfg.input_parent_dir, cfg.subreddit, "comments"))
    sub_df = read_df_from_shards(os.path.join(cfg.input_parent_dir, cfg.subreddit, "submissions"))

    sub_df = sub_df[[c for c in sub_df.columns if c in sub_columns_to_keep]]
    com_df = com_df[[c for c in com_df.columns if c in com_columns_to_keep]]

    # Group comments by submission id and merge with submissions
    log.info("Merging submissions and comments")
    com_df = com_df.groupby("parent_id").agg(
        {"id": lambda x: list(x), "body": lambda x: list(x), "score": lambda x: list(x)}
    )
    merged_df = sub_df.merge(com_df, how="left", left_on="name", right_on="parent_id", suffixes=(None, "_comments"))
    original_len = len(merged_df)
    del com_df
    del sub_df

    # Drop rows with no found comments
    merged_df = merged_df[~merged_df["body"].isna()].reset_index(drop=True)

    log.info("Processing the merged dataframe")
    os.makedirs(os.path.join(cfg.output_dir, f"{cfg.subreddit}/"), exist_ok=True)
    modified_lines = []
    shard_num = 0
    kept_count = 0
    log.info(f"Processing {len(merged_df)} submissions from {original_len} original submissions.")

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        title = eli5_pattern.sub("", row["title"])
        selftext = removed_pattern.sub("", row["selftext"])
        comments = [{"text": row["body"][j], "score": row["score_comments"][j]} for j in range(len(row["body"]))]

        # Filter out based on number of comments for a given submission
        if len(comments) < cfg.min_comments:
            continue

        # Find the best comment based on score and length as tiebreaker. Filter based on submission / comment scores
        comments.sort(key=lambda x: x.get("score", 0), reverse=True)
        best_comment, best_score = comments[0]["text"], comments[0]["score"]
        if best_score < cfg.min_comment_score or row["score"] < cfg.min_submission_score:
            continue

        for c in comments:
            if c["score"] < best_score:
                break
            if len(c["text"]) > len(best_comment):
                best_comment = c["text"]

        # Get rid of the [removed] / [deleted] pattern and check if the comment is long enough
        best_comment = removed_pattern.sub("", best_comment)
        if len(best_comment.strip()) < cfg.min_comment_length:
            continue

        text = title
        if selftext:
            text += " " + selftext
        text += "\n\n" + best_comment
        text = gt_pattern.sub("", text)
        text = hyperlink_pattern.sub(r"\1", text)
        text = url_regex.sub("", text)
        text = text.strip()

        row["text"] = text
        dolma_formatted_row = get_dolma_formatted_row(row)
        modified_lines.append(dolma_formatted_row)
        kept_count += 1

        if cfg.shard_size and len(modified_lines) == cfg.shard_size:
            write_jsonl(
                modified_lines,
                os.path.join(cfg.output_dir, f"{cfg.subreddit}/{cfg.subreddit}_shard{shard_num}.jsonl.gz"),
            )
            shard_num += 1
            modified_lines = []

    # Write the last shard
    write_jsonl(
        modified_lines, os.path.join(cfg.output_dir, f"{cfg.subreddit}/{cfg.subreddit}_shard{shard_num}.jsonl.gz")
    )
    log.info(f"Kept {kept_count} pages out of {original_len} original pages")


@draccus.wrap()
def main(cfg: ProcessSubredditConfig):
    ray.init()

    response = process_subreddit.remote(cfg)

    try:
        ray.get(response)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
