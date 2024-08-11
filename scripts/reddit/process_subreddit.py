"""
Adapted from DCLM's code for filtering subreddit dumps.

This script takes the unfiltered subreddit shards processed from extract_subreddit.py and 
filters out posts that don't meet the minimum comment and submission score thresholds.

We adapt this to work with the marin filesystem in Google cloud storage and with Ray
because this is quite memory-intensive.

Usage:
python scripts/reddit/process_subreddit.py
"""
import fsspec
import glob
import ray
import re
from tqdm import tqdm
import os
import json
import pandas as pd
import argparse
import subprocess

def remote_glob(input_dir):
    file_path = os.path.join(input_dir, "*.jsonl")
        
     # Use fsspec to get a list of files
    fs = fsspec.core.url_to_fs(file_path)[0]
    protocol = fsspec.core.split_protocol(file_path)[0]

    def join_protocol(file):
        if protocol:
            return f"{protocol}://{file}"
        return file

    return [join_protocol(file) for file in fs.glob(file_path)]
    
def read_df_from_shards(shard_dir):
    shards = []
    for f in tqdm(remote_glob(shard_dir)):
        df = pd.read_json(f, lines=True)
        shards.append(df)
    df = pd.concat(shards)
    return df

def write_jsonl(lines, file):
    with fsspec.open(file, 'w') as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

def get_url_regex(args):
    from retrie.retrie import Blacklist

    # Run the subprocess and wait for it to finish
    tlds_filepath = os.path.expanduser(args.tlds_filepath)
    subprocess.run(["wget", "-O", tlds_filepath, "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"], check=True)

    # Get rid of standalone urls with a regex based on top-level domains (taken from IANA)
    with open(tlds_filepath, "r") as file:
        tlds_list = [re.escape(tld) for tld in file.read().splitlines()]
    tlds_regex = Blacklist(tlds_list, match_substrings=True).compiled
    url_regex = re.compile(rf'\s{{0,10}}(?:((https?|ftp)://))?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}\.({tlds_regex.pattern})\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
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
    "title", "selftext", "score", "ups", "downs",
    "url", "subreddit_id", "subreddit", "media", "over_18",
    "is_self", "Locked", "created_utc", "gilded", "distinguished",
    "num_comments", "name", "id", "is_original_content", "is_video", "selftext_html"
    ] 
    COM_COLUMNS_TO_KEEP = ["parent_id", "id", "body", "score"]
    return SUB_COLUMNS_TO_KEEP, COM_COLUMNS_TO_KEEP

@ray.remote(memory=200 * 1024 * 1024 * 1024, runtime_env= {"pip": ["retrie"]})
def process_subreddit(args):
    eli5_pattern, removed_pattern, gt_pattern, hyperlink_pattern = get_regex_patterns_to_remove()
    sub_columns_to_keep, com_columns_to_keep = get_columns_to_keep()
    url_regex = get_url_regex(args)

    # Read single dfs for comments and submissions and keep only the necessary columns
    print("Reading in data")
    com_df = read_df_from_shards(os.path.join(args.input_parent_dir, args.subreddit, "comments"))
    sub_df = read_df_from_shards(os.path.join(args.input_parent_dir, args.subreddit, "submissions"))

    sub_df = sub_df[[c for c in sub_df.columns if c in sub_columns_to_keep]]
    com_df = com_df[[c for c in com_df.columns if c in com_columns_to_keep]]

    # Group comments by submission id and merge with submissions
    print("Merging submissions and comments")
    com_df = com_df.groupby('parent_id').agg({'id': lambda x: list(x), 'body':lambda x: list(x), 'score': lambda x: list(x)})
    merged_df = sub_df.merge(com_df, how='left', left_on='name', right_on='parent_id', suffixes=(None, '_comments'))
    original_len = len(merged_df)
    del com_df
    del sub_df

    # Drop rows with no found comments
    merged_df = merged_df[~merged_df['body'].isna()].reset_index(drop=True)

    print("Processing the merged dataframe")
    os.makedirs(os.path.join(args.output_dir, f"{args.subreddit}/"), exist_ok=True)
    modified_lines = []
    shard_num = 0
    kept_count = 0
    print(f"Processing {len(merged_df)} submissions from {original_len} original submissions.")

    for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        title = eli5_pattern.sub("", row['title'])
        selftext = removed_pattern.sub("", row['selftext'])
        comments = [{'text': row['body'][j] , 'score': row['score_comments'][j]} for j in range(len(row['body']))]
        
        # Filter out based on number of comments for a given submission
        if len(comments) < args.min_comments:
            continue
        
        # Find the best comment based on score and length as tiebreaker. Filter based on submission / comment scores
        comments.sort(key=lambda x: x.get('score', 0), reverse=True)
        best_comment, best_score = comments[0]['text'], comments[0]['score']
        if best_score < args.min_comment_score or row['score'] < args.min_submission_score:
            continue
        
        for c in comments:
            if c['score'] < best_score:
                break
            if len(c['text']) > len(best_comment):
                best_comment = c['text']
        
        # Get rid of the [removed] / [deleted] pattern and check if the comment is long enough
        best_comment = removed_pattern.sub("", best_comment)
        if len(best_comment.strip()) < args.min_comment_length:
            continue

        text = title
        if selftext:
            text += " " + selftext
        text += "\n\n" + best_comment
        text = gt_pattern.sub("", text)
        text = hyperlink_pattern.sub(r"\1", text)
        text = url_regex.sub("", text)
        text = text.strip()

        row['text'] = text
        modified_lines.append(dict(row))
        kept_count += 1
        
        if (args.shard_size and len(modified_lines) == args.shard_size):
            write_jsonl(modified_lines, os.path.join(args.output_dir, f"{args.subreddit}/{args.subreddit}_shard{shard_num}.jsonl"))
            shard_num += 1
            modified_lines = []

    # Write the last shard
    write_jsonl(modified_lines, os.path.join(args.output_dir, f"{args.subreddit}/{args.subreddit}_shard{shard_num}.jsonl"))
    print(f"Kept {kept_count} pages out of {original_len} original pages")


def main(args):
    ray.init()

    response = process_subreddit.remote(args)

    try:
        ray.get(response)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parent_dir", type=str, default="gs://marin-us-central2/raw/", help="Path to the data directory containing all subreddits")
    parser.add_argument("--subreddit", type=str, default='explainlikeimfive', help="Subreddit to process")
    parser.add_argument("--output_dir", type=str, default="gs://marin-us-central2/documents/dclm/", help="The directory to write the) output")
    parser.add_argument("--tlds_filepath", type=str, default="~/iana_tlds.txt", help="The file containing the top-level domains")
    parser.add_argument("--shard_size", type=int, default=None, help="Number of documents in each shard")
    parser.add_argument("--min_comments", type=int, default=3, help="Minimum number of comments to consider a post")
    parser.add_argument("--min_comment_score", type=int, default=5, help="Minimum score of the best comment to consider a post")
    parser.add_argument("--min_submission_score", type=int, default=0, help="Minimum score of the best comment to consider a post")
    parser.add_argument("--min_comment_length", type=int, default=10, help="Minimum length of the best comment to consider a post")
    args = parser.parse_args()

    main(args)