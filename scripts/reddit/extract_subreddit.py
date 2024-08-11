"""
Adapted from DCLM's code for extracting subreddit dumps from Academic Torrents.

This script extracts a compressed .zst dump from Academic Torrents and outputs the uncompressed contents
as shards of .jsonl files. We adapt it to work with the marin filesystem in Google cloud storage.

You can obtain the initial .zst dump by using Academic Torrents with the transmission client
and selecting the specific subreddit dumps you want. In this case, we select the explainlikeimfive subreddit.
"""

import fsspec
import zstandard
import os
import json
from datetime import datetime
import logging.handlers
import argparse

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line, file_handle.tell()
            buffer = lines[-1]
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", type=str, default='explainlikeimfive', help="Subreddit to process")
    parser.add_argument("--input_parent_dir", help="path to the data directory containing all subreddits", default="~/reddit/subreddits23") 
    parser.add_argument("--shard_size", type=int, help="The number of documents in each shard", default=100000)
    parser.add_argument("--output_dir", help="The directory to write the output", default="gs://marin-us-central2/raw")

    args = parser.parse_args()

    parent_dir = args.input_parent_dir
    subreddit = args.subreddit
    shard_size = args.shard_size

    for dump_type in ['submissions', 'comments']:
        file_path = os.path.join(os.path.expanduser(parent_dir), f"{subreddit}_{dump_type}.zst")
        file_size = os.stat(file_path).st_size
        
        file_lines = 0
        file_bytes_processed = 0
        created = None
        bad_lines = 0
        shard_num = 0
        lines = []

        output_dir = os.path.join(args.output_dir, subreddit, dump_type)
        os.makedirs(output_dir, exist_ok=True)
        for line, file_bytes_processed in read_lines_zst(file_path):
            try:
                obj = json.loads(line)
                created = datetime.utcfromtimestamp(int(obj['created_utc']))
            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1
            file_lines += 1
            lines.append(line)
            if shard_size and file_lines % shard_size == 0:
                log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")
                output_filepath = os.path.join(output_dir, f"{dump_type}_shard{shard_num}.jsonl")
                with fsspec.open(output_filepath, 'w') as f:
                    for line in lines:
                        f.write(line+"\n")
                lines = []
                shard_num += 1
                print(f"Wrote {output_dir}/{dump_type}_shard{shard_num-1}.jsonl")
            
        # Write the remaining shard
        output_filepath = os.path.join(output_dir, f"{dump_type}_shard{shard_num}.jsonl")
        with fsspec.open(output_filepath, 'w') as f:
            for line in lines:
                f.write(line+"\n")

        log.info(f"Complete : {file_lines:,} : {bad_lines:,}")
