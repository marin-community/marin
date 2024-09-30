"""
Adapted from DCLM's code that extracts subreddit dumps from Academic Torrents.

This script accepts a directory containing all of the subreddit dumps with an example directory structure:
reddit/
│
├── subreddits23/
│   ├── explainlikeimfive_comments.zst # subreddit comments
│   ├── explainlikeimfive_submissions.zst # subreddit submissions
|   |── {subreddit}_{dump)_type}.zst

The script then extracts each of the .zst reddit dumps, decodes the file, and outputs a jsonl file for each subreddit.

You can obtain the initial .zst dump by using Academic Torrents with the transmission client
and selecting the specific subreddit dumps you want. In this case, we select the explainlikeimfive subreddit.
"""

import json
import logging.handlers
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import draccus
import fsspec
import zstandard

from marin.utils import fsspec_mkdirs

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


@dataclass
class ExtractSubredditConfig:
    """Configuration class for extracting subreddit jsonl files from Academic Torrents' .zst reddit dumps.

    Attributes:
        subreddit (str): The subreddit to extract (e.g. "explainlikeimfive")
        input_parent_dir (str): The path to the directory containing the subreddit dumps (e.g. "~/reddit/subreddits23")
        shard_size (int): The number of documents in each shard
        output_dir (str): The directory to write the output which could be a local or remote directory
    """

    subreddit: str = "explainlikeimfive"
    input_parent_dir: str = "~/reddit/subreddits23"
    shard_size: int = 100000
    output_dir: str = "gs://marin-us-central2/raw"


def read_and_decode(
    reader: zstandard.ZstdDecompressor,
    chunk_size: int,
    max_window_size: int,
    previous_chunk: str | None = None,
    bytes_read: int = 0,
) -> str:
    """Read and decode a chunk of the zst file.

    Args:
        reader (zstandard.ZstdDecompressor): The zstandard decompressor.
        chunk_size (int): The size of the chunk to read.
        max_window_size (int): The maximum window size.
        previous_chunk (str | None): The previous chunk to decode.
        bytes_read (int): The number of bytes read.

    Returns:
        str: The decoded chunk.
    """
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError as e:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes") from e
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name: str) -> Iterator[tuple[str, int]]:
    """
    Read lines from a zst file.

    Args:
        file_name (str): The path to the zst file.

    Returns:
        Iterator[tuple[str, int]]: An iterator that yields a tuple of the line and the number of bytes read.
    """
    with open(file_name, "rb") as file_handle:
        buffer = ""
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


@draccus.wrap()
def main(cfg: ExtractSubredditConfig):
    parent_dir = cfg.input_parent_dir
    subreddit = cfg.subreddit
    shard_size = cfg.shard_size

    for dump_type in ["submissions", "comments"]:
        file_path = os.path.join(os.path.expanduser(parent_dir), f"{subreddit}_{dump_type}.zst")
        file_size = os.stat(file_path).st_size

        file_lines = 0
        file_bytes_processed = 0
        created = None
        bad_lines = 0
        shard_num = 0
        lines = []

        output_dir = os.path.join(cfg.output_dir, subreddit, dump_type)
        fsspec_mkdirs(output_dir, exist_ok=True)
        for line, file_bytes_processed in read_lines_zst(file_path):
            try:
                obj = json.loads(line)
                created = datetime.utcfromtimestamp(int(obj["created_utc"]))
            except (KeyError, json.JSONDecodeError):
                bad_lines += 1
            file_lines += 1
            lines.append(line)
            if shard_size and file_lines % shard_size == 0:
                log.info(
                    f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : \
                        {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%"
                )
                output_filepath = os.path.join(output_dir, f"{dump_type}_shard{shard_num}.jsonl")
                with fsspec.open(output_filepath, "w") as f:
                    for line in lines:
                        f.write(line + "\n")
                lines = []
                shard_num += 1
                print(f"Wrote {output_dir}/{dump_type}_shard{shard_num-1}.jsonl")

        # Write the remaining shard
        output_filepath = os.path.join(output_dir, f"{dump_type}_shard{shard_num}.jsonl")
        with fsspec.open(output_filepath, "w") as f:
            for line in lines:
                f.write(line + "\n")

        log.info(f"Complete : {file_lines:,} : {bad_lines:,}")


if __name__ == "__main__":
    main()
