'''Convert fineweb to markdown'''
import argparse
import glob
import logging
import os
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
import s3fs
from warcio.archiveiterator import ArchiveIterator

from markdown import to_markdown
from markweb.markweb import convert_page

# Initialize S3 file system
s3 = s3fs.S3FileSystem()
logging.basicConfig(level=logging.INFO)


def process_wrac_file(input_file, output_file):
    '''
    Takes in the input file and processes it to get the html and md content.
    It scans the s3 bucket in input_file and returns the content of the urls in the input_file
    Args:
    input_file (str): The input file to process
    output_file (str): The output file to write the processed content to
    '''

    df = pd.read_parquet(input_file)

    assert df["file_path"].unique().size == 1, "There should be only one file_path per group"

    df["md"] = None
    df["html"] = None

    urls = df["url"].tolist()
    s3_url = df["file_path"].iloc[0]
    url_idx = 0  # All urls are arrange in the same order in parquet as in the CC

    # Logging variables
    logging.info(f"Processing {s3_url}")
    length_url_inp_list = len(urls)
    length_warc = 0

    with s3.open(s3_url, 'rb') as f:
        for record in ArchiveIterator(f):
            # Check if it's a response record
            if record.rec_type == 'response':
                # Process the record
                url = record.rec_headers.get_header('WARC-Target-URI')
                length_warc += 1
                if length_warc % 1000 == 0: logging.info(f"Processed {length_warc} records")
                if url == urls[url_idx]:
                    url_idx += 1
                    try:
                        # TODO: Is it ok to ignore errors here? I got errors sometimes
                        # Read the response body
                        content = record.content_stream().read()
                        html_decoded = content.decode(errors='ignore')
                        markdown = convert_page(html_decoded, url)
                        df.iloc[url_idx]["html"] = html_decoded
                        df.iloc[url_idx]["md"] = markdown["content"].encode('utf-8', 'ignore').decode('utf-8')
                    except Exception as e:
                        logging.error(f"This is the process_wrac_file, will be ignored if it is a CSS stylesheet error")
                        if not "CSS stylesheet" in str(e):
                            # Ignore CSS stylesheet errors
                            logging.error(f"Error processing {url} in {s3_url}: {e}")

    logging.info(
        f"Processed {s3_url}, found {length_warc} records, {length_url_inp_list} urls, {length_warc}/{length_url_inp_list} ratio")
    df.to_parquet(output_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert fineweb to markdown.")
    parser.add_argument('input_path', type=str, help='Path to the fineweb parquet file, with a single warc')
    parser.add_argument('--remove_input', action='store_true', help="if we should remove the input file")
    args = parser.parse_args()

    args.output_path = ".".join(args.input_path.split(".")[:-1]) + "_markdownified.parquet"
    process_wrac_file(args.input_path, args.output_path)
    if args.remove_input:
        os.remove(args.input_path)
