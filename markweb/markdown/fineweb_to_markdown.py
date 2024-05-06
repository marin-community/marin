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

def process_wrac_file(s3_url, url_inp_list):
    """
    Takes in s3_url and returns the html content as well as md content.
    It searches for the url_inp_list in the s3_url file and returns the content of that url_list

    Parameters:
    s3_url (str): The S3 URL to process
    url_inp_list (list): List of URLs to search for in the s3_url file

    Returns:
    list: HTML content of the URLs found
    list: Markdown content of the URLs found
    """
    htmls_mds = []

    length_url_inp_list = len(url_inp_list)
    length_warc = 0
    logging.info(f"Processing {s3_url}")
    with s3.open(s3_url, 'rb') as f:
        # TODO: I think the code was a bit faster with this.
        # stream = gzip.GzipFile(fileobj=f)  # Decompress data with gzip
        for record in ArchiveIterator(f, block_size=1024*1024):
            # Check if it's a response record
            if record.rec_type == 'response':
                # Process the record
                url = record.rec_headers.get_header('WARC-Target-URI')
                length_warc += 1
                if length_warc % 1000 == 0: logging.info(f"Processed {length_warc} records")
                if url in url_inp_list:
                    try:
                        # TODO: Is it ok to ignore errors here? I got errors sometimes
                        # Read the response body
                        content = record.content_stream().read()
                        html_decoded = content.decode(errors='ignore')
                        markdown = convert_page(html_decoded, url)
                        htmls_mds.append((html_decoded,
                                          markdown["content"].encode('utf-8', 'ignore').decode('utf-8')))
                    except Exception as e:
                        logging.error(f"This is the real ERROR")
                        if not "CSS stylesheet" in str(e):
                            # Ignore CSS stylesheet errors
                            logging.error(f"Error processing {url} in {s3_url}: {e}")
                        htmls_mds.append(("None", "None"))
    logging.info(f"Processed {s3_url}, found {length_warc} records, {length_url_inp_list} urls, {length_warc}/{length_url_inp_list} ratio")
    return htmls_mds


def fineweb_to_markdown(input_path, output_path=None):
    """
    Converts fineweb files to html and markdown. This will essentially take in fineweb and split different groups based
    on file_path and write all those file paths to a new folder and then run sbatch for each group

    Parameters:
    input_path (str): Path to the fineweb file, will be processed using glob
    output_path (str): Path to the markdown file
    """

    if os.path.isdir(input_path): # If input_path is a directory
        input_path = os.path.join(input_path, "*.parquet")

    for file_path in glob.glob(input_path):
        logging.info(f"Processing {file_path}")
        df = pd.read_parquet(file_path)

        # Get the folder name (without extension)
        folder_name = os.path.splitext(os.path.basename(file_path))[0]

        # Get the absolute path for the folder
        folder_path = os.path.join(os.path.dirname(file_path), folder_name)

        # Create the folder with the same name as the file (without extension)
        os.makedirs(folder_path, exist_ok=True)

        # Get the relative path to the batch script
        script_name = "process_warc_file.py"

        # file_path is s3 url
        grouped = df.groupby("file_path")


        for index, (file_url, group_df) in enumerate(grouped):
            filename = os.path.join(folder_path, f"{index}.parquet")

            # Save the group to a parquet file
            group_df.to_parquet(filename)

            abs_script_path = os.path.join(os.path.dirname(__file__), script_name)
            command_to_run = (f"nlprun -a crfm -o fineweb_{folder_name}_{index} -q john -c 1 "
                              f"'python {abs_script_path} {filename} --remove_input'")

            logging.info(f"Running command: {command_to_run}")
            os.system(command_to_run)
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert fineweb to markdown.")
    parser.add_argument('input_path', type=str, help='Path to the fineweb file,if its a directory, all files will be processed')
    parser.add_argument('output_path', type=str, help='Path to the markdown file', default=None, nargs='?')
    args = parser.parse_args()
    fineweb_to_markdown(args.input_path, args.output_path)
