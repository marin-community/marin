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
    Converts fineweb files to html and markdown

    Parameters:
    input_path (str): Path to the fineweb file, will be processed using glob
    output_path (str): Path to the markdown file
    """
    if os.path.isdir(input_path): # If input_path is a directory
        input_path = os.path.join(input_path, "*.parquet")

    for file in glob.glob(input_path):
        logging.info(f"Processing {file}")
        df = pd.read_parquet(file)
        # The below line groups by file_path and then applies the process_wrac_file function
        # such that urls are all the urls in that file_path
        modified_df = df.groupby('file_path')["url"].transform(lambda x:
                                                               process_wrac_file(x.name, x.tolist()))
        df[["html", "md"]] = pd.DataFrame(modified_df.tolist(), index=df.index)

        # Save the modified df to parquet with same name as input file but with "_markdownified" appended
        file_name = os.path.basename(file).split(".")[0] + "_markdownified" + ".parquet"
        if output_path is None:
            output_path = os.path.dirname(file)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, file_name)

        df.to_parquet(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert fineweb to markdown")
    parser.add_argument('input_path', type=str, help='Path to the fineweb file,if its a directory, all files will be processed')
    parser.add_argument('output_path', type=str, help='Path to the markdown file', default=None, nargs='?')
    args = parser.parse_args()
    fineweb_to_markdown(args.input_path, args.output_path)
