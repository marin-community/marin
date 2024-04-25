'''Convert fineweb to markdown'''
import argparse
import os
import glob
import s3fs, gzip
import pandas as pd
from markdown import to_markdown
import logging
from warcio.archiveiterator import ArchiveIterator

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
    if not s3_url == "s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/warc/CC-MAIN-20240220211055-20240221001055-00000.warc.gz":
        return [(s3_url + "_" + url, s3_url + "211_" + url) for url in url_inp_list]

    htmls_mds = []
    with s3.open(s3_url, 'rb') as f:
        stream = gzip.GzipFile(fileobj=f)  # Decompress data with gzip
        for record in ArchiveIterator(stream):
            # Check if it's a response record
            if record.rec_type == 'response':
                # Process the record
                url = record.rec_headers.get_header('WARC-Target-URI')

                # Read the response body
                content = record.content_stream().read()
                if url in url_inp_list:
                    # TODO: Is it ok to ignore errors here? I got errors sometimes
                    html_decoded = content.decode(errors='ignore')
                    htmls_mds.append((html_decoded, to_markdown(html_decoded)))
    logging.info(f"Processed {s3_url}")
    return htmls_mds


def fineweb_to_markdown(input_path, output_path=None):
    """
    Converts fineweb files to html and markdown

    Parameters:
    input_path (str): Path to the fineweb file, will be processed using glob
    output_path (str): Path to the markdown file
    """
    for file in glob.glob(input_path):
        df = pd.read_parquet(file)
        # The below line groups by file_path and then applies the process_wrac_file function
        # such that urls are all the urls in that file_path
        modified_df = df.groupby('file_path')["url"].transform(lambda x:
                                                               process_wrac_file(x.name, x.tolist()))
        df["html", "md"] = pd.DataFrame(modified_df.tolist(), index=df.index)
        import pdb;
        pdb.set_trace()

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
    parser.add_argument('input_path', type=str, help='Path to the fineweb file, will be processed using glob')
    parser.add_argument('output_path', type=str, help='Path to the markdown file', default=None, nargs='?')
    args = parser.parse_args()
    fineweb_to_markdown(args.input_path, args.output_path)
