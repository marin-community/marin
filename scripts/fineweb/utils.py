# Utils specific to fw processing

def get_warc_parquet_success_path(input_file_path: str):
    """ Given a parquet file path, return the path to the output md file and success file """
    output_file = input_file_path.replace(".parquet", "_processed_md.jsonl.gz")
    success_file = output_file + ".SUCCESS"
    return output_file, success_file
