import fsspec


def gcs_file_exists(file_path):
    """
    Check if a file exists in a Google Cloud Storage bucket.

    Args:
        file_path (str): The path of the file in the GCS bucket.

    Returns:
        bool: True if the file exists, False otherwise.
    """

    # Use fsspec to check if the file exists
    fs = fsspec.filesystem('gcs')
    return fs.exists(file_path)
