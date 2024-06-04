import fsspec
import functools


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


def get_gcs_path(file_path):
    """
    Get the GCS path. If a path starts from gs:// then it returns the path as it is. else appends gs:// to the path.
    """
    if file_path.startswith("gs://"):
        return file_path
    return f"gs://{file_path}"


def make_sential(success_suffix="success"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(input_file_path, output_file_path, *args, **kwargs):
            # output_file is file for md output and success_file is the Ledger file to
            success_file = output_file_path + f".{success_suffix}"

            # If the ledger file exists, then we do not process the file again
            if gcs_file_exists(success_file):
                print(f"Output file already processed. Skipping {input_file_path}")
                return True

            # Execute the main function
            func(input_file_path, output_file_path, *args, **kwargs)

            # Write the success file, so that we don't have to process it next time
            with fsspec.open(success_file, 'w') as f:
                f.write("SUCCESS")

            print(f"Processed {input_file_path}")
            return True

        return wrapper

    return decorator
