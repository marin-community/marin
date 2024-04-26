import pandas as pd
import os
import sys


def dump_html_md(parquet_path):
    # Load the Parquet file
    df = pd.read_parquet(parquet_path)

    # Output directory to store HTML and md files
    parquet_dir, parquet_file = os.path.split(parquet_path)
    output_dir = os.path.join(parquet_dir, f"{os.path.splitext(parquet_file)[0]}_files")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract data from the current row
        url = row['url']
        html_content = row['html']
        md_content = row['md']


        # Generate file names
        html_file_name = os.path.join(output_dir, f"{index}.html")
        md_file_name = os.path.join(output_dir, f"{index}.md")

        # Write HTML content to file
        with open(html_file_name, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

        # Write md content to file
        with open(md_file_name, 'w', encoding='utf-8') as md_file:
            md_file.write(md_content)

    print("Files dumped successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <parquet_path>")
        sys.exit(1)

    parquet_path = sys.argv[1]
    if not os.path.exists(parquet_path):
        print("Error: Parquet file does not exist.")
        sys.exit(1)

    dump_html_md(parquet_path)
