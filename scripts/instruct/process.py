import argparse
import json
import os
import gzip
import fsspec
from tqdm import tqdm
from marin.web.convert import convert_page

def html_to_md(input_file_path, md_output_dir, html_output_dir):
    """
    Convert HTML files to Markdown format and render them locally.

    Args:
        input_file_path (str): Path to the input HTML file.
        md_output_dir (str): Directory to save the converted Markdown files.
        html_output_dir (str): Directory to save the rendered HTML files.
    """
    
    with gzip.open(input_file_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, desc="Processing lines")):
            data = json.loads(line)
            base_name = os.path.splitext(os.path.basename(input_file_path))[0]

            output_file_name = f"{base_name}_{idx}.md"
            md_output_file_path = os.path.join(md_output_dir, output_file_name)
            html_output_file_path = os.path.join(html_output_dir, output_file_name.replace(".md", ".html"))

            try:
                md_content = ""
                html_content = ""

                for item in data:
                    role = item.get("role", "")
                    text = item.get("content", "")

                    out = convert_page(text, url="")
                    title = out["title"]
                    md = out["content"]
                    html = out["html"]

                    if role:
                        md = f"# {role}\n\n{md}"
                        html = f"<h1>{role}</h1>\n{html}"

                    md_content += md + "\n\n"
                    html_content += html + "\n\n"

                with open(md_output_file_path, "w") as md_output:
                    print(md_content, file=md_output)

                with open(html_output_file_path, "w") as html_output:
                    print(html_content, file=html_output)

            except Exception as e:
                print(f"Error {e} in processing {idx = }, file: {input_file_path}")
                continue

    return True


def main(input_dir, output_dir):
    """
    Process and render HTML/MD files locally.

    Args:
        input_dir (str): Path to the directory containing HTML files.
        output_dir (str): Path to the output directory.

    The function converts HTML files to Markdown format and saves them in the 'markdown' subdirectory of the output directory.
    It also renders the HTML files and saves them in the 'html' subdirectory of the output directory.
    """
    md_output_dir = os.path.join(output_dir, "markdown")
    html_output_dir = os.path.join(output_dir, "html")

    os.makedirs(md_output_dir, exist_ok=True)
    os.makedirs(html_output_dir, exist_ok=True)

    for file_path in tqdm(fsspec.open_files(os.path.join(input_dir, "**/*.jsonl.gz"), mode="rb"), desc="Processing files"):
        local_file_path = os.path.join(output_dir, os.path.basename(file_path.path))

        with open(local_file_path, "wb") as local_file:
            local_file.write(file_path.open().read())

        try:
            html_to_md(local_file_path, md_output_dir, html_output_dir)
        except Exception as e:
            print(f"Error processing file: {local_file_path}, {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert HTML to markdown.")
    parser.add_argument('--input_dir', type=str, help='Path to HTML directory on GCS', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to local output directory', default="output")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)