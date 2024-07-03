import argparse
import json
import os
import gzip
import fsspec
from tqdm import tqdm
import ray
from marin.web.convert import convert_page

@ray.remote
def html_to_md(input_file_path, md_output_dir, html_output_dir):
    """
    Convert HTML files to Markdown format and render them locally.

    Args:
        input_file_path (str): Path to the input HTML file.
        md_output_dir (str): Directory to save the converted Markdown files.
        html_output_dir (str): Directory to save the rendered HTML files.
    """
    
    with fsspec.open(input_file_path, "rt", compression="gzip") as f:
        data = json.load(f)
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]

        for idx, item in enumerate(data):
            output_file_name = f"{base_name}_{idx}.md"
            md_output_file_path = os.path.join(md_output_dir, output_file_name)
            html_output_file_path = os.path.join(html_output_dir, output_file_name.replace(".md", ".html"))

            try:
                id = item.get("id", "")
                html = item.get("text", "")
                source = item.get("source", "")
                url = item.get("metadata", {}).get("fineweb_metadata", {}).get("url", "")

                out = convert_page(html, url)
                title = out["title"]
                md = out["content"]
                html_content = out["html"]

                md_content = f"# {title}\n\n{md}"
                
                with fsspec.open(md_output_file_path, "w") as md_output:
                    md_output.write(md_content)

                with fsspec.open(html_output_file_path, "w") as html_output:
                    html_output.write(html_content)

            except Exception as e:
                print(f"Error {e} in processing {idx = }, file: {input_file_path}")
                continue

    return True

def main(input_dir, output_dir):
    """
    Process and render HTML/MD files using Ray.

    Args:
        input_dir (str): Path to the directory containing HTML files on GCP.
        output_dir (str): Path to the output directory on GCP.
    """
    ray.init()

    md_output_dir = os.path.join(output_dir, "markdown")
    html_output_dir = os.path.join(output_dir, "html")

    # Use GCP-compatible file operations
    fs, _ = fsspec.core.url_to_fs(input_dir)
    fs.makedirs(md_output_dir, exist_ok=True)
    fs.makedirs(html_output_dir, exist_ok=True)

    input_files = fs.glob(os.path.join(input_dir, "**/*.jsonl.gz"))
    
    tasks = []
    for file_path in input_files:
        task = html_to_md.remote(file_path, md_output_dir, html_output_dir)
        tasks.append(task)

    try:
        ray.get(tasks)
    except Exception as e:
        print(f"Error processing: {e}")
        # Add retry logic here if needed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert HTML to markdown using Ray on GCP.")
    parser.add_argument('--input_dir', type=str, help='Path to HTML directory on GCS', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to output directory on GCS', required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)