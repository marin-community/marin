import argparse
import json
import os
import gzip
import fsspec

from marin.web.convert import convert_page


def html_to_md(input_file_path, output_dir):
    with gzip.open(input_file_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # file is a jsonl so each line is a list with one json
            import ipdb; ipdb.set_trace()
            data = json.loads(line)[0]
            # Extract relevant information from the data
            id = data.get("id", f"{idx}")  # Set a default value if "id" is not present
            text = data.get("content", "")  # Use "content" as the text field
            source = "instruct"  # Set the source to "instruct" or any other appropriate value
            role = data.get("role", "")  # Extract the role information

            # Create the metadata dictionary
            metadata = {
                "role": role
            }

            # Create the Dolma-formatted document
            dolma_doc = {
                "id": id,
                "text": text,
                "source": source,
                "metadata": metadata
            }

            base_name = os.path.splitext(os.path.basename(input_file_path))[0]
            md_output_file_path = os.path.join(output_dir, f"{base_name}_{id}.md")
            html_output_file_path = os.path.join(output_dir, f"{base_name}_{id}.html")

            try:
                out = convert_page(text, url="")
                title = out["title"]
                md = out["content"]
                html = out["html"]

                # Add role to the beginning of the markdown content
                if role:
                    md = f"# {role}\n\n{md}"

                # Write the markdown to the output file
                with open(md_output_file_path, "w") as md_output:
                    print(md, file=md_output)

                # Write the HTML to the output file
                with open(html_output_file_path, "w") as html_output:
                    print(html, file=html_output)

            except Exception as e:
                print(f"Error {e} in processing {id = }, file: {input_file_path}")
                continue

    return True


def main(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all the files in the input directory on GCS
    for file_path in fsspec.open_files(os.path.join(input_dir, "**/*.jsonl.gz"), mode="rb"):
        # Generate the local file path
        local_file_path = os.path.join(output_dir, os.path.basename(file_path.path))

        # Save the file locally
        import ipdb; ipdb.set_trace()
        with open(local_file_path, "wb") as local_file:
            local_file.write(file_path.open().read())

        # Process the local file
        try:
            html_to_md(local_file_path, output_dir)
        except Exception as e:
            print(f"Error processing file: {local_file_path}, {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to convert HTML to markdown.")
    parser.add_argument('--input_dir', type=str, help='Path to HTML directory on GCS', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to local output markdown directory', default="output")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)