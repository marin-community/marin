"""
Usage:
python -m marin.processing.fasttext.example_server
"""

import gradio as gr
import fsspec
import json
import random

from marin.utils import fsspec_glob


# gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2020-10/000_00000/0_processed_md.jsonl.gz
# gs://marin-data/scratch/chrisc/attributes/fasttext-quality/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/0_processed_md.jsonl.gz
# TODO(chris): fix for all files? why are not all files processed
def sample_and_display():
    # Find all JSON files in fineweb
    data_jsons = fsspec_glob("gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2024-10/000_00000/**/0_*_md.jsonl.gz")

    if not data_jsons:
        return "No JSON files found in gs://marin-data/", "", "No files found"

    # Randomly select a JSON file
    selected_json = random.choice(data_jsons)

    # Read the selected JSON file
    data_content = []
    with fsspec.open(selected_json, "rt", compression="gzip") as f:
        for line in f:
            data_content.append(json.loads(line))

    # Get corresponding attributes JSON file
    attr_json = selected_json.replace(
        "gs://marin-data/processed/", "gs://marin-data/scratch/chrisc/attributes/fasttext-quality/"
    )
    print(attr_json)

    # Read the attributes JSON file
    attr_content = []
    with fsspec.open(attr_json, "r", compression="gzip") as f:
        for line in f:
            attr_content.append(json.loads(line))

    print(len(attr_content))

    # Randomly select an index
    random_index = random.randint(0, len(data_content) - 1)

    # Get text and fasttext-quality label
    text = data_content[random_index].get("text", "No text found")
    fasttext_quality = attr_content[random_index].get("attributes", "No label found")

    return text, fasttext_quality, f"File: {selected_json}, Index: {random_index}"


# Create Gradio interface
iface = gr.Interface(
    fn=sample_and_display,
    inputs=None,
    outputs=[
        gr.Markdown(label="Sampled Text"),
        gr.Textbox(label="Fasttext Quality"),
        gr.Textbox(label="Source Information"),
    ],
    title="Model-based Quality Filtering Results",
    description="Click the button to randomly sample text from fineweb and its associated fasttext quality",
)

# Launch the server
iface.launch(share=True)
