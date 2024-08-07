"""
Usage:
python -m marin.processing.quality.eval.annotations_server --input-file gs://marin-data/filtered/fineweb-edu-quality-3.0/fineweb/fw-v1.0/md/CC-MAIN-2020-10/000_00000/0_processed.jsonl.gz --attributes-file gs://marin-data/processed/fineweb/fw-v1.0/attributes_md/fineweb-edu-quality/CC-MAIN-2020-10/000_00000/0_processed.jsonl.gz
"""

import argparse
import gradio as gr
import fsspec
import json
import random
import os

from marin.utils import fsspec_glob


def sample_and_display(data_content, attr_content, input_filename, attributes_filename):
    if not data_content:
        with fsspec.open(input_filename, "rt", compression="gzip") as f:
            for line in f:
                data_content.append(json.loads(line))

    if not attr_content:
        with fsspec.open(attributes_filename, "rt", compression="gzip") as f:
            for line in f:
                attr_content.append(json.loads(line))

    if not data_content or not attr_content:
        return "No JSON files found in gs://marin-data/", "", "No files found", None, None, None

    random_index = random.randint(0, len(data_content) - 1)

    text = data_content[random_index].get("text", "No text found")
    data_content_id = data_content[random_index].get("id", "No id found")

    attribute_quality = None
    for attr in attr_content:
        if attr["id"] == data_content_id:
            attribute_quality = attr["attributes"]["fineweb-edu-quality"]
            break

    source_info = f"File: {input_filename}, Index: {random_index}"

    return text, attribute_quality, source_info, data_content_id, random_index, input_filename


def downvote(id, index, json_name):
    downvote_data = {"id": id, "index": index, "json_name": json_name}

    output_file = "downvotes.json"

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(downvote_data)

    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=2)

    return f"Downvoted: ID {id}, Index {index}, File {json_name}"

def build_demo(input_file: str, attributes_file: str):
    # Create Gradio interface

    with gr.Blocks() as iface:
        input_filename = gr.State(input_file)
        attributes_filename = gr.State(attributes_file)
        data_content = gr.State([])
        attr_content = gr.State([])

        with gr.Row():
            sample_button = gr.Button("Sample")

        with gr.Row():
            text_output = gr.Markdown(label="Sampled Text")

        with gr.Row():
            attribute_quality_output = gr.Textbox(label="Attribute Quality")

        with gr.Row():
            source_info_output = gr.Textbox(label="Source Information")

        with gr.Row():
            id_output = gr.Textbox(label="ID", visible=False)
            index_output = gr.Number(label="Index", visible=False)
            json_name_output = gr.Textbox(label="JSON Name", visible=False)

        with gr.Row():
            downvote_button = gr.Button("Downvote")
            downvote_result = gr.Textbox(label="Downvote Result")

        sample_button.click(
            sample_and_display,
            inputs=[data_content, attr_content, input_filename, attributes_filename],
            outputs=[text_output, attribute_quality_output, source_info_output, id_output, index_output, json_name_output],
        )

        downvote_button.click(downvote, inputs=[id_output, index_output, json_name_output], outputs=downvote_result)

    return iface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--attributes-file", type=str, default=None)
    args = parser.parse_args()

    iface = build_demo(args.input_file, args.attributes_file)

    iface.launch(share=True)
