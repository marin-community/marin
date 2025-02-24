import json
import os

import fsspec
import gradio as gr

from marin.utils import fsspec_glob

SCORE_OPTIONS = ["Great", "Good", "Okay", "Poor", "Useless"]


def get_files(directory):
    """Given a directory, recursively finds all .json files."""
    files = fsspec_glob(os.path.join(directory, "**/*.json"))
    return files


def load_file(file_path, index=0):
    """Loads a single JSON file and returns its contents at given index."""
    try:
        with fsspec.open(file_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return "File is empty", "", 0, 0
            if index >= len(lines):
                index = 0
            content = json.loads(lines[index])
            # Extract keys; default to empty string if not found
            prompt = content.get("prompt", "")
            generated_text = content.get("generated_text", "")
            return prompt, generated_text, index, len(lines)
    except Exception as e:
        return f"Error reading file {file_path}: {e}", "", 0, 0


def load_and_store(directory):
    """Gets list of files and returns a status message and the file list."""
    files = get_files(directory)
    return f"Found {len(files)} files.", gr.update(choices=files)


def display_file_contents(file_path, score_filter, index):
    """Loads and displays contents of selected file if it matches score filter."""
    if not file_path:
        return "", "", 0, 0

    prompt, generated_text, curr_index, total_rows = load_file(file_path, index)
    if score_filter and len(score_filter) < len(SCORE_OPTIONS):
        # Only show content if no score filter is applied or if content matches filter
        # In a real implementation, you would need to store/load scores for each generation
        return "", "Please select a score filter to view content", curr_index, total_rows

    return f"### Prompt\n{prompt}", f"### Generated Text\n{generated_text}", curr_index, total_rows


def next_row(file_path, score_filter, current_index):
    """Displays the next row in the file."""
    return display_file_contents(file_path, score_filter, current_index + 1)


with gr.Blocks() as demo:
    gr.Markdown("# Gradio File Browser")

    with gr.Row():
        directory_input = gr.Textbox(label="Directory Path", placeholder="Enter directory path")
        load_button = gr.Button("Load Data")

    status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        file_dropdown = gr.Dropdown(label="Select File", choices=[])
        score_filter = gr.CheckboxGroup(label="Filter by Score", choices=SCORE_OPTIONS, value=SCORE_OPTIONS)

    prompt_output = gr.Markdown()
    generated_output = gr.Markdown()

    # Hidden state for tracking current row index
    current_index = gr.State(value=0)
    total_rows = gr.State(value=0)

    with gr.Row():
        next_button = gr.Button("Next Row")
        row_counter = gr.Textbox(label="Current Row", value="0 / 0", interactive=False)

    load_button.click(fn=load_and_store, inputs=directory_input, outputs=[status_text, file_dropdown])

    def update_row_counter(index, total):
        return f"{index + 1} / {total}"

    file_dropdown.change(
        fn=display_file_contents,
        inputs=[file_dropdown, score_filter, current_index],
        outputs=[prompt_output, generated_output, current_index, total_rows],
    ).then(fn=update_row_counter, inputs=[current_index, total_rows], outputs=row_counter)

    score_filter.change(
        fn=display_file_contents,
        inputs=[file_dropdown, score_filter, current_index],
        outputs=[prompt_output, generated_output, current_index, total_rows],
    ).then(fn=update_row_counter, inputs=[current_index, total_rows], outputs=row_counter)

    next_button.click(
        fn=next_row,
        inputs=[file_dropdown, score_filter, current_index],
        outputs=[prompt_output, generated_output, current_index, total_rows],
    ).then(fn=update_row_counter, inputs=[current_index, total_rows], outputs=row_counter)

demo.launch(share=True)
