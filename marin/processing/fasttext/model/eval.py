"""
Usage:
python marin/processing/fasttext/model/eval.py
"""

import os
import argparse
import fasttext

EVAL_DATASET_FILE_PATH = os.path.expanduser("~/data/fasttext_test.txt")
OUTPUT_MODEL_PATH = os.path.expanduser("~/dolma_fasttext_model/model.bin")
# OUTPUT_MODEL_PATH = os.path.expanduser("~/model/fasttext_model.bin")

def predict(model, text):
    return model.predict(text)

def print_results(num_examples, precision, recall):
    print(f"Number of examples: {num_examples}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-file", type=str, help="The validation file")
    parser.add_argument("--model-path", type=str, help="The output model path")

    args = parser.parse_args()

    model = fasttext.load_model(OUTPUT_MODEL_PATH)
    print(model.get_dimension())

    print_results(*model.test(EVAL_DATASET_FILE_PATH))