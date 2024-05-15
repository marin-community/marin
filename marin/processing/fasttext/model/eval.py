"""
Usage:
nlprun -m jagupard36 -c 16 -r 40G 'python marin/processing/fasttext/model/eval.py --val-file /nlp/scr/cychou/fasttext.val --model-path /nlp/scr/cychou/fasttext_model.bin'
"""

import argparse
import fasttext

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

    model = fasttext.load_model(args.model_path)

    print_results(*model.test(args.val_file))