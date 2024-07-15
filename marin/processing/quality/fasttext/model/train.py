"""
Usage:
python marin/processing/fasttext/model/train.py
"""

import argparse

import fasttext
import fsspec

TRAIN_DATASET_FILE_PATH = "/home/gcpuser/data/fasttext_train.txt"
OUTPUT_MODEL_PATH = "/home/gcpuser/model/fasttext_model.bin"


def train(train_file, output_model_path, word_ngrams):
    model = fasttext.train_supervised(input=train_file, wordNgrams=word_ngrams)
    model.save_model(output_model_path)

    return model


def predict(model, text):
    return model.predict(text)


def print_results(num_examples, precision, recall):
    print(f"Number of examples: {num_examples}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-model-path", type=str, help="The output model path")
    parser.add_argument("--word-ngrams", type=int, default=2, help="The number of word n-grams to use")

    args = parser.parse_args()

    model = train(TRAIN_DATASET_FILE_PATH, OUTPUT_MODEL_PATH, args.word_ngrams)
