"""
Usage:
nlprun -m jagupard36 -c 16 -r 40G 'python marin/processing/fasttext/model/train.py --train-file /nlp/scr/cychou/fasttext.train --output-model-path /nlp/scr/cychou/fasttext_model.bin'
"""

import argparse
import fasttext

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
    parser.add_argument("--train-file", type=str, help="The training file")
    parser.add_argument("--output-model-path", type=str, help="The output model path")
    parser.add_argument("--word-ngrams", type=int, default=2, help="The number of word n-grams to use")

    args = parser.parse_args()

    model = train(args.train_file, args.output_model_path, args.word_ngrams)