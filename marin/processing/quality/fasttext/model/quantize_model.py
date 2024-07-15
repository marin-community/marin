"""
Usage:

python -m marin.processing.fasttext.model.quantize_model --model-path /home/gcpuser/dolma_fasttext_model/model.bin
"""

import argparse
import os

import fasttext


def main(model_path: str):
    model = fasttext.load_model(model_path)
    model.quantize(
        input=None,
        qout=False,
        cutoff=0,
        retrain=False,
        epoch=None,
        lr=None,
        thread=None,
        verbose=None,
        dsub=2,
        qnorm=False,
    )

    directory = os.path.dirname(model_path)
    model.save_model(os.path.join(directory, "model_quantized.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)

    args = parser.parse_args()

    main(args.model_path)
