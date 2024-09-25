"""
Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.model.train \
    --input-file <input_file> \
    --output-model-path <output_model_path> \
    [other options]
"""

import argparse
import os
import tempfile

import fsspec
import ray


def _cleanup(training_file_name, local_output_file_name):
    os.remove(training_file_name)
    os.remove(local_output_file_name)


@ray.remote
def train(
    input_file,
    lr,
    dim,
    ws,
    epoch,
    wordNgrams,
    minCount,
    minCountLabel,
    minn,
    maxn,
    neg,
    loss,
    bucket,
    thread,
    lrUpdateRate,
    t,
    label,
    verbose,
    pretrainedVectors,
    output_model_path,
):
    import fasttext

    training_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    local_model_output_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)

    try:
        with fsspec.open(input_file, "rt", compression="gzip") as f_in:
            training_file.write(f_in.read())
            training_file.flush()

        model = fasttext.train_supervised(
            input=training_file.name,
            lr=lr,
            dim=dim,
            ws=ws,
            epoch=epoch,
            wordNgrams=wordNgrams,
            minCount=minCount,
            minCountLabel=minCountLabel,
            minn=minn,
            maxn=maxn,
            neg=neg,
            loss=loss,
            bucket=bucket,
            thread=thread,
            lrUpdateRate=lrUpdateRate,
            t=t,
            label=label,
            verbose=verbose,
            pretrainedVectors=pretrainedVectors,
        )

        model.save_model(local_model_output_file.name)

        with open(local_model_output_file.name, "rb") as f_in:
            with fsspec.open(output_model_path, "wb") as f_out:
                f_out.write(f_in.read())

    except Exception as e:
        print(f"Error during model training: {e}")
    finally:
        _cleanup(training_file.name, local_model_output_file.name)

    print("Finished training fasttext model.")


def main(
    input_file,
    lr,
    dim,
    ws,
    epoch,
    wordNgrams,
    minCount,
    minCountLabel,
    minn,
    maxn,
    neg,
    loss,
    bucket,
    thread,
    lrUpdateRate,
    t,
    label,
    verbose,
    pretrainedVectors,
    output_model_path,
):

    response = train.options(num_cpus=thread, runtime_env=ray.runtime_env.RuntimeEnv(pip=["fasttext"])).remote(
        input_file,
        lr,
        dim,
        ws,
        epoch,
        wordNgrams,
        minCount,
        minCountLabel,
        minn,
        maxn,
        neg,
        loss,
        bucket,
        thread,
        lrUpdateRate,
        t,
        label,
        verbose,
        pretrainedVectors,
        output_model_path,
    )

    # Wait for the training to complete and handle any exceptions
    try:
        ray.get(response)
        print(f"Model training completed. Model saved at: {output_model_path}")
    except Exception as e:
        print(f"Error during model training: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a FastText model for text classification")

    # Fasttext specific parameters
    parser.add_argument("--input-file", type=str, required=True, help="The train file path")
    parser.add_argument("--lr", type=float, default=0.1, help="The learning rate")
    parser.add_argument("--dim", type=int, default=100, help="Size of the word vectors")
    parser.add_argument("--ws", type=int, default=5, help="The size of the context window")
    parser.add_argument("--epoch", type=int, default=5, help="The number of epochs to train for")
    parser.add_argument("--wordNgrams", type=int, default=1, help="Max length of word ngram")
    parser.add_argument("--minCount", type=int, default=1, help="Minimal number of word occurrences")
    parser.add_argument("--minCountLabel", type=int, default=1, help="Minimal number of label occurrences")
    parser.add_argument("--minn", type=int, default=0, help="Min length of char ngram")
    parser.add_argument("--maxn", type=int, default=0, help="Max length of char ngram")
    parser.add_argument("--neg", type=int, default=5, help="Number of negatives sampled")
    parser.add_argument(
        "--loss", type=str, default="softmax", choices=["ns", "hs", "softmax", "ova"], help="Loss function"
    )
    parser.add_argument("--bucket", type=int, default=2000000, help="Number of buckets")
    parser.add_argument(
        "--thread", type=int, default=115, help="Number of threads"
    )  # Uses the number of roughly CPUs on a TPU
    parser.add_argument("--lrUpdateRate", type=int, default=100, help="Change the rate of updates for the learning rate")
    parser.add_argument("--t", type=float, default=0.0001, help="Sampling threshold")
    parser.add_argument("--label", type=str, default="__label__", help="Label prefix")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose")
    parser.add_argument(
        "--pretrainedVectors", type=str, default="", help="Pretrained word vectors (.vec file) for supervised learning"
    )

    # Parameters for marin
    parser.add_argument("--output-model-path", type=str, required=True, help="The output model path")

    args = parser.parse_args()

    ray.init()

    main(
        args.input_file,
        args.lr,
        args.dim,
        args.ws,
        args.epoch,
        args.wordNgrams,
        args.minCount,
        args.minCountLabel,
        args.minn,
        args.maxn,
        args.neg,
        args.loss,
        args.bucket,
        args.thread,
        args.lrUpdateRate,
        args.t,
        args.label,
        args.verbose,
        args.pretrainedVectors,
        args.output_model_path,
    )
