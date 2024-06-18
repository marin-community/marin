"""
Usage:
ray job submit --working-dir . --no-wait -- python marin/processing/fasttext/model/inference.py
"""
import datetime
import json
import os

# import fasttext
import fsspec
import ray

from marin.core.runtime import cached_or_construct_output
from marin.processing.fasttext.download_dolma_classifier import download_file

MODEL_PATH = os.path.expanduser("~/dolma_fasttext_model/model.bin")
FILENAMES = [
    "gs://marin-data/processed/fineweb/fw-v1.0/CC-MAIN-2023-50/000_00000/0_processed_md.jsonl.gz"
]

class BaseQualityClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict(self, document):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError

class DummyQualityClassiifer(BaseQualityClassifier):
    def __init__(self, model):
        self.model = model

    def predict(self, document):
        label, score = "__label__test", 1.0
        return label, score
    
    def __call__(self, row):
        label, score = self.predict(row["text"])
        row.update({
            "label": label,
            "score": score,
        })
        return row


class FasttextQualityClassifier(BaseQualityClassifier):
    def __init__(self, model):
        import fasttext

        model_path = download_file()
        self.model = fasttext.load_model(model_path)

    def predict(self, document):
        return self.model.predict(document)

    def __call__(self, row):
        text = row["text"].replace("\n", " ")
        label, score_arr = self.predict(text)
        row.update({
            "label": label,
            "score": score_arr[0],
        })

        return row


@ray.remote(memory= 5 * 1024 * 1024 * 1024, runtime_env={"pip": ["fasttext"]})
# @cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename, output_filename):
    # Load fasttext classifier in the global object store
    # model = "DUMMY RAY MODEL"
    # model = fasttext.load_model(MODEL_PATH)
    # model_ref = ray.put(model)

    ds = ray.data.read_json(input_filename)

    print(f"[*] Reading in dataset {input_filename}")

    print("[*] Beginning quality classification")
    
    # NOTE(chris): Could we parallelize using map_batches?
    results = ds.map(
        FasttextQualityClassifier,
        num_gpus=0,
        concurrency=(1,16),
        fn_constructor_args=("",)
    )

    print("[*] Finished quality classification")
    
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
            for row in results.iter_rows():
                
                # TODO(chris): support more documents, not just fineweb
                if "fineweb_metadata" in row['metadata'] and "date" in row['metadata']["fineweb_metadata"]:
                    row['metadata']['fineweb_metadata']["date"] = row['metadata']['fineweb_metadata']['date'].strftime('%m/%d/%Y')

                json_row = json.dumps(row)
                f_out.write(json_row + "\n")

def main():
    ray.init()

    # model = fasttext.load_model(MODEL_PATH)
    # model_path = download_file()

    # # Load fasttext classifier in the global object store
    # # model = "DUMMY RAY MODEL"
    # model = fasttext.load_model(MODEL_PATH)
    # model_ref = ray.put(model)

    output_filename = "gs://marin-data/scratch/chrisc/0_processed_md.jsonl.gz"
    for filename in FILENAMES:
        ray.get(process_file.remote(filename, output_filename))



if __name__ == "__main__":
    main()