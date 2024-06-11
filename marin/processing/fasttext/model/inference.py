"""
Usage:
ray job submit --working-dir . --no-wait -- python marin/processing/fasttext/model/inference.py
"""
import json
import os

# import fasttext
import fsspec
import ray

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
    def __call__(self, batch):
        return [self.model.predict(sentence) for sentence in batch]


@ray.remote
def process_file(model, filename):
    output_filename = "gs://marin-data/scratch/chrisc/0_processed_md.jsonl.gz"
    ds = ray.data.read_json(filename)

    print(f"[*] Reading in dataset {filename}")
    # Load fasttext classifier in the global object store
    model = "DUMMY RAY MODEL"
    model_ref = ray.put(model)

    print("[*] Beginning quality clasisfication")
    
    # NOTE(chris): Could we parallelize using map_batches?
    results = ds.map(
        DummyQualityClassiifer,
        num_gpus=0,
        concurrency=(1,16),
        fn_constructor_args=(model_ref,)
    )
    print(results.schema())
    print("[*] Finished quality classification")
    
    # with fsspec.open(filename, "rt", compression="gzip") as f_in:
    # with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
    #     json.dump(results, output_filename, indent=4)
    results.write_json(output_filename)
            
    with fsspec.open(f"{output_filename}.success", "wt") as f_out:
        f_out.write("success")

def main():
    ray.init()

    # model = fasttext.load_model(MODEL_PATH)
    for filename in FILENAMES:
        ray.get(process_file.remote("", filename))



if __name__ == "__main__":
    main()