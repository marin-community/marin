import os
import tempfile

from google.cloud import storage

from marin.processing.fasttext.download_dolma_classifier import download_file

class BaseQualityClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict(self, document):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError

class DummyQualityClassifier(BaseQualityClassifier):
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
    GCS_BUCKET_NAME = "marin-data"
    GCS_BLOB_NAME = "scratch/chrisc/dolma_fasttext_model/model.bin"
    HF_REPO_ID = "allenai/dolma-1_7-fasttext-quality-filter"
    HF_FILENAME = "model.bin"

    def __init__(self, model_ref):
        # self.model = self.download_dolma_classifier(model_path)    
        self.model = self.load_model(model_ref)

    def load_model(self, model_ref):
        from fasttext.FastText import _FastText

        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp_file:
            temp_filename = temp_file.name
            
            # Write the model content to the temporary file
            temp_file.write(model_ref.getvalue())
        try:
            model = _FastText(temp_filename)
        finally:
            os.unlink(temp_filename)

        return model

    def download_dolma_classifier(self, model_path):
        from huggingface_hub import hf_hub_download
        from marin.processing.utils import download_huggingface_file_with_backoff, download_gcs_file_with_backoff

        print(f"Trying to download classifier to {model_path}", flush=True)

        directory, basename = os.path.dirname(model_path), os.path.basename(model_path)
        os.makedirs(directory, exist_ok=True)
        
        # NOTE(chris): 1 try theoretically should work but there are sometimes some SystemExceptions from either FastText loading or Huggingface about metadata
        try:
            try:
                print("Download using huggingface start.")
                download_huggingface_file_with_backoff(self.HF_REPO_ID, self.HF_FILENAME, directory)
                print("Downloaded from huggingface.")
            except Exception as e:
                print(e, flush=True)
                
                try:
                    print("Download using GCS start.")
                    download_gcs_file_with_backoff(self.GCS_BUCKET_NAME, self.GCS_BLOB_NAME, model_path)
                    print("Download from GCS.")
                except Exception as e:
                    print(e, flush=True)
            
            print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024 / 1024} GB")
            model = _FastText(model_path)
        except Exception as e:
            raise e

        return model


    def predict(self, documents):
        return self.model.predict(documents)

    def __call__(self, batch):
        texts = []
        for text in list(batch["text"]):
            if text:
                text = text.replace("\n", " ")
            else:
                text = ""
            texts.append(text)

        label_arr, score_arr = self.predict(texts)

        attributes_arr = []
        for i, row in enumerate(list(batch["text"])):
            fasttext_quality_dict = {}
            for label, score, in zip(label_arr[i], score_arr[i]):
                fasttext_quality_dict.update({
                    label: score
                })

            attributes_arr.append({
                "fasttext-quality" : fasttext_quality_dict
            })
        
        batch.update({
            "attributes" : attributes_arr
        })

        return batch