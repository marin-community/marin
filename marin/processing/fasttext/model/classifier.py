import os
import tempfile

import ray
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
    LOCAL_FILEPATH = os.path.expanduser("~/dolma_fasttext_model/model.bin")

    def __init__(self, model_ref):
        self.model = self.load_model(model_ref)

    def load_model(self, model_ref):
        from fasttext.FastText import _FastText

        try:
            if not os.path.exists(self.LOCAL_FILEPATH):
                directory, basename = os.path.dirname(self.LOCAL_FILEPATH), os.path.basename(self.LOCAL_FILEPATH)
                os.makedirs(directory, exist_ok=True)

                with open(self.LOCAL_FILEPATH, "wb") as f:
                    f.write(model_ref.getvalue())

            print(f"Size of model: {os.path.getsize(self.LOCAL_FILEPATH)}")
            model = _FastText(self.LOCAL_FILEPATH)
        except Exception as e:
            print(e)
            print("failed to load model")
            
        return model

    def predict(self, document):
        return self.model.predict(document)

    def __call__(self, row):
        if row["text"]:
            text = row["text"].replace("\n", " ")
        else:
            text = ""

        label_arr, score_arr = self.predict(text)
        fasttext_quality_dict = {}
        for label, score in zip(label_arr, score_arr):
            fasttext_quality_dict.update({
                label: score
            })

        row.update({
            "attributes" : {
                "fasttext-quality" : fasttext_quality_dict
            }
        })

        return row

class BatchFasttextQualityClassifier(BaseQualityClassifier):
    LOCAL_FILEPATH = os.path.expanduser("~/dolma_fasttext_model/model_quantized.bin")

    def __init__(self, model_ref):
        self.model = self.load_model(model_ref)

    def load_model(self, model_ref):
        from fasttext.FastText import _FastText

        try:
            if not os.path.exists(self.LOCAL_FILEPATH):
                directory, basename = os.path.dirname(self.LOCAL_FILEPATH), os.path.basename(self.LOCAL_FILEPATH)
                os.makedirs(directory, exist_ok=True)

                with open(self.LOCAL_FILEPATH, "wb") as f:
                    f.write(model_ref.getvalue())

            print(f"Size of model: {os.path.getsize(self.LOCAL_FILEPATH)}")
            model = _FastText(self.LOCAL_FILEPATH)
        except Exception as e:
            print(e)
            print("failed to load model")

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