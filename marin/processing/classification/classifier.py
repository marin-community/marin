import os
from typing import Any, ClassVar, Dict, List

import fsspec
from huggingface_hub import hf_hub_download


class BaseClassifier:
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: List[str]):
        raise NotImplementedError

    def __call__(self, batch: Dict[str, Any]):
        raise NotImplementedError


class DummyClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: List[str]):
        score = 1.0
        return [{"score": score} for _ in range(len(documents))]

    def __call__(self, batch: Dict[str, Any]):
        scores = self.predict(batch["text"])
        batch.update({"attributes": [{"dummy-quality": score} for score in scores]})
        return batch


class FasttextClassifier(BaseClassifier):
    _MODEL_NAME_TO_MODEL_FILENAME_DICT: ClassVar[Dict[str, str]] = {
        "mlfoundations/fasttext-oh-eli5": "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        "allenai/dolma-1_7-fasttext-quality-filter": "model.bin",
    }

    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.model = self.load_model()

    def load_model(self):
        from fasttext.FastText import _FastText

        # Classifier is stored in a remote storage.
        if "://" in self.model_name:
            from google.cloud import storage

            # Parse bucket name and blob name from the path
            protocol, path = fsspec.core.split_protocol(self.model_name)

            # Sample filepath is: gs://bucket_name/file/to/blob_name
            # bucket_name is bucket_name
            # blob_name is file/to/blob_name
            split_path = path.split("/")
            bucket_name = split_path[0]
            blob_name = "/".join(split_path[1:])

            # Check if the local path exists
            local_path = os.path.expanduser(f"~/{blob_name}")
            local_success_file = f"{local_path}.success"

            # We make sure that the success file exists as well since it's possible that the
            # file was partially downloaded.
            if os.path.exists(local_path) and os.path.exists(local_success_file):
                print(f"Using existing model from {local_path}")
                model = _FastText(local_path)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Download the file from Google Cloud Storage
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(local_path)

                with open(local_success_file, "wt") as f:
                    f.write("success")

                model = _FastText(local_path)
        else:
            # Classifier is stored in HuggingFace Hub.
            model_path = hf_hub_download(
                repo_id=self.model_name, filename=self._MODEL_NAME_TO_MODEL_FILENAME_DICT[self.model_name]
            )
            model = _FastText(model_path)

        return model

    def predict(self, documents: List[str]):
        # TODO(chris): Add support for multi-class k > 2.
        return self.model.predict(documents, k=2)

    def __call__(self, batch: Dict[str, Any]):
        texts = []
        for text in list(batch["text"]):
            if text:
                # FastText uses newline to delimit new examples. Thus, if one examples
                # has multiple newlines then it will appear as if it were
                # multiple examples. So, we replace `\n` with a space to avoid this.
                # https://arc.net/l/quote/mfbqvtry
                text = text.replace("\n", " ")
            else:
                text = ""
            texts.append(text)

        label_arr, score_arr = self.predict(texts)

        attributes_arr = []
        for i in range(list(batch["text"])):
            fasttext_quality_dict = dict(zip(label_arr[i], score_arr[i], strict=False))
            attributes_arr.append({self.attribute_name: fasttext_quality_dict})

        res = {"id": batch["id"], "source": batch["source"], "attributes": attributes_arr}
        batch.update({"attributes": attributes_arr})

        return res


class BERTClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

        self.model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.attribute_name = attribute_name

    def predict(self, documents: List[str]) -> List[float]:
        inputs = self.tokenizer(documents, return_tensors="jax", padding="longest", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)
        logits_list = logits.tolist()
        return logits_list

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class FinewebEduClassifier(BERTClassifier):
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        scores = self.predict(batch["text"])

        # Fineweb edu classifier is scored on educational value from 0 to 5, so we want to round to the nearest integer.
        int_scores = [int(round(max(0, min(score, 5)))) for score in scores]
        batch.update(
            {
                "attributes": [
                    {self.attribute_name: {"score": score, "int_score": int_score}}
                    for score, int_score in zip(scores, int_scores, strict=False)
                ]
            }
        )

        return batch


class AutoClassifier(BaseClassifier):
    _MODEL_NAME_TO_CLS_DICT: ClassVar[Dict[str, BaseClassifier]] = {
        "fasttext": FasttextClassifier,
        "fineweb": FinewebEduClassifier,
    }

    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.cls = self.from_model_path(model_name, attribute_name, *args, **kwargs)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.cls.__call__(batch)

    @classmethod
    def from_model_path(cls, model_name: str, attribute_name: str, *args, **kwargs) -> BaseClassifier:
        for key in cls._MODEL_NAME_TO_CLS_DICT.keys():
            if key in model_name:
                print(f"Using {key} model")
                return cls._MODEL_NAME_TO_CLS_DICT[key](model_name, attribute_name, *args, **kwargs)

        raise ValueError(f"Model name {model_name} not supported")
