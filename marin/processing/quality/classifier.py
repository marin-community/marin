import os
import tempfile
from typing import List, Dict, Any

import ray
from google.cloud import storage
from huggingface_hub import hf_hub_download


class BaseQualityClassifier:
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: List[str]):
        raise NotImplementedError

    def __call__(self, batch: Dict[str, Any]):
        raise NotImplementedError


class DummyQualityClassifier(BaseQualityClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: List[str]):
        label, score = "__label__test", 1.0
        return [{"score": score} for _ in range(len(documents))]

    def __call__(self, batch: Dict[str, Any]):
        scores = self.predict(batch["text"])
        batch.update({"attributes": [{"dummy-quality": score} for score in scores]})
        return batch


class FasttextQualityClassifier(BaseQualityClassifier):
    _MODEL_NAME_TO_MODEL_FILENAME_DICT = {
        "mlfoundations/fasttext-oh-eli5": "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        "allenai/dolma-1_7-fasttext-quality-filter": "model.bin",
    }

    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        from fasttext.FastText import _FastText

        model_path = hf_hub_download(repo_id=model_name, filename=self._MODEL_NAME_TO_MODEL_FILENAME_DICT[model_name])
        self.model = _FastText(model_path)
        self.attribute_name = attribute_name

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
        for i, row in enumerate(list(batch["text"])):
            fasttext_quality_dict = dict(zip(label_arr[i], score_arr[i]))

            # pyarrow schema does not like it if you do not have a key. We make it non-optional to not have the key
            # if self.model_type == "dolma":
            #     if "__label__hq" not in fasttext_quality_dict:
            #         fasttext_quality_dict.update({"__label__hq": 1.0 - fasttext_quality_dict.get("__label__lq")})
            #     elif "__label__lq" not in fasttext_quality_dict:
            #         fasttext_quality_dict.update({"__label__lq": 1.0 - fasttext_quality_dict.get("__label__hq")})
            # elif self.model_type == "dclm":
            #     if "__label__cc" not in fasttext_quality_dict:
            #         fasttext_quality_dict.update({"__label__cc": 1.0 - fasttext_quality_dict.get("__label__hq")})
            #     elif "__label__hq" not in fasttext_quality_dict:
            #         fasttext_quality_dict.update({"__label__hq": 1.0 - fasttext_quality_dict.get("__label__cc")})

            attributes_arr.append({self.attribute_name: fasttext_quality_dict})

        res = {"id": batch["id"], "source": batch["source"], "attributes": attributes_arr}
        batch.update({"attributes": attributes_arr})

        return res


class BERTQualityClassifier(BaseQualityClassifier):
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


class FinewebEduQualityClassifier(BERTQualityClassifier):
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        scores = self.predict(batch["text"])

        # Fineweb edu classifier is scored on educational value from 0 to 5, so we want to round to the nearest integer.
        int_scores = [int(round(max(0, min(score, 5)))) for score in scores]
        batch.update(
            {
                "attributes": [
                    {self.attribute_name: {"score": score, "int_score": int_score}}
                    for score, int_score in zip(scores, int_scores)
                ]
            }
        )

        return batch


class AutoClassifier(BaseQualityClassifier):
    _MODEL_NAME_TO_CLS_DICT = {
        "fasttext": FasttextQualityClassifier,
        "fineweb": FinewebEduQualityClassifier,
    }

    def __init__(self, model_name, attribute_name, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.cls = self.from_model_path(model_name, attribute_name, *args, **kwargs)

    def __call__(self, batch):
        return self.cls.__call__(batch)

    @classmethod
    def from_model_path(cls, model_name, attribute_name, *args, **kwargs):
        for key in cls._MODEL_NAME_TO_CLS_DICT.keys():
            if key in model_name:
                print(f"Using {key} model")
                return cls._MODEL_NAME_TO_CLS_DICT[key](model_name, attribute_name, *args, **kwargs)

        raise ValueError(f"Model name {model_name} not supported")
