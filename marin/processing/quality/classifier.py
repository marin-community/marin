import os
import tempfile

import ray
from google.cloud import storage
from huggingface_hub import hf_hub_download


class BaseQualityClassifier:
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def predict(self, documents):
        raise NotImplementedError

    def __call__(self, batch):
        raise NotImplementedError


class DummyQualityClassifier(BaseQualityClassifier):
    def __init__(self, model):
        self.model = model

    def predict(self, documents):
        label, score = "__label__test", 1.0
        return [{"score": score} for _ in range(len(documents))]

    def __call__(self, batch):
        scores = self.predict(batch["text"])
        batch.update({"attributes": [{"dummy-quality": score} for score in scores]})
        return batch

class BatchFasttextQualityClassifier(BaseQualityClassifier):
    _MODEL_NAME_TO_MODEL_FILENAME_DICT = {
        "mlfoundations/fasttext-oh-eli5" : "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        "allenai/dolma-1_7-fasttext-quality-filter" : "model.bin",
    }

    def __init__(self, model_name, *args, **kwargs):
        from fasttext.FastText import _FastText

        model_path = hf_hub_download(repo_id=model_name, filename=self._MODEL_NAME_TO_MODEL_FILENAME_DICT[model_name])
        self.model = _FastText(model_path)
        self.model_type = ""

        if "dolma" in model_name:
            self.model_type = "dolma"
        elif "mlfoundations" in model_name:
            self.model_type = "dclm"

    # def load_model(self, model_ref):
    #     import tempfile

    #     from fasttext.FastText import _FastText

    #     try:
    #         if not os.path.exists(self.local_filepath):
    #             directory, basename = os.path.dirname(self.local_filepath), os.path.basename(self.local_filepath)
    #             os.makedirs(directory, exist_ok=True)

    #             with open(self.local_filepath, "wb") as f:
    #                 f.write(model_ref.getvalue())

    #         print(f"Size of model: {os.path.getsize(self.local_filepath)}")
    #         model = _FastText(self.local_filepath)
    #     except Exception as e:
    #         print(e)
    #         print("failed to load model")

    #     return model

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
            for (
                label,
                score,
            ) in zip(label_arr[i], score_arr[i]):
                fasttext_quality_dict.update({label: score})

            # pyarrow schema does not like it if you do not have a key. We make it non-optional to not have the key
            if self.model_type == "dolma":
                if "__label__hq" not in fasttext_quality_dict:
                    fasttext_quality_dict.update({"__label__hq": 1.0 - fasttext_quality_dict.get("__label__lq")})
                elif "__label__lq" not in fasttext_quality_dict:
                    fasttext_quality_dict.update({"__label__lq": 1.0 - fasttext_quality_dict.get("__label__hq")})
            elif self.model_type == "dclm":
                if "__label__cc" not in fasttext_quality_dict:
                    fasttext_quality_dict.update({"__label__cc": 1.0 - fasttext_quality_dict.get("__label__hq")})
                elif "__label__hq" not in fasttext_quality_dict:
                    fasttext_quality_dict.update({"__label__hq": 1.0 - fasttext_quality_dict.get("__label__cc")})
                pass

            fasttext_key_name = "fasttext-quality"

            if self.model_type != "":
                fasttext_key_name = f"{self.model_type}-fasttext-quality"

            attributes_arr.append({fasttext_key_name: fasttext_quality_dict})

        res = {"id": batch["id"], "source": batch["source"], "attributes": attributes_arr}

        batch = res

        return res


class BERTQualityClassifier(BaseQualityClassifier):
    def __init__(self, model_name, *args, **kwargs):
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

        self.model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, documents):
        inputs = self.tokenizer(documents, return_tensors="jax", padding="longest", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)
        logits_list = logits.tolist()
        return logits_list

    def __call__(self, batch):
        scores = self.predict(batch["text"])
        int_scores = [int(round(max(0, min(score, 5)))) for score in scores]
        batch.update({"attributes": [{"fineweb-edu-quality": {"score": score, "int_score": int_score}} for score, int_score in zip(scores, int_scores)]})

        return batch


class AutoClassifier(BaseQualityClassifier):
    _MODEL_NAME_TO_CLS_DICT = {
        "fasttext": BatchFasttextQualityClassifier,
        "fineweb": BERTQualityClassifier,
    }

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        self.cls = self.from_model_path(model_name, *args, **kwargs)

    def __call__(self, batch):
        return self.cls.__call__(batch)

    @classmethod
    def from_model_path(cls, model_name, *args, **kwargs):
        for key in cls._MODEL_NAME_TO_CLS_DICT.keys():
            if key in model_name:
                print(f"Using {key} model")
                return cls._MODEL_NAME_TO_CLS_DICT[key](model_name, *args, **kwargs)

        raise ValueError(f"Model name {model_name} not supported")
