import os
import tempfile

import ray
from google.cloud import storage


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
        row.update(
            {
                "label": label,
                "score": score,
            }
        )
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
            fasttext_quality_dict.update({label: score})

        row.update({"attributes": {"fasttext-quality": fasttext_quality_dict}})

        return row


class BatchFasttextQualityClassifier(BaseQualityClassifier):
    def __init__(self, model_ref, local_filepath):
        self.local_filepath = local_filepath
        if "~" in self.local_filepath:
            self.local_filepath = os.path.expanduser(self.local_filepath)

        self.model = self.load_model(model_ref)

    def load_model(self, model_ref):
        from fasttext.FastText import _FastText

        try:
            if not os.path.exists(self.local_filepath):
                directory, basename = os.path.dirname(self.local_filepath), os.path.basename(self.local_filepath)
                os.makedirs(directory, exist_ok=True)

                with open(self.local_filepath, "wb") as f:
                    f.write(model_ref.getvalue())

            print(f"Size of model: {os.path.getsize(self.local_filepath)}")
            model = _FastText(self.local_filepath)
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
            for (
                label,
                score,
            ) in zip(label_arr[i], score_arr[i]):
                fasttext_quality_dict.update({label: score})

            # pyarrow schema does not like it if you do not have a key. We make it non-optional to not have the key
            if "__label__hq" not in fasttext_quality_dict:
                fasttext_quality_dict.update({"__label__hq": 1.0 - score})
            elif "__label__lq" not in fasttext_quality_dict:
                fasttext_quality_dict.update({"__label__lq": 1.0 - score})

            attributes_arr.append({"fasttext-quality": fasttext_quality_dict})

        res = {"id": batch["id"], "source": batch["source"], "attributes": attributes_arr}

        batch = res

        return res


class BERTQualityClassifier(BaseQualityClassifier):
    def __init__(self, model_name):
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

        self.model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, documents):
        inputs = self.tokenizer(documents, return_tensors="jax", padding="longest", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)
        logits_list = logits.tolist()
        scores = [int(round(max(0, min(logit, 5)))) for logit in logits_list]
        return scores

    def __call__(self, batch):
        scores = self.predict(batch["text"])

        batch.update({"attributes": [{"fineweb-edu-quality": score} for score in scores]})

        return batch
