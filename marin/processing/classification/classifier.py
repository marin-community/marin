import atexit
import hashlib
import os
import re
import time
import urllib.parse
from typing import Any, ClassVar

import fsspec

from marin.generation.llm_generation import LLMProvider


class BaseClassifier:
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: list[str]):
        raise NotImplementedError

    def __call__(self, batch: dict[str, Any]):
        raise NotImplementedError


class DummyClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name

    def predict(self, documents: list[str]):
        score = 1.0
        return [{"score": score} for _ in range(len(documents))]

    def __call__(self, batch: dict[str, Any]):
        scores = self.predict(batch["text"])
        batch.update({"attributes": [{"dummy-quality": score} for score in scores]})
        return batch


class FasttextClassifier(BaseClassifier):
    _MODEL_NAME_TO_MODEL_FILENAME_DICT: ClassVar[dict[str, str]] = {
        "mlfoundations/fasttext-oh-eli5": "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        "allenai/dolma-1_7-fasttext-quality-filter": "model.bin",
        "open-web-math/filtering-models": "math_score.bin",
    }

    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.model = self.load_model()

    def load_model(self):
        from fasttext.FastText import _FastText
        from filelock import FileLock
        from huggingface_hub import hf_hub_download, repo_exists

        # Classifier is stored in a remote storage.

        is_remote_or_local_path: bool = urllib.parse.urlparse(self.model_name).scheme or os.path.exists(self.model_name)
        try:
            is_huggingface_path: bool = not os.path.exists(self.model_name) and repo_exists(self.model_name)
        except Exception:
            print(
                f"Exception checking if {self.model_name} is a Hugging Face path. \
                This is normal for remote paths. Setting is_huggingface_path to False"
            )
            is_huggingface_path = False

        fs, fs_path = fsspec.core.url_to_fs(self.model_name)

        if not fs_path.endswith(".bin"):
            fs_path = os.path.join(fs_path, "model.bin")

        model_descriptor = hashlib.md5(self.model_name.encode()).hexdigest()
        lock_file = f"/tmp/{model_descriptor}.lock"
        success_file = f"/tmp/{model_descriptor}.success"

        model_basename = os.path.basename(fs_path)

        if is_remote_or_local_path:
            local_filepath = f"/tmp/{model_descriptor}/{model_basename}"
        else:
            local_filepath = f"/tmp/{model_descriptor}/{self._MODEL_NAME_TO_MODEL_FILENAME_DICT[self.model_name]}"

        if os.path.exists(success_file) and not os.path.exists(local_filepath):
            print(
                f"Warning: Success file found for {fs_path}, but model file not found. \
                    Removing stale success file {success_file}"
            )
            os.unlink(success_file)

        with FileLock(lock_file):
            if not os.path.exists(success_file):
                fs.makedirs(f"/tmp/{model_descriptor}", exist_ok=True)

                if is_remote_or_local_path:
                    fs.get(fs_path, local_filepath)
                elif is_huggingface_path:
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=self._MODEL_NAME_TO_MODEL_FILENAME_DICT[self.model_name],
                        local_dir=f"/tmp/{model_descriptor}",
                    )

                atexit.register(lambda: os.unlink(local_filepath))
                print(f"Downloaded model from {fs_path} to {local_filepath}")

                with open(success_file, "w") as f:
                    f.write("success")

                atexit.register(lambda: os.unlink(success_file))
            else:
                print(f"Model already downloaded to {local_filepath}")

        # Wait for the file to be ready, with a timeout
        timeout_s = 300  # 5 minutes
        start_time = time.time()
        while not os.path.exists(success_file):
            if time.time() - start_time > timeout_s:
                raise TimeoutError(f"Timeout waiting for {success_file}")
            time.sleep(1)

        assert os.path.exists(success_file) and os.path.exists(local_filepath), f"Model file {local_filepath} not found"

        model = _FastText(local_filepath)

        return model

    def predict(self, documents: list[str]):
        # TODO(chris): Add support for multi-class k > 2.
        return self.model.predict(documents, k=2)

    def __call__(self, batch: dict[str, Any]):
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
        for i in range(len(list(batch["text"]))):
            fasttext_quality_dict = dict(zip(label_arr[i], score_arr[i], strict=False))
            attributes_arr.append({self.attribute_name: fasttext_quality_dict})

        res = {"id": batch["id"], "attributes": attributes_arr}
        batch.update({"attributes": attributes_arr})

        return res


class BERTClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

        self.model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.attribute_name = attribute_name

    def predict(self, documents: list[str]) -> list[float]:
        inputs = self.tokenizer(documents, return_tensors="jax", padding="longest", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)
        logits_list = logits.tolist()
        return logits_list

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class FinewebEduClassifier(BERTClassifier):
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
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


class VLLMClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        super().__init__(model_name, attribute_name)

        self.preset_prompt = kwargs.get("preset_prompt", None)

        if self.preset_prompt is None:
            raise Exception(
                "The prompt format should be passed in with the key <preset-prompt> \
                so that we can score examples based on this rubric."
            )

        generation_kwargs = kwargs.get("generation_kwargs", {})
        engine_kwargs = kwargs.get("engine_kwargs", {})

        self.llm = LLMProvider(model_name=model_name, generation_kwargs=generation_kwargs, engine_kwargs=engine_kwargs)

    def predict(self, documents: list[str]) -> list[str]:
        generated_texts = self.llm.generate(prompts=documents)
        return generated_texts

    def extract_score_from_response(self, document: str) -> int:
        # Use regular expression to find the last occurrence of "Score:" followed by numbers
        matches = list(re.finditer(r"Score:\s*(\d+)", document))
        if matches:
            return int(matches[-1].group(1))  # Extract the number part from the last match. This is because sometimes
            # the model will include the word "Score" in its COT.
        else:
            # NOTE(chris): Maybe raise an error if the output response has no score? Implement retry?
            return -1

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        prompts = [self.preset_prompt.format(example=example) for example in batch["text"]]
        responses = self.predict(prompts)

        scores = [self.extract_score_from_response(response) for response in responses]
        batch.update(
            {
                "attributes": [
                    {self.attribute_name: {"score": score, "response": response}}
                    for score, response in zip(scores, responses, strict=False)
                ]
            }
        )

        return batch


class AutoClassifier(BaseClassifier):
    _MODEL_NAME_TO_CLS_DICT: ClassVar[dict[str, BaseClassifier]] = {
        "fasttext": FasttextClassifier,
        "fineweb": FinewebEduClassifier,
        "vllm": VLLMClassifier,
    }

    def __init__(self, model_name: str, attribute_name: str, model_type: str | None, *args, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.attribute_name = attribute_name
        self.cls = self.from_model_path(model_name, attribute_name, model_type, *args, **kwargs)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.cls.__call__(batch)

    @classmethod
    def from_model_path(
        cls, model_name_or_path: str, attribute_name: str, model_type: str | None, *args, **kwargs
    ) -> BaseClassifier:
        if model_type is None:
            for key in cls._MODEL_NAME_TO_CLS_DICT.keys():
                if key in model_name_or_path:
                    print(f"Using {key} model")
                    break
            else:
                raise ValueError(
                    f"Model type must be specified for model {model_name_or_path} or must have "
                    f"one of {cls._MODEL_NAME_TO_CLS_DICT.keys()} in the name."
                )
        else:
            key = model_type.lower()

        try:
            return cls._MODEL_NAME_TO_CLS_DICT[key](model_name_or_path, attribute_name, model_type, *args, **kwargs)
        except KeyError as e:
            raise ValueError(
                f"Model name {model_name_or_path} not supported. "
                f"Must have one of {cls._MODEL_NAME_TO_CLS_DICT.keys()} in the name."
            ) from e
