# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import copy
import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import ray
from typing import Any, ClassVar

import torch

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

import fsspec
import lz4.frame

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


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
        "julien-c/fasttext-language-id": "lid.176.bin",
    }

    def __init__(self, model_name: str, attribute_name: str, k: int = 2, *args, **kwargs):
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.model = self.load_model()
        self.k = k

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
        return self.model.predict(documents, k=self.k)

    def __call__(self, batch: dict[str, Any]):
        texts = []
        for text in list(batch["text"]):
            if text:
                # FastText uses newline to delimit new examples. Thus, if one examples
                # has multiple newlines then it will appear as if it were
                # multiple examples. So, we replace `\n` with a space to avoid this.
                # https://arc.net/l/quote/mfbqvtry
                text = text.replace("\n", " ")
            texts.append(text)

        label_arr, score_arr = self.predict(texts)

        attributes_arr = []
        for i in range(len(list(batch["text"]))):
            fasttext_quality_dict = dict(zip(label_arr[i], score_arr[i], strict=False))
            attributes_arr.append({self.attribute_name: fasttext_quality_dict})

        batch.update({"attributes": attributes_arr})

        return batch


class BERTClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, max_label: int = 5, *args, **kwargs):
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

        self.model = FlaxAutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.attribute_name = attribute_name
        self.max_label = max_label

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
        int_scores = [round(max(0, min(score, self.max_label))) for score in scores]
        batch.update(
            {
                "attributes": [
                    {self.attribute_name: {"score": score, "int_score": int_score}}
                    for score, int_score in zip(scores, int_scores, strict=False)
                ]
            }
        )

        return batch


class BERTQualityClassifier(BaseClassifier):
    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        print(f"Loading model from {model_name}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            fs, fs_path = fsspec.core.url_to_fs(model_name)
            fs.get(fs_path + "/*", tmp_dir)

            device = xm.xla_device()
            self.model = AutoModelForSequenceClassification.from_pretrained(tmp_dir).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            with fsspec.open(os.path.join(tmp_dir, "label_index.json"), "r") as f:
                label_idx = json.load(f)
                self.labels = [k for k, v in sorted(label_idx.items(), key=lambda item: item[1])]

        self.attribute_name = attribute_name

    @torch.no_grad()
    def predict(self, documents: list[str]) -> list[float]:
        inputs = self.tokenizer(
            documents,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        xm.mark_step()
        probs = probs.squeeze(-1)
        return probs.tolist()

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        probs = self.predict(batch["text"])

        attributes = []
        for i in range(len(list(batch["text"]))):
            quality_dict = dict(zip(self.labels, probs[i], strict=False))
            attributes.append({self.attribute_name: quality_dict})

        res = {"id": batch["id"], "attributes": attributes}
        batch.update({"attributes": attributes})

        return res


class GTEClassifier(FinewebEduClassifier):
    """Classifier that uses the Alibaba-NLP/gte-base-en-v1.5 model to classify documents"""

    def __init__(self, model_name: str, attribute_name: str, max_length: int, max_label: int = 5, *args, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = xm.xla_device()

        is_file_path: bool = urllib.parse.urlparse(model_name).scheme or os.path.exists(model_name)
        if is_file_path:
            print(f"Downloading model from {model_name}")
            with tempfile.TemporaryDirectory() as tmp_dir:
                fs, fs_path = fsspec.core.url_to_fs(model_name)
                fs.get(fs_path + "/*", tmp_dir)

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    tmp_dir, trust_remote_code=True, output_hidden_states=False
                ).to(device)
                self.tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            print(f"Model downloaded to {tmp_dir}")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, output_hidden_states=False
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.attribute_name = attribute_name
        self.max_length = max_length
        self.max_label = max_label

    @torch.no_grad()
    def predict(self, documents: list[str]) -> list[float]:
        inputs = self.tokenizer(
            documents, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1)
        xm.mark_step()
        return logits.tolist()


class CompressionClassifier(BaseClassifier):
    """A classifier that calculates LZ4 compression ratios for text documents.

    The compression ratio is calculated as (compressed_size / original_size).
    Higher ratios indicate text that is harder to compress (potentially more random/noisy),
    while lower ratios indicate text that compresses well (potentially more structured/repetitive).
    """

    def __init__(self, model_name: str, attribute_name: str, *args, **kwargs):
        super().__init__(model_name, attribute_name)

    def __call__(self, batch):
        compression_ratios = []
        for text in batch["text"]:
            text_bytes = text.encode("utf-8")
            # Handle empty text case
            if len(text_bytes) == 0:
                # Set a default ratio value for empty strings
                ratio = 2.0
            else:
                compressed = lz4.frame.compress(text_bytes)
                ratio = len(compressed) / len(text_bytes)

            compression_ratios.append({self.attribute_name: ratio})
        return {"attributes": compression_ratios}


class vLLMClassifier(BaseClassifier):
    """A classifier that uses vLLM for text generation-based classification.

    This classifier uses the vLLMTextGeneration pipeline to generate classification
    scores or labels by prompting the model with input text and a classification template.
    Uses dataset processor functions to parse the generated text into scores.
    """

    def __init__(
        self,
        model_name: str,
        attribute_name: str,
        template: str,
        post_process_fn: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        save_original_generation: bool = False,
        max_doc_tokens: int = 7000,
        apply_chat_template: bool = True,
        prompt_column: str = "text",
        save_templated_prompt: bool = False,
        generated_text_column_name: str = "generated_text",
        *args,
        **kwargs,
    ):
        super().__init__(model_name, attribute_name)

        # Import vLLM text generation pipeline
        from marin.generation.pipeline import vLLMTextGeneration

        self.is_ready = False
        self.text_generator = vLLMTextGeneration(
            model_name=model_name,
            engine_kwargs=engine_kwargs,
            generation_kwargs=generation_kwargs,
            template=template,
            prompt_column=prompt_column,
            apply_chat_template=apply_chat_template,
            save_templated_prompt=save_templated_prompt,
            max_doc_tokens=max_doc_tokens,
            generated_text_column_name=generated_text_column_name,
        )
        self.is_ready = True  # after vllm instantiation, we set this to true
        self.save_original_generation = save_original_generation
        self.post_process_fn = post_process_fn
        self.generated_text_column_name = generated_text_column_name
        self.prompt_column = prompt_column

        if post_process_fn is None:
            self.save_original_generation = True

    def ping(self):
        return self.is_ready

    def _default_post_process_fn(self, generation: str) -> dict[str, Any]:
        """Default score extractor that looks for numbers 0-5."""
        import re

        # Try to extract a number from the generation
        numbers = re.findall(r"\b([0-5])\b", generation.strip())
        if numbers:
            return {"score": int(numbers[0])}
        else:
            # Default to middle score if no valid number found
            return {"score": -1}

    def _default_parse_strategy_fn(self, generation: str) -> dict[str, Any]:
        import re

        # Extract strategy names from lines that start with '##'
        # Example input:
        # ## X
        # ## Y
        # ## Z
        matches = list(re.finditer(r"(?m)^\s*##\s*(.+?)\s*$", generation))
        strategies: list[str] = []
        seen: set[str] = set()

        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(generation)
            body = generation[start:end].strip()
            strategy = match.group(1).strip()
            if body:
                strategy = f"{strategy}\n{body}"
            if strategy and strategy not in seen:
                strategies.append(strategy)
                seen.add(strategy)

        return {"strategies": strategies}

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process a batch of documents."""
        # Use the text generation pipeline to generate responses
        batch_with_generations = self.text_generator(batch)

        # Process the generated text to extract classification scores
        attributes = []
        generated_texts = batch_with_generations[self.generated_text_column_name]

        if self.post_process_fn is not None:
            if self.post_process_fn == "score":
                post_process_fn = self._default_post_process_fn
            elif self.post_process_fn == "active_reading":
                post_process_fn = self._default_parse_strategy_fn
            else:
                raise ValueError(f"Invalid post_process_fn: {self.post_process_fn}")

            for generated_text in generated_texts:
                attributes.append({self.attribute_name: post_process_fn(generated_text)})

            batch.update({"attributes": attributes})

        if self.save_original_generation:
            batch[self.generated_text_column_name] = generated_texts

        # print("Save original generation: ", self.save_original_generation)
        # print(batch)

        return batch


class vLLMClassifierWithDocsAndAttributes(vLLMClassifier):
    """LLM generation based on a template, attribute, and document

    The idea is that there is a function that we will have to apply
    that applies a transformation on the template based on the
    attributes of the given doc.

    In symbolic form:
    vLLM classifier: text -> M(text, template) where template is the
    given template, text is the document text, and M is the model

    vLLM classifier with docs and attributes:
    template + attributes -> f(template, attributes) -> template'
    then template' + text -> M(text, template') -> generated_text
    We apply some tranformation f on the template + attribute of the
    given document to get a new template then
    apply that new template to the text.
    """

    def __init__(
        self,
        model_name: str,
        attribute_name: str,
        template: str,
        attribute_list_within_attributes_name: str,
        post_process_fn: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        save_original_generation: bool = False,
        max_doc_tokens: int = 7000,
        apply_chat_template: bool = True,
        prompt_column: str = "text",
        save_templated_prompt: bool = False,
        generated_text_column_name: str = "generated_text",
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name,
            attribute_name,
            template,
            post_process_fn,
            engine_kwargs,
            generation_kwargs,
            save_original_generation,
            max_doc_tokens,
            apply_chat_template,
            prompt_column,
            save_templated_prompt,
            generated_text_column_name,
            *args,
            **kwargs,
        )
        self.attribute_list_within_attributes_name = attribute_list_within_attributes_name
        self.template = template

    def _change_template_based_on_attributes(self, attribute: str) -> str:
        # NOTE(Chris) If we don't clean this up, it will break if we have a placeholder in the template
        # since it expects a string to replace it. for example, imagine the attribute contained \boxed{10}
        # it would expect us to have a string.format(10=...) which is not valid.
        cleaned_attribute = attribute.replace("{", "").replace("}", "")
        return self.template.format(attribute=cleaned_attribute, example="{example}")  # keep example as placeholder

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process a batch of documents."""
        # Use the text generation pipeline to generate responses
        # batch_with_generations = self.text_generator(batch)

        # Process the generated text to extract classification scores
        attributes = []
        attribute_dict_list = batch["attributes"]

        # We assume that we want to change the prompt template based on what each attribute is.
        # Number of strategies * number of text documents = number of flattened batches
        templates = []

        # requires attributes as an input key
        flattened_batch_keys_to_values = {key: [] for key in batch.keys()}
        current_index = 0
        for i in range(len(batch["text"])):
            for attribute_idx, attribute in enumerate(
                attribute_dict_list[i][self.attribute_name][self.attribute_list_within_attributes_name]
            ):
                for key, value in batch.items():
                    if isinstance(value[i], dict):
                        deepcopied_value = copy.deepcopy(value[i])
                        flattened_batch_keys_to_values[key].append(deepcopied_value)
                    else:
                        flattened_batch_keys_to_values[key].append(value[i])
                flattened_batch_keys_to_values["attributes"][current_index]["attribute_idx"] = attribute_idx
                templates.append(self._change_template_based_on_attributes(attribute))
                current_index += 1

        self.text_generator.template = templates

        # Generate text for each flattened batch
        batch_with_generations = self.text_generator(flattened_batch_keys_to_values)
        generated_texts = batch_with_generations[self.generated_text_column_name]
        if self.post_process_fn is not None:
            for generated_text in generated_texts:
                attributes.append({self.attribute_name: self.post_process_fn(generated_text)})

            flattened_batch_keys_to_values.update({"attributes": attributes})

        if self.save_original_generation:
            flattened_batch_keys_to_values[self.generated_text_column_name] = generated_texts

        return flattened_batch_keys_to_values


class AutoClassifier(BaseClassifier):
    _MODEL_NAME_TO_CLS_DICT: ClassVar[dict[str, BaseClassifier]] = {
        "fasttext": FasttextClassifier,
        "fineweb": FinewebEduClassifier,
        "gte": GTEClassifier,
        "compression": CompressionClassifier,
        "bert": BERTQualityClassifier,
        "vllm": vLLMClassifier,
        "dummy": DummyClassifier,
        "vllm_with_docs_and_attributes": vLLMClassifierWithDocsAndAttributes,
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
            return cls._MODEL_NAME_TO_CLS_DICT[key](model_name_or_path, attribute_name, *args, **kwargs)
        except KeyError as e:
            raise ValueError(
                f"Model name {model_name_or_path} not supported. "
                f"Must have one of {cls._MODEL_NAME_TO_CLS_DICT.keys()} in the name."
            ) from e


@ray.remote(concurrency_groups={"ping": 2})
class AutoClassifierRayActor(AutoClassifier):
    """Ray Actor Wrapper around the AutoClassifier class.

    The mapping operation f(x) is preserved, we just add a ping method to this actor. This ping method
    is used to check if the actor is alive from the autoscaler in .autoscaler. We set the concurrecy group
    to 2 which will allow multiple threads to access the ping method concurrently, so when a batch is running
    on the actor, the ping method can still be accessed by the autoscaler.
    """

    def __init__(self, model_name: str, attribute_name: str, model_type: str | None, *args, **kwargs):
        super().__init__(model_name, attribute_name, model_type, *args, **kwargs)

    @ray.method(concurrency_group="ping")
    def ping(self):
        return True
