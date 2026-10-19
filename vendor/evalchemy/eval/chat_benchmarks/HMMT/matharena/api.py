from loguru import logger
import re
import os
from tqdm import tqdm
from google import genai
from google.genai import types
from openai import OpenAI
from together import Together
import anthropic
from anthropic.types import ThinkingBlock, TextBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import base64
import requests
import json

# Import tempfile to create temporary files
import tempfile


def encode_image(image_path):
    image_type = image_path.split(".")[-1]
    with open(image_path, "rb") as image_file:
        return image_type, base64.b64encode(image_file.read()).decode("utf-8")


class APIQuery:
    def __init__(
        self,
        model,
        timeout=6000,
        max_tokens=None,
        api="openai",
        max_retries=50,
        concurrent_requests=30,
        is_chat=True,
        no_system_messages=False,
        read_cost=1,
        write_cost=1,
        sleep_on_error=60,
        sleep_after_request=0.1,
        throw_error_on_failure=False,
        max_tokens_param="max_tokens",
        reasoning_effort=None,
        batch_processing=False,
        openai_responses=False,
        **kwargs,
    ):
        # if "think" in model and api == "google":
        #     logger.info("Google Think model does not allow chat.")
        #     is_chat = False # think model cannot handle chat
        #     max_tokens_param = "max_output_tokens"
        if ("o1" in model or "o3" in model or "o4" in model) and api == "openai":
            logger.info("Not using system messages for o1/o3/o4 model.")
            no_system_messages = True  # o1 model cannot handle system messages
            max_tokens_param = "max_completion_tokens"
            if "--" in model:
                model, reasoning_effort = model.split("--")
                logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if api == "anthropic" and "claude-3-7" not in model:
            logger.info("Setting max tokens to 8192 for Anthropic API.")
            max_tokens = min(8192, max_tokens)
        if api == "deepseek":
            logger.info("Setting max tokens to 8192 for DeepSeek API.")
            max_tokens = min(8192, max_tokens)
        if api not in ["anthropic", "openai"] and batch_processing:
            logger.warning("Batch processing is only supported for the Anthropic API and OpenAI API.")
            batch_processing = False
        if openai_responses:
            max_tokens_param = "max_output_tokens"

        self.kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.is_chat = is_chat
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.write_cost = write_cost
        self.batch_processing = batch_processing
        self.openai_responses = openai_responses

        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.openai_responses:
                self.kwargs["reasoning_effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        self.api = api
        self.api_key = None
        self.base_url = None

        self.initialize_api_keys()

    def kwarg_remover(self, api, model, kwargs):
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def initialize_api_keys(self):
        if self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            # if not "think" in self.model:
            #     self.api = "openai"
            #     self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "hyperbolic":
            self.api_key = os.getenv("HYPERBOLIC_API_KEY")
            self.base_url = "https://api.hyperbolic.xyz/v1"
            self.api = "openai"
        elif self.api == "sambanova":
            self.api_key = os.getenv("SAMBA_API_KEY")
            self.base_url = "https://api.sambanova.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.api == "fireworks":
            self.api_key = os.getenv("FIREWORKS_API_KEY")
            self.base_url = "https://api.fireworks.ai/inference/v1"
            self.api = "openai"
        elif self.api == "vllm":
            self.api_key = "token-abc123"
            self.api = "openai"
            self.base_url = f"http://localhost:8000/v1"
            # command = f"vllm serve {self.model} --dtype auto --api-key token-abc123"
            # Launch the command in the background.
            # subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Poll the server until it's running.
        else:
            raise ValueError(f"API {self.api} not supported.")

        assert self.api_key is not None, f"API key not found."

    def prepare_query(self, query):
        query, image_path = query
        if not self.is_chat:
            output_query = query[0]["content"]
            for message in query:
                output_query += f"\n\n{'=' * 20}{message['role']}{'=' * 20}\n\n{message['content']}"
            return output_query, image_path
        elif self.no_system_messages:
            # convert system role to user role
            query = [
                {"role": message["role"] if message["role"] != "system" else "user", "content": message["content"]}
                for message in query
            ]
        return query, image_path

    def get_cost(self, response):
        cost = response["input_tokens"] * self.read_cost + response["output_tokens"] * self.write_cost
        return cost / (10**6)

    def run_queries(self, queries):
        queries_actual = []
        for query in queries:
            if not isinstance(query, tuple):
                queries_actual.append((query, None))
            else:
                queries_actual.append(query)
        if self.api == "vllm":
            while True:
                try:
                    response = requests.get(f"{self.base_url}", timeout=1)
                    if response.status_code == 401:  # unauthorized, because no api key here
                        break
                except Exception:
                    pass
                time.sleep(5)
                logger.info("Waiting for VLLM server to start...")
            logger.info("VLLM server started.")

        logger.info(f"Running {len(queries_actual)} queries.")

        if self.batch_processing:
            if self.api == "openai":
                processed_results = self.openai_batch_processing(queries_actual)
            else:
                processed_results = self.anthropic_batch_processing(queries_actual)
            for idx, result in enumerate(processed_results):
                detailed_cost = {
                    "cost": self.get_cost(result),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                yield idx, result["output"], detailed_cost
        else:
            with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
                future_to_index = {
                    executor.submit(self.run_query_with_retry, query): i for i, query in enumerate(queries_actual)
                }
                for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
                    idx = future_to_index[future]
                    result = future.result()
                    detailed_cost = {
                        "cost": self.get_cost(result),
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                    }
                    yield idx, result["output"], detailed_cost

    def run_query_with_retry(self, query):
        i = 0
        while i < self.max_retries:
            try:
                output = self.run_query(query)
                time.sleep(self.sleep_after_request)
                return output
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.sleep_on_error)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    i += 1
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max retries reached.")
        else:
            return {
                "output": "",
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def run_query(self, query):
        query = self.prepare_query(query)
        if self.api == "openai":
            return self.openai_query(query)
        elif self.api == "together":
            return self.together_query(query)
        elif self.api == "google":
            return self.google_query(query)
        elif self.api == "anthropic":
            return self.anthropic_query(query)
        elif self.api == "openrouter":
            return self.openrouter_query(query)

    def postprocess_anthropic_result(self, result):
        output_text = ""

        for content in result.content:
            if isinstance(content, ThinkingBlock):
                output_text += "<think>\n" + content.thinking + "</think>\n\n"
            elif isinstance(content, TextBlock):
                output_text += content.text
                break
        return {
            "output": output_text,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        }

    def anthropic_batch_processing(self, queries, error_repetition=0):
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                for _ in range(len(queries))
            ]

        text_queries = [query[0] for query in queries]
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []

        for i, text_query in enumerate(text_queries):
            kwargs_here = self.kwargs.copy()
            if text_query[0]["role"] == "system":
                kwargs_here["system"] = text_query[0]["content"]
                text_query = text_query[1:]

            request = Request(
                custom_id=f"apiquery-{i}",
                params=MessageCreateParamsNonStreaming(model=self.model, messages=text_query, **kwargs_here),
            )
            requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except:
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s.")
                pass
            if any(
                [
                    current_request_counts[key] != dict(message_batch.request_counts)[key]
                    for key in current_request_counts
                ]
            ):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(
                    f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}"
                )
            if message_batch.processing_status == "ended":
                break
            time.sleep(10)

        outputs = []
        repeat_indices = []

        while True:
            try:
                results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to Anthropic: {e}. Retrying in 10 seconds.")
                time.sleep(10)

        for i, result in enumerate(results):
            if result.result.type == "succeeded":
                outputs.append(self.postprocess_anthropic_result(result.result.message))
            else:
                outputs.append(None)
                repeat_indices.append(i)
                if result.result.type == "errored":
                    logger.error(result.result.error)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.anthropic_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output

        return outputs

    def anthropic_query(self, query):
        query, image_path = query
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        system_message = anthropic.NOT_GIVEN
        if query[0]["role"] == "system":
            system_message = query[0]["content"]
            query = query[1:]
        result = client.messages.create(model=self.model, messages=query, system=system_message, **self.kwargs)

        return self.postprocess_anthropic_result(result)

    def openrouter_query(self, query):
        query, image_path = query
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        query_key = "messages" if self.is_chat else "prompt"

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={"model": self.model, query_key: query, **self.kwargs},
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        json_response = response.json()

        if "choices" not in json_response:
            raise Exception(f"Error: {json_response}")

        if self.is_chat:
            output = json_response["choices"][0]["message"]["content"]
            if (
                "reasoning_content" in json_response["choices"][0]["message"]
                and json_response["choices"][0]["message"]["reasoning_content"] is not None
            ):
                output = json_response["choices"][0]["message"]["reasoning_content"] + "</think>\n\n" + output
            return {
                "output": output,
                "input_tokens": json_response["usage"]["prompt_tokens"],
                "output_tokens": json_response["usage"]["completion_tokens"],
            }
        else:
            output = json_response["choices"][0]["text"]
            output = self.skip_repetition(output, query)

            reasoning_content = ""
            if (
                "reasoning_content" in json_response["choices"][0]
                and json_response["choices"][0]["reasoning_content"] is not None
            ):
                reasoning_content = json_response["choices"][0]["reasoning_content"]
            if "reasoning" in json_response["choices"][0] and json_response["choices"][0]["reasoning"] is not None:
                reasoning_content = json_response["choices"][0]["reasoning"]

            text = "</think>\n\n"
            if len(output) == 0:
                text = ""
            output = reasoning_content + text + output

            return {
                "output": output,
                "input_tokens": json_response["usage"]["prompt_tokens"],
                "output_tokens": json_response["usage"]["completion_tokens"],
            }

    def google_query(self, query):
        client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})
        query, image_path = query
        parts = []
        if image_path is not None:
            file = client.files.upload(file=image_path)
            assert len(query) == 1
            parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
        parts.append(types.Part.from_text(text=query[0]["content"]))
        query = [types.Content(role="user", parts=parts)]

        # if "think" in self.model:
        #     config['thinking_config'] = {'include_thoughts': True}
        # config = None
        response = client.models.generate_content(model=self.model, contents=query, **self.kwargs)
        return {
            "output": "\n\n".join(
                [response.candidates[0].content.parts[i].text for i in range(len(response.candidates[0].content.parts))]
            ),
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }

    def together_query(self, query):
        client = Together()
        query, image_path = query
        response = client.chat.completions.create(model=self.model, messages=query, **self.kwargs)
        output = response.choices[0].message.content
        if hasattr(response.choices[0].message, "reasoning_content"):
            output = response.choices[0].message.reasoning_content + "\n\n" + output
        return {
            "output": output,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def openai_batch_processing(self, queries, error_repetition=0):
        if error_repetition >= self.max_retries:
            return [
                {
                    "output": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                for _ in range(len(queries))
            ]
        text_queries = [query[0] for query in queries]
        jsonl_queries = []

        for i, query in enumerate(text_queries):
            request = {
                "custom_id": f"apiquery-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": self.model, "messages": query, **self.kwargs},
            }
            jsonl_queries.append(request)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(file=open(tmp.name, "rb"), purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(
            f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}."
        )

        request_counts = dict(batch.request_counts)

        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:
                logger.warning(f"Error connecting to OpenAI. Retrying in 10s.")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(
                    f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}"
                )
            if batch.status == "completed":
                break
            time.sleep(10)

        while True:
            try:
                file_response = client.files.content(batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to OpenAI: {e}. Retrying in 10 seconds.")
                time.sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        outputs = [None for _ in range(len(queries))]
        repeat_indices = []

        for result in json_response:
            index = int(result["custom_id"].split("-")[-1])
            if result["response"]["status_code"] != 200:
                repeat_indices.append(index)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    outputs[index] = {
                        "output": result["response"]["body"]["choices"][0]["message"]["content"],
                        "input_tokens": result["response"]["body"]["usage"]["prompt_tokens"],
                        "output_tokens": result["response"]["body"]["usage"]["completion_tokens"],
                    }
                except Exception as e:
                    logger.error(f"Error: {e}")
                    repeat_indices.append(index)
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            repeat_outputs = self.openai_batch_processing(repeat_queries, error_repetition + 1)
            for i, output in zip(repeat_indices, repeat_outputs):
                outputs[i] = output

        return outputs

    def openai_query(self, query):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        query, image_path = query
        if image_path is not None:
            image_type, base64_image = encode_image(image_path)
            query.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{base64_image}"}}
                    ],
                }
            )

        if any([kw in self.model for kw in ["o1", "o3", "o4"]]) and "temperature" in self.kwargs:
            self.kwargs.pop("temperature")

        if not self.openai_responses:
            response = client.chat.completions.create(
                model=self.model, messages=query, timeout=self.timeout, **self.kwargs
            )
            output = response.choices[0].message.content
            if (
                hasattr(response.choices[0].message, "reasoning_content")
                and response.choices[0].message.reasoning_content is not None
            ):
                output = response.choices[0].message.reasoning_content + "\n\n" + output
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            if self.base_url is not None and "api.x.ai" in self.base_url:
                output_tokens += response.usage.completion_tokens_details.reasoning_tokens
        else:
            response = client.responses.create(model=self.model, input=query, timeout=self.timeout, **self.kwargs)
            try:
                output = response.output[-1].content[0].text
            except Exception as e:
                if response.incomplete_details.reason == "max_output_tokens":
                    logger.info(
                        "Found incomplete response because of max output tokens. Setting output to the empty string information."
                    )
                    output = "<Empty response because model reached the maximum output tokens limit.>"
                else:
                    raise e
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
