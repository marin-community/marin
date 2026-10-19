import os
import requests
import json
import time
from openai._exceptions import RateLimitError


class _CompletionOutput:
    def __init__(self, choices=[]):
        self.choices = choices


class _Message:
    def __init__(self, role="", content=""):
        self.role = role
        self.content = content


class _Choice:
    def __init__(self, message=_Message()):
        self.message = message


class _Completions:
    def __init__(self, key):
        self.url = "https://fast-api.snova.ai/v1/chat/completions"
        self.headers = {"Authorization": f"Basic {key}", "Content-Type": "application/json"}
        self.NUM_SECONDS_TO_SLEEP = 30
        self.key = key

    def create(self, model, response_format, max_tokens, messages):
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "model": model,
            "response_format": response_format,
            "stream": True,
            "stream_options": {"include_usage": True},
            "stop": ["[INST", "[INST]", "[/INST]", "[/INST]"],
        }
        return self._query_url(payload)

    def format_output(self, response_text):
        return _CompletionOutput(choices=[_Choice(message=_Message(content=response_text))])

    def _query_url(self, payload):
        post_response = requests.post(self.url, json=payload, headers=self.headers, stream=True)
        if (
            post_response.status_code == 503
            or post_response.status_code == 504
            or post_response.status_code == 401
            or post_response.status_code == 429
        ):
            print(post_response.content)
            print(f"Attempt failed due to rate limit or gate timeout. Trying again...")
            time.sleep(self.NUM_SECONDS_TO_SLEEP)
            raise RateLimitError("Rate limit or gateway timeout", response=post_response, body=None)
        response_text = ""
        for line in post_response.iter_lines():
            if line.startswith(b"data: "):
                data_str = line.decode("utf-8")[6:]
                try:
                    line_json = json.loads(data_str)

                    if (
                        "choices" in line_json
                        and len(line_json["choices"]) > 0
                        and "content" in line_json["choices"][0]["delta"]
                    ):
                        response_text += line_json["choices"][0]["delta"]["content"]

                except json.JSONDecodeError as e:
                    pass
        output = self.format_output(response_text)
        return output


class _Chat:
    def __init__(self, key):
        self.completions = _Completions(key)


class SnovaClient:
    def __init__(self, key):
        self.chat = _Chat(key)


if __name__ == "__main__":
    fake_message = _Message("system", "Hello, how are you?")
    print(fake_message)
