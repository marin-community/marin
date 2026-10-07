# Marin inference components

This package provides serving backends, request types and provider transports.
[serve.py](serve.py), [config.py](config.py) and [iris.py](iris.py) configure and run
Iris-backed inference; [backend.py](backend.py) defines the backend interface.
The [dashboard](dashboard/README.md) provides the serving UI.

[openai_chat.py](openai_chat.py) sends one OpenAI-compatible chat request with an
explicit endpoint, credential and timeout. Its caller owns batching and bounded
retries. [openai_batch.py](openai_batch.py) supplies the batch API transport.
Dataset review decisions, rubrics and inference reuse belong to the calling
[task curation pipeline](../../../../../experiments/post_training/task_curation/README.md).
These transports contain no source quality policy.
