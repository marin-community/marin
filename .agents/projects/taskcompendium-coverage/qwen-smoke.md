# Qwen Harbor smoke run

This is a model-serving smoke check for the first-wave Harbor lowerings. It is
not a benchmark result or a verifier-quality audit.

The endpoint was `http://spark-0143.local:8001/v1`, model
`Qwen3.8-27B-FP8`, with `max_tokens=4096` and
`chat_template_kwargs={"enable_thinking": false}`. The server was relaunched
with automatic tool choice and a Qwen-compatible tool parser.

The corrected tool-capable run passed end to end with `max_tokens=4096`:

- predicted action: reward `1.0`;
- Workplace single-step: reward `1.0`;
- derived Workplace three-step: steps 1--3 and aggregate reward `1.0`.

The six instruction-following variants produced valid submissions with
`max_tokens=1024`. Binary variants scored `0.0`; fractional variants scored
`0.5`. The three code-answer variants completed with `max_tokens=512`: raw code
scored `0.0`, while JSON and XML failed their declared extraction protocols.
These answer results are model smoke outcomes, not task-quality conclusions.

The successful tool run is retained at
`/var/folders/9h/dpqcjkwn5lx0sp3162765y780000gn/T/taskcompendium-qwen-tools-leszv0vw/summary.json`.
The six-variant format run is retained at
`/var/folders/9h/dpqcjkwn5lx0sp3162765y780000gn/T/taskcompendium-qwen-ifeval-nothink-r1jmilyy/summary.json`.
The three-variant code run is retained at
`/var/folders/9h/dpqcjkwn5lx0sp3162765y780000gn/T/taskcompendium-qwen-code-nothink-eyqv6wg8/summary.json`.

The serving controls are exposed through `DirectChatAgent`'s optional
`chat_template_kwargs` argument and are passed through to the
OpenAI-compatible request.
