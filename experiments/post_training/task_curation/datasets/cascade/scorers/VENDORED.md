# Cascade terminal-answer scorers

The source is [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym/tree/39d2bde61e426f93e4bc00c0255fc1d839f859e6)
at `39d2bde61e426f93e4bc00c0255fc1d839f859e6`, Apache-2.0. `extraction.json`
records upstream and extracted-file SHA256 values. `LICENSE` preserves the primary repository grant.

`mcqa.py` keeps every top-level helper/constant and the original `_programmatic_score` method.
The webserver superclass is removed, configuration is supplied through a small constructor, and
transport-only response models use `SimpleNamespace`; their `reward` field is consumed unchanged.
`grade.py` supplies the source request's actual grading fields through a Pydantic record.

`structured_outputs.py` keeps the source `SchemaType` and its parsing, strict-schema,
XML/CSV coercion and evaluation methods. Only server initialization is replaced with configuration
injection. XML coercion stays enabled. There is no added Markdown-fence fallback or schema repair.

`math_with_judge.py` keeps every top-level mathematical helper and the source class's output-muting
and delimiter-removal methods. The primary `resources_servers/math_with_judge/configs/math_with_judge.yaml`
at this pin explicitly sets `should_use_judge: false`; format-tolerant extraction defaults false.
The adapter preserves its boxed-answer gate, source math-verify extraction, ten-second child-process
limit and zero reward on a symbolic timeout. It does not use the similarly named generic verifyit
math comparator or an unconfigured judge fallback.

No original passing response is claimed as an oracle. All source grading contracts remain private.
IF's required verifier package failed the binding license checker; workplace requires its original
27-tool live stateful runtime; SWE's released request inputs are empty and require its reconstruction
runtime. These components retain explicit conversion exclusions instead of terminal-answer tasks.
