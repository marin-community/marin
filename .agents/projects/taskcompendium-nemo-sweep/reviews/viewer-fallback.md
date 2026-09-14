# Viewer fallback review

The fallback sampler returned two first rows for six datasets after ordinary Viewer size/range requests failed. Revisions, configs, splits, and offsets are recorded in `viewer-fallback.json`.

The two instruction-following rows are accepted through the existing `import_instruction_following` importer: each has a complete user prompt, parallel constraint IDs/kwargs, and a supported binary constraint verifier. OpenMathReasoning and math Stack Overflow are deferred because expected answers do not include a pinned NeMo verifier or normalization contract. Structured-Outputs-v2 is deferred because its XML/YAML JSON Schemas need a source-specific validator. ReasoningGym lacks an inspectable deterministic verifier, and SysBench depends on persona plus hidden LLM-judge criteria; both are deferred.
