# NeMo Gym two-row ingestion sweep

`lib/taskcompendium/examples/nemo_gym_sampling.py` selects two deterministic source
rows from each non-blend dataset in the current NVIDIA NeMo Gym collection. Each
sample records the Hub revision, Viewer config/split, selected row offsets, raw
fields, and Viewer feature metadata.

Each corpus review writes one JSON object with the corpus identity, both sample
offsets, and one disposition per sample: `accepted`, `rejected`, or `deferred`.
An accepted sample must name the existing TaskCompendium verifier contract and a
complete model-visible task, resources, capability requirements, and rendering.
It must not expose verifier, reward, judge, test, or reference-answer details in
the prompt. A rejected or deferred sample must name the source evidence and the
general blocker, such as missing source facts, unsound verifier, reactive user
simulation, unavailable environment state, or unsupported private verifier.

Do not accept a row only because it resembles another NeMo Gym dataset. Source
tool schemas, state, verifiers, and response history determine the semantic task.
Do not infer agent tool access from a source Docker image or an evaluator that
executes code privately. Preserve source revision and row offset in every record.

`summary.json` is the final disposition index. It supersedes any preliminary
finding in an individual review when a follow-up source-format review establishes
that the public projection would lose required state.
