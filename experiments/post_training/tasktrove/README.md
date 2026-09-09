# TaskTrove conversion

Converts the [open-thoughts/TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove)
Harbor task binaries to the Marin verifier format: a `[verifier]` table in `task.toml`, one of
four shared Dockerfile tiers, and a three-line `tests/test.sh` that hands off to the
`tasktrove-verify` tool baked into the image. `verifier_spec.py` is the contract.

```
python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --stage templates --run
python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --run
```

Stages: `raw` (HF download) → `fingerprints` (one template id per task) → `templates` (index plus
one exemplar per template) → `converted` (apply converters) → `validated` (ledger; fails on
malformed output). Sources are routed by `source_verdicts.json`: `drop` sources are skipped,
`rewrite` sources are recorded as `rewrite_source` for the agent rewrite queue, `keep` sources
go through a converter.

## Writing a converter

The corpus is about 100 templates stamped onto 1.7M tasks; 100 templates cover 99.4% of rows and
the long tail is per-repository SWE setup scripts. One converter serves one template id.

1. Open `templates.md` in the `templates` artifact. Pick a template with no entry in
   `converters/registry.py`; its row names the sources, the template code files, and whether an
   oracle solution ships.
2. Read `templates/<template_id>/exemplar/`: the old `tests/test.sh`, verifier code, and the
   per-task data file(s) the converter must map.
3. Write `convert_<name>(task: TaskFiles) -> ConvertedTask` in a module under `converters/`,
   following `converters/nemotron_gym.py`. Read only the per-task data files; pass
   `instruction.md` through; choose the image tier; raise `ValueError` for a task the template
   cannot grade soundly (a multi-character MCQ gold, zero test cases) so it lands in the ledger
   as `converter_error` rather than silently converting.
4. Register the template id in the module's `CONVERTERS` dict and merge it in `registry.py`.
5. Run the `validated` stage on a `dev` version and check `ledger.json`: the template's rows should
   move from `no_converter` to `converted` with no `invalid` entries.

Container gates (oracle passes, no-op scores 0, old and new graders agree) run in the verifier
image and are a separate stage, not part of static validation.
