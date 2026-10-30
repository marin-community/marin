# LAB-Bench research run source

Source snapshots for [issue #10575](https://github.com/marin-community/marin/issues/10575).
The `.py.txt` suffix preserves the exact executed source as evidence without
registering these historical, side-effecting workers as maintained entrypoints.
Rename to `.py` to execute in the pinned environment after configuring a new
output destination. Existing run prefixes refuse overwrites.

| Source | Run |
| --- | --- |
| [Luna worker](luna-full-worker.py.txt), [launcher](luna-launcher.py.txt) | Original 741-question GPT 6 Luna run: high reasoning, 128000 output tokens, concurrency 8. |
| [GLM worker](glm-panel216-worker.py.txt) | Frozen 216-question GLM-5.3 panel: medium reasoning, 8192 thinking tokens, 16384 total output tokens, concurrency 1. |
| [Selection and aggregation](subset.py.txt) | Five SeqQA questions per subtask, seed 42, score-independent SHA-256 ranking; full ProtocolQA and CloningScenarios. |

Workers use the Evalchemy prompt builder and shared grader from Marin commit
`6ae3623e76259a768314a440a4e9e7a43e87447c`. These are direct endpoint requests,
not native Marin-to-EvalDash validation. Private endpoint-discovery and
credential-loading wrappers are excluded. No credential values are included.
The GLM worker expects `LABBENCH_GLM_BASE` and `OPENAI_API_KEY` from the authorized
launcher; the Luna worker uses `OPENROUTER_API_KEY`. Both require object-storage
access and `IRIS_OUTPUT_DIR`. The independent evaluator environment is installed
with `uv run --directory vendor/evalchemy --frozen --all-extras`.

The original Luna worker is recovered byte-for-byte from its launch archive,
not the later raw-response-capture fix. Its two failed requests were retried in
a separate diagnostic run; the combined scores preserve that attempt provenance.
The selection script consumes the saved full-run inputs and the combined grade
report. It does not generate model responses.

Source archives (Marin object-storage access required):

- Original Luna: `s3://marin-us-east-08a/marin/eval-validation/labbench-10575/luna-full-20261026/launch-evidence.tar.gz`, MD5 `5072499030f1e02148b297521c60ac78`.
- Luna retries: `s3://marin-us-east-08a/marin/eval-validation/labbench-10575/luna-diagnostic-20261026/evidence.tar.gz`, MD5 `8227ad24b530153c73d9c321e59cb98d`.
- GLM panel: `s3://marin-us-east-08a/marin/eval-validation/labbench-10575/glm-panel216-thinking8k-total16k-20261030/launcher.tar.gz`, MD5 `28106dd3d9ea6aaaff0240090454f662`.
- Frozen panel: `s3://marin-us-east-08a/marin/eval-validation/labbench-10575/seq75-20261030/manifest.json`, SHA-256 `ac9d45e65cac3a2485d448b11b7ecba3da1ce02f11c5b1db18147aace1fd5c9b`.

Per-file SHA-256 values are in [SHA256SUMS](SHA256SUMS).
