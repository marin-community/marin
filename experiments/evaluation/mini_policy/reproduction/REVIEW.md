# Archive review disposition

The advisory review ran on the committed archival branch. The following historical script patterns are retained because this PR records the executed experiment and will be closed without merging. They are not proposed main-branch library interfaces.

| Finding | Locations | Disposition |
| --- | --- | --- |
| `ml-tuple-return-shape` | `cohort_automl.py`, `costs.py`, `svd_baseline.py` | Retain the historical private return/serialization shapes and their existing callers. A future maintained implementation should use named records. |
| `ml-overloaded-function` | `cohort_automl.py:evaluate` | Preserve the aggregate-study driver as executed; its separate analysis stages and results are documented in issue #10804. |
| `ml-module-globals` | `calibration.py`, `experiment.py`, `gsm8k_parity.py`, `materialize.py` | These are one-shot CLI scripts run in separate Linux Iris processes. Do not call successive modes in one imported Python process or assume their process-pool configuration works unchanged under another multiprocessing start method. |
| `ml-duplicate-logic-block` | `experiment.py` | Preserve independently recorded method arms. Refactoring the selection loops would require validating their task sets and predictions again. |
| `ml-naive-path-join` | `assemble.py`, `export_mirt.py`, `joint_intervals.py`, `native_probe.py`, `svd_baseline.py` | Reproduce with the recorded literal S3 prefixes and no added trailing slash. These scripts do not promise arbitrary object-key normalization. Published data/configs remain immutable; a maintained replacement should use the storage-path API. |
| `ml-unused-param` | `inventory.py:inspect_cell` | Retain the unused historical output-directory argument in this snapshot; it does not affect evidence collection. |

Mechanical formatting, license headers and newline fixes do not change the analysis or grading. The dated runtime-config tarball retains exact original bytes. The frozen published runtime, source patch and verification receipts remain authoritative for exact reproduction.
