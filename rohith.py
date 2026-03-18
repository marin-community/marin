from marin.execution.executor import Executor
from experiments.pretraining_datasets.nemotron import tokenize_nemotron

steps = tokenize_nemotron()
low_actual_step = steps["nemotron_cc/low_actual"]
e = Executor(prefix="gs://marin-us-central2", executor_info_base_path="gs://marin-us-central2/experiments")
e.run([low_actual_step], dry_run=True)
print(e.configs[low_actual_step].train_paths)