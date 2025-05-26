from dataclasses import replace
from experiments.datashop.default_configs import default_text_generation_config_kwargs
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.tutorials.exp1096_datashop_example import datashop_runner_config
from marin.resources import GpuConfig

text_generation_config_kwargs = default_text_generation_config_kwargs
text_generation_config_kwargs["engine_kwargs"]["tensor_parallel_size"] = 1

datashop_gpu_runner_config = replace(datashop_runner_config, 
    experiment_name="datashop-tutorial-gpu",
    labeler_resource_config=GpuConfig(gpu_count=1),
    text_generation_inference_config_kwargs=text_generation_config_kwargs,
)

datashop_gpu_runner = DatashopRunner(datashop_gpu_runner_config)

if __name__ == "__main__":
    datashop_gpu_runner.run_eval_cluster_steps()