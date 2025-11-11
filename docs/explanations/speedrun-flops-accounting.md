# Speedrun FLOPs

In [Marin Speedrun](../explanations/speedrun.md), we track the FLOPs (floating-point operations) used to train a model. This guide attempts to explain how we calculate FLOPs for your speedrun.

## Total Hardware Training FLOPs

We calculate total hardware training FLOPs as:

$$
F_{total} = \left(\sum_{i=1}^{n} t_i\right) \times d \times f_{peak}
$$

where:
- $F_{total}$ is the total hardware FLOPs used
- $t_i$ is the time taken by *training step* $i$ in seconds*
- $n$ is the total number of training steps
- $d$ is the number of accelerator devices (eg. 8 GPUs, or 64 chips in a v4-128 TPU pod)
- $f_{peak}$ is the peak FLOPs/second per device. Example: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf tells us that for an A100, $f_{peak}$ = 312 TFLOPS/s for BF16 precision.


## Model FLOPs

We also track model FLOPs, which is the total number of FLOPs required to train the model. The definition/code for this can be found in the [compute_model_flops()](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/speedrun/speedrun.py#L116) function.


Before running a speedrun, we can use the model FLOPs to give you a rough estimate of the estimated training HW FLOPs, by computing the ratio of model FLOPs to the assumed model FLOPs utilization (MFU). We plug in a couple of plausible MFU values (0.2 and 0.5) to give you a rough sense of how many hardware FLOPs your training run will use.

$$
F_{total} = \frac{F_{model}}{MFU}
$$

This estimate is shown both when calling [speedrun_config.print_run_info()](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/speedrun/speedrun.py#L76), which you can call before running training, and is also logged when running the speedrun (i.e., training). It can be used to guide your choice of model, training hyperparameters, and hardware configuration.
