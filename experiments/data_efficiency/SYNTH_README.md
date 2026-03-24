# Data-efficient pre-training by scaling synthetic megadocs

[Paper link](https://arxiv.org/abs/2603.18534) 

[WandB report](https://wandb.ai/stanford-mercury/suhas-data-efficiency/reports/Data-efficient-pre-training-by-scaling-synthetic-megadocs--VmlldzoxNjIyNjExNA)

[Issue #3905](https://github.com/marin-community/marin/issues/3905)

## Description

Compute is growing faster than internet data, so we're interested in data-efficient pre-training algorithms, which typically aim to better model the web distribution by achieving lower loss. Synthetic data is a promising solution, but most works focus on its ability to transform data into formats that are higher quality or closer to the downstream task. We study how useful synthetic data is for modeling the original web data distribution, the core problem of pre-training.


## Using this directory

### Training code

We release code that captures the configs passed into our pre-training runs for {epoching, parameter scaling, regularization, ensembling, synthetic data}. All code should work with standard marin instructions. Relevant experiment files: 

```
- Baselines
    | - synth_data_regularized_baselines.py
    | - synth_data_sd.py
- Ensembles 
    | - synth_data_regularized_ensembles.py
    | - synth_data_synth_ensembles.py
- Ablations
    | - synth_data_unmasking.py
    | - synth_data_wrap_unmasking.py
    | - synth_data_iid_mixing.py
    | - synth_data_var.py 
    | - synth_data_student_scaling.py
- Scaling
    | - synth_data_copy_scaling.py
    | - synth_data_final_convex_tune.py
- Evals
    | - synth_data_eval_models.py
    | - synth_data_eval_downstream.py
- Plots
    | - synth_data_efficiency_plot_utils.py
    | - synth_data_final_plots.py
```



### Plotting code

`synth_data_final_plots.py` which is a behemoth that can reproduce all of our plots programatically from our WandB runs. Instructions on usage:

Please reach out to Konwoo or Suhas if you have any questions, more than happy to help!