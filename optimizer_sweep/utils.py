import wandb
# Initialize the WandB API
api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "marin-optimizer"
ths_hold = 1e-3
# Retrieve the run directly using its full path


def get_status_and_loss(run_id):
    run = api.run(f"{username}/{project}/{run_id}")
    if run.state != 'finished':
        return False, None
    else:
        return True, run.summary['eval/paloma/c4_en/loss']


def sweeping(run_id_list, baseline_run_id):
    # return
    # 1. status of sweeping
    # 2. unfinished run / config better than baseline
    losses = {}    

    for run_id in run_id_list:
        status, loss = get_status_and_loss(run_id)
        if status:
            losses[run_id] = loss
    if baseline_run_id in losses:
        best_id = min(losses, key = lambda x: losses[x])
        if(best_id != baseline_run_id):
            return 'Next Iteration', best_id
        if(len(losses) == len(run_id_list)):
            return 'Success!', best_id    
    return 'Unfinished Iteration', None


import os
def get_wandb_id(steps):
    return [os.path.basename(step.name) for step in steps]
