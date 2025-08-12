import copy

import wandb

# Lazy-initialize the WandB API (avoid top-level side effects)
_wandb_api = None

def get_wandb_api():
    global _wandb_api
    if _wandb_api is None:
        _wandb_api = wandb.Api()
    return _wandb_api
# Define your details
username = "marin-community"
project = "optimizer-scaling"


# Retrieve the run directly using its full path
def convert_run_to_config(run, keys):
    if "optimizer" not in run.config or "trainer" not in run.config:
        # too early
        return None
    return {
        key: run.config["optimizer"][key] if (key != "train_batch_size") else run.config["trainer"][key] for key in keys
    }


def grab_best_run(keys, tags, return_loss=False, thshold=3e-3):
    filters = {"tags": {"$all": tags}}
    print(tags)
    runs = get_wandb_api().runs(f"{username}/{project}", filters=filters)
    min_loss = 10000
    min_run = None
    for run in runs:
        if "trainer" not in run.config:
            continue
        loss = run.summary.get("eval/paloma/c4_en/loss", None)
        if type(loss) is float and loss < min_loss:
            min_loss = loss
            min_run = run
    print(min_loss)
    approximate_min_runs = []
    for run in runs:
        loss = run.summary.get("eval/paloma/c4_en/loss", None)
        if "trainer" not in run.config:
            continue
        if type(loss) is float and loss < min_loss + thshold:
            approximate_min_runs.append(run)
    best_config = {}
    if min_run is not None:
        best_config = convert_run_to_config(min_run, keys)
        approximate_best_list = []
        for run in approximate_min_runs:
            approximate_best_list.append(convert_run_to_config(run, keys))
        if return_loss:
            return best_config, approximate_best_list, min_loss
        else:
            return best_config, approximate_best_list
    else:
        if return_loss:
            return None, [], None
        else:
            return None, []


def bad_number(x):
    return (type(x) is not float) or (x > 10000)


def bad_run(run):
    history = run.history(keys=["eval/paloma/c4_en/loss"], pandas=True)
    if len(history) > 0:
        history = history.fillna(20)
        max_step = history["_step"].max()
        min_loss = history["eval/paloma/c4_en/loss"].min()
        if max_step >= 2000 and min_loss >= 6:
            return True
    return False


def actually_finish(run, strict=False):
    if strict:
        key = "lm_eval/wsc273/acc"
    else:
        key = "eval/paloma/c4_en/loss"
    history = run.history(keys=[key], pandas=True)
    step = run.config["trainer"]["num_train_steps"]
    if len(history) > 0:
        max_step = history["_step"].max()
        if max_step >= step - 10:
            return True
    return False


def approximate(baseline, config):
    if config is None:
        return False
    for key in baseline:
        if type(baseline[key]) is not float:
            if baseline[key] != config[key]:
                return False
        else:
            if abs(baseline[key] - config[key]) > 1e-25:
                return False
    return True


def check_baseline_run(baseline, tags, strict=True, return_loss=False):
    filters = {"tags": {"$all": tags}}
    runs = get_wandb_api().runs(f"{username}/{project}", filters=filters)
    for run in runs:
        config = convert_run_to_config(run, baseline.keys())
        if approximate(baseline, config) and (
            bad_number(run.summary.get("eval/paloma/c4_en/loss", 0.0))
            or actually_finish(run, strict=strict)
            or bad_run(run)
        ):
            # diverge before finish
            # or have some crazyness: finished eval multiple time
            if return_loss:
                return True, run.summary.get("eval/paloma/c4_en/loss", 0.0)
            else:
                return True
    if return_loss:
        return False, None
    else:
        return False


def grab_run(baseline, tags):
    filters = {"tags": {"$all": tags}}
    runs = get_wandb_api().runs(f"{username}/{project}", filters=filters)
    print(runs)

    for run in runs:
        config = convert_run_to_config(run, baseline.keys())
        print(config)
        print(baseline)
        if approximate(baseline, config) and actually_finish(run, strict=False):
            # diverge before finish
            # or have some crazyness: finished eval multiple time
            print("Found run")
            return run
    return None


def create_configs(baseline_config, sweep_grids, target_data=5120000):
    config_in_dict = []
    target_steps = []
    batch_size = baseline_config["train_batch_size"]
    target_step = target_data // (4096 * batch_size)
    target_steps.append(target_step)
    config_in_dict.append(baseline_config)
    for key in sweep_grids:
        for value in sweep_grids[key]:
            new_config = copy.copy(baseline_config)
            if baseline_config[key] != (value):
                new_config[key] = value
                batch_size = new_config["train_batch_size"]
                target_step = target_data // (4096 * batch_size)
                target_steps.append(target_step)
                if (new_config["warmup"]) <= target_step:
                    config_in_dict.append(new_config)
    return target_steps, config_in_dict


expected_params = {
    "130m": 134217728,  # 32 * (512*2048*3 + 512*512*4)
    "300m": 301989888,  # 32 * (768*3072*3 + 768*768*4)
    "520m": 536870912,  # 32 * (1024*4096*3 + 1024*1024*4)
    "1.2b": 1207959552,  # 32 * (1536*6144*3 + 1536*1536*4)
}


def calculate_chinchilla_from_tag(tag):
    return expected_params[tag] * 20


def calculate_data_tag(tag, target_chinchilla):
    chinchilla = calculate_chinchilla_from_tag(tag)
    target_data = target_chinchilla * chinchilla
    data_size = f"{target_data // 1_000_000_000}B"
    return target_data, data_size
