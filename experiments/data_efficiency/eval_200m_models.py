from argparse import ArgumentParser
from dataclasses import dataclass
import os
import glob
import json

STEPS_TO_SEED_TOKENS = {
    800: '209M',
    1600: '418M',
    3200: '838M',
    6400: '1.7B',
}

@dataclass
class ModelConfig:
    seed_steps: int
    num_epochs: int
    lr: float
    wd: float
    model_size: str
    num_seeds: int
    seed_in_name: bool = True
    append_bs: bool = False
    teacher_data: str = None
    teacher_ratio: float = None

def get_base_name(model_config: ModelConfig) -> str:
    # NOTE: adds extra 0 for lr and wd to match format of the name
    lr = f"{model_config.lr}0" if model_config.lr != 3e-4 else model_config.lr
    return f"{model_config.model_size}-{STEPS_TO_SEED_TOKENS[model_config.seed_steps]}x{model_config.num_epochs}-dclm{f'+{model_config.teacher_data}^{model_config.teacher_ratio}' if model_config.teacher_data else ''}-cos-lr{lr}-wd{model_config.wd}0{f'-bs64' if model_config.append_bs else ''}"

def get_names(model_config: ModelConfig) -> list[str]:
    base_name = get_base_name(model_config)
    if model_config.seed_in_name:
        return [f"{base_name}-seed{i}" for i in range(model_config.num_seeds)]
    else:
        assert model_config.num_seeds == 1, "Only one seed is allowed when seed_in_name is False"
        return [base_name]

def total_steps(model_config: ModelConfig) -> int:
    if model_config.teacher_data:
        return int(model_config.num_epochs * model_config.seed_steps * 1 / (1 - model_config.teacher_ratio))
    return model_config.num_epochs * model_config.seed_steps

VANILLA_MODEL_SCALING = [
    ModelConfig(seed_steps=800, num_epochs=16, lr=3e-3, wd=0.8, model_size="150m4k", num_seeds=1),
    ModelConfig(seed_steps=800, num_epochs=16, lr=3e-3, wd=1.6, model_size="300m4k", num_seeds=1),
    ModelConfig(seed_steps=800, num_epochs=8, lr=1e-3, wd=3.2, model_size="600m4k", num_seeds=1),
    ModelConfig(seed_steps=800, num_epochs=8, lr=1e-3, wd=3.2, model_size="1_4b4k", num_seeds=1),
]

ENSEMBLE_MODEL_SCALING = [
    ModelConfig(seed_steps=800, num_epochs=32, lr=3e-3, wd=0.4, model_size="150m4k", num_seeds=5),
    ModelConfig(seed_steps=800, num_epochs=32, lr=3e-3, wd=0.8, model_size="300m4k", num_seeds=5),
    ModelConfig(seed_steps=800, num_epochs=16, lr=1e-3, wd=1.6, model_size="600m4k", num_seeds=5),
    ModelConfig(seed_steps=800, num_epochs=16, lr=1e-3, wd=1.6, model_size="1_4b4k", num_seeds=5),
]

UNREGULARIZED_MODEL_SCALING = [
    ModelConfig(seed_steps=800, num_epochs=8, lr=3e-3, wd=0.1, model_size="150m4k", num_seeds=1, seed_in_name=False, append_bs=True),
    ModelConfig(seed_steps=800, num_epochs=8, lr=1e-3, wd=0.1, model_size="300m4k", num_seeds=1, seed_in_name=False, append_bs=True),
    ModelConfig(seed_steps=800, num_epochs=4, lr=1e-3, wd=0.1, model_size="600m4k", num_seeds=1, seed_in_name=False, append_bs=True),
    ModelConfig(seed_steps=800, num_epochs=4, lr=3e-4, wd=0.1, model_size="1_4b4k", num_seeds=1, seed_in_name=False, append_bs=True),
    ModelConfig(seed_steps=800, num_epochs=4, lr=1e-3, wd=0.1, model_size="1_5b4k", num_seeds=1, seed_in_name=False, append_bs=True),
    ModelConfig(seed_steps=800, num_epochs=4, lr=3e-4, wd=0.1, model_size="3_2b4k", num_seeds=1, seed_in_name=False, append_bs=True),
]

DISTILL_MODELS = [
    ModelConfig(seed_steps=800, num_epochs=16, lr=3e-3, wd=0.1, model_size="300m4k", num_seeds=1, seed_in_name=False, append_bs=True, teacher_data="ens8x0730", teacher_ratio=0.9),
    ModelConfig(seed_steps=800, num_epochs=16, lr=3e-3, wd=0.1, model_size="300m4k", num_seeds=1, seed_in_name=False, append_bs=True, teacher_data="sd0805", teacher_ratio=0.75),
]

def main(args):
    if args.mode in ['download', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING + UNREGULARIZED_MODEL_SCALING + DISTILL_MODELS:
            for name in get_names(model):
                print("Downloading: ", name)
                path = 'models/' + name
                os.makedirs(path, exist_ok=True)
                os.system(f"gcloud storage cp -r gs://marin-us-central2/checkpoints/data_efficiency/{name}/hf/step-{total_steps(model) - 1}/* {path}")
    

    if args.mode in ['rearrange', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING + UNREGULARIZED_MODEL_SCALING + DISTILL_MODELS:
            base_path = 'models/' + get_base_name(model)
            os.makedirs(base_path, exist_ok=True)
            for i, name in enumerate(get_names(model)):
                if model.seed_in_name:
                    existing_path = 'models/' + name
                    os.system(f"mv {existing_path} {base_path}/seed{i}")
                else:
                    existing_path = 'models/' + name
                    tmp_name = name + '_tmp'
                    tmp_path = 'models/' + tmp_name
                    os.system(f"mv {existing_path} {tmp_path}")
                    os.makedirs(f'{base_path}/seed{i}', exist_ok=True)
                    os.system(f"mv {tmp_path} {base_path}/seed0")
                    os.system(f"mv {base_path}/seed0/{tmp_name}/* {base_path}/seed0")
                    os.system(f"rm -r {base_path}/seed0/{tmp_name}")

    if args.mode in ['eval', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING + UNREGULARIZED_MODEL_SCALING + DISTILL_MODELS:
            base_name = get_base_name(model)
            results_path = 'results/' + base_name
            os.system(f"mkdir {results_path}")
            for i in range(model.num_seeds):
                os.system(f"./scripts/eval_pt_ensemble.sh models/{base_name} {i + 1}")
                latest_json = max(glob.glob(f"results/*.json"), key=os.path.getctime)
                os.system(f"mv {latest_json} {results_path}/{i+1}_seeds.json")


    if args.mode in ['collect', 'all']:
        all_results = {}
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING + UNREGULARIZED_MODEL_SCALING + DISTILL_MODELS:
            base_name = get_base_name(model)
            results_path = 'results/' + base_name
            all_results[base_name] = {}
            for i in range(model.num_seeds):
                with open(f"{results_path}/{i+1}_seeds.json", "r") as f:
                    data = json.load(f)

                results = {}
                for task, task_results in data['results'].items():
                    results[task] = {
                        'acc': task_results['acc,none'],
                        'acc_stderr': task_results['acc_stderr,none'],
                    }
                all_results[base_name][i+1] = results

        with open('results/200m_benchmark_results.json', 'w') as f:
            json.dump(all_results, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['download', 'eval', 'all', 'rearrange', 'collect'], default='eval')
    args = parser.parse_args()
    main(args)