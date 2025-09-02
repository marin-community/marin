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

def get_base_name(model_config: ModelConfig) -> str:
    # NOTE: adds extra 0 for lr and wdto match format of the name
    return f"{model_config.model_size}-{STEPS_TO_SEED_TOKENS[model_config.seed_steps]}x{model_config.num_epochs}-dclm-cos-lr{model_config.lr}0-wd{model_config.wd}0"

def get_names(model_config: ModelConfig) -> list[str]:
    base_name = get_base_name(model_config)
    return [f"{base_name}-seed{i}" for i in range(model_config.num_seeds)]

def total_steps(model_config: ModelConfig) -> int:
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

def main(args):
    if args.mode in ['download', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING:
            for name in get_names(model):
                print("Downloading: ", name)
                path = 'models/' + name
                os.makedirs(path, exist_ok=True)
                os.system(f"gcloud storage cp -r gs://marin-us-central2/checkpoints/data_efficiency/{name}/hf/step-{total_steps(model) - 1}/* {path}")
    

    if args.mode in ['rearrange', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING:
            base_path = 'models/' + get_base_name(model)
            os.system(f"mkdir {base_path}")
            for i, name in enumerate(get_names(model)):
                existing_path = 'models/' + name
                os.system(f"mv {existing_path} {base_path}/seed{i}")

    if args.mode in ['eval', 'all']:
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING:
            base_name = get_base_name(model)
            results_path = 'results/' + base_name
            os.system(f"mkdir {results_path}")
            for i in range(model.num_seeds):
                os.system(f"./eval_ensemble.sh models/{base_name} {i + 1}")
                latest_json = max(glob.glob(f"results/*.json"), key=os.path.getctime)
                os.system(f"mv {latest_json} {results_path}/{i+1}_seeds.json")


    if args.mode in ['collect', 'all']:
        # results/150m4k-209Mx16-dclm-cos-lr0.0030-wd0.80/1_seeds.json
        all_results = {}
        for model in VANILLA_MODEL_SCALING + ENSEMBLE_MODEL_SCALING:
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