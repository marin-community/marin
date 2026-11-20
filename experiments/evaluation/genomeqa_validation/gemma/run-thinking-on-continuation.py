"""Private, bounded GenomeQA Gemma comparison; no public launcher registration."""
import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import urllib.request

from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from levanter.compat.hf_checkpoints import load_tokenizer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import RolloutLimits
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext
from experiments.evaluation.benchmarks.genomeqa import sources
from experiments.evaluation.benchmarks.environments import BASIC_GRADER
from experiments.post_training.task_curation.pipeline import source_recipe, environment_requirements
from experiments.post_training.task_curation.images.build import build_environment, EnvironmentBuild, identity_digest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.evaluation.task_worker import execute_with_retries
from marin.inference.iris import remote_inference
from marin.rollouts.evaluation import EvaluationRunConfig, ModelIdentity

ROOT = 's3://marin-us-east-02a/marin/experiments/genomeqa/gemma-paired-fe813e34-20261112'
MODEL_FILE = Path('experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs/google-gemma-4-26B-A4B-it.yaml')


def save(name, value):
    (StoragePath(ROOT) / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'artifact': name}), flush=True)


def prepare():
    dataset = sources()[0].config
    model = load_model_config(MODEL_FILE)
    tokenizer = load_tokenizer(model.location, revision=model.revision, trust_remote_code=False)
    messages = [{'role': 'user', 'content': 'Choose A or B. Reply with A.'}]
    rendered = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enabled) for enabled in (False, True)]
    assert rendered[0] != rendered[1], 'Thinking switch has no effect on the pinned chat template'
    save('tokenizer-preflight.json', {'revision': model.revision, 'template_sha256': [hashlib.sha256(s.encode()).hexdigest() for s in rendered], 'thinking_switch_changes_prompt': True})
    inputs = StoragePath(ROOT) / 'inputs'
    for filename in dataset.recipe.source.files:
        local = hf_hub_download(dataset.recipe.source.repo, filename, repo_type='dataset', revision=dataset.recipe.source.revision)
        (inputs / filename).write_bytes(Path(local).read_bytes())
    prompt = dataset.recipe.inputs['prompt']
    with urllib.request.urlopen(prompt.url, timeout=60) as response:
        contents = response.read()
    assert hashlib.sha256(contents).hexdigest() == 'bc932144c20e02ccb18e4d8407f90814ca4f8fdc389d9309cd0cbc1525db58a3'
    (inputs / 'run.py').write_bytes(contents)
    built = build_environment(BASIC_GRADER, EnvironmentBuild(identity_digest(BASIC_GRADER), 'unused-lock-only', ROOT + '/environment'))
    requirements = environment_requirements(BASIC_GRADER, built)
    recipe = source_recipe(dataset.recipe, {'prompt': str(inputs)}, requirements)
    with ZephyrContext(max_workers=1, resources=ResourceConfig(cpu=2, ram='4g'),
                       coordinator_resources=ResourceConfig(cpu=1, ram='2g'),
                       chunk_storage_prefix=ROOT + '/chunks', max_execution_retries=0,
                       name='genomeqa-paired-preparation') as context:
        for mode, enabled in (('off', False), ('on', True)):
            protocol = dataset.protocol.model_copy(update={'sampling': dataset.protocol.sampling.model_copy(update={'chat_template_kwargs': {'enable_thinking': enabled}})})
            manifest = publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context,
                str(inputs), ROOT + '/corpus-' + mode, protocol_name='genomeqa-thinking-' + mode)
            save('publication-' + mode + '.json', manifest.model_dump(mode='json'))
    save('prepared.json', {'population_per_mode': 5200, 'model': model.location, 'revision': model.revision, 'code_revision': 'fe813e34c2a0d047f10a1604a12da35f162031cd'})


def run():
    assert (StoragePath(ROOT) / 'prepared.json').exists(), 'Complete CPU preflight first'
    model = load_model_config(MODEL_FILE)
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type='H100', gpu_count=2)
    serving = inference_config_for_model(model, accelerator, env_vars={}, priority=0, api_model=model.location, instances=1, request_timeout_seconds=1200)
    serving = replace(serving, iris=replace(serving.iris, max_retries_failure=0, max_retries_preemption=0))
    identity = ModelIdentity(name=model.location, revision=model.revision, tokenizer=model.location,
        tokenizer_revision=model.revision, context_window=73728,
        serving={'backend': 'vllm', 'model_config': str(MODEL_FILE), 'gpu': '2xH100', 'top_k': '64 (checkpoint default)', 'code_revision': 'fe813e34'})
    config = EvaluationRunConfig(model=identity, max_tokens=16384, concurrency=32,
        limits=RolloutLimits(max_turns=1, max_total_tokens=None, model_turn_timeout=1200,
        tool_turn_timeout=30, total_turn_timeout=1230, attempt_timeout=1500, verifier_timeout=120, cleanup_timeout=30))
    save('run-config.json', config.model_dump(mode='json'))
    save('budget-thinking-on-continuation.json', {'parent_timeout_seconds': 14400, 'gpu_count': 2, 'max_gpu_hours': 8, 'modes': ['on'], 'questions_per_mode': 5200, 'max_tokens_per_question': 16384, 'max_native_passes_per_mode': 3})
    with remote_inference(serving) as session:
        session.wait_until_ready()
        endpoint = session.model.endpoint
        for mode in ('on',):
            print('STARTING_MODE ' + mode, flush=True)
            execute_with_retries((ROOT + '/corpus-' + mode,), ROOT + '/results-' + mode,
                        config, endpoint.base_url, endpoint.api_key or 'unused', None, max_attempts=3)
            save('completed-' + mode + '.json', {'mode': mode, 'runner_returned_successfully': True})
    assert (StoragePath(ROOT) / 'completed-off.json').exists()
    save('completed.json', {'modes': ['off', 'on'], 'serving_context_closed': True})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    configure_coreweave_s3()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
