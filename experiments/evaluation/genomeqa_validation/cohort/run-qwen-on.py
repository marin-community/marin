"""Bounded cohort evaluation using the existing GenomeQA pipeline and native runner."""
import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from finestore.reader import ReadView
from levanter.compat.hf_checkpoints import load_tokenizer
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import RolloutLimits
from taskcompendium.evaluation.corpus import EvaluationManifest, MANIFEST_FILENAME
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.evaluation.task_worker import execute_with_retries
from marin.inference.iris import remote_inference
from marin.rollouts.evaluation import EvaluationRunConfig, ModelIdentity

BASE = 's3://marin-us-east-02a/marin/experiments/genomeqa/gemma-paired-fe813e34-20261112'
ROOT = 's3://marin-us-east-02a/marin/experiments/genomeqa/cohort-fe813e34-20261116'
MODELS = {'qwen-on': ('Qwen-Qwen3.6-35B-A3B', 4, {'enable_thinking': True})}


def save(root, name, value):
    (StoragePath(root)/name).write_text(json.dumps(value, indent=2)+'\n')
    print('ARTIFACT '+name, flush=True)


def score(output):
    view = ReadView(output)
    cells = defaultdict(Counter)
    seen = set()
    for row in view.iter_rows('samples', columns=['doc_id','doc','metrics']):
        if not row['metrics']:
            continue
        assert row['doc_id'] not in seen
        seen.add(row['doc_id'])
        doc = json.loads(row['doc']) if isinstance(row['doc'], str) else row['doc']
        cell = cells[doc['groups']['cell']]
        cell['count'] += 1
        metrics = dict(row['metrics'])
        for key in ('accuracy','truncation','parse_failure','unanswered','refusal'):
            cell[key] += metrics[key]
    assert len(seen) == 5200 and len(cells) == 12
    for name,cell in cells.items():
        assert cell['count'] == (100 if name.startswith('6_motif/') else 500)
    accuracy = {k:v['accuracy']/v['count'] for k,v in cells.items()}
    return {'graded': len(seen), 'macro': sum(accuracy.values())/12,
        'format_macro': {fmt:sum(v for k,v in accuracy.items() if k.endswith('/'+fmt))/6 for fmt in ('binary','mcq')},
        'subtask_macro': {task:sum(v for k,v in accuracy.items() if k.startswith(task+'/'))/2 for task in sorted({k.split('/')[0] for k in cells})}, 'cells':dict(cells)}


def run(label):
    model_name, gpu_count, kwargs = MODELS[label]
    root = ROOT + '/' + label
    model_file = Path('experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/model-configs')/(model_name+'.yaml')
    model = load_model_config(model_file)
    tokenizer = load_tokenizer(model.location, revision=model.revision, trust_remote_code=False)
    messages = [{'role':'user','content':'Choose A or B. Reply with A.'}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    nonthinking = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert rendered != nonthinking and rendered.endswith('<think>\n')
    corpus = BASE + '/corpus-on'
    original_corpus = BASE + '/corpus-off'
    manifest = EvaluationManifest.model_validate_json((StoragePath(corpus)/MANIFEST_FILENAME).read_text())
    original = EvaluationManifest.model_validate_json((StoragePath(original_corpus)/MANIFEST_FILENAME).read_text())
    assert manifest.rows == original.rows == 5200
    assert [(s.tasks.sha256,s.items.sha256,s.rows) for s in manifest.shards] == [(s.tasks.sha256,s.items.sha256,s.rows) for s in original.shards]
    protocol = json.loads((StoragePath(corpus)/'protocol.json').read_text())
    off_protocol = json.loads((StoragePath(original_corpus)/'protocol.json').read_text())
    assert protocol['sampling']['chat_template_kwargs'] == kwargs
    assert off_protocol['sampling']['chat_template_kwargs'] == {'enable_thinking': False}
    off_protocol['sampling']['chat_template_kwargs'] = kwargs
    assert protocol == off_protocol
    assert protocol['sampling']['max_tokens'] == 16384
    assert protocol['sampling']['temperature'] == 0.7 and protocol['sampling']['top_p'] == 0.95
    assert protocol['trials'] == {'count': 1, 'seed': 42}
    save(root, 'preflight.json', {'model':model.location, 'revision':model.revision, 'template_kwargs':kwargs, 'rendered_probe':rendered, 'template_sha256':hashlib.sha256(rendered.encode()).hexdigest(), 'same_questions_as_off':True, 'protocol_only_thinking_changed':True})
    accelerator = AcceleratorChoice(platform=Platform.GPU,gpu_type='H100',gpu_count=gpu_count)
    serving = inference_config_for_model(model,accelerator,env_vars={},priority=0,api_model=model.location,instances=1,request_timeout_seconds=1200)
    assert serving.broker is not None
    serving = replace(serving,iris=replace(serving.iris,max_retries_failure=0,max_retries_preemption=0),broker=replace(serving.broker,max_retries_failure=0,max_retries_preemption=0))
    identity = ModelIdentity(name=model.location,revision=model.revision,tokenizer=model.location,tokenizer_revision=model.revision,context_window=73728,serving={'backend':'vllm','model_config':str(model_file),'gpu':f'{gpu_count}xH100','code_revision':'fe813e34','mode':label})
    config = EvaluationRunConfig(model=identity,max_tokens=16384,concurrency=64,limits=RolloutLimits(max_turns=1,max_total_tokens=None,model_turn_timeout=1200,tool_turn_timeout=30,total_turn_timeout=1230,attempt_timeout=1500,verifier_timeout=120,cleanup_timeout=30))
    save(root,'run-config.json',config.model_dump(mode='json'))
    save(root,'budget.json',{'timeout_seconds':14400,'gpu_count':gpu_count,'max_gpu_hours':gpu_count*4,'max_native_passes':3,'corpus':corpus,'template_kwargs':kwargs})
    with remote_inference(serving) as session:
        session.wait_until_ready()
        endpoint = session.model.endpoint
        execute_with_retries((corpus,),root+'/results',config,endpoint.base_url,endpoint.api_key or 'unused',None,max_attempts=3)
    summary = score(root+'/results')
    save(root,'scores.json',summary)
    usage = {}
    for row in ReadView(root+'/results').iter_rows('evaluation_attempts', columns=['task_id','trial','retry','phase','outcome','prompt_tokens','completion_tokens']):
        if row['phase'] == 'finished' and row['outcome'] == 'graded':
            key = (row['task_id'], row['trial'])
            assert key not in usage
            assert row['prompt_tokens'] is not None and row['completion_tokens'] is not None
            usage[key] = (row['prompt_tokens'], row['completion_tokens'])
    assert len(usage) == 5200
    save(root, 'usage.json', {'graded':len(usage), 'input_tokens':sum(x[0] for x in usage.values()), 'output_tokens':sum(x[1] for x in usage.values()), 'scope':'final graded attempts; output includes reasoning and final answer'})
    save(root,'completed.json',{'graded':5200,'serving_context_closed':True})
    print('FINAL_SCORE '+json.dumps(summary),flush=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    configure_coreweave_s3()
    parser = argparse.ArgumentParser()
    parser.add_argument('label',choices=MODELS)
    run(parser.parse_args().label)
