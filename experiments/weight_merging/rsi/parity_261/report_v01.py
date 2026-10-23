"""Summarize downloaded task261 v0.1 result documents and Prometheus deltas."""
import ast
import csv
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
baselines = {r['benchmark']: r for r in csv.DictReader(open(sys.argv[2]))}
specs = [('math500','accuracy',500,.962),('gsm8k-0shot','exact_match,flexible-extract',1319,.929492),('boolq','acc,none',3270,.8336),('piqa','acc,none',1838,.82046),('humanevalplus','python_pass@1',164,145/164),('aime24','accuracy_avg',30,.7266666667)]
rows = []
for name,key,n,single in specs:
    p = root/name
    doc = json.loads(json.loads((p/'results.json').read_text())[0]['document'])
    outcome = next(iter(doc['task_outcomes'].values()))
    assert outcome['status'] == 'succeeded'
    assert outcome['expected_count'] == outcome['generated_count'] == outcome['scored_count'] == n
    metrics = next(iter(doc['results'].values()))
    score = metrics[key]
    # Use question-level binomial SEM for AIME, not 300 independent questions.
    sem = math.sqrt(score*(1-score)/n)
    ref = float(baselines[name]['score'])
    refsem = float(baselines[name]['sem'])
    z = (score-ref)/math.hypot(sem,refsem)
    def counter(file, metric, reason=None):
        return sum(float(line.rsplit(' ',1)[1]) for line in (p/file).read_text().splitlines() if line.startswith(metric+'{') and (reason is None or f'finished_reason="{reason}"' in line))
    tokens = int(counter('metrics-after.txt','vllm:generation_tokens_total')-counter('metrics-before.txt','vllm:generation_tokens_total'))
    finishes = {r:int(counter('metrics-after.txt','vllm:request_success_total',r)-counter('metrics-before.txt','vllm:request_success_total',r)) for r in ('stop','length','abort','error','repetition')}
    wall = float(doc['total_evaluation_time_seconds'])
    rows.append(dict(task=name,metric=key,score=score,n=n,single_host_TPU=single,h100=ref,h100_sem=refsem,z=z,parity=abs(z)<=2,eval_seconds=wall,generated_tokens=tokens,tokens_per_second=tokens/wall,finish_reasons=finishes,repeat_sem=metrics.get('accuracy_std_err')))
hbm = {}
for line in re.sub(r'\x1b\[[0-9;]*m','',(root/'server.log').read_text()).splitlines():
    if 'TASK261_POST_LOAD_HBM ' in line:
        for device,stats in ast.literal_eval(line.split('TASK261_POST_LOAD_HBM ',1)[1]):
            hbm[device]=stats
assert len(hbm)==16
summary=dict(rows=rows,post_load_HBM=hbm,wall=json.loads((root/'host-0/wall.json').read_text()),gate=json.loads((root/'gate.json').read_text()))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Task261 v0.1 multi-host parity','', 'One reserved v4-32, four hosts / 16 chips, BF16 TP16/PP1; all four tasks succeeded and resources released. Server context 73728, effective evaluator cap 73664, concurrency 16, Evalchemy c1d5c62c and task249 campaign YAMLs/client. Verified 7 global layers / 19 local layers and four KV groups.','', '| Task | v4-32 | Single-host TPU | H100 | n | Combined z vs H100 | Verdict | Eval seconds | Generated tokens | Stop / length |','|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
for r in rows:
    f=r['finish_reasons']
    lines.append(f"| {r['task']} | {r['score']:.4f} | {r['single_host_TPU']:.4f} | {r['h100']:.4f} | {r['n']} | {r['z']:+.2f} | {'PARITY' if r['parity'] else 'OUTSIDE'} | {r['eval_seconds']:.1f} | {r['generated_tokens']:,} | {f['stop']} / {f['length']} |")
lines += ['', 'All task outcomes succeeded with full coverage. AIME is 10 repeats of 30 questions (300 responses), not 300 independent questions; the z calculation uses question-level binomial SEM and the supplied H100 SEM. Its across-repeat SEM is 0.0110. Single-host values are task249 references (some rounded).', '', 'BoolQ/PIQA use loglikelihood: their one generated token per choice is discarded from scoring; length finishes are expected, not truncated answers. Prometheus deltas exclude warmup. Eval seconds include harness/scoring overhead; throughput is not pure decode throughput.', '', f"Total leader worker wall: {summary['wall']['elapsed_seconds']:.1f} s. Post-load HBM across 16 chips: {min(x['bytes_in_use'] for x in hbm.values()):,}–{max(x['bytes_in_use'] for x in hbm.values()):,} bytes; device limit {sorted({x['bytes_limit'] for x in hbm.values()})} bytes. This is a post-load snapshot, not peak runtime memory.", '', 'Artifacts: gs://marin-us-central2/scratch/karan/task261/v01-r1/; each task has results.json, settings.json, process.json, metrics-before/after.txt and FineStore data/. Job /karan/rsi261-v01-v4-32-r1. The v4 tiling remains a job-local patch; this does not establish stock upstream v4 support.']
(root/'parity-table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
