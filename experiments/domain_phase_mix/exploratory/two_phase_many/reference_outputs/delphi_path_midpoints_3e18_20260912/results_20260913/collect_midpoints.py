# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs"]
# ///
"""Collect verified midpoint artifacts without fitting or altering frozen predictors."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import fsspec

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.experiments/ledger.sqlite').exists())
SOURCE = ROOT / '.experiments/artifacts/status_refresh_20260913_1404'
FREEZE = HERE.parent / 'prediction_freeze'
RAW = HERE / 'raw'
RAW.mkdir(exist_ok=True)


def archive(uri, destination):
    """Archive small immutable result metadata and retain an input receipt."""
    if destination.exists():
        data = destination.read_bytes()
    else:
        with fsspec.open(uri, 'rb') as stream:
            data = stream.read()
        destination.write_bytes(data)
    return {'uri': uri, 'path': str(destination), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def collect_train(row):
    candidate = re.search(r'mp50_[ut](?:9)?_\w+_cap64', row['path']).group()
    base = row['path']
    receipts = [archive(base + '/checkpoints/eval_metrics.jsonl', RAW / f'{candidate}.eval_metrics.jsonl'),
                archive(base + '/checkpoints/step-3006/metadata.json', RAW / f'{candidate}.metadata.json'),
                archive(base + '/.executor_status', RAW / f'{candidate}.train_status.json')]
    metadata = json.loads(Path(receipts[1]['path']).read_text())
    assert metadata == row['metadata'] and metadata['step'] == 3006 and not metadata['is_temporary']
    metrics = [json.loads(s) for s in Path(receipts[0]['path']).read_text().splitlines() if s]
    selected = [m for m in metrics if m.get('step') == 3006]
    assert selected and selected[-1] == row['final_metrics']
    final = selected[-1]
    components = {k: v for k,v in final.items() if k.startswith('eval/uncheatable_eval/') and k.endswith('/bpb') and k.count('/') == 3}
    assert len(components) == 7 and all(math.isfinite(v) for v in components.values())
    return {'candidate_id': candidate, 'checkpoint_step': metadata['step'], 'checkpoint_uri': base + '/checkpoints/step-3006',
            'inline_uri': receipts[0]['uri'], 'inline_schema': final['eval/bpb_schema_version'],
            'raw_uncheatable_bpb': final['eval/uncheatable_eval/bpb'], 'uncheatable_components': components,
            'final_metrics': final, 'receipts': receipts}


def collect_eval(row):
    candidate = re.search(r'mp50_[ut](?:9)?_\w+_cap64', row['path']).group()
    base = row['path'].rsplit('/', 1)[0]
    receipts = [archive(row['path'], RAW / f'{candidate}.native_eval_results.json'),
                archive(base + '/olmo_base_eval_table9_results.json', RAW / f'{candidate}.native_details.json'),
                archive(base + '/.executor_status', RAW / f'{candidate}.eval_status.json')]
    raw = json.loads(Path(receipts[0]['path']).read_text())
    details = json.loads(Path(receipts[1]['path']).read_text())
    components = {k: v for k,v in raw.items() if k.startswith('olmo_base_easy/table9/') and k.endswith('/bpb')}
    assert len(components) == 51 and components == row['components']
    assert len(details['num_instances']) == len(details['task_bpb']) == 104
    assert all(math.isfinite(v) for v in components.values())
    assert math.isclose(sum(components.values()) / 51, row['macro_bpb'], rel_tol=0, abs_tol=1e-14)
    assert row['macro_bpb'] == details['table9_macro_bpb']
    assert details['checkpoint_path'].endswith('/hf/step-3006') and candidate in details['checkpoint_path']
    assert details['provenance']['source_run_name'] == candidate
    return {'candidate_id': candidate, 'table9_macro_bpb': row['macro_bpb'], 'native_eval_uri': row['path'],
            'native_details': details, 'receipts': receipts}


def main():
    receipt = json.loads((FREEZE / 'summary.json').read_text())
    predictions_file = Path(receipt['midpoint_predictions_csv'])
    assert hashlib.sha256(predictions_file.read_bytes()).hexdigest() == receipt['midpoint_predictions_sha256']
    candidates = HERE.parent / 'candidate_weights.csv'
    assert hashlib.sha256(candidates.read_bytes()).hexdigest() == receipt['candidate_weights_sha256']
    with ThreadPoolExecutor(max_workers=6) as pool:
        training = list(pool.map(collect_train, json.loads((SOURCE / 'midpoints.json').read_text())))
        evaluations = list(pool.map(collect_eval, json.loads((SOURCE / 'midpoint_evaluations.json').read_text())))
    assert len(training) == len(evaluations) == 10
    keyed_train = {r['candidate_id']: r for r in training}
    keyed_eval = {r['candidate_id']: r for r in evaluations}
    assert keyed_train.keys() == keyed_eval.keys() and len(keyed_train) == 10
    predictions = list(csv.DictReader(predictions_file.open()))
    assert len(predictions) == 40
    assert {r['candidate_id'] for r in predictions} == keyed_train.keys()
    combined = []
    rows = []
    for candidate in sorted(keyed_train):
        train = keyed_train[candidate]
        evaluation = keyed_eval[candidate]
        selected = [p for p in predictions if p['candidate_id'] == candidate and p['coordinate'] == 'runtime']
        assert len(selected) == 2
        target = selected[0]['target']; comparator = selected[0]['baseline']
        model = {p['predictor']: float(p['prediction_bpb']) for p in selected}
        value = train['raw_uncheatable_bpb'] if target == 'uncheatable' else evaluation['table9_macro_bpb']
        rows.append({'target': target, 'comparator': comparator, 'candidate_id': candidate, 'position': .5,
                     'trainer_seed': 0, 'data_seed': 666200 if target == 'uncheatable' else 662009,
                     'raw_measured_bpb': value, 'mariner_prediction': model['mariner'],
                     'baseline_prediction': model[comparator], 'metric_comparable': target == 'table9'})
        combined.append({**train, **evaluation, 'receipts': train['receipts'] + evaluation['receipts']})
    (HERE / 'collected_midpoints.json').write_text(json.dumps(combined, indent=2) + '\n')
    with (HERE / 'raw_midpoint_results_not_for_plot.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'collected_utc': datetime.now(UTC).isoformat(), 'training_count': len(training), 'evaluation_count': len(evaluations),
               'checkpoint_step': 3006, 'all_permanent': True, 'uncheatable_component_count': 7, 'table9_component_count': 51,
               'native_leaf_task_count': 104, 'frozen_prediction_sha256': receipt['midpoint_predictions_sha256'],
               'metrics_comparability': 'OBE native scoring unchanged; raw Uncheatable schema2 aggregate differs from frozen objective; bridge required.',
               'training_or_model_refit': False}
    (HERE / 'collection_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
