# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Score measured midpoints against the unchanged frozen runtime predictions."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'prediction_freeze'
FIT = HERE.parent.parent / 'delphi_frozen_procedure_validation_3e18_20260908/fits/fit_uncheatable.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = json.loads((FROZEN / 'summary.json').read_text())
    assert sha256(FROZEN / 'midpoint_predictions.csv') == freeze['midpoint_predictions_sha256']
    fit = json.loads(FIT.read_text())
    # Retain the serialized fitted aggregation weights without renormalizing.
    assert abs(sum(fit['task_weights']) - 1) < 1e-7
    for name in ['runtime_identity_audit.json', 'native_obe_comparability.json']:
        assert json.loads((HERE / name).read_text())['pass']
    bridge = json.loads((HERE / 'uncheatable_metric_bridge_audit.json').read_text())
    assert fit['task_weights'] == bridge['frozen_task_weights']
    assert [t['component'] for t in fit['tasks']] == bridge['component_keys']
    predictions = list(csv.DictReader((FROZEN / 'midpoint_predictions.csv').open()))
    measured = json.loads((HERE / 'collected_midpoints.json').read_text())
    output = []
    for row in measured:
        candidate = row['candidate_id']
        selected = [p for p in predictions if p['candidate_id'] == candidate and p['coordinate'] == 'runtime']
        assert len(selected) == 2
        target = selected[0]['target']; comparator = selected[0]['baseline']
        model = {p['predictor']: float(p['prediction_bpb']) for p in selected}
        if target == 'uncheatable':
            value = sum(w * row['final_metrics'][t['component']] for w, t in zip(fit['task_weights'], fit['tasks'], strict=True))
            metric = 'frozen_weighted_seven_component_pooled_bpb'
        else:
            value = row['table9_macro_bpb']
            metric = 'native_51_component_macro_bpb'
        assert all(Path(r['path']).read_text() == 'SUCCESS' for r in row['receipts'] if '_status.json' in r['path'])
        m, b = model['mariner'], model[comparator]
        output.append(dict(target=target, comparator=comparator, candidate_id=candidate, position=.5,
                           trainer_seed=0, data_seed=666200 if target == 'uncheatable' else 662009,
                           measured_bpb=value, mariner_prediction=m, baseline_prediction=b,
                           mariner_signed_error_bpb=m-value, baseline_signed_error_bpb=b-value,
                           mariner_absolute_error_bpb=abs(m-value), baseline_absolute_error_bpb=abs(b-value),
                           mariner_closer=abs(m-value)<abs(b-value), metric=metric,
                           raw_schema2_parent_bpb=row['raw_uncheatable_bpb'], source_inline_uri=row['inline_uri'],
                           source_native_eval_uri=row['native_eval_uri']))
    assert len(output) == 10 and len({r['candidate_id'] for r in output}) == 10
    destination = HERE / 'midpoint_scored_results.csv'
    with destination.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    observed_delta = max(abs(r['aggregate_difference']) for r in bridge['comparisons'])
    smallest_advantage = min(r['baseline_absolute_error_bpb']-r['mariner_absolute_error_bpb'] for r in output if r['target']=='uncheatable')
    assert smallest_advantage > 2 * observed_delta
    receipt = dict(pass_=True, comparable_at_plot_precision=True, exact_legacy_uncheatable_metric=False,
                   candidate_count=10, trained_checkpoint_step=3006, training_or_predictor_refit=False,
                   all_ten_runtime_predictions_unchanged=True, mariner_closer_count=sum(r['mariner_closer'] for r in output),
                   uncheatable_primary_metric='Frozen task weights applied to measured schema2 per-component BPB; no empirical offset',
                   uncheatable_estimator_caveat='Historical components used token-weighted batch BPB, new components pool scored bytes. The empirical 98-checkpoint difference is small but not a universal bound or exact legacy reconstruction.',
                   max_observed_legacy_estimator_delta_bpb=observed_delta,
                   smallest_uncheatable_absolute_error_advantage_bpb=smallest_advantage,
                   midpoint_outcomes_single_trainer_seed=True,
                   scored_csv_sha256=sha256(destination),
                   input_sha256={str(p): sha256(p) for p in [FIT,FROZEN/'midpoint_predictions.csv',HERE/'collected_midpoints.json',HERE/'runtime_identity_audit.json',HERE/'native_obe_comparability.json',HERE/'uncheatable_metric_bridge_audit.json']})
    receipt['pass'] = receipt.pop('pass_')
    for target in ['uncheatable','table9']:
        subset=[r for r in output if r['target']==target]
        receipt[target+'_mae']={key:sum(r[key+'_absolute_error_bpb'] for r in subset)/5 for key in ['mariner','baseline']}
    (HERE / 'midpoint_scoring_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
