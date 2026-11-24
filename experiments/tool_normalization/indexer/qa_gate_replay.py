"""200-row QA gate on the pool-replay archives: verify_adapter re-runs on every
pool-converted row + the status/receipt distribution. Writes qa-gate.json to S3.
"""
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from rigging.filesystem.buckets import filesystem_for  # noqa: E402

from experiments.tool_normalization.verify import verify_adapter  # noqa: E402

BASE = 's3://marin-us-east-08a/marin/users/benfeuer/tool-normalization'
FLEET = 'ortet-24-20261112'
REPLICA = 'r1'
OUT = 's3://marin-us-east-02a/marin/users/benfeuer/tool-normalization/tool-conversion-index/qa/qa-gate-replay.json'
VERIFIED = {'converted_adapter_checked', 'rule_converted_adapter_checked', 'header_normalized_no_calls'}
WORK = Path('/tmp/qa-gate')
RULE_FILE = Path('/app/experiments/glm_tool_serving/production_data/rule-approval-v3.json')

log = lambda m: print(f'{time.strftime("%H:%M:%SZ", time.gmtime())} gate: {m}', flush=True)


def main():
    fs, _ = filesystem_for(BASE + '/x')
    prefix = f'{BASE}/{FLEET}/{REPLICA}/batches'
    entries = [str(e) for e in fs.ls(prefix, detail=False) if '-replay-' in str(e)]
    log(f'{len(entries)} replay archives')
    results = []
    rows = 0
    for uri in entries:
        name = uri.split('/')[-1][:-7]
        try:
            raw = fs.cat(uri)
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                names = tar.getnames()
                disp_m = next(n for n in names if n.endswith('/dispositions.json'))
                base = disp_m[:-len('/dispositions.json')]
                rec_root = f'{base}/run/records'
                manifest = [json.loads(l) for l in tar.extractfile(f'{base}/manifest.jsonl').read().decode().splitlines() if l.strip()]
                recon = None
                recon_m = next((n for n in names if n.endswith('/reconciliation.json')), None)
                if recon_m:
                    recon = json.load(tar.extractfile(recon_m))
                cfg = json.load(tar.extractfile('config.json'))
                if cfg.get('rule_approval'):
                    cfg['rule_approval']['path'] = str(RULE_FILE)
                cfg['model']['api_key_file'] = '/tmp/qa-api-key.txt'
                for m in manifest:
                    rid = m['id']
                    rp = f'{rec_root}/{rid}'
                    rows += 1
                    try:
                        result = json.load(tar.extractfile(f'{rp}/result.json'))
                        status = result.get('status')
                    except KeyError:
                        results.append({'archive': name, 'id': rid, 'status': 'NO_RESULT'})
                        continue
                    fact = {'archive': name, 'id': rid, 'status': status,
                            'verified_in_driver': result.get('verification') is not None}
                    rec_m = next((n for n in names if n == f'{rp}/review/receipt.json'), None)
                    if rec_m:
                        try:
                            fact['receipt_base_url'] = json.load(tar.extractfile(rec_m)).get('base_url')
                        except Exception:
                            pass
                    if status in VERIFIED:
                        # extract the record dir + sources, rewrite the source path, run the verifier
                        rdir = WORK / rid
                        if rdir.exists():
                            shutil.rmtree(rdir)
                        rdir.mkdir(parents=True)
                        for mm in tar.getmembers():
                            if mm.name.startswith(rp + '/') and not mm.isdir():
                                dest = rdir / mm.name[len(rp) + 1:]
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(tar.extractfile(mm).read())
                        sr_file = rdir / 'source-reference.json'
                        if sr_file.exists():
                            sr = json.loads(sr_file.read_text())
                            src_key = Path(sr['path']).parent.name  # .../sources/<key>/source.json
                            sdir = WORK / 'sources' / src_key
                            sdir.mkdir(parents=True, exist_ok=True)
                            marker = f'/sources/{src_key}/'
                            for mm in tar.getmembers():
                                if marker in mm.name and not mm.isdir():
                                    rel = mm.name.split(marker, 1)[1]
                                    d = sdir / rel
                                    d.parent.mkdir(parents=True, exist_ok=True)
                                    d.write_bytes(tar.extractfile(mm).read())
                            sr['path'] = str(sdir / 'source.json')
                            sr_file.write_text(json.dumps(sr))
                        try:
                            verdict = verify_adapter(rdir, cfg)
                            fact['verify_pass'] = True
                            fact['verify_summary'] = {k: str(v)[:80] for k, v in verdict.items() if not isinstance(v, dict)} if isinstance(verdict, dict) else None
                        except Exception as error:
                            fact['verify_pass'] = False
                            fact['verify_error'] = f'{type(error).__name__}: {str(error)[:200]}'
                        fact['preserved_in_reconciliation'] = bool(recon and recon.get('preserved', {}).get(rid))
                    results.append(fact)
        except Exception as error:
            log(f'archive fail {name}: {error!r}')
        if rows >= 260:
            break
    dist = Counter(r['status'] for r in results)
    verified_rows = [r for r in results if r['status'] in VERIFIED]
    passes = sum(1 for r in verified_rows if r.get('verify_pass'))
    report = {'rows': len(results), 'status_distribution': dict(dist),
              'verified_rows': len(verified_rows),
              'verifier_pass': passes, 'verifier_fail': len(verified_rows) - passes,
              'receipt_base_urls': dict(Counter(r.get('receipt_base_url', 'none') for r in results)),
              'results': results}
    blob = json.dumps(report, indent=1)
    handle, _ = filesystem_for(OUT)
    handle.put(str(_w(blob)), OUT)
    log(f"rows={len(results)} verified={len(verified_rows)} verifier pass={passes}/{len(verified_rows)}")
    log('report uploaded')


def _w(text):
    import tempfile
    p = Path(tempfile.gettempdir()) / 'qa-gate-replay.json'
    p.write_text(text)
    return p


if __name__ == '__main__':
    main()
