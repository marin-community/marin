"""Replay driver for pool-era infrastructure_deferred rows.

Reads the replay inventory (per-replica deferred batches), and for each batch:
  1. downloads the original archive (preserved sibling outcomes + config/provenance),
  2. reconstructs source.json for every deferred row (TreeStore + tokenizer),
  3. runs the standard conversion pipeline (driver.run) on a manifest of deferred rows,
  4. assembles a -replay- archive in the recovery format the indexer understands
     (dispositions + re-run records + reconciliation.json with preserved statuses),
  5. commits it to the replica's batches/ prefix.

Idempotent: a batch whose -replay- archive already exists is skipped, so pods can
process the same queue in parallel and crashes resume cleanly. Exactly-once holds at
the conversion level: the original attempt produced no conversion output (HTTP 401
before any model response completed), and the replay archive is committed once.

Env: GLM_POOL_BASE_URL / GLM_POOL_MODEL (optional endpoint override), REPLAY_REPLICAS
(comma-separated subset), REPLAY_CONFIG (fleet config json), REPLAY_LIMIT (max batches).
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from rigging.filesystem.buckets import filesystem_for  # noqa: E402

from experiments.tool_normalization.driver import run  # noqa: E402

BASE = 's3://marin-us-east-08a/marin/users/benfeuer/tool-normalization'
INVENTORY = 's3://marin-us-east-02a/marin/users/benfeuer/tool-normalization/tool-conversion-index/replay/replay-inventory.json'
REPLAY_OUT = 's3://marin-us-east-02a/marin/users/benfeuer/tool-normalization/tool-conversion-index/replay'
TOKENIZER_S3 = f'{BASE}/h100-eight-20261105/production_data/sft_tokenizer'
WORK = Path('/tmp/replay-work')
FLEETS = {'ortet-24-20261112': 24, 'h100-eight-20261105': 8}

log = lambda m: print(f'{time.strftime("%H:%M:%SZ", time.gmtime())} replay: {m}', flush=True)


def fs():
    handle, _ = filesystem_for(BASE + '/x')
    return handle


def load_inventory(handle):
    handle.get(INVENTORY, str(WORK / 'replay-inventory.json'))
    return json.loads((WORK / 'replay-inventory.json').read_text())


def select_replicas(inv):
    subset = os.environ.get('REPLAY_REPLICAS')
    if not subset:
        return list(inv['replicas'].keys())
    wanted = set(subset.split(','))
    return [k for k in inv['replicas'] if k in wanted]


def reconstruct_sources(batch, tokenizer, workdir: Path) -> dict:
    """Rebuild source.json for each deferred row; returns {row_id: source_file_path}."""
    sources = {}
    out = workdir / 'sources'
    out.mkdir(parents=True, exist_ok=True)
    for entry in batch['deferred']:
        row = {'source': entry['source'], 'shard': entry['shard'], 'row': entry['row']}
        key = hashlib.sha256(f'{row["source"]}\n{row["shard"]}\n{row["row"]}'.encode()).hexdigest()
        directory = out / key
        directory.mkdir(parents=True, exist_ok=True)
        source_file = directory / 'source.json'
        if not source_file.exists():
            import numpy as np
            from levanter.store.tree_store import TreeStore
            store = TreeStore.open({'input_ids': np.zeros(0, dtype=np.int32)}, entry['shard'], mode='r', cache_metadata=True)
            ids = store.tree['input_ids'].get_batch_sync([entry['row']])[0]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            saved = {'source': row['source'], 'shard': row['shard'], 'row': row['row'],
                     'id': key, 'tokens': len(ids),
                     'input_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                     'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'text': text,
                     'source_loss_masks': 'not stored in source TreeStore; must reconstruct and validate from original template',
                     'length_disposition': 'over_limit' if len(ids) > 262144 else 'eligible'}
            (directory / 'input_ids.i32').write_bytes(ids.tobytes())
            source_file.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
        sources[entry['id']] = str(source_file)
    return sources


def fix_pod_paths(cfg, handle):
    """Rewrite the original pod-local config paths for the replay pod."""
    prod = Path(os.environ.get('REPLAY_PROD_DIR', '/app/experiments/glm_tool_serving/production_data'))
    prod.mkdir(parents=True, exist_ok=True)
    for name, s3key in (('sft_tokenizer', 'sft_tokenizer'), ('serving_tokenizer', 'serving_tokenizer/tokenizer.json')):
        dst = prod / s3key
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            handle.get(f'{BASE}/h100-eight-20261105/production_data/{s3key}', str(dst))
            log(f'fetched {s3key} ({dst.stat().st_size} bytes)')
    serving_dir = prod / 'serving_tokenizer'
    if not serving_dir.exists():
        serving_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(prod / 'serving_tokenizer/tokenizer.json'), str(serving_dir / 'tokenizer.json'))
    rules = prod / 'rule-approval-v3.json'
    if not rules.exists():
        handle.get(f'{BASE}/h100-eight-20261105/production_data/rule-approval-v3.json', str(rules))
        log(f'fetched rule-approval-v3.json ({rules.stat().st_size} bytes)')
    if cfg.get('rule_approval'):
        cfg['rule_approval']['path'] = str(rules)
    cfg['sft_tokenizer'] = str(prod / 'sft_tokenizer')
    cfg['serving_tokenizer'] = str(prod / 'serving_tokenizer')
    cap = WORK / 'capacity.json'
    generation = cfg['model'].get('generation', 'replay')
    cap.write_text(json.dumps({'generation': generation, 'conversion_slots': 24,
                               'science_slots': 0, 'expires_at_unix': time.time() + 30 * 3600}))
    cfg['model']['capacity_file'] = str(cap)
    ledger = WORK / 'ledger.sqlite3'
    cfg['model']['ledger'] = str(ledger)
    for evidence in cfg.get('adapter_evidence', []):
        name = Path(evidence['path']).name
        candidate = prod / name
        if candidate.exists():
            evidence['path'] = str(candidate)


def replay_batch(handle, replica_key, batch, cfg, tokenizer, workdir: Path) -> str:
    """Re-run one deferred batch; returns the replay archive URI."""
    fleet = replica_key.split('/r')[0]
    replica = int(replica_key.split('/r')[-1])
    batch_name = batch['batch']
    replay_name = batch_name + '-replay-' + hashlib.sha256(batch_name.encode()).hexdigest()[:12]
    dest_uri = f'{BASE}/{fleet}/r{replica}/batches/{replay_name}.tar.gz'
    try:
        handle.ls(dest_uri)
        return f'SKIP exists: {dest_uri}'  # already replayed
    except Exception:
        pass

    original_uri = f'{BASE}/{fleet}/r{replica}/batches/{batch["archive"]}'
    original = fs().cat(original_uri) if False else handle.cat(original_uri)
    workdir.mkdir(parents=True, exist_ok=True)
    bdir = workdir / replay_name
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir()
    preserved_statuses = {}
    with tarfile.open(fileobj=io.BytesIO(original)) as tar:
        for m in tar.getmembers():
            if m.isdir():
                continue
            tail = m.name
            if tail.startswith(batch_name + '/'):
                tail = tail[len(batch_name) + 1:]
            # Only whitelisted metadata is carried over. The parent mkdir MUST stay inside
            # this branch: creating parents for every member would recreate the original
            # run/records/<id> dirs as empty dirs and every replayed row would die on
            # mkdir with FileExistsError before any model call.
            if tail in ('config.json', 'lineage.json', 'provenance.json', 'dispositions.json', 'manifest.jsonl'):
                dest = bdir / tail
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(m).read())
    for p in batch['preserved']:
        preserved_statuses[p['id']] = p['status']

    sources = reconstruct_sources(batch, tokenizer, bdir)
    manifest_path = bdir / 'replay-manifest.jsonl'
    with manifest_path.open('w') as f:
        for entry in batch['deferred']:
            rid = entry['id']
            if rid not in sources:
                log(f'  source reconstruction failed for {rid[:12]}; skipping row')
                continue
            f.write(json.dumps({'id': rid, 'source_file': sources[rid]}) + '\n')

    run_root = bdir / 'run'
    run_root.mkdir(exist_ok=True)
    config_path = bdir / 'config.json'
    # run() re-reads the config from disk: persist the path-fixed cfg over the extracted original.
    config_path.write_text(json.dumps(cfg, indent=2))
    run(config_path, manifest_path, run_root)

    summary = {}
    summary_path = run_root / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    reconciliation = {
        'replica': replica,
        'batch': batch_name,
        'original_archive': batch['archive'],
        'decision': ('replay of pool-era infrastructure_deferred rows (HTTP 401 before any '
                     'conversion output; coordinator-approved classification); sibling rows '
                     'preserved with their committed outcomes'),
        'preserved': preserved_statuses,
        'replayed_ids': [e['id'] for e in batch['deferred'] if e['id'] in sources],
        'original_outcomes': batch['outcomes'],
        'replayed_at': time.time(),
    }
    (bdir / 'reconciliation.json').write_text(json.dumps(reconciliation, indent=2))

    archive_path = workdir / (replay_name + '.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(bdir, arcname=replay_name)
        tar.add(config_path, arcname='config.json')
    handle.put(str(archive_path), dest_uri)
    shutil.rmtree(bdir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)
    converted = sum(1 for s in summary.values() if isinstance(s, int))
    log(f'  {replay_name}: committed ({converted} outcome entries)')
    return f'COMMITTED: {dest_uri}'


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    handle = fs()
    inv = load_inventory(handle)
    replicas = select_replicas(inv)
    limit = int(os.environ.get('REPLAY_LIMIT', '0'))
    tokenizer = None
    tok_path = WORK / 'sft_tokenizer.json'
    if not tok_path.exists():
        handle.get(TOKENIZER_S3, str(tok_path))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))

    total = sum(sum(len(b['deferred']) for b in inv['replicas'][k]) for k in replicas)
    log(f'replaying {total} deferred rows across {len(replicas)} replicas'
        + (f' (limit {limit} batches)' if limit else ''))

    done = 0
    for replica_key in replicas:
        batches = sorted(inv['replicas'][replica_key], key=lambda b: b['batch'])
        cfg = None
        for batch in batches:
            if limit and done >= limit:
                log(f'batch limit {limit} reached')
                return
            try:
                if cfg is None:
                    original_uri = f'{BASE}/{replica_key}/batches/{batch["archive"]}'
                    raw = handle.cat(original_uri)
                    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                        cfg = json.load(tar.extractfile('config.json'))
                    fix_pod_paths(cfg, handle)
                    manifest_ref = os.environ.get('GLM_POOL_MANIFEST_REF')
                    if manifest_ref:
                        m = json.loads(handle.cat(manifest_ref).decode())
                        entry = (m[0] if isinstance(m, list) else m['entries'][0])
                        cfg['model']['base_url'] = entry['base_url']
                        key_ref = entry['api_key_ref']
                        key_data = handle.cat(key_ref).decode().strip()
                        if not key_data:
                            raise RuntimeError(f'EMPTY credential at {key_ref} (fail-fast)')
                        key_path = WORK / 'pool-api-key.txt'
                        key_path.write_text(key_data)
                        os.chmod(key_path, 0o600)
                        cfg['model']['api_key_file'] = str(key_path)
                        log(f'pool credential wired from manifest ({len(key_data)} bytes, not printed)')
                    if os.environ.get('GLM_POOL_MODEL'):
                        cfg['model']['model'] = os.environ['GLM_POOL_MODEL']
                    cfg['workers'] = int(os.environ.get('REPLAY_WORKERS', '16'))
                log(f'{replica_key} {batch["batch"]}: {len(batch["deferred"])} deferred rows')
                result = replay_batch(handle, replica_key, batch, cfg, tokenizer, WORK / 'scratch')
                log(f'  -> {result}')
                done += 1
            except Exception as error:
                import traceback
                log(f'  BATCH FAILED {batch["batch"]}: {error!r}')
                log(traceback.format_exc()[-500:])
    log(f'replay pass complete: {done} batches processed')


if __name__ == '__main__':
    main()
