"""Check required workflow inputs without importing numerical or GPU libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

WORKFLOWS = {
    'reports': [('FULL_NUISANCE_BALANCE_OUTPUT', 'clay_full_sweep/metrics.csv'),
                ('FULL_NUISANCE_BALANCE_OUTPUT', 'baseline_metrics.csv'),
                ('FINAL_MODELS_DIR', 'metrics.csv')],
    'training': [('CANONICAL_CHIPS_H5', ''), ('CLAY_CKPT', ''), ('CLAY_META', '')] +
                [('FULL_NUISANCE_BALANCE_OUTPUT', f'seed_{seed}/{name}')
                 for seed in [42, 123, 456] for name in ['selected.csv', f'split_seed{seed}.json']],
    'final': [('FULL_NUISANCE_EMBEDDINGS', ''),
              ('FULL_NUISANCE_EMBEDDINGS', '@protocol')] +
             [('FULL_NUISANCE_BALANCE_OUTPUT', f'seed_{seed}/selected.csv') for seed in [42, 123, 456]],
    'field': [('CLAY_CKPT', ''), ('CLAY_META', ''), ('FIELD_OBSERVATIONS', ''),
              ('S2_CAMPAIGNS_DIR', 'pixel_date_labels.csv')] +
             [('S2_CAMPAIGNS_DIR', f'{month}/scene.npz') for month in ['june', 'july', 'september']] +
             [('FINAL_CHECKPOINTS_DIR', f'{name}/model.pt') for name in ['small_mlp_pool5', 'mlp_pool1']],
}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def check(workflow, hashes=False):
    problems = []
    for key, child in WORKFLOWS[workflow]:
        try:
            path = config.get_path(key)
            if child == '@protocol':
                path = path.with_name('protocol.json')
            elif child:
                path = path / child
            if not path.is_file():
                problems.append(f'{key}' + (' companion protocol.json' if child == '@protocol' else f'/{child}' if child else '') + ': missing file')
        except (ValueError, FileNotFoundError, KeyError) as exc:
            problems.append(str(exc))
    if hashes and not problems:
        if workflow in {'training', 'field'}:
            protocol = json.loads((config.REPO_ROOT / 'training/full_nuisance_balance/clay_full_sweep/protocol.json').read_text())
            for key, field in [('CLAY_CKPT', 'checkpoint_sha256'), ('CLAY_META', 'clay_metadata_sha256')]:
                if digest(config.get_path(key)) != protocol[field]:
                    problems.append(f'{key}: SHA-256 differs from the published encoder')
        if workflow == 'field':
            manifest = json.loads((config.get_path('FINAL_MODELS_DIR') / 'model_manifest.json').read_text())
            for model in manifest['models']:
                if digest(config.get_path('FINAL_CHECKPOINTS_DIR') / model['relative_path']) != model['sha256']:
                    problems.append(f"Checkpoint hash mismatch: {model['relative_path']}")
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', choices=WORKFLOWS)
    parser.add_argument('--hashes', action='store_true', help='Also verify applicable backbone/metadata/checkpoint hashes')
    args = parser.parse_args()
    problems = check(args.workflow, args.hashes)
    for problem in problems:
        print('MISSING OR INVALID:', problem)
    print(f'{args.workflow}: ' + ('input checks failed' if problems else 'required files present'))
    raise SystemExit(bool(problems))
