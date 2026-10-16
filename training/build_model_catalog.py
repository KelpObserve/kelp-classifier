"""Build the comparison-model catalog from fitted checkpoints and selection records."""
from pathlib import Path
import hashlib
import json
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from encoder_geometry import validate_geometry


def main():
    root = config.get_path('FULL_NUISANCE_BALANCE_OUTPUT')
    weights = config.get_path('CHECKPOINTS_DIR')
    protocol = json.loads((root/'clay_full_sweep/protocol.json').read_text())
    validate_geometry(protocol)
    models = []
    for family, pools in [('mlp', [1, 2, 3, 5]), ('small_mlp', [1, 5]), ('logistic', [1, 5])]:
        for pool in pools:
            for seed in [42, 123, 456]:
                folder = f'clay_full_sweep/pool_{pool}/seed_{seed}' if family == 'mlp' else f'simple_heads/{family}_p{pool}_seed{seed}'
                filename = 'model.joblib' if family == 'logistic' else 'model.pt'
                path = weights/folder/filename
                if family == 'mlp':
                    selection = torch.load(path, weights_only=True, map_location='cpu')
                else:
                    selection = json.loads((root/folder/'selection.json').read_text())
                models.append(dict(model=f'{family}_pool{pool}_seed{seed}', architecture=family,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                    input_dim=2048, pool_size=pool, seed=seed,
                    validation_threshold=selection['threshold'], best_epoch=selection.get('best_epoch'),
                    regularization=selection.get('regularization', .0001),
                    relative_path=f'{folder}/{filename}', path_key='CHECKPOINTS_DIR',
                    split=f'training/full_nuisance_balance/seed_{seed}/split_seed{seed}.json',
                    results=f'training/full_nuisance_balance/{"clay_full_sweep" if family == "mlp" else "simple_heads"}/metrics.csv',
                    status='current_evaluation_checkpoint', encoder_gsd_m=protocol['gsd_m']))
    output = root.parent/'model_manifest.json'
    output.write_text(json.dumps(dict(experiment='full_nuisance_balance',
        backbone='Frozen Clay v1.5 large; external checkpoint',
        input='Full native 128x128 resized to224x224; six bands; zero time/location; CLS plus centrally pooled tokens',
        weights_location='Resolve CHECKPOINTS_DIR and each relative_path',
        final_fit='See training/final_models/model_manifest.json',models=models), indent=2)+'\n')
    print(f'Catalogued {len(models)} fitted comparison models.')


if __name__ == '__main__':
    main()
