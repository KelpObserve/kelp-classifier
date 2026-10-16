"""Copy the published partition evidence into a fresh training destination."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination)
    if not destination.is_absolute():
        destination = ROOT / destination
    source = ROOT / 'training/full_nuisance_balance'
    files = [source / name for name in ['protocol.json', 'feature_balance.csv', 'selected.csv', 'split_counts.csv']]
    for seed in [42, 123, 456]:
        files.extend(source / f'seed_{seed}' / name for name in ['selected.csv', f'split_seed{seed}.json'])
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    print(prepare(parser.parse_args().destination))
