"""Verify externally supplied published artifacts against the versioned inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(manifest, groups):
    problems = []
    for item in manifest['artifacts']:
        if groups and item['group'] not in groups:
            continue
        label = item['id']
        try:
            path = config.get_path(item['path_key'])
            if item.get('relative_path'):
                path = path / item['relative_path']
            if not path.is_file():
                problems.append(f'{label}: missing')
            elif digest(path) != item['sha256']:
                problems.append(f'{label}: checksum mismatch')
        except (ValueError, FileNotFoundError, KeyError) as exc:
            problems.append(f'{label}: {type(exc).__name__}')
    return problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', choices=['training', 'comparison', 'final', 'field', 'sources'])
    args = parser.parse_args()
    manifest = json.loads((config.REPO_ROOT / 'configs/artifacts.json').read_text())
    problems = verify(manifest, args.group)
    for problem in problems:
        print(problem)
    print(f'Artifact verification: {len(problems)} errors')
    raise SystemExit(bool(problems))
