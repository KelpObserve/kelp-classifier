"""Print configured artifact paths and access links without loading model data."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    selected = Path(os.environ.get('KELP_CONFIG', config.REPO_ROOT / 'config.json')).expanduser()
    if not selected.is_absolute():
        selected = config.REPO_ROOT / selected
    local = json.loads(selected.read_text()) if selected.exists() else {}
    defaults = json.loads((config.REPO_ROOT / 'example_config.json').read_text())
    resources = {key: {**value, **local.get('resources', {}).get(key, {})}
                 for key, value in defaults['resources'].items()}
    resources.update({key: value for key, value in local.get('resources', {}).items() if key not in resources})
    result = {}
    for key, resource in resources.items():
        try:
            path = config.get_path(key)
            result[key] = {'path': str(path), 'exists_here': path.exists(),
                           'access_uri': resource.get('access_uri', ''),
                           'description': resource.get('description', '')}
        except ValueError as exc:
            result[key] = {'configuration_needed': str(exc)}
    print(json.dumps({'resources': result}, indent=2))


if __name__ == '__main__':
    main()
