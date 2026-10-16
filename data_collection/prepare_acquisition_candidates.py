"""Split policy-filtered observations into class-specific acquisition inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def prepare(source, output):
    source, output = Path(source), Path(output)
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    required = {'observation_id', 'gbifID', 'decimalLatitude', 'decimalLongitude',
                'eventDate', 'species', 'occurrenceStatus', 'absence_check_passed'}
    if not fields or not required.issubset(fields):
        raise ValueError('Use the policy-filtered observation export, not basic deduplication output.')
    if not rows or any(r['occurrenceStatus'] not in {'PRESENT', 'ABSENT'} for r in rows):
        raise ValueError('Expected nonempty PRESENT/ABSENT observations.')
    ids = [r['observation_id'] for r in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Observation IDs must be nonempty and unique.')
    if any(r['occurrenceStatus'] == 'ABSENT' and r['absence_check_passed'].lower() != 'true' for r in rows):
        raise ValueError('Every ABSENT candidate must have passed the absence check.')
    groups = {label: [r for r in rows if r['occurrenceStatus'] == label]
              for label in ('PRESENT', 'ABSENT')}
    if any(not group for group in groups.values()):
        raise ValueError('Both classes are required.')
    output.mkdir(parents=True, exist_ok=False)
    for label, group in groups.items():
        with (output / f'{label.lower()}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(group)
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'counts': {label: len(group) for label, group in groups.items()},
                'policy': 'Preserves the supplied policy-filtered observations without reselection.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=config.get_path('DATA_COLLECTION_DIR') / 's2_sesimbra_like_observations.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
