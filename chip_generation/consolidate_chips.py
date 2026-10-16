"""Consolidate fetch directories, retaining every observation-to-array association.

Usage: python chip_generation/consolidate_chips.py --output runs/chips INPUT_DIR ...
Input directories contain chips.h5 and chips_metadata.csv from the fetchers.
A fresh output directory is required. Array ordering follows input order; this
creates a new collection and does not recreate published HDF5 indices.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def consolidate(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays, associations, labels = {}, [], {}
    fields = ['h5_index', 'chip_id', 'sha256', 'label_conflict']
    with h5py.File(output / 'chips.h5', 'w') as target:
        chips = target.create_dataset('chips', (0, 8, 128, 128),
                                     maxshape=(None, 8, 128, 128),
                                     dtype='uint16', chunks=(1, 8, 128, 128), compression='lzf')
        target.attrs['bands'] = np.array(['B01', 'B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'SCL'], dtype='S')
        target.attrs['chip_size'] = 128
        target.attrs['pixel_m'] = 10
        for directory in map(Path, inputs):
            with (directory / 'chips_metadata.csv').open(newline='') as stream, h5py.File(directory / 'chips.h5', 'r') as source:
                for row in csv.DictReader(stream):
                    if row.get('status', '').lower() != 'ok':
                        continue
                    index = int(row['chipIndex'])
                    if not 0 <= index < int(source.attrs.get('n_chips', len(source['chips']))):
                        raise ValueError(f'Invalid input index in {directory.name}: {index}')
                    array = source['chips'][index]
                    if array.shape != (8, 128, 128) or array.dtype != np.uint16:
                        raise ValueError('Expected uint16 arrays shaped (8, 128, 128)')
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    if digest not in arrays:
                        destination = len(arrays)
                        chips.resize(destination + 1, axis=0)
                        chips[destination] = array
                        arrays[digest] = destination
                        labels[digest] = set()
                    tag = row.get('class_tag', '').lower()
                    if tag not in {'present', 'absent'}:
                        raise ValueError('Every accepted row requires class_tag present or absent')
                    labels[digest].add(tag)
                    # Updated fetchers preserve the observation ID before adding scene suffixes.
                    if not row.get('observation_id'):
                        raise ValueError('Missing observation_id; use outputs from the current fetchers')
                    association = dict(row, source_collection=directory.name,
                                       h5_index=arrays[digest], sha256=digest, chip_id=digest)
                    associations.append(association)
        target.attrs['n_chips'] = len(arrays)
    with (output / 'chips_metadata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for digest, index in arrays.items():
            writer.writerow(dict(h5_index=index, chip_id=digest, sha256=digest,
                                 label_conflict=len(labels[digest]) > 1))
    with (output / 'observation_chips.csv').open('w', newline='') as stream:
        fields = list(dict.fromkeys(k for row in associations for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(associations)
    summary = dict(arrays=len(arrays), associations=len(associations),
                   conflicting_arrays=sum(len(v) > 1 for v in labels.values()),
                   inputs=[Path(p).name for p in inputs])
    (output / 'consolidation.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.inputs, args.output), indent=2))
