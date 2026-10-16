"""Render hash-verified current selected chips as RGB and SCL inspection panels."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--out', type=Path, default=config.REPO_ROOT / 'runs/chip_inspection')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    frame = pd.read_csv(config.get_path('TRAINING_SELECTION')).drop_duplicates('chip_id')
    # Interleave labels so the default panel selection covers both classes.
    groups = [frame[frame.label.eq(label)] for label in [0, 1]]
    chosen = []
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group):
                chosen.append(group.iloc[index])
        if len(chosen) >= args.limit:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.get_path('CANONICAL_CHIPS_H5'), 'r') as source:
        for index, row in enumerate(chosen[:args.limit]):
            chip = source['chips'][int(row.h5_index)]
            if hashlib.sha256(chip.tobytes()).hexdigest() != row.sha256:
                raise ValueError(f'Array hash mismatch: {row.chip_id}')
            rgb = chip[[3, 2, 1]].astype(float).transpose(1, 2, 0)
            lo, hi = np.percentile(rgb, [1, 99], axis=(0, 1))
            rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(rgb)
            axes[0].set_title('RGB, per-band 1–99% stretch')
            image = axes[1].imshow(chip[7], vmin=0, vmax=11, cmap='tab20', interpolation='nearest')
            axes[1].set_title('Scene classification (SCL)')
            fig.colorbar(image, ax=axes[1], ticks=range(12), fraction=.046)
            for axis in axes:
                axis.add_patch(Rectangle((61.5, 61.5), 5, 5, fill=False, edgecolor='yellow'))
                axis.set_axis_off()
            fig.suptitle(f"{'PRESENT' if row.label else 'ABSENT'} · {row.chip_id}", fontsize=9)
            fig.tight_layout()
            destination = args.out / f'{index:02d}_label{int(row.label)}.png'
            fig.savefig(destination, dpi=140)
            plt.close(fig)
            print(destination)


if __name__ == '__main__':
    main()
