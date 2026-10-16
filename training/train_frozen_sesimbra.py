"""Frozen Clay v1.5 + repository MLP on immutable Sesimbra splits.

Uses full-chip input. Geography/time encodings are constant.
Preserves every split row and the 2:1 class ratio. No test-driven selection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, json, random, sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from model_head import MLPHead, sigmoid_focal_loss
from encoder_geometry import NATIVE_SIZE, NATIVE_GSD_M, ENCODER_SIZE, ENCODER_GSD_M, validate_geometry

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metrics(y, p, t):
    pred = p >= t
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(threshold=float(t), roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, pred)), macro_f1=float(f1.mean()), absent_recall=float(rec[0]), present_recall=float(rec[1]), absent_precision=float(prec[0]), present_precision=float(prec[1]), tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--metadata', type=Path)
    ap.add_argument('--pools', nargs='+', type=int, default=[1, 2, 3, 5])
    ap.add_argument('--embedding-cache', type=Path)
    ap.add_argument('--chips', type=Path, help='Canonical chips.h5; otherwise use CANONICAL_CHIPS_H5 from config')
    args = ap.parse_args()
    args.dataset = args.dataset or config.get_path('FULL_NUISANCE_BALANCE_OUTPUT')
    args.checkpoint = args.checkpoint or config.get_path('CLAY_CKPT')
    args.metadata = args.metadata or config.get_path('CLAY_META')
    args.chips = args.chips or config.get_path('CANONICAL_CHIPS_H5')
    root = args.dataset.resolve()
    out = root / 'clay_full_sweep'
    out.mkdir(exist_ok=False)
    assert len(set(args.pools)) == len(args.pools) and all((p in [1, 2, 3, 5, 7] for p in args.pools))
    pools = args.pools
    from split_groups import group_rows
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    frames = {}
    lineage = {}
    for seed in [42, 123, 456]:
        f = pd.read_csv(root / f'seed_{seed}/selected.csv')
        sp_path = root / f'seed_{seed}/split_seed{seed}.json'
        sp = json.loads(sp_path.read_text())
        f['group'] = group_rows(f)
        assert f.chip_id.is_unique and f.groupby('group').split.nunique().max() == 1
        assert f.label.isin([0, 1]).all()
        for split in ['train', 'val', 'test']:
            z = f[f.split.eq(split)]
            assert set(z.chip_id) == set(sp[split + '_chip_ids'])
            assert int(z.label.sum()) == 2 * int(z.label.eq(0).sum())
        for p in [sp_path, root / f'seed_{seed}/selected.csv']:
            lineage[str(p.relative_to(root))] = sha(p)
        duplicate_split = root / f'splits/split_seed{seed}.json'
        if duplicate_split.exists():
            assert json.loads(duplicate_split.read_text()) == sp
            lineage[str(duplicate_split.relative_to(root))] = sha(duplicate_split)
        frames[seed] = f
    allrows = pd.concat(frames.values()).drop_duplicates('chip_id').sort_values('chip_id').reset_index(drop=True)
    assert allrows.chip_id.is_unique and allrows.groupby('chip_id').sha256.nunique().max() == 1
    n_arrays = len(allrows)
    protocol = dict(chips_path='{CANONICAL_CHIPS_H5}', scope='full', seeds=[42, 123, 456], clay_checkpoint='{CLAY_CKPT}', checkpoint_sha256=sha(args.checkpoint), script_sha256=sha(__file__), clay_metadata_sha256=sha(args.metadata), embedding='Frozen Clay v1.5 large; CLS concatenated with pooled patch token(s), 2048 dimensions', spatial_input='Full native 128x128 resized to 224x224; CLS plus centre pooled patch tokens; encoder spacing preserves the 1280 m footprint', gsd_m=ENCODER_GSD_M, native_gsd_m=NATIVE_GSD_M, native_size=NATIVE_SIZE, resized_size=ENCODER_SIZE, pool_sizes=pools, caveat='Resizing preserves the native footprint; it does not add image detail. Time/location remain constant and training retains 2:1.', metadata_inputs='Time and latitude/longitude encodings held at all-zero constants for every image. No depth or other tabular predictors.', bands=['B02', 'B03', 'B04', 'B05', 'B08', 'B11'], head='2048 -> 512 ReLU dropout0.3 -> 128 ReLU dropout0.3 -> 1', training='All train rows, no resampling; focal alpha=.5 gamma=2; Adam lr=.001 weight_decay=.0001; batch64; max100epochs; patience25; no embedding scaling', selection='Checkpoint selected by validation macroF1 at .5; threshold selected by validation macroF1 over .05:.95 step.01; test evaluated only after selection', ratio='2 PRESENT : 1 ABSENT in every split', limitations='Exploratory dataset; selection and split balancing previously used full-pool covariates. Few independent ABSENT groups in held-out sets.', input_hashes=lineage)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.embedding_cache:
        import shutil
        cache = np.load(args.embedding_cache)
        assert np.array_equal(cache['chip_ids'], allrows.chip_id.to_numpy(str))
        assert np.array_equal(cache['sha256'], allrows.sha256.to_numpy(str))
        cls_cache = cache['cls']
        patch_cache = cache['centre_patches']
        assert cls_cache.shape == (n_arrays, 1024) and patch_cache.shape == (n_arrays, 7, 7, 1024)
        assert np.isfinite(cls_cache).all() and np.isfinite(patch_cache).all()
        source_protocol = json.loads(args.embedding_cache.with_name('protocol.json').read_text())
        validate_geometry(source_protocol)
        for field in ['checkpoint_sha256', 'clay_metadata_sha256', 'scope', 'bands', 'metadata_inputs']:
            assert protocol[field] == source_protocol[field], field
        protocol['embedding_cache_source'] = args.embedding_cache.name
        protocol['embedding_cache_sha256'] = sha(args.embedding_cache)
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        shutil.copy2(args.embedding_cache, out / 'embeddings.npz')
        torch.set_float32_matmul_precision('medium')
        print('Reused verified full-chip cache:', len(cls_cache), 'arrays', flush=True)
    else:
        from claymodel.module import ClayMAEModule
        model = ClayMAEModule.load_from_checkpoint(str(args.checkpoint), model_size='large', shuffle=False, mask_ratio=0.0, metadata_path=str(args.metadata), map_location='cuda').eval().to('cuda')
        stats = yaml.safe_load(args.metadata.read_text())['sentinel-2-l2a']['bands']
        keys = ['blue', 'green', 'red', 'rededge1', 'nir', 'swir16']
        mean = torch.tensor([stats['mean'][k] for k in keys], device='cuda')[None, :, None, None]
        std = torch.tensor([stats['std'][k] for k in keys], device='cuda')[None, :, None, None]
        waves = torch.tensor([0.493, 0.56, 0.665, 0.704, 0.842, 1.61], device='cuda')
        vectors = []
        cls_cache = []
        patch_cache = []
        with h5py.File(args.chips, 'r') as h, torch.inference_mode():
            for start in range(0, len(allrows), 4):
                batch = allrows.iloc[start:start + 4]
                arrays = []
                for row in batch.itertuples():
                    a = h['chips'][int(row.h5_index)]
                    assert sha_array(a) == row.sha256
                    c = a[:, 62:67, 62:67]
                    assert ((c[7] == 6) & (c[1] >= 10) & (c[2] >= 10)).sum() >= 24
                    a = a[[1, 2, 3, 4, 5, 6]]
                    arrays.append(a)
                pix = torch.tensor(np.stack(arrays).astype(np.float32), device='cuda')
                size = ENCODER_SIZE
                pix = F.interpolate(pix, size=(size, size), mode='bilinear', align_corners=False)
                pix = (pix - mean) / std
                n = len(batch)
                encoded = model.model.encoder(dict(pixels=pix, time=torch.zeros(n, 4, device='cuda'), latlon=torch.zeros(n, 4, device='cuda'), gsd=torch.tensor(ENCODER_GSD_M, device='cuda'), waves=waves, platform='sentinel-2-l2a'))[0]
                patches = encoded[:, 1:].reshape(n, 28, 28, 1024)
                cls_cache.append(encoded[:, 0].cpu().numpy())
                patch_cache.append(patches[:, 10:17, 10:17].cpu().numpy())
                pool = patches[:, 11:16, 11:16].mean((1, 2))
                vectors.append(torch.cat([encoded[:, 0], pool], dim=1).cpu().numpy())
                if start % 80 == 0:
                    print('Embedded', start + n, '/', len(allrows), flush=True)
        emb = np.concatenate(vectors)
        assert emb.shape == (n_arrays, 2048) and np.isfinite(emb).all()
        cls_cache = np.concatenate(cls_cache)
        patch_cache = np.concatenate(patch_cache)
        assert patch_cache.shape == (n_arrays, 7, 7, 1024) and np.isfinite(patch_cache).all()
        np.savez(out / 'embeddings.npz', chip_ids=allrows.chip_id.to_numpy(str), sha256=allrows.sha256.to_numpy(str), cls=cls_cache, centre_patches=patch_cache)
        del model
        torch.cuda.empty_cache()
    idx = {cid: i for i, cid in enumerate(allrows.chip_id)}
    results = []
    predictions = []
    from patch_pool import cache_local_slice
    for pool_size in pools:
        pool_out = out / f'pool_{pool_size}'
        pool_out.mkdir()
        lo, hi = cache_local_slice(pool_size)
        emb = np.concatenate([cls_cache, patch_cache[:, lo:hi, lo:hi].mean((1, 2))], axis=1)
        print('Training pool', pool_size, flush=True)
        for seed, f in frames.items():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            np.random.seed(seed)
            sets = {s: f[f.split.eq(s)].copy() for s in ['train', 'val', 'test']}

            def tensors(s):
                z = sets[s]
                return (torch.tensor(emb[[idx[c] for c in z.chip_id]], device='cuda'), torch.tensor(z.label.to_numpy(np.float32), device='cuda'))
            xt, yt = tensors('train')
            xv, yv = tensors('val')
            yval = sets['val'].label.to_numpy()
            head = MLPHead().cuda()
            opt = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=0.0001)
            best = -1.0
            stale = 0
            history = []
            for epoch in range(1, 101):
                head.train()
                order = torch.randperm(len(yt), device='cuda')
                loss_sum = 0.0
                for i in range(0, len(yt), 64):
                    b = order[i:i + 64]
                    loss = sigmoid_focal_loss(head(xt[b]), yt[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_sum += float(loss) * len(b)
                head.eval()
                with torch.no_grad():
                    vp = torch.sigmoid(head(xv)).cpu().numpy()
                score = metrics(yval, vp, 0.5)['macro_f1']
                history.append(dict(epoch=epoch, train_loss=loss_sum / len(yt), val_macro_f1=score))
                if score > best + 1e-05:
                    best = score
                    best_epoch = epoch
                    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if epoch % 10 == 0:
                    print('Seed', seed, 'epoch', epoch, 'val_macro_f1', round(score, 4), flush=True)
                if stale >= 25:
                    break
            head.load_state_dict(state)
            head.eval()
            with torch.no_grad():
                vp = torch.sigmoid(head(xv)).cpu().numpy()
            grid = [0.5] + [float(t) for t in np.linspace(0.05, 0.95, 91) if not np.isclose(t, 0.5)]
            threshold = max(grid, key=lambda t: metrics(yval, vp, t)['macro_f1'])
            dest = pool_out / f'seed_{seed}'
            dest.mkdir()
            torch.save(dict(state_dict=state, threshold=threshold, scope='full', pool_size=pool_size, seed=seed, best_epoch=best_epoch, embedding_dim=2048, encoder_gsd_m=ENCODER_GSD_M), dest / 'model.pt')
            for split in ['train', 'val', 'test']:
                x, y = tensors(split)
                z = sets[split]
                with torch.no_grad():
                    p = torch.sigmoid(head(x)).cpu().numpy()
                yi = z.label.to_numpy()
                groups = z.group.to_numpy()
                weights = np.array([1 / np.sum((groups == g) & (yi == label)) for g, label in zip(groups, yi)])
                boot = []
                rng = np.random.default_rng(seed)
                ug = np.unique(groups)
                if split != 'train':
                    for _ in range(1000):
                        sampled = rng.choice(ug, len(ug), replace=True)
                        ix = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
                        if len(np.unique(yi[ix])) == 2:
                            boot.append(roc_auc_score(yi[ix], p[ix]))
                for mode, t in [('default', 0.5), ('validation_tuned', threshold)]:
                    results.append(dict(pool_size=pool_size, seed=seed, split=split, threshold_mode=mode, N=len(z), P=int(yi.sum()), A=int((yi == 0).sum()), best_epoch=best_epoch, **metrics(yi, p, t), group_equal_auc=float(roc_auc_score(yi, p, sample_weight=weights)), auc_ci_low=float(np.quantile(boot, 0.025)) if boot else None, auc_ci_high=float(np.quantile(boot, 0.975)) if boot else None))
                for row, prob in zip(z.itertuples(), p):
                    predictions.append(dict(pool_size=pool_size, seed=seed, split=split, chip_id=row.chip_id, label=row.label, group=row.group, probability=float(prob)))
            (dest / 'history.json').write_text(json.dumps(history, indent=2))
            print('Finished seed', seed, 'best epoch', best_epoch, 'threshold', threshold, flush=True)
    assert all((sha(root / p) == digest for p, digest in lineage.items()))
    pd.DataFrame(results).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    (out / 'verification.json').write_text(json.dumps(dict(unique_arrays=n_arrays, all_array_hashes_verified=not bool(args.embedding_cache), cached_array_hashes_match_selected_rows=True, frozen_inputs_unchanged=True, no_missing_embeddings=True, exact_2_to_1_in_every_split=True, no_leakage_group_overlap=True), indent=2))
    r = pd.DataFrame(results)
    tab = r[r.split.eq('test')]
    html = '<meta charset="utf-8"><title>Clay classifier</title><style>body{font:16px system-ui;margin:40px;max-width:1300px}td,th{padding:7px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>Frozen Clay + trained MLP: ' + 'full' + '</h1><p>All three saved splits retained with 2 PRESENT per ABSENT in every partition. Checkpoint and threshold chosen using validation only. No depth, geography or date predictors. Clay encoder frozen; MLP head trained.</p><h2>Test metrics</h2>' + tab.round(3).to_html(index=False) + '<h2>Validation and training</h2>' + r[~r.split.eq('test')].round(3).to_html(index=False) + '<h2>Protocol</h2><pre>' + json.dumps(protocol, indent=2) + '</pre><p>Bootstrap intervals resample held-out leakage groups and are conditional on fitted models; few ABSENT groups make uncertainty substantial. These are exploratory results, not an independent external validation.</p>'
    (out / 'report.html').write_text(html)
    print(tab.to_string(index=False), flush=True)

def sha_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()
if __name__ == '__main__':
    main()
