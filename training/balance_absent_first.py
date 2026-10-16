"""Balance ABSENT partitions first, then match PRESENT marginal distributions.

Retain every ABSENT chip, approximately 70/15/15 partitions, 2:1 class counts and whole
50 m groups. Optimize feature distributions, never classifier test scores.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.stats import ks_2samp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from baselines import evaluate

def partition_absent(a, old, seed, cols):
    """Seeded group-swap search using ABSENT marginal CDFs only."""
    groups = np.unique(a.group)
    target_counts = np.array([150, 32, 32]) if len(a) == 214 else np.array([0.7, 0.15, 0.15]) * len(a)
    lower_counts = np.floor(target_counts) - [10, 5, 5]
    upper_counts = np.ceil(target_counts) + [10, 5, 5]
    blocks = []
    owners = []
    for j, col in enumerate(cols):
        values = a[col].to_numpy()
        cuts = np.unique(values) if col == 'depth_m' and any((c.startswith('nuisance_') for c in cols)) else np.unique(np.quantile(values, np.linspace(0, 1, 31)))
        for cut in cuts:
            blocks.append((values <= cut).astype(float))
            owners.append(j)
    M = np.column_stack(blocks)
    owners = np.array(owners)
    starts = np.r_[0, np.flatnonzero(np.diff(owners)) + 1]
    totals = np.array([M[a.group.eq(g)].sum(axis=0) for g in groups])
    sizes = np.array([a.group.eq(g).sum() for g in groups])
    target = M.mean(axis=0)
    oldmap = old[old.label.eq(0)].set_index('chip_id').split
    initial = np.array([['train', 'val', 'test'].index(oldmap.loc[a[a.group.eq(g)].chip_id].iloc[0]) for g in groups])
    assert all((oldmap.loc[a[a.group.eq(g)].chip_id].nunique() == 1 for g in groups))

    def score(assign):
        counts = np.bincount(assign, weights=sizes, minlength=3)
        if np.any(counts < lower_counts) or np.any(counts > upper_counts) or np.min(np.bincount(assign, minlength=3)) < 5:
            return np.inf
        delta = np.abs(np.array([totals[assign == k].sum(axis=0) / counts[k] for k in range(3)]) - target)
        distances = np.maximum.reduceat(delta, starts, axis=1)
        return float(distances.mean() + distances.max() + 0.02 * np.mean(np.abs(counts - target_counts) / target_counts))
    rng = np.random.default_rng(seed)
    best = initial.copy()
    best_score = score(best)
    for restart in range(5):
        current = best.copy()
        value = score(current)
        for step in range(8000):
            proposal = current.copy()
            i, j = rng.choice(len(groups), 2, replace=False)
            if rng.random() < 0.8:
                proposal[i], proposal[j] = (proposal[j], proposal[i])
            else:
                proposal[i] = rng.integers(3)
            v = score(proposal)
            temperature = 0.025 * (1 - step / 8000) ** 2 + 5e-05
            if v < value or rng.random() < np.exp(min(0, (value - v) / temperature)):
                current, value = (proposal, v)
            if value < best_score:
                best, best_score = (current.copy(), value)
    allocation = {g: ['train', 'val', 'test'][best[i]] for i, g in enumerate(groups)}
    return (a.group.map(allocation), dict(method='Five seeded annealing restarts, 8000 group proposals each; heuristic, no optimality guarantee', objective=best_score, initial_objective=score(initial), group_counts=np.bincount(best, minlength=3).tolist()))

def main():
    initialization = config.get_path('SPLIT_INITIALIZATION')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    pool = config.get_path('FULL_NUISANCE_MATCHING_POOL')
    out = config.get_path('FULL_NUISANCE_BALANCE_OUTPUT')
    prefix = 'nuisance_'
    out.mkdir(parents=True, exist_ok=args.resume)
    raw = pd.read_csv(pool)
    f = raw[raw.depth_m.between(-20, 0) & raw.depth_disagreement.le(1) & ~raw.label_ambiguous & np.where(raw.label.eq(1), raw.abs_days_off.le(10), raw.abs_days_off.le(30))].sort_values('chip_id').reset_index(drop=True)
    assert f.chip_id.is_unique and f.label.eq(1).sum() >= 2 * f.label.eq(0).sum()
    f['group'] = group_rows(f)
    p = f[f.label.eq(1)].reset_index(drop=True)
    a = f[f.label.eq(0)].reset_index(drop=True)
    groups = np.unique(f.group)
    gi = {g: i for i, g in enumerate(groups)}
    n = len(p)
    G = len(groups)
    cols = [c for c in f if c.startswith(prefix)] + ['depth_m']
    allrows = pd.concat([p, a], ignore_index=True)
    num = allrows[cols].fillna(allrows[cols].median()).fillna(0)
    sd = num.std().replace(0, 1)
    z = ((num - num.mean()) / sd).clip(-5, 5).to_numpy()
    blocks = [z]
    names = ['mean:' + c for c in cols]
    weights = [1 / len(cols)] * len(cols)
    iscdf = [False] * len(cols)
    thresholds = []
    for j, c in enumerate(cols):
        q = np.unique(num[c].quantile(np.linspace(0, 1, 21)))
        if c == 'depth_m':
            ad = a.depth_m.to_numpy()
            q = np.unique(np.r_[q, ad, np.nextafter(ad, -np.inf)])
        for cut in q:
            blocks.append((num[c].to_numpy() <= cut)[:, None].astype(float))
            names.append(f'cdf:{c}:{cut:.9g}')
            thresholds.append((c, cut))
            iscdf.append(True)
    weights += [2 / len(thresholds)] * len(thresholds)
    M = np.column_stack(blocks)
    K = M.shape[1]
    weights = np.array(weights)
    iscdf = np.array(iscdf)
    for i, name in enumerate(names):
        if 'depth_m' in name:
            weights[i] *= 3
        elif any((c in name for c in [prefix + 'cloud', prefix + 'land', prefix + 'water'])):
            weights[i] *= 1.5
    records = pd.read_csv(out / 'baseline_metrics.csv').to_dict('records') if args.resume and (out / 'baseline_metrics.csv').exists() else []
    balance = pd.read_csv(out / 'feature_balance.csv').to_dict('records') if args.resume and (out / 'feature_balance.csv').exists() else []
    counts = []
    lineage = {}
    solver = []
    splits = ['train', 'val', 'test']
    targets = [300, 64, 64]
    importance = [1, 1, 1]
    for seed in [42, 123, 456]:
        initial = pd.read_csv(initialization)
        initial = initial[initial.seed.eq(seed)].drop(columns='seed')
        old = f[f.label.eq(0)].merge(initial, on='chip_id', validate='one_to_one')
        assert len(old) == len(a), 'Initialization must cover every eligible ABSENT chip'
        lineage[str(seed)] = hashlib.sha256(initialization.read_bytes()).hexdigest()
        existing = out / f'seed_{seed}/selected.csv'
        if args.resume and existing.exists():
            selected = pd.read_csv(existing)
            for split in ['train', 'val', 'test']:
                part = selected[selected.split.eq(split)]
                counts.append(dict(seed=seed, split=split, P=int(part.label.sum()), A=int(part.label.eq(0).sum())))
            solver.append(json.loads((existing.parent / 'solver.json').read_text()))
            continue
        amap, absent_protocol = partition_absent(a, old, seed, cols)
        targets = [2 * int(amap.eq(sp).sum()) for sp in splits]
        oldA = pd.Series(amap.to_numpy(), index=a.chip_id)
        print(f'Seed {seed}: ABSENT counts {[t // 2 for t in targets]}, partition objective {absent_protocol}', flush=True)
        fixed = {}
        for g, sp in zip(a.group, amap):
            k = splits.index(sp)
            assert g not in fixed or fixed[g] == k
            fixed[g] = k
        xsize = 3 * n
        ystart = xsize
        dstart = ystart + 3 * G
        tstart = dstart + 3 * K
        fstart = tstart + 3
        V = fstart + 3 * len(cols)
        ri = []
        ci = []
        vv = []
        lower = []
        upper = []

        def add(ids, vals, lo=-np.inf, hi=np.inf):
            row = len(lower)
            ri.extend([row] * len(ids))
            ci.extend(ids)
            vv.extend(vals)
            lower.append(lo)
            upper.append(hi)
        for i in range(n):
            add([s * n + i for s in range(3)], [1] * 3, hi=1)
        for g in groups:
            j = gi[g]
            add([ystart + s * G + j for s in range(3)], [1] * 3, hi=1)
            if g in fixed:
                add([ystart + fixed[g] * G + j], [1], lo=1, hi=1)
        for s in range(3):
            for i, g in enumerate(p.group):
                add([s * n + i, ystart + s * G + gi[g]], [1, -1], hi=0)
            add(list(range(s * n, (s + 1) * n)), [1] * n, lo=targets[s], hi=targets[s])
            aim = M[n:][amap.eq(splits[s]).to_numpy()].mean(axis=0)
            for k in range(K):
                vals = M[:n, k] / targets[s]
                nz = np.flatnonzero(vals)
                ids = (s * n + nz).tolist()
                add(ids + [dstart + s * K + k], vals[nz].tolist() + [-1], hi=aim[k])
                add(ids + [dstart + s * K + k], (-vals[nz]).tolist() + [-1], hi=-aim[k])
                if iscdf[k]:
                    add([dstart + s * K + k, tstart + s], [1, -1], hi=0)
                    feature_index = cols.index(names[k].split(':')[1])
                    add([dstart + s * K + k, fstart + s * len(cols) + feature_index], [1, -1], hi=0)
        c = np.zeros(V)
        for s, w in enumerate(importance):
            c[dstart + s * K:dstart + (s + 1) * K] = weights * w
            c[tstart + s] = 2 * w
            c[fstart + s * len(cols):fstart + (s + 1) * len(cols)] = w / len(cols)
        c[:xsize] = np.random.default_rng(seed).uniform(0, 1e-07, xsize)
        integrality = np.zeros(V)
        integrality[:dstart] = 1
        ub = np.full(V, np.inf)
        ub[:dstart] = 1
        matrix = coo_matrix((vv, (ri, ci)), shape=(len(lower), V)).tocsr()
        print(f'Seed {seed}: optimizing {n} PRESENT candidates against fixed ABSENT partitions; {K} distribution features', flush=True)
        result = milp(c, integrality=integrality, bounds=Bounds(np.zeros(V), ub), constraints=LinearConstraint(matrix, lower, upper), options={'time_limit': 180, 'mip_rel_gap': 0.015})
        fallback = False
        if result.x is None:
            print('Retrying the same objective with a longer time limit', flush=True)
            result = milp(c, integrality=integrality, bounds=Bounds(np.zeros(V), ub), constraints=LinearConstraint(matrix, lower, upper), options={'time_limit': 600, 'mip_rel_gap': 0.025})
        assert result.x is not None, result.message
        selected = []
        for s, split in enumerate(splits):
            take = np.flatnonzero(result.x[s * n:(s + 1) * n] > 0.5)
            assert len(take) == targets[s]
            part = pd.concat([p.iloc[take], a[amap.eq(split)]], ignore_index=True)
            part['split'] = split
            selected.append(part)
            counts.append(dict(seed=seed, split=split, P=len(take), A=int(part.label.eq(0).sum())))
        selected = pd.concat(selected).sort_values('chip_id').reset_index(drop=True)
        assert len(selected) == 3 * len(a) and selected.chip_id.is_unique and (selected.groupby('group').split.nunique().max() == 1)
        assert selected[selected.label.eq(0)].set_index('chip_id').split.to_dict() == oldA.to_dict()
        folder = out / f'seed_{seed}'
        folder.mkdir()
        selected.to_csv(folder / 'selected.csv', index=False)
        (folder / f'split_seed{seed}.json').write_text(json.dumps({s + '_chip_ids': selected[selected.split.eq(s)].chip_id.tolist() for s in splits}, indent=2))
        solver.append(dict(seed=seed, status=int(result.status), message=result.message, objective=float(result.fun), gap=float(result.mip_gap), fixed_group_fallback=fallback, absent_partition=absent_protocol))
        for variant, frame in [('balanced', selected)]:
            for split in splits:
                aa = frame[frame.split.eq(split) & frame.label.eq(0)]
                pp = frame[frame.split.eq(split) & frame.label.eq(1)]
                for col in cols:
                    av = aa[col].dropna()
                    pv = pp[col].dropna()
                    scale = np.sqrt((av.var() + pv.var()) / 2)
                    balance.append(dict(variant=variant, seed=seed, split=split, feature=col, ks=float(ks_2samp(pv, av).statistic), smd=float((pv.mean() - av.mean()) / scale) if scale else 0))
            for split in ['val', 'test']:
                records.extend(evaluate(frame, split, seed, variant, image_prefix='full_', image_family='full'))
                nuisance_records = evaluate(frame, split, seed, variant, image_prefix='nuisance_', image_family='quality')
                records.extend((r for r in nuisance_records if r['family'] != 'depth_only'))
        (folder / 'solver.json').write_text(json.dumps(solver[-1], indent=2))
        pd.DataFrame(records).to_csv(out / 'baseline_metrics.csv', index=False)
        pd.DataFrame(balance).to_csv(out / 'feature_balance.csv', index=False)
        print(f'Seed {seed} complete; solver gap {result.mip_gap:.3f}', flush=True)
    pd.DataFrame(counts).to_csv(out / 'split_counts.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(pool.read_bytes()).hexdigest(), initialization_hashes=lineage, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pixel_scope='full native 128x128', predictors=cols, all_absent_retained=True, absent_assignments_unchanged=False, ratio='2:1', priority=dict(train=1, val=1, test=1), distribution_objective='ABSENT group partitioning first, then PRESENT marginal means and CDFs, per-feature and global maximum CDF gaps. No feature combinations. Extra weight for depth and SCL fractions.', selection='No classifier predictions used to choose chips or splits.', solver=solver, limitations='Exploratory covariate-balanced population; all-pool covariates used for dataset construction, including evaluation partitions. Both ABSENT assignments and PRESENT test chips change; not an untouched model test.'), indent=2))
    print(pd.DataFrame(records).groupby(['variant', 'split', 'family', 'model'])[['macro_f1', 'roc_auc']].mean().round(3).to_string(), flush=True)
if __name__ == '__main__':
    main()
