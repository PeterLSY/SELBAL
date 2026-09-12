"""
analyze_merge_data.py  —  Phase 2: behavioral features & first predictor.

Place in the SESiL repo ROOT and run:

    conda activate sesil
    python analyze_merge_data.py

Reads merge_dataset_match_tensors_permute.csv (from collect_merge_data.py),
then:
  1. builds behavioral features per pair (shared classes, parent accuracies...),
  2. builds targets (child per-task acc; per-class retention = child - parent),
  3. prints Spearman correlations feature -> target,
  4. fits a cross-validated regressor (leave-one-out, 45 samples),
  5. saves scatter plots to ./analysis_figs/.
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'merge_dataset_match_tensors_permute.csv'
FIG_DIR = 'analysis_figs'
os.makedirs(FIG_DIR, exist_ok=True)


def classes_of(label_str):
    return sorted(set(int(x) for x in label_str.split('_')))


def build_features(df):
    rows = []
    for _, r in df.iterrows():
        ca, cb = classes_of(r['labels_a']), classes_of(r['labels_b'])
        pa = json.loads(r['a_per_class'])
        pb = json.loads(r['b_per_class'])
        pc = json.loads(r['child_per_class'])

        shared = sorted(set(ca) & set(cb))
        union = sorted(set(ca) | set(cb))
        disjoint = [c for c in union if c not in shared]

        # parent accuracy on a class (max over parents that know it)
        def parent_acc(c):
            vals = []
            if c in ca and pa[c] is not None: vals.append(pa[c])
            if c in cb and pb[c] is not None: vals.append(pb[c])
            return max(vals) if vals else np.nan

        # retention per class: child - best parent
        ret = {c: (pc[c] - parent_acc(c)) if pc[c] is not None else np.nan
               for c in union}
        ret_shared = [ret[c] for c in shared if not np.isnan(ret[c])]
        ret_disj = [ret[c] for c in disjoint if not np.isnan(ret[c])]

        feats = {
            # identifiers
            'pair': f"{r['labels_a']}|{r['labels_b']}",
            # ---------- behavioral features (Paper-1 style) ----------
            'n_shared': len(shared),
            'n_union': len(union),
            'a_per_task': r['a_per_task'],
            'b_per_task': r['b_per_task'],
            'mean_parent_acc': (r['a_per_task'] + r['b_per_task']) / 2,
            'min_parent_acc': min(r['a_per_task'], r['b_per_task']),
            'parent_acc_gap': abs(r['a_per_task'] - r['b_per_task']),
            'shared_acc_mean': np.mean([parent_acc(c) for c in shared]) if shared else np.nan,
            'shared_acc_gap': (np.mean([abs(pa[c] - pb[c]) for c in shared])
                               if shared else np.nan),
            # ---------- targets ----------
            'child_per_task': r['child_per_task'],
            'mean_retention': np.nanmean(list(ret.values())),
            'min_retention': np.nanmin(list(ret.values())),
            'retention_shared': np.mean(ret_shared) if ret_shared else np.nan,
            'retention_disjoint': np.mean(ret_disj) if ret_disj else np.nan,
        }
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV)
    print(f'Loaded {len(df)} merge outcomes from {CSV}\n')
    F = build_features(df)
    F.to_csv('merge_features.csv', index=False)

    features = ['n_shared', 'mean_parent_acc', 'min_parent_acc',
                'parent_acc_gap', 'shared_acc_mean', 'shared_acc_gap']
    targets = ['child_per_task', 'mean_retention', 'min_retention',
               'retention_shared', 'retention_disjoint']

    # ---------- 1) Spearman correlations ----------
    print('=== Spearman correlation (feature -> target) ===')
    header = f'{"feature":<18}' + ''.join(f'{t:>20}' for t in targets)
    print(header)
    for f in features:
        line = f'{f:<18}'
        for t in targets:
            mask = F[f].notna() & F[t].notna()
            if mask.sum() > 4:
                rho, p = spearmanr(F.loc[mask, f], F.loc[mask, t])
                star = '*' if p < 0.05 else ' '
                line += f'{rho:>18.3f}{star}'
            else:
                line += f'{"n/a":>19} '
        print(line)
    print('(* = p < 0.05)\n')

    # ---------- 2) group stats: shared vs disjoint retention ----------
    print('=== Retention by shared-class count ===')
    print(F.groupby('n_shared')[['child_per_task', 'mean_retention',
                                 'retention_shared', 'retention_disjoint']]
            .mean().round(4), '\n')

    # ---------- 3) leave-one-out predictor baseline ----------
    print('=== Leave-one-out prediction of child_per_task ===')
    X = F[features].fillna(F[features].mean()).values
    y = F['child_per_task'].values
    for name, model in [('Ridge', Ridge(alpha=1.0)),
                        ('GradBoost', GradientBoostingRegressor(
                            n_estimators=100, max_depth=2, random_state=0))]:
        pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
        rho, _ = spearmanr(pred, y)
        mae = np.mean(np.abs(pred - y))
        print(f'{name:<10} Spearman={rho:.3f}  MAE={mae:.4f}')

        plt.figure(figsize=(5, 5))
        plt.scatter(y, pred)
        lim = [min(y.min(), pred.min()) - .02, max(y.max(), pred.max()) + .02]
        plt.plot(lim, lim, 'k--', lw=1)
        plt.xlabel('actual child per-task acc')
        plt.ylabel('LOO-predicted')
        plt.title(f'{name}: Spearman={rho:.3f}')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/loo_{name}.png', dpi=150)
        plt.close()

    # feature importance from a full-data fit (indicative only, n=45)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                   random_state=0).fit(X, y)
    imp = sorted(zip(features, gb.feature_importances_),
                 key=lambda t: -t[1])
    print('\n=== Feature importance (GradBoost, full fit) ===')
    for f, v in imp:
        print(f'{f:<18}{v:.3f}')

    # ---------- 4) headline scatter plots ----------
    for f, t in [('n_shared', 'child_per_task'),
                 ('mean_parent_acc', 'child_per_task'),
                 ('shared_acc_gap', 'retention_shared'),
                 ('n_shared', 'retention_disjoint')]:
        mask = F[f].notna() & F[t].notna()
        plt.figure(figsize=(5, 4))
        plt.scatter(F.loc[mask, f], F.loc[mask, t])
        plt.xlabel(f); plt.ylabel(t)
        rho, p = spearmanr(F.loc[mask, f], F.loc[mask, t])
        plt.title(f'rho={rho:.3f} (p={p:.3f})')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/{f}__vs__{t}.png', dpi=150)
        plt.close()

    print(f'\nFigures saved to ./{FIG_DIR}/, features to merge_features.csv')


if __name__ == '__main__':
    main()