"""Paper 2 statistical revision analysis.

Addresses reviewer concerns using the already-saved lesion-grouped MC predictions:
1) primary uncertainty score selected using validation AURC only;
2) validation-selected thresholds evaluated across multiple target coverages;
3) lesion-cluster bootstrap 95% CIs;
4) paired lesion-cluster bootstrap differences between uncertainty scores;
5) melanoma false-negative escape analysis;
6) secondary lesion-level probability aggregation.

No test labels are used to select the primary uncertainty score or thresholds.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

EPS = 1e-12
B = 1000
BOOT_SEED = 20260811
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MEL = CLASS_NAMES.index('mel')
METHODS = ['one_minus_msp', 'predictive_entropy', 'expected_entropy', 'mutual_information', 'variation_ratio']
CONTINUOUS_METHODS = ['one_minus_msp', 'predictive_entropy', 'expected_entropy', 'mutual_information']
COVERAGES = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

BASE = Path('/kaggle/working/paper2_data')
OUT = BASE / 'grouped_mc_seed2026'
NPZ_PATH = OUT / 'paper2_grouped_mc_predictions.npz'
SPLIT_CSV = BASE / 'splits' / 'ham10000_lesion_group_split_seed2026.csv'
RESULT_JSON = OUT / 'paper2_statistical_revision.json'


def norm_probs(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, EPS, 1.0)
    return x / x.sum(axis=-1, keepdims=True)


def uncertainty_scores(mc):
    p = norm_probs(mc)
    mean_p = p.mean(axis=1)
    pred_ent = -np.sum(mean_p * np.log(mean_p), axis=1)
    exp_ent = -np.mean(np.sum(p * np.log(p), axis=2), axis=1)
    mi = np.maximum(pred_ent - exp_ent, 0.0)
    msp = 1.0 - mean_p.max(axis=1)
    votes = p.argmax(axis=2)
    vr = np.empty(len(votes), dtype=float)
    for i, row in enumerate(votes):
        vr[i] = 1.0 - np.bincount(row, minlength=mean_p.shape[1]).max() / len(row)
    return mean_p, {
        'one_minus_msp': msp,
        'predictive_entropy': pred_ent,
        'expected_entropy': exp_ent,
        'mutual_information': mi,
        'variation_ratio': vr,
    }


def aurc(score, correct):
    order = np.argsort(score, kind='mergesort')
    c = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(c) + 1)
    cov = k / len(c)
    risk = 1.0 - np.cumsum(c) / k
    return float(np.trapezoid(np.r_[risk[0], risk], np.r_[0.0, cov]))


def err_metrics(y, mean_p, score):
    pred = mean_p.argmax(1)
    err = (pred != y).astype(int)
    if len(np.unique(err)) < 2:
        return {'auroc': np.nan, 'auprc': np.nan}
    return {
        'auroc': float(roc_auc_score(err, score)),
        'auprc': float(average_precision_score(err, score)),
    }


def threshold_from_validation(score, target):
    s = np.sort(np.asarray(score, dtype=float), kind='mergesort')
    k = max(1, min(len(s), int(round(target * len(s)))))
    return float(s[k - 1])


def basic_metrics(y, mean_p):
    pred = mean_p.argmax(1)
    mel = y == MEL
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'melanoma_recall': float(np.mean(pred[mel] == MEL)) if np.any(mel) else None,
    }


def selective_metrics(y, mean_p, score, threshold):
    pred = mean_p.argmax(1)
    keep = np.asarray(score) <= threshold
    if not np.any(keep):
        return None
    yk, pk = y[keep], pred[keep]
    mel_all = y == MEL
    mel_keep = yk == MEL
    full_mel_fn = mel_all & (pred != MEL)
    retained_mel_fn = keep & full_mel_fn
    full_mel_fn_n = int(full_mel_fn.sum())
    retained_mel_fn_n = int(retained_mel_fn.sum())
    return {
        'retained_n': int(keep.sum()),
        'referred_n': int((~keep).sum()),
        'coverage': float(keep.mean()),
        'referral_rate': float((~keep).mean()),
        'accuracy': float(accuracy_score(yk, pk)),
        'macro_f1': float(f1_score(yk, pk, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(yk, pk, average='weighted', zero_division=0)),
        'melanoma_total_n': int(mel_all.sum()),
        'melanoma_retained_n': int(mel_keep.sum()),
        'melanoma_coverage': float(mel_keep.sum() / mel_all.sum()) if np.any(mel_all) else None,
        'selective_melanoma_recall': float(np.mean(pk[mel_keep] == MEL)) if np.any(mel_keep) else None,
        'full_melanoma_false_negatives': full_mel_fn_n,
        'retained_melanoma_false_negatives': retained_mel_fn_n,
        'referred_melanoma_false_negatives': int(full_mel_fn_n - retained_mel_fn_n),
        'malignant_false_negative_escape_rate': float(retained_mel_fn_n / full_mel_fn_n) if full_mel_fn_n else 0.0,
    }


def cluster_index_map(lesion_ids):
    lesion_ids = np.asarray(lesion_ids).astype(str)
    unique = np.unique(lesion_ids)
    return unique, {g: np.where(lesion_ids == g)[0] for g in unique}


def sample_cluster_indices(rng, unique_lesions, index_map):
    sampled = rng.choice(unique_lesions, size=len(unique_lesions), replace=True)
    return np.concatenate([index_map[g] for g in sampled])


def q95(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [None, None]
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def cluster_bootstrap_full_and_selective(y, mean_p, score, threshold, lesion_ids, b=B, seed=BOOT_SEED):
    unique, imap = cluster_index_map(lesion_ids)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in [
        'accuracy','macro_f1','melanoma_recall','coverage','selective_accuracy',
        'selective_macro_f1','melanoma_coverage','selective_melanoma_recall',
        'malignant_false_negative_escape_rate'
    ]}
    for _ in range(b):
        idx = sample_cluster_indices(rng, unique, imap)
        yy, pp, ss = y[idx], mean_p[idx], score[idx]
        base = basic_metrics(yy, pp)
        sel = selective_metrics(yy, pp, ss, threshold)
        vals['accuracy'].append(base['accuracy'])
        vals['macro_f1'].append(base['macro_f1'])
        vals['melanoma_recall'].append(base['melanoma_recall'])
        if sel is None:
            for k in ['coverage','selective_accuracy','selective_macro_f1','melanoma_coverage','selective_melanoma_recall','malignant_false_negative_escape_rate']:
                vals[k].append(np.nan)
        else:
            vals['coverage'].append(sel['coverage'])
            vals['selective_accuracy'].append(sel['accuracy'])
            vals['selective_macro_f1'].append(sel['macro_f1'])
            vals['melanoma_coverage'].append(sel['melanoma_coverage'])
            vals['selective_melanoma_recall'].append(sel['selective_melanoma_recall'])
            vals['malignant_false_negative_escape_rate'].append(sel['malignant_false_negative_escape_rate'])
    return {k: q95(v) for k, v in vals.items()}


def paired_cluster_deltas(y, mean_p, scores, primary, lesion_ids, b=B, seed=BOOT_SEED + 1):
    unique, imap = cluster_index_map(lesion_ids)
    rng = np.random.default_rng(seed)
    out = {}
    pred = mean_p.argmax(1)
    for other in CONTINUOUS_METHODS:
        if other == primary:
            continue
        vals = {'delta_aurc_primary_minus_other': [], 'delta_auroc_primary_minus_other': [], 'delta_auprc_primary_minus_other': []}
        for _ in range(b):
            idx = sample_cluster_indices(rng, unique, imap)
            yy = y[idx]; mp = mean_p[idx]; corr = pred[idx] == yy
            ap = aurc(scores[primary][idx], corr)
            ao = aurc(scores[other][idx], corr)
            ep = err_metrics(yy, mp, scores[primary][idx])
            eo = err_metrics(yy, mp, scores[other][idx])
            vals['delta_aurc_primary_minus_other'].append(ap - ao)
            vals['delta_auroc_primary_minus_other'].append(ep['auroc'] - eo['auroc'])
            vals['delta_auprc_primary_minus_other'].append(ep['auprc'] - eo['auprc'])
        out[other] = {
            k: {'estimate': None, 'ci95': q95(v)} for k, v in vals.items()
        }
        # point estimates from original held-out test set
        corr0 = pred == y
        ep0 = err_metrics(y, mean_p, scores[primary])
        eo0 = err_metrics(y, mean_p, scores[other])
        out[other]['delta_aurc_primary_minus_other']['estimate'] = aurc(scores[primary], corr0) - aurc(scores[other], corr0)
        out[other]['delta_auroc_primary_minus_other']['estimate'] = ep0['auroc'] - eo0['auroc']
        out[other]['delta_auprc_primary_minus_other']['estimate'] = ep0['auprc'] - eo0['auprc']
    return out


def attach_lesions(image_ids, split_df, expected_split):
    frame = split_df[split_df['split'] == expected_split][['image_id','lesion_id']].copy()
    frame['image_id'] = frame['image_id'].astype(str)
    mapper = dict(zip(frame['image_id'], frame['lesion_id'].astype(str)))
    image_ids = [str(x) for x in image_ids]
    missing = [x for x in image_ids if x not in mapper]
    if missing:
        raise KeyError(f'{len(missing)} image IDs missing lesion mapping; examples={missing[:5]}')
    return np.array([mapper[x] for x in image_ids], dtype=str)


def lesion_level_aggregate(y, mean_p, lesion_ids):
    rows_y, rows_p, rows_id = [], [], []
    for lesion in np.unique(lesion_ids):
        idx = np.where(lesion_ids == lesion)[0]
        labels = np.unique(y[idx])
        if len(labels) != 1:
            raise RuntimeError(f'Lesion {lesion} has inconsistent labels {labels.tolist()}')
        rows_y.append(int(labels[0]))
        rows_p.append(mean_p[idx].mean(axis=0))
        rows_id.append(str(lesion))
    return np.asarray(rows_y, int), np.asarray(rows_p, float), np.asarray(rows_id, str)


def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(SPLIT_CSV)
    d = np.load(NPZ_PATH, allow_pickle=False)
    split_df = pd.read_csv(SPLIT_CSV)

    val_mc = d['val_mc_probs']; val_y = d['val_labels'].astype(int)
    test_mc = d['id_mc_probs']; test_y = d['id_labels'].astype(int)
    val_ids = d['val_image_ids'].astype(str); test_ids = d['id_image_ids'].astype(str)
    val_lesions = attach_lesions(val_ids, split_df, 'validation')
    test_lesions = attach_lesions(test_ids, split_df, 'test')

    val_mean, val_scores = uncertainty_scores(val_mc)
    test_mean, test_scores = uncertainty_scores(test_mc)
    val_pred = val_mean.argmax(1); test_pred = test_mean.argmax(1)
    val_correct = val_pred == val_y; test_correct = test_pred == test_y

    # Primary score selection is performed using validation AURC only.
    validation_comparison = {}
    for m in METHODS:
        validation_comparison[m] = {
            'aurc': aurc(val_scores[m], val_correct),
            'error_detection': err_metrics(val_y, val_mean, val_scores[m]),
        }
    primary = min(CONTINUOUS_METHODS, key=lambda m: validation_comparison[m]['aurc'])

    test_comparison = {}
    for m in METHODS:
        test_comparison[m] = {
            'aurc': aurc(test_scores[m], test_correct),
            'error_detection': err_metrics(test_y, test_mean, test_scores[m]),
        }

    operating_points = {}
    for cov in COVERAGES:
        thr = threshold_from_validation(val_scores[primary], cov)
        sel = selective_metrics(test_y, test_mean, test_scores[primary], thr)
        operating_points[f'{int(round(cov*100))}%'] = {
            'target_validation_coverage': cov,
            'threshold': thr,
            'validation_actual_coverage': float(np.mean(val_scores[primary] <= thr)),
            'test': sel,
            'lesion_cluster_bootstrap_95ci': cluster_bootstrap_full_and_selective(
                test_y, test_mean, test_scores[primary], thr, test_lesions, B, BOOT_SEED + int(cov*100)
            ),
        }

    # Main 60% point kept for continuity, but it is no longer the sole analysis.
    main_thr = threshold_from_validation(val_scores[primary], 0.60)

    # Secondary lesion-level aggregation: average predictive probabilities over all images of each lesion.
    lesion_y, lesion_p, lesion_ids = lesion_level_aggregate(test_y, test_mean, test_lesions)
    lesion_primary_score = None
    if primary == 'one_minus_msp':
        lesion_primary_score = 1.0 - lesion_p.max(axis=1)
    elif primary == 'predictive_entropy':
        lesion_primary_score = -np.sum(np.clip(lesion_p, EPS, 1.0) * np.log(np.clip(lesion_p, EPS, 1.0)), axis=1)
    # Expected entropy / MI are not recoverable exactly after image aggregation from mean probabilities alone;
    # do not fabricate a lesion-level equivalent for them.

    lesion_level = {
        'n_lesions': int(len(lesion_y)),
        'classification': basic_metrics(lesion_y, lesion_p),
    }
    if lesion_primary_score is not None:
        lesion_level['primary_score_aurc'] = aurc(lesion_primary_score, lesion_p.argmax(1) == lesion_y)
        lesion_level['primary_score_error_detection'] = err_metrics(lesion_y, lesion_p, lesion_primary_score)
    else:
        lesion_level['primary_score_note'] = 'Primary score depends on within-image MC decomposition; lesion-level score not reconstructed from averaged probabilities.'

    result = {
        'design': {
            'split': 'lesion-grouped with zero lesion overlap',
            'split_seed': 2026,
            'bootstrap_unit': 'lesion cluster',
            'bootstrap_replicates': B,
            'primary_score_selected_on': 'validation AURC only',
            'thresholds_selected_on': 'validation only',
            'candidate_primary_scores': CONTINUOUS_METHODS,
            'variation_ratio_handling': 'reported comparatively but excluded from primary selection because discrete ties prevent reliable scalar-threshold fixed coverage',
            'coverage_targets': COVERAGES,
            'test_images': int(len(test_y)),
            'test_unique_lesions': int(len(np.unique(test_lesions))),
        },
        'selected_primary_uncertainty_score': primary,
        'validation_uncertainty_comparison': validation_comparison,
        'heldout_test_uncertainty_comparison': test_comparison,
        'full_test_classification': basic_metrics(test_y, test_mean),
        'operating_points': operating_points,
        'paired_lesion_cluster_bootstrap_test_differences': paired_cluster_deltas(
            test_y, test_mean, test_scores, primary, test_lesions, B, BOOT_SEED + 999
        ),
        'secondary_lesion_level_analysis': lesion_level,
        'main_60pct_threshold_for_continuity': main_thr,
    }

    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    print('PAPER2_STAT_REVISION_JSON_BEGIN')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print('PAPER2_STAT_REVISION_JSON_END')
    print('RESULT_JSON|', RESULT_JSON)


main()
