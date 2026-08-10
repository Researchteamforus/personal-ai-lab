"""Paper 2 uncertainty, calibration, OOD, and selective-risk analysis.

Expected real-data NPZ keys when available:
  id_mc_probs: float array [N_id, T, C]
  id_labels:   int array [N_id]
  ood_mc_probs: optional float array [N_ood, T, C]
  ood_labels:   optional int array [N_ood]

The current remote run executes a synthetic smoke test so the metric pipeline can be
validated before the original Paper 2 predictions/model assets are mounted.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as exc:
    raise RuntimeError("scikit-learn is required for AUROC/AUPRC") from exc

EPS = 1e-12


def _validate_mc_probs(mc_probs):
    x = np.asarray(mc_probs, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"mc_probs must have shape [N,T,C], got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("mc_probs contains NaN/Inf")
    x = np.clip(x, EPS, 1.0)
    x = x / x.sum(axis=-1, keepdims=True)
    return x


def uncertainty_scores(mc_probs):
    """Return predictive mean and five uncertainty scores per sample."""
    p = _validate_mc_probs(mc_probs)
    mean_p = p.mean(axis=1)

    predictive_entropy = -np.sum(mean_p * np.log(mean_p), axis=1)
    expected_entropy = -np.mean(np.sum(p * np.log(p), axis=2), axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    one_minus_msp = 1.0 - np.max(mean_p, axis=1)

    votes = np.argmax(p, axis=2)
    n, t = votes.shape
    variation_ratio = np.empty(n, dtype=np.float64)
    for i in range(n):
        counts = np.bincount(votes[i], minlength=mean_p.shape[1])
        variation_ratio[i] = 1.0 - counts.max() / float(t)

    return mean_p, {
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "one_minus_msp": one_minus_msp,
        "variation_ratio": variation_ratio,
    }


def multiclass_nll(mean_p, labels):
    labels = np.asarray(labels, dtype=int)
    return float(-np.mean(np.log(np.clip(mean_p[np.arange(len(labels)), labels], EPS, 1.0))))


def multiclass_brier(mean_p, labels):
    labels = np.asarray(labels, dtype=int)
    y = np.zeros_like(mean_p)
    y[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((mean_p - y) ** 2, axis=1)))


def ece_top1(mean_p, labels, n_bins=15):
    labels = np.asarray(labels, dtype=int)
    conf = mean_p.max(axis=1)
    pred = mean_p.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def safe_binary_metrics(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }


def risk_coverage_curve(uncertainty, correct):
    """Selective error risk as lowest-uncertainty samples are retained."""
    uncertainty = np.asarray(uncertainty, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(uncertainty, kind="mergesort")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    coverage = k / float(len(c))
    risk = 1.0 - np.cumsum(c) / k
    coverage_i = np.concatenate([[0.0], coverage])
    risk_i = np.concatenate([[risk[0]], risk])
    aurc = float(np.trapezoid(risk_i, coverage_i))
    return coverage, risk, aurc


def selective_at_coverage(mean_p, labels, uncertainty, target_coverage=0.60):
    labels = np.asarray(labels, dtype=int)
    pred = mean_p.argmax(axis=1)
    n_keep = max(1, int(round(len(labels) * target_coverage)))
    keep = np.argsort(uncertainty, kind="mergesort")[:n_keep]
    retained_labels = labels[keep]
    retained_pred = pred[keep]

    out = {
        "target_coverage": float(target_coverage),
        "actual_coverage": float(len(keep) / len(labels)),
        "retained_n": int(len(keep)),
        "retained_accuracy": float(np.mean(retained_pred == retained_labels)),
        "class_metrics": {},
    }

    for cls in sorted(np.unique(labels).tolist()):
        total_cls = int(np.sum(labels == cls))
        retained_cls_mask = retained_labels == cls
        retained_cls_n = int(np.sum(retained_cls_mask))
        sensitivity = (
            float(np.mean(retained_pred[retained_cls_mask] == cls))
            if retained_cls_n else None
        )
        out["class_metrics"][str(cls)] = {
            "total_n": total_cls,
            "retained_n": retained_cls_n,
            "class_coverage": float(retained_cls_n / total_cls) if total_cls else None,
            "selective_sensitivity": sensitivity,
        }
    return out


def analyze_id(id_mc_probs, id_labels, target_coverage=0.60):
    labels = np.asarray(id_labels, dtype=int)
    mean_p, scores = uncertainty_scores(id_mc_probs)
    pred = mean_p.argmax(axis=1)
    correct = pred == labels
    error = (~correct).astype(int)

    result = {
        "n": int(len(labels)),
        "accuracy": float(correct.mean()),
        "nll": multiclass_nll(mean_p, labels),
        "brier": multiclass_brier(mean_p, labels),
        "ece15": ece_top1(mean_p, labels, 15),
        "uncertainty_methods": {},
    }

    for name, score in scores.items():
        rc_cov, rc_risk, aurc = risk_coverage_curve(score, correct)
        result["uncertainty_methods"][name] = {
            "error_detection": safe_binary_metrics(error, score),
            "aurc": aurc,
            "risk_at_60pct_nearest": float(rc_risk[np.argmin(np.abs(rc_cov - target_coverage))]),
            "selective": selective_at_coverage(mean_p, labels, score, target_coverage),
        }
    return result, mean_p, scores


def analyze_ood(id_scores, ood_mc_probs):
    _, ood_scores = uncertainty_scores(ood_mc_probs)
    result = {}
    for name in id_scores:
        score = np.concatenate([id_scores[name], ood_scores[name]])
        y = np.concatenate([
            np.zeros(len(id_scores[name]), dtype=int),
            np.ones(len(ood_scores[name]), dtype=int),
        ])
        result[name] = safe_binary_metrics(y, score)
    return result


def analyze_npz(path, target_coverage=0.60):
    data = np.load(path, allow_pickle=False)
    required = {"id_mc_probs", "id_labels"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Missing required NPZ keys: {sorted(missing)}")

    id_result, _, id_scores = analyze_id(
        data["id_mc_probs"], data["id_labels"], target_coverage=target_coverage
    )
    result = {"source": str(path), "id": id_result}
    if "ood_mc_probs" in data.files:
        result["ood_detection"] = analyze_ood(id_scores, data["ood_mc_probs"])
    return result


def synthetic_smoke_test(seed=2026):
    rng = np.random.default_rng(seed)
    n, t, c = 300, 30, 7
    labels = rng.integers(0, c, size=n)
    mc = np.empty((n, t, c), dtype=np.float64)

    hard = rng.random(n) < 0.28
    for i in range(n):
        base = np.ones(c) * (0.35 if hard[i] else 0.08)
        base[labels[i]] = 1.6 if hard[i] else 6.0
        for j in range(t):
            alpha = base.copy()
            if hard[i] and rng.random() < 0.45:
                wrong = rng.choice([k for k in range(c) if k != labels[i]])
                alpha[wrong] += 2.5
            mc[i, j] = rng.dirichlet(alpha)

    ood = np.empty((150, t, c), dtype=np.float64)
    for i in range(len(ood)):
        for j in range(t):
            alpha = rng.uniform(0.25, 0.9, size=c)
            ood[i, j] = rng.dirichlet(alpha)

    id_result, _, id_scores = analyze_id(mc, labels, target_coverage=0.60)
    return {
        "mode": "synthetic_smoke_test",
        "id": id_result,
        "ood_detection": analyze_ood(id_scores, ood),
    }


def main(argv=None):
    argv = [] if argv is None else list(argv)
    if argv:
        result = analyze_npz(Path(argv[0]))
    else:
        result = synthetic_smoke_test()

    print("PAPER2_ANALYSIS_JSON_BEGIN")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print("PAPER2_ANALYSIS_JSON_END")
    return 0


# Remote lab workers execute code via exec(), so call main explicitly at top level.
main([])
