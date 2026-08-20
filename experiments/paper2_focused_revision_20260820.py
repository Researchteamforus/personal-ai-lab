import gc, hashlib, json, random, socket, tarfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE = Path('/kaggle/working/paper2_data')
IMG_DIR = BASE / 'HAM10000_images'
DL = BASE / 'downloads'
SPLIT_CSV = BASE / 'splits/ham10000_lesion_group_split_seed2026.csv'
OUTROOT = BASE / 'focused_revision_20260820'
OUTROOT.mkdir(parents=True, exist_ok=True)
DL.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

HOST = socket.gethostname()
HOST_ROLE = {
    'b44f0ce87fe6': 'crossfit_outer_0_1_2',
    '836b08d4b34d': 'crossfit_outer_3_4_augmentation',
}
ROLE = HOST_ROLE.get(HOST, 'unknown')
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
C2I = {c: i for i, c in enumerate(CLASSES)}
MEL = C2I['mel']
EPS = 1e-12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH = 32
WORKERS = 4
LR = 1e-4
EPOCHS = 20
PATIENCE = 5
MC_T = 30
INNER_FOLDS = 3
UCB_BOOT = 2000

HAM_FILES = {
    'HAM10000_images_part_1.zip': (3172585, 1366522108),
    'HAM10000_images_part_2.zip': (3172584, 1403566547),
    'HAM10000_metadata.tab': (4338392, 830428),
}

print(f'FOCUSED_REVISION_START|host={HOST}|role={ROLE}|device={DEVICE}', flush=True)
if DEVICE.type == 'cuda':
    print('GPU|' + torch.cuda.get_device_name(0), flush=True)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def norm(p):
    p = np.clip(np.asarray(p, float), EPS, None)
    return p / p.sum(-1, keepdims=True)


def entropy(p):
    p = np.clip(np.asarray(p, float), EPS, 1)
    return -(p * np.log(p)).sum(-1)


def nll(p, y):
    p = norm(p)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def brier(p, y):
    p = norm(p)
    return float(np.mean(np.sum((p - np.eye(len(CLASSES))[y]) ** 2, axis=1)))


def ece(p, y, bins=15):
    p = norm(p)
    pred = p.argmax(1)
    conf = p.max(1)
    ok = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    v = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i+1]) if i < bins - 1 else (conf <= edges[i+1]))
        if m.any():
            v += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(v)


def aurc_order(correct):
    c = np.asarray(correct, float)
    k = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / k
    cov = k / len(c)
    return float(np.trapezoid(np.r_[risk[0], risk], np.r_[0, cov]))


def aurc(score, pred, y):
    o = np.argsort(score, kind='mergesort')
    return aurc_order(pred[o] == y[o])


def eaurc(score, pred, y):
    return aurc(score, pred, y) - aurc_order(np.sort((pred == y).astype(int))[::-1])


def threshold_for_coverage(score, target=.60):
    s = np.sort(np.asarray(score), kind='mergesort')
    k = max(1, min(len(s), int(round(target * len(s)))))
    return float(s[k-1])


def metrics(p, y, score):
    p = norm(p)
    pr = p.argmax(1)
    err = (pr != y).astype(int)
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pr)),
        'macro_f1': float(f1_score(y, pr, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, pr, average='weighted', zero_division=0)),
        'melanoma_recall': float(recall_score(y == MEL, pr == MEL, zero_division=0)),
        'nll': nll(p, y),
        'brier': brier(p, y),
        'ece15': ece(p, y),
        'aurc': aurc(score, pr, y),
        'eaurc': eaurc(score, pr, y),
        'error_auroc': float(roc_auc_score(err, score)) if len(np.unique(err)) > 1 else np.nan,
        'error_auprc': float(average_precision_score(err, score)) if len(np.unique(err)) > 1 else np.nan,
    }


def safety(p, y, score, thr):
    p = norm(p)
    pr = p.argmax(1)
    keep = np.asarray(score) <= thr
    mel = y == MEL
    mk = keep & mel
    fn = mel & (pr != MEL)
    rfn = fn & keep
    rn = int(mk.sum())
    mn = int(mel.sum())
    return {
        'coverage': float(keep.mean()),
        'selective_accuracy': float(accuracy_score(y[keep], pr[keep])) if keep.any() else np.nan,
        'selective_macro_f1': float(f1_score(y[keep], pr[keep], average='macro', zero_division=0)) if keep.any() else np.nan,
        'melanoma_n': mn,
        'melanoma_coverage': float(rn / mn) if mn else np.nan,
        'retained_melanoma_n': rn,
        'retained_melanoma_fn': int(rfn.sum()),
        'retained_melanoma_sensitivity': float((mk & (pr == MEL)).sum() / rn) if rn else np.nan,
        'automatic_melanoma_miss_rate': float(rfn.sum() / mn) if mn else np.nan,
        'melanoma_fn_escape_rate': float(rfn.sum() / fn.sum()) if fn.sum() else 0.0,
    }


def download_ham():
    for name, (fid, size) in HAM_FILES.items():
        dest = DL / name
        if not dest.exists() or dest.stat().st_size != size:
            if dest.exists():
                dest.unlink()
            print('HAM_DOWNLOAD_START|' + name, flush=True)
            with requests.get(f'https://dataverse.harvard.edu/api/access/datafile/{fid}', stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                tmp = Path(str(dest) + '.part')
                if tmp.exists():
                    tmp.unlink()
                with open(tmp, 'wb') as f:
                    for ch in r.iter_content(8 * 1024 * 1024):
                        if ch:
                            f.write(ch)
                tmp.replace(dest)
            if dest.stat().st_size != size:
                raise RuntimeError(f'{name} size mismatch')
            print('HAM_DOWNLOAD_DONE|' + name + '|' + str(size), flush=True)
    for name in ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip']:
        marker = IMG_DIR / (Path(name).stem + '.extracted')
        if not marker.exists() or len(list(IMG_DIR.glob('*.jpg'))) < 10000:
            print('HAM_EXTRACT_START|' + name, flush=True)
            with zipfile.ZipFile(DL / name) as z:
                for member in z.namelist():
                    if member.lower().endswith(('.jpg', '.jpeg', '.png')):
                        with z.open(member) as src, open(IMG_DIR / Path(member).name, 'wb') as dst:
                            while True:
                                b = src.read(4 * 1024 * 1024)
                                if not b:
                                    break
                                dst.write(b)
            marker.write_text('ok')
    n = len(list(IMG_DIR.glob('*.jpg')))
    print('HAM_IMAGES|' + str(n), flush=True)
    if n < 10015:
        raise RuntimeError('HAM images incomplete')


def metadata():
    p = DL / 'HAM10000_metadata.tab'
    df = pd.read_csv(p, sep='\t')
    if len(df.columns) == 1:
        df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ['lesion_id', 'image_id', 'dx']:
        df[c] = df[c].astype(str).str.strip()
    return df


BASE_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

AUG_TFM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.90, 1.00), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class HamDS(Dataset):
    def __init__(self, df, tfm=BASE_TFM):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        from PIL import Image
        with Image.open(IMG_DIR / f'{r.image_id}.jpg') as im:
            x = self.tfm(im.convert('RGB'))
        return x, C2I[r.dx], r.image_id


def loader(df, shuffle=False, batch=BATCH, tfm=BASE_TFM):
    return DataLoader(
        HamDS(df, tfm), batch_size=batch, shuffle=shuffle,
        num_workers=WORKERS, pin_memory=True,
        persistent_workers=(WORKERS > 0),
    )


class HeadResNet50(nn.Module):
    def __init__(self, p=.2, weights=True):
        super().__init__()
        b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if weights else None)
        d = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.fc1 = nn.Linear(d, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(512, 7)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(self.backbone(x)))))


def class_weights(df):
    counts = df.dx.value_counts().reindex(CLASSES).values.astype(float)
    w = len(df) / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def save_fp16(model, path, meta=None):
    sd = {
        k: (v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    torch.save({'state_dict_fp16': sd, 'meta': meta or {}}, path)


@torch.no_grad()
def det_probs(model, ld):
    model.eval()
    ps, ys, ids = [], [], []
    for x, y, i in ld:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            z = model(x)
        ps.append(torch.softmax(z.float(), 1).cpu().numpy())
        ys.extend(y.numpy())
        ids.extend(list(i))
    return np.concatenate(ps), np.asarray(ys, int), np.asarray(ids, dtype='U20')


def enable_mc(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_probs(model, ld, T=MC_T):
    enable_mc(model)
    arr = []
    yref = None
    iref = None
    for t in range(T):
        ps, ys, ids = [], [], []
        for x, y, i in ld:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                z = model(x)
            ps.append(torch.softmax(z.float(), 1).cpu().numpy())
            ys.extend(y.numpy())
            ids.extend(list(i))
        arr.append(np.concatenate(ps))
        if yref is None:
            yref = np.asarray(ys, int)
            iref = np.asarray(ids, dtype='U20')
        if (t + 1) % 10 == 0 or t == 0:
            print(f'MC|{t+1}/{T}', flush=True)
    return np.stack(arr).transpose(1, 0, 2), yref, iref


def mc_scores(mc):
    p = norm(mc)
    mp = p.mean(1)
    pe = entropy(mp)
    ee = entropy(p).mean(1)
    return mp, {
        'pe': pe,
        'ee': ee,
        'mi': np.maximum(pe - ee, 0),
        'msp': 1 - mp.max(1),
    }


def train_one(model, tr, val, out, seed, train_tfm=BASE_TFM, epochs=EPOCHS):
    seed_all(seed)
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE)
    trld = loader(tr, True, tfm=train_tfm)
    vald = loader(val, False, tfm=BASE_TFM)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    best = -1.0
    best_epoch = 0
    best_sd = None
    hist = []
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        losses, yy, pp = [], [], []
        for x, y, _ in trld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                z = model(x)
                loss = criterion(z, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()) * len(y))
            yy.extend(y.detach().cpu().numpy())
            pp.extend(z.detach().argmax(1).cpu().numpy())
        vp, vy, _ = det_probs(model, vald)
        vf = float(f1_score(vy, vp.argmax(1), average='macro', zero_division=0))
        row = {
            'epoch': ep,
            'train_loss': sum(losses) / len(tr),
            'train_macro_f1': float(f1_score(yy, pp, average='macro', zero_division=0)),
            'val_macro_f1': vf,
            'val_accuracy': float(accuracy_score(vy, vp.argmax(1))),
        }
        hist.append(row)
        print('TRAIN|' + json.dumps(row), flush=True)
        if vf > best + 1e-6:
            best = vf
            best_epoch = ep
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if ep >= 8 and stale >= PATIENCE:
            print('EARLY_STOP|' + str(ep), flush=True)
            break
    model.load_state_dict(best_sd)
    pd.DataFrame(hist).to_csv(out / 'training_history.csv', index=False)
    save_fp16(model, out / 'best_model_fp16.pt', {'best_val_macro_f1': best, 'best_epoch': best_epoch, 'seed': seed})
    return model, best, best_epoch


def train_fixed_epochs(model, tr, out, seed, epochs, train_tfm=BASE_TFM):
    seed_all(seed)
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE)
    trld = loader(tr, True, tfm=train_tfm)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        losses, yy, pp = [], [], []
        for x, y, _ in trld:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                z = model(x)
                loss = criterion(z, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()) * len(y))
            yy.extend(y.detach().cpu().numpy())
            pp.extend(z.detach().argmax(1).cpu().numpy())
        row = {
            'epoch': ep,
            'train_loss': sum(losses) / len(tr),
            'train_macro_f1': float(f1_score(yy, pp, average='macro', zero_division=0)),
        }
        hist.append(row)
        print('FINAL_TRAIN|' + json.dumps(row), flush=True)
    pd.DataFrame(hist).to_csv(out / 'training_history.csv', index=False)
    save_fp16(model, out / 'final_model_fp16.pt', {'epochs': epochs, 'seed': seed})
    return model


def lesion_aggregate_det(p, y, ids, frame):
    pos = {str(i): j for j, i in enumerate(ids)}
    rows = []
    for lesion, g in frame.groupby('lesion_id', sort=True):
        ix = [pos[str(i)] for i in g.image_id]
        yy = np.asarray([y[j] for j in ix])
        if len(np.unique(yy)) != 1:
            raise RuntimeError('lesion label mismatch ' + str(lesion))
        rows.append((str(lesion), p[ix].mean(0), int(yy[0])))
    lesion_ids = np.asarray([r[0] for r in rows], dtype='U32')
    probs = np.stack([r[1] for r in rows])
    labels = np.asarray([r[2] for r in rows], int)
    return probs, labels, lesion_ids


def lesion_aggregate_mc(mc, y, ids, frame):
    pos = {str(i): j for j, i in enumerate(ids)}
    rows = []
    for lesion, g in frame.groupby('lesion_id', sort=True):
        ix = [pos[str(i)] for i in g.image_id]
        yy = np.asarray([y[j] for j in ix])
        if len(np.unique(yy)) != 1:
            raise RuntimeError('lesion label mismatch ' + str(lesion))
        rows.append((str(lesion), mc[ix].mean(0), int(yy[0])))
    lesion_ids = np.asarray([r[0] for r in rows], dtype='U32')
    probs = np.stack([r[1] for r in rows])
    labels = np.asarray([r[2] for r in rows], int)
    return probs, labels, lesion_ids


def fit_temperature(p, y):
    from scipy.optimize import minimize_scalar
    logp = np.log(np.clip(norm(p), EPS, 1))
    def obj(logt):
        t = np.exp(logt)
        z = logp / t
        z = z - z.max(1, keepdims=True)
        q = np.exp(z)
        q = q / q.sum(1, keepdims=True)
        return nll(q, y)
    res = minimize_scalar(obj, bounds=(np.log(.05), np.log(10.0)), method='bounded')
    return float(np.exp(res.x))


def apply_temperature(p, t):
    logp = np.log(np.clip(norm(p), EPS, 1)) / t
    logp = logp - logp.max(1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(1, keepdims=True)


def ucb_threshold_lesion(score, p, y, cap=.10, B=UCB_BOOT, seed=20260820):
    score = np.asarray(score)
    p = norm(p)
    pr = p.argmax(1)
    mel_ix = np.where(y == MEL)[0]
    if len(mel_ix) == 0:
        raise RuntimeError('no melanoma lesions')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(mel_ix), size=(B, len(mel_ix)), dtype=np.int32)
    s = score[mel_ix]
    wrong = (pr[mel_ix] != MEL).astype(np.int8)
    thresholds = np.unique(np.sort(score))
    def upper(th):
        retained_wrong = (wrong * (s <= th)).astype(np.int8)
        vals = retained_wrong[draws].mean(1)
        return float(np.quantile(vals, .95))
    lo, hi, best = 0, len(thresholds) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if upper(thresholds[mid]) <= cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best < 0:
        best = 0
    th = float(thresholds[best])
    return th, upper(th)


def select_mc_score(lesion_mc, y):
    mp, sc = mc_scores(lesion_mc)
    vals = {k: eaurc(v, mp.argmax(1), y) for k, v in sc.items()}
    best = min(vals, key=vals.get)
    return best, vals, mp, sc


def evaluation_bundle(val_frame, test_frame, val_det, val_mc, val_y, val_ids, test_det, test_mc, test_y, test_ids):
    vd_l, vy_l, vlids = lesion_aggregate_det(val_det, val_y, val_ids, val_frame)
    vm_l, _, _ = lesion_aggregate_mc(val_mc, val_y, val_ids, val_frame)
    td_l, ty_l, tlids = lesion_aggregate_det(test_det, test_y, test_ids, test_frame)
    tm_l, _, _ = lesion_aggregate_mc(test_mc, test_y, test_ids, test_frame)

    mc_name, selection_eaurc, vmp_l, vsc_l = select_mc_score(vm_l, vy_l)
    tmp_l, tsc_l = mc_scores(tm_l)
    det_score_l = 1 - vd_l.max(1)
    tdet_score_l = 1 - td_l.max(1)
    mc_score_l = vsc_l[mc_name]
    tmc_score_l = tsc_l[mc_name]

    det_60 = threshold_for_coverage(det_score_l, .60)
    mc_60 = threshold_for_coverage(mc_score_l, .60)
    det_ucb, det_u95 = ucb_threshold_lesion(det_score_l, vd_l, vy_l, cap=.10, seed=20260820)
    mc_ucb, mc_u95 = ucb_threshold_lesion(mc_score_l, vmp_l, vy_l, cap=.10, seed=20260821)

    det_t = fit_temperature(vd_l, vy_l)
    mc_t = fit_temperature(vmp_l, vy_l)
    td_l_cal = apply_temperature(td_l, det_t)
    tmp_l_cal = apply_temperature(tmp_l, mc_t)

    vmp_img, vsc_img = mc_scores(val_mc)
    tmp_img, tsc_img = mc_scores(test_mc)

    return {
        'selected_mc_score': mc_name,
        'oof_score_selection_eaurc': selection_eaurc,
        'thresholds': {
            'lesion_det_60': det_60,
            'lesion_mc_60': mc_60,
            'lesion_det_ucb10': det_ucb,
            'lesion_mc_ucb10': mc_ucb,
            'lesion_det_validation_u95': det_u95,
            'lesion_mc_validation_u95': mc_u95,
        },
        'temperatures': {'det': det_t, 'mc': mc_t},
        'image_level': {
            'deterministic': metrics(test_det, test_y, 1 - test_det.max(1)),
            'mc': metrics(tmp_img, test_y, tsc_img[mc_name]),
        },
        'lesion_level': {
            'deterministic': metrics(td_l, ty_l, tdet_score_l),
            'mc': metrics(tmp_l, ty_l, tmc_score_l),
            'deterministic_calibrated': {
                'nll': nll(td_l_cal, ty_l), 'brier': brier(td_l_cal, ty_l), 'ece15': ece(td_l_cal, ty_l)
            },
            'mc_calibrated': {
                'nll': nll(tmp_l_cal, ty_l), 'brier': brier(tmp_l_cal, ty_l), 'ece15': ece(tmp_l_cal, ty_l)
            },
            'det_60': safety(td_l, ty_l, tdet_score_l, det_60),
            'mc_60': safety(tmp_l, ty_l, tmc_score_l, mc_60),
            'det_ucb10': safety(td_l, ty_l, tdet_score_l, det_ucb),
            'mc_ucb10': safety(tmp_l, ty_l, tmc_score_l, mc_ucb),
            'n_lesions': int(len(ty_l)),
        },
        'test_lesion_ids': tlids,
    }


def run_crossfit_outer(df, fold):
    root = OUTROOT / 'crossfit' / f'outer_{fold}'
    root.mkdir(parents=True, exist_ok=True)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    splits = list(outer.split(df.image_id, df.dx, df.lesion_id))
    tv_idx, test_idx = splits[fold]
    tv = df.iloc[tv_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=7200 + fold)

    oof_det = np.zeros((len(tv), len(CLASSES)), dtype=np.float32)
    oof_mc = np.zeros((len(tv), MC_T, len(CLASSES)), dtype=np.float16)
    filled = np.zeros(len(tv), dtype=bool)
    best_epochs = []
    inner_rows = []

    for inner_fold, (tr_idx, val_idx) in enumerate(inner.split(tv.image_id, tv.dx, tv.lesion_id)):
        tr = tv.iloc[tr_idx].reset_index(drop=True)
        val = tv.iloc[val_idx].reset_index(drop=True)
        leak = len(set(tr.lesion_id) & set(val.lesion_id))
        if leak:
            raise RuntimeError(f'inner lesion leak outer={fold} inner={inner_fold}')
        idir = root / f'inner_{inner_fold}'
        seed = 12000 + fold * 100 + inner_fold
        print(f'CROSSFIT_INNER|outer={fold}|inner={inner_fold}|tr={len(tr)}|val={len(val)}|seed={seed}', flush=True)
        model, best, best_epoch = train_one(HeadResNet50(.2), tr, val, idir, seed, BASE_TFM)
        vd, vy, vids = det_probs(model, loader(val, False))
        vm, _, _ = mc_probs(model, loader(val, False), MC_T)
        pos = {str(i): j for j, i in enumerate(tv.image_id.astype(str).values)}
        place = np.asarray([pos[str(i)] for i in vids], int)
        if filled[place].any():
            raise RuntimeError('OOF overwrite')
        oof_det[place] = vd.astype(np.float32)
        oof_mc[place] = vm.astype(np.float16)
        filled[place] = True
        best_epochs.append(best_epoch)
        inner_rows.append({'inner_fold': inner_fold, 'seed': seed, 'best_val_macro_f1': best, 'best_epoch': best_epoch, 'n_train': len(tr), 'n_val': len(val)})
        del model, vd, vm
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not filled.all():
        raise RuntimeError(f'OOF incomplete outer={fold}: {filled.mean()}')

    final_epochs = int(np.clip(round(float(np.median(best_epochs))), 1, EPOCHS))
    final_seed = 13000 + fold
    fdir = root / 'final_model'
    print(f'CROSSFIT_FINAL|outer={fold}|epochs={final_epochs}|seed={final_seed}|tv={len(tv)}|test={len(test)}', flush=True)
    final_model = train_fixed_epochs(HeadResNet50(.2), tv, fdir, final_seed, final_epochs, BASE_TFM)
    td, ty, tids = det_probs(final_model, loader(test, False))
    tm, _, _ = mc_probs(final_model, loader(test, False), MC_T)

    bundle = evaluation_bundle(
        tv, test,
        oof_det, oof_mc.astype(np.float32), np.asarray([C2I[x] for x in tv.dx], int), tv.image_id.astype(str).values,
        td, tm, ty, tids,
    )
    bundle['outer_fold'] = fold
    bundle['inner_folds'] = INNER_FOLDS
    bundle['best_epochs'] = best_epochs
    bundle['final_epochs'] = final_epochs
    bundle['final_seed'] = final_seed
    bundle['oof_n_images'] = int(len(tv))
    bundle['test_n_images'] = int(len(test))
    bundle['test_n_lesions'] = int(test.lesion_id.nunique())
    test_lesion_ids = bundle.pop('test_lesion_ids')

    pd.DataFrame(inner_rows).to_csv(root / 'inner_training_summary.csv', index=False)
    pd.concat([tv.assign(split='outer_trainval'), test.assign(split='outer_test')]).to_csv(root / 'outer_split.csv', index=False)
    np.savez_compressed(
        root / 'crossfit_predictions.npz',
        oof_det=oof_det,
        oof_mc=oof_mc,
        oof_y=np.asarray([C2I[x] for x in tv.dx], int),
        oof_ids=tv.image_id.astype(str).values,
        oof_lesion_ids=tv.lesion_id.astype(str).values,
        test_det=td.astype(np.float32),
        test_mc=tm.astype(np.float16),
        test_y=ty,
        test_ids=tids,
        test_lesion_ids=np.asarray(test_lesion_ids),
    )
    (root / 'summary.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    del final_model, td, tm
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return bundle


def load_fixed_split(df):
    if SPLIT_CSV.exists():
        s = pd.read_csv(SPLIT_CSV)
    else:
        sg = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=2026)
        fold = np.empty(len(df), int)
        for f, (_, idx) in enumerate(sg.split(df.image_id, df.dx, df.lesion_id)):
            fold[idx] = f
        s = df.copy()
        s['group_fold'] = fold
        s['split'] = 'train'
        s.loc[s.group_fold == 8, 'split'] = 'validation'
        s.loc[s.group_fold == 9, 'split'] = 'test'
        SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(SPLIT_CSV, index=False)
    return s[s.split == 'train'].copy(), s[s.split == 'validation'].copy(), s[s.split == 'test'].copy()


def eval_fixed_model(model, val, test, out):
    vd, vy, vids = det_probs(model, loader(val, False))
    vm, _, _ = mc_probs(model, loader(val, False), MC_T)
    td, ty, tids = det_probs(model, loader(test, False))
    tm, _, _ = mc_probs(model, loader(test, False), MC_T)
    bundle = evaluation_bundle(val, test, vd, vm, vy, vids, td, tm, ty, tids)
    test_lesion_ids = bundle.pop('test_lesion_ids')
    np.savez_compressed(
        out / 'predictions.npz',
        val_det=vd.astype(np.float32), val_mc=vm.astype(np.float16), val_y=vy, val_ids=vids,
        test_det=td.astype(np.float32), test_mc=tm.astype(np.float16), test_y=ty, test_ids=tids,
        test_lesion_ids=np.asarray(test_lesion_ids),
    )
    return bundle


def run_augmentation_pair(df):
    tr, val, test = load_fixed_split(df)
    root = OUTROOT / 'augmentation_sensitivity'
    root.mkdir(parents=True, exist_ok=True)
    seed = 9601
    results = {}
    for name, tfm in [('no_additional_augmentation', BASE_TFM), ('conventional_augmentation', AUG_TFM)]:
        d = root / name
        print(f'AUGMENTATION_MODEL|{name}|seed={seed}', flush=True)
        model, best, best_epoch = train_one(HeadResNet50(.2), tr, val, d, seed, tfm)
        bundle = eval_fixed_model(model, val, test, d)
        bundle['best_val_macro_f1'] = best
        bundle['best_epoch'] = best_epoch
        bundle['seed'] = seed
        bundle['training_transform'] = name
        (d / 'summary.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
        results[name] = bundle
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (root / 'paired_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def archive_component(tag, path):
    archive = BASE / f'Paper2_FOCUSED_{tag}.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=2) as tf:
        tf.add(path, arcname=tag)
    h = hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive) + '.sha256').write_text(f'{h}  {archive.name}\n')
    print(f'FOCUSED_ARCHIVE|{archive}|{archive.stat().st_size}|{h}', flush=True)
    return archive


def main():
    if ROLE == 'unknown':
        raise RuntimeError('Unknown worker hostname ' + HOST)
    download_ham()
    df = metadata()
    summary = {'host': HOST, 'role': ROLE, 'started': time.time(), 'mc_T': MC_T, 'inner_folds': INNER_FOLDS, 'ucb_bootstrap_reps': UCB_BOOT}
    if ROLE == 'crossfit_outer_0_1_2':
        folds = [0, 1, 2]
        summary['crossfit'] = {}
        for f in folds:
            summary['crossfit'][str(f)] = run_crossfit_outer(df, f)
            archive_component(f'crossfit_outer_{f}', OUTROOT / 'crossfit' / f'outer_{f}')
    elif ROLE == 'crossfit_outer_3_4_augmentation':
        folds = [3, 4]
        summary['crossfit'] = {}
        for f in folds:
            summary['crossfit'][str(f)] = run_crossfit_outer(df, f)
            archive_component(f'crossfit_outer_{f}', OUTROOT / 'crossfit' / f'outer_{f}')
        summary['augmentation_sensitivity'] = run_augmentation_pair(df)
        archive_component('augmentation_sensitivity', OUTROOT / 'augmentation_sensitivity')
    summary['finished'] = time.time()
    role_dir = OUTROOT / f'role_{ROLE}'
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / 'role_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    archive_component(f'{ROLE}_summary', role_dir)
    print('FOCUSED_REVISION_DONE|' + ROLE, flush=True)


if __name__ == '__main__':
    main()
