"""Download the original HAM10000 training set from Harvard Dataverse and
create a lesion-level leakage-safe 80/10/10 split.

This is a new robustness protocol for Paper 2. It intentionally does not claim to
reconstruct the manuscript's original image-level split because the original split
indices/random seed were not available on the workers.
"""

import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path('/kaggle/working/paper2_data')
DL = BASE / 'downloads'
IMG_DIR = BASE / 'HAM10000_images'
SPLIT_DIR = BASE / 'splits'
DL.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'HAM10000_images_part_1.zip': {
        'id': 3172585,
        'size': 1366522108,
    },
    'HAM10000_images_part_2.zip': {
        'id': 3172584,
        'size': 1403566547,
    },
    'HAM10000_metadata.tab': {
        'id': 4338392,
        'size': 830428,
    },
}


def download_file(name, meta):
    dest = DL / name
    expected = int(meta['size'])
    if dest.exists() and dest.stat().st_size == expected:
        print(f'DOWNLOAD_SKIP|{name}|bytes={expected}')
        return dest
    if dest.exists():
        print(f'DOWNLOAD_RESTART|{name}|existing={dest.stat().st_size}|expected={expected}')
        dest.unlink()

    url = f"https://dataverse.harvard.edu/api/access/datafile/{meta['id']}"
    print(f'DOWNLOAD_START|{name}|expected_bytes={expected}')
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + '.part')
        if tmp.exists():
            tmp.unlink()
        written = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if written % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                        print(f'DOWNLOAD_PROGRESS|{name}|{written}/{expected}')
        tmp.replace(dest)
    actual = dest.stat().st_size
    print(f'DOWNLOAD_DONE|{name}|bytes={actual}')
    if actual != expected:
        raise RuntimeError(f'Unexpected size for {name}: {actual} != {expected}')
    return dest


def extract_zip(path):
    marker = IMG_DIR / (path.stem + '.extracted')
    if marker.exists():
        print(f'EXTRACT_SKIP|{path.name}')
        return
    print(f'EXTRACT_START|{path.name}')
    with zipfile.ZipFile(path, 'r') as zf:
        members = zf.namelist()
        print(f'ZIP_MEMBERS|{path.name}|{len(members)}')
        for member in members:
            # Flatten only image files into one directory.
            if member.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = zf.open(member)
                dest = IMG_DIR / Path(member).name
                with src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
    marker.write_text('ok\n', encoding='utf-8')
    print(f'EXTRACT_DONE|{path.name}')


def read_metadata(path):
    # Dataverse may return tab-separated metadata; fall back to comma if needed.
    df = pd.read_csv(path, sep='\t')
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    # Strip Dataverse table-export quoting/whitespace quirks.
    df.columns = [str(c).strip() for c in df.columns]
    for col in ['lesion_id', 'image_id', 'dx']:
        if col not in df.columns:
            raise KeyError(f'Missing required metadata column {col}; columns={df.columns.tolist()}')
        df[col] = df[col].astype(str).str.strip()
    return df


def make_group_split(df, seed=2026):
    # 10 grouped folds; use 8 train, 1 validation, 1 test.
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold = pd.Series(index=df.index, dtype='int64')
    X = df[['image_id']]
    y = df['dx']
    groups = df['lesion_id']
    for fold_id, (_, idx) in enumerate(sgkf.split(X, y, groups)):
        fold.iloc[idx] = fold_id
    out = df.copy()
    out['group_fold'] = fold.astype(int)
    out['split'] = 'train'
    out.loc[out['group_fold'] == 8, 'split'] = 'validation'
    out.loc[out['group_fold'] == 9, 'split'] = 'test'
    return out


def verify(split_df):
    print('METADATA_ROWS|', len(split_df))
    print('UNIQUE_IMAGES|', split_df['image_id'].nunique())
    print('UNIQUE_LESIONS|', split_df['lesion_id'].nunique())
    print('CLASS_COUNTS_TOTAL|', split_df['dx'].value_counts().sort_index().to_dict())
    print('SPLIT_COUNTS|', split_df['split'].value_counts().to_dict())
    print('SPLIT_CLASS_COUNTS|')
    print(pd.crosstab(split_df['dx'], split_df['split']).to_string())

    # Leakage check: each lesion belongs to exactly one split.
    n_splits_per_lesion = split_df.groupby('lesion_id')['split'].nunique()
    leaked = n_splits_per_lesion[n_splits_per_lesion > 1]
    print('LESION_LEAKAGE_COUNT|', len(leaked))
    if len(leaked):
        raise RuntimeError('Lesion-level leakage detected in generated split')

    # Image availability check.
    expected = set(split_df['image_id'].astype(str) + '.jpg')
    actual = {p.name for p in IMG_DIR.glob('*.jpg')}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    print('IMAGE_FILES_FOUND|', len(actual))
    print('MISSING_METADATA_IMAGES|', len(missing))
    print('EXTRA_IMAGES|', len(extra))
    if missing:
        print('MISSING_EXAMPLES|', missing[:20])
        raise RuntimeError('Some metadata images are missing after extraction')


def main():
    print('=== PREPARE ORIGINAL HAM10000 + LESION-GROUP SPLIT ===')
    paths = {name: download_file(name, meta) for name, meta in FILES.items()}
    extract_zip(paths['HAM10000_images_part_1.zip'])
    extract_zip(paths['HAM10000_images_part_2.zip'])

    df = read_metadata(paths['HAM10000_metadata.tab'])
    out = make_group_split(df, seed=2026)
    verify(out)

    split_csv = SPLIT_DIR / 'ham10000_lesion_group_split_seed2026.csv'
    out.to_csv(split_csv, index=False)
    print('SPLIT_CSV|', split_csv)
    print('PREPARE_HAM10000_DONE')


# Workers execute submitted code via exec().
main()
