"""Package all Paper 2 model/checkpoint/prediction/result artifacts before lab shutdown.

This script is intentionally stored at repository root so creating/updating it does not
trigger the current Paper 2 GitHub Actions workflow (whose push paths are limited to
workflow, github_dispatch.py, and experiments/**).

It scans /kaggle/working/paper2_data recursively, writes a SHA-256 manifest, and creates
separate TAR archives for logical result directories. It does not delete or move the
original files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

BASE = Path('/kaggle/working/paper2_data')
EXPORT = Path('/kaggle/working/paper2_exports')
EXPORT.mkdir(parents=True, exist_ok=True)

# Preserve everything that can materially reproduce the analysis, while avoiding the
# raw HAM10000 image directory (the original dataset is publicly recoverable and huge).
KEEP_SUFFIXES = {
    '.pt', '.pth', '.ckpt', '.npz', '.npy', '.json', '.csv', '.tsv', '.txt',
    '.png', '.jpg', '.jpeg', '.pdf', '.svg', '.log', '.yaml', '.yml',
}
EXCLUDE_DIR_NAMES = {'HAM10000_images', '__pycache__'}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def wanted(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return False
    return path.suffix.lower() in KEEP_SUFFIXES


def safe_name(rel: Path) -> str:
    s = '__'.join(rel.parts) if rel.parts else 'root'
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)


def build_manifest(files: list[Path]) -> dict:
    rows = []
    total = 0
    for p in files:
        size = p.stat().st_size
        total += size
        rows.append({
            'relative_path': str(p.relative_to(BASE)),
            'size_bytes': int(size),
            'sha256': sha256_file(p),
            'modified_unix': float(p.stat().st_mtime),
        })
    return {
        'base': str(BASE),
        'created_unix': time.time(),
        'hostname': os.uname().nodename,
        'file_count': len(rows),
        'total_bytes': int(total),
        'files': rows,
    }


def group_files(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in files:
        rel = p.relative_to(BASE)
        # Keep each top-level result directory separately so archives remain manageable.
        key = rel.parts[0] if len(rel.parts) > 1 else 'root_files'
        groups.setdefault(key, []).append(p)
    return groups


def make_tar(name: str, files: list[Path]) -> Path:
    out = EXPORT / f'{safe_name(Path(name))}.tar'
    with tarfile.open(out, mode='w') as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(BASE)), recursive=False)
    return out


def main():
    if not BASE.exists():
        raise FileNotFoundError(BASE)

    files = sorted(p for p in BASE.rglob('*') if wanted(p))
    if not files:
        raise RuntimeError(f'No Paper 2 artifacts found under {BASE}')

    manifest = build_manifest(files)
    manifest_path = EXPORT / 'paper2_artifact_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')

    archives = []
    for group, group_paths in sorted(group_files(files).items()):
        archive = make_tar(group, group_paths)
        archives.append({
            'group': group,
            'path': str(archive),
            'size_bytes': archive.stat().st_size,
            'sha256': sha256_file(archive),
            'file_count': len(group_paths),
        })
        print('BACKUP_ARCHIVE|' + json.dumps(archives[-1], sort_keys=True), flush=True)

    index = {
        'manifest_path': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'archives': archives,
        'source_file_count': manifest['file_count'],
        'source_total_bytes': manifest['total_bytes'],
    }
    index_path = EXPORT / 'paper2_backup_index.json'
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding='utf-8')

    print('PAPER2_BACKUP_INDEX_BEGIN', flush=True)
    print(json.dumps(index, indent=2, sort_keys=True), flush=True)
    print('PAPER2_BACKUP_INDEX_END', flush=True)
    print('PAPER2_BACKUP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
