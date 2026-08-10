import os
import sys

print('=== HAM10000 HF MIRROR SCHEMA CHECK ===')
print('cwd:', os.getcwd())
print('python:', sys.version.split()[0])

from datasets import load_dataset

try:
    ds = load_dataset('marmal88/skin_cancer')
    print('SPLITS:', list(ds.keys()))
    for split, part in ds.items():
        print(f'SPLIT|{split}|N={len(part)}|COLUMNS={part.column_names}')
        print('FEATURES|', part.features)
        if len(part):
            row = part[0]
            summary = {}
            for k, v in row.items():
                if k == 'image':
                    summary[k] = {
                        'type': type(v).__name__,
                        'size': getattr(v, 'size', None),
                        'mode': getattr(v, 'mode', None),
                    }
                else:
                    text = repr(v)
                    summary[k] = text[:300]
            print('FIRST_ROW|', summary)
except Exception as exc:
    print('LOAD_DATASET_ERROR:', type(exc).__name__, repr(exc))

print('HAM10000_SCHEMA_CHECK_DONE')
