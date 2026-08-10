import json
import os
import sys
from pathlib import Path

import requests

print('=== PAPER 2 DATA ACCESS CHECK ===')
print('cwd:', os.getcwd())
print('python:', sys.version.split()[0])

# Check direct internet access to a public HAM10000 mirror metadata endpoint.
api_url = 'https://huggingface.co/api/datasets/marmal88/skin_cancer'
try:
    r = requests.get(api_url, timeout=30)
    print('HF_API_STATUS:', r.status_code)
    if r.ok:
        data = r.json()
        print('HF_DATASET_ID:', data.get('id'))
        siblings = data.get('siblings', [])
        print('HF_FILE_COUNT:', len(siblings))
        for item in siblings[:30]:
            print('HF_FILE:', item.get('rfilename'))
except Exception as exc:
    print('HF_API_ERROR:', repr(exc))

# Check whether common data libraries are already installed.
for pkg in ['datasets', 'huggingface_hub', 'kagglehub', 'torchvision', 'pandas', 'sklearn']:
    try:
        mod = __import__(pkg)
        print(f'PKG|{pkg}|OK|{getattr(mod, "__version__", "unknown")}')
    except Exception as exc:
        print(f'PKG|{pkg}|MISSING|{type(exc).__name__}: {exc}')

# Try a very small metadata-only HTTP request from the original Harvard Dataverse landing API.
try:
    url = 'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T'
    r = requests.get(url, timeout=30)
    print('HARVARD_API_STATUS:', r.status_code)
    if r.ok:
        j = r.json()
        latest = j.get('data', {}).get('latestVersion', {})
        print('HARVARD_DATASET_VERSION:', latest.get('versionNumber'), latest.get('versionMinorNumber'))
        files = latest.get('files', [])
        print('HARVARD_FILE_COUNT:', len(files))
        for f in files[:20]:
            df = f.get('dataFile', {})
            print('HARVARD_FILE:', df.get('filename'), '|id=', df.get('id'), '|bytes=', df.get('filesize'))
except Exception as exc:
    print('HARVARD_API_ERROR:', repr(exc))

print('PAPER2_DATA_ACCESS_CHECK_DONE')
